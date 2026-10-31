from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import ipaddress
import json
import logging
import re
import threading
import time
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import jwt
from jwt import PyJWK
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.index import rebuild_index
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.rag.retrieval import retrieve


logger = logging.getLogger(__name__)
_ALLOWED_JWT_ALGORITHMS = frozenset(
    {"RS256", "RS384", "RS512", "PS256", "ES256", "ES384", "EdDSA"}
)
_DISCOVERY_TTL_SECONDS = 300
_MAX_METADATA_BYTES = 1_000_000
_SCOPE_PATTERN = re.compile(r"^[\x21\x23-\x5B\x5D-\x7E]+$")


class MCPSettings(BaseSettings):
    """Settings for the optional remote MCP search endpoint."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MCP_",
        env_ignore_empty=True,
        extra="ignore",
    )

    issuer_url: str
    resource_server_url: str
    read_scope: str = "notes:read"
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    allowed_hosts: str | None = None

    @field_validator("issuer_url", "resource_server_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("MCP issuer and resource URLs must be valid HTTP URLs")
        if parsed.scheme != "https" and not _is_loopback(parsed.hostname):
            raise ValueError("MCP issuer and resource URLs must use HTTPS off loopback")
        return value

    @field_validator("resource_server_url")
    @classmethod
    def validate_resource_endpoint(cls, value: str) -> str:
        if urlsplit(value).path.rstrip("/") != "/mcp":
            raise ValueError("MCP_RESOURCE_SERVER_URL must identify the /mcp endpoint")
        return value.rstrip("/")

    @field_validator("read_scope")
    @classmethod
    def validate_read_scope(cls, value: str) -> str:
        if not _SCOPE_PATTERN.fullmatch(value):
            raise ValueError("MCP_READ_SCOPE must be one valid OAuth scope token")
        return value

    @model_validator(mode="after")
    def validate_listener(self) -> MCPSettings:
        if not _is_loopback(self.host) and not self.allowed_hosts_list:
            raise ValueError(
                "MCP_ALLOWED_HOSTS is required when MCP_HOST is not loopback"
            )
        return self

    @property
    def allowed_hosts_list(self) -> list[str]:
        if not self.allowed_hosts:
            return []
        hosts = [host.strip() for host in self.allowed_hosts.split(",") if host.strip()]
        if any("*" in host for host in hosts):
            raise ValueError("MCP_ALLOWED_HOSTS cannot contain a wildcard")
        return hosts


class RankingProvenance(BaseModel):
    final_rank: int = Field(ge=1)
    rrf_rank: int | None = Field(default=None, ge=1)
    rerank_rank: int | None = Field(default=None, ge=1)
    vector_rank: int | None = Field(default=None, ge=1)
    vector_score: float | None = Field(default=None, allow_inf_nan=False)
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = Field(default=None, allow_inf_nan=False)
    fusion_score: float | None = Field(default=None, allow_inf_nan=False)
    reranker_score: float | None = Field(default=None, allow_inf_nan=False)


class NoteSearchResult(BaseModel):
    text: str
    source: str
    chunk_id: int | str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    ranking: RankingProvenance

    @model_validator(mode="after")
    def validate_line_range(self) -> NoteSearchResult:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class SearchNotesResponse(BaseModel):
    query: str
    top_k: int = Field(ge=1)
    results: list[NoteSearchResult]


@dataclass(frozen=True)
class _CachedDocument:
    body: dict[str, Any]
    fetched_at: float


class OIDCJwksTokenVerifier(TokenVerifier):
    """Verify issuer-signed JWT access tokens through OIDC discovery and JWKS."""

    def __init__(
        self,
        *,
        issuer_url: str,
        resource_server_url: str,
        fetch_json: Callable[[str], dict[str, Any]] | None = None,
        cache_ttl_seconds: int = _DISCOVERY_TTL_SECONDS,
    ) -> None:
        self.issuer_url = issuer_url
        self.resource_server_url = resource_server_url.rstrip("/")
        self._fetch_json = fetch_json or _fetch_json
        self._cache_ttl_seconds = cache_ttl_seconds
        self._discovery: _CachedDocument | None = None
        self._jwks: dict[str, _CachedDocument] = {}
        self._cache_lock = threading.RLock()

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            return await asyncio.to_thread(self._verify_token, token)
        except Exception as exc:
            # Authentication failures must fail closed; token contents are never logged.
            logger.info("MCP bearer token rejected (%s)", type(exc).__name__)
            return None

    def _verify_token(self, token: str) -> AccessToken | None:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg")
        key_id = header.get("kid")
        if algorithm not in _ALLOWED_JWT_ALGORITHMS or not isinstance(key_id, str):
            return None

        jwk = self._find_signing_key(key_id)
        if jwk is None or jwk.get("alg") not in (None, algorithm):
            return None
        if jwk.get("use") not in (None, "sig"):
            return None
        if "key_ops" in jwk and "verify" not in jwk["key_ops"]:
            return None

        signing_key = PyJWK.from_dict(jwk, algorithm=algorithm).key
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=[algorithm],
            issuer=self.issuer_url,
            audience=self.resource_server_url,
            options={"require": ["exp", "iss", "aud"]},
            leeway=30,
        )

        raw_scopes = claims.get("scope", claims.get("scp", []))
        if isinstance(raw_scopes, str):
            scopes = raw_scopes.split()
        elif isinstance(raw_scopes, list) and all(
            isinstance(scope, str) for scope in raw_scopes
        ):
            scopes = raw_scopes
        else:
            return None
        client_id = claims.get("client_id") or claims.get("azp") or claims.get("sub")
        if not isinstance(client_id, str) or not client_id:
            return None

        expires_at = claims.get("exp")
        if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
            return None

        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(expires_at),
            resource=self.resource_server_url,
            subject=claims.get("sub"),
            claims=claims,
        )

    def _find_signing_key(self, key_id: str) -> dict[str, Any] | None:
        discovery = self._get_discovery()
        jwks_uri = discovery.get("jwks_uri")
        if not isinstance(jwks_uri, str):
            raise ValueError("OIDC discovery metadata has no JWKS URI")
        _validate_metadata_url(jwks_uri)
        if (
            urlsplit(jwks_uri).scheme != "https"
            and urlsplit(self.issuer_url).scheme != "http"
        ):
            raise ValueError("Remote issuers must publish JWKS over HTTPS")

        jwks = self._get_jwks(jwks_uri, refresh=False)
        found = _matching_jwk(jwks, key_id)
        if found is not None:
            return found

        # A single refresh permits standards-based key rotation without accepting an
        # unknown key or retaining stale keys after the issuer removes them.
        jwks = self._get_jwks(jwks_uri, refresh=True)
        return _matching_jwk(jwks, key_id)

    def _get_discovery(self) -> dict[str, Any]:
        with self._cache_lock:
            now = time.monotonic()
            cached = self._discovery
            if cached and now - cached.fetched_at < self._cache_ttl_seconds:
                return cached.body

            metadata = self._fetch_json(
                f"{self.issuer_url.rstrip('/')}/.well-known/openid-configuration"
            )
            if metadata.get("issuer") != self.issuer_url:
                raise ValueError("OIDC issuer metadata does not match configured issuer")
            if not isinstance(metadata.get("jwks_uri"), str):
                raise ValueError("OIDC issuer metadata has no JWKS URI")
            self._discovery = _CachedDocument(metadata, now)
            return metadata

    def _get_jwks(self, jwks_uri: str, *, refresh: bool) -> dict[str, Any]:
        with self._cache_lock:
            now = time.monotonic()
            cached = self._jwks.get(jwks_uri)
            if (
                not refresh
                and cached
                and now - cached.fetched_at < self._cache_ttl_seconds
            ):
                return cached.body

            jwks = self._fetch_json(jwks_uri)
            if not isinstance(jwks.get("keys"), list):
                raise ValueError("JWKS document has no keys array")
            self._jwks[jwks_uri] = _CachedDocument(jwks, now)
            return jwks


def create_mcp_server(
    *,
    search: Callable[[str], Sequence[Mapping[str, Any]]],
    settings: MCPSettings,
    max_chunks: int,
    token_verifier: TokenVerifier | None = None,
) -> MCPServer:
    """Build the one-tool MCP server; authentication is always enabled."""
    if max_chunks < 1:
        raise ValueError("max_chunks must be positive")

    verifier = token_verifier or OIDCJwksTokenVerifier(
        issuer_url=settings.issuer_url,
        resource_server_url=settings.resource_server_url,
    )
    auth = AuthSettings(
        issuer_url=settings.issuer_url,
        resource_server_url=settings.resource_server_url,
        required_scopes=[settings.read_scope],
        # The verifier checks aud against resource_server_url itself. This supports
        # standard issuer-specific API audiences as well as RFC 8707 resource URLs.
        validate_token_resource=False,
    )
    server = MCPServer(
        "RAG Notes Helper",
        version="0.2.0",
        instructions=(
            "Search the configured local note corpus and return ranked evidence. "
            "This server does not generate answers or expose filesystem access."
        ),
        token_verifier=verifier,
        auth=auth,
    )
    search_lock = threading.Lock()

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def search_notes(
        query: str,
        top_k: int | None = None,
        ctx: Context | None = None,
    ) -> SearchNotesResponse:
        """Search local notes and return the final ranked evidence chunks."""
        _reject_unexpected_arguments(ctx)
        normalized_query = query.strip()
        if not normalized_query:
            raise ToolError("query must not be empty")
        if len(normalized_query) > 2_000:
            raise ToolError("query exceeds the 2000 character limit")
        if isinstance(top_k, bool) or (top_k is not None and top_k < 1):
            raise ToolError("top_k must be a positive integer")

        limit = max_chunks if top_k is None else min(top_k, max_chunks)
        with search_lock:
            candidates = search(normalized_query)
        results: list[NoteSearchResult] = []
        for final_rank, candidate in enumerate(candidates, start=1):
            if final_rank > limit:
                break
            try:
                results.append(_to_public_result(candidate, final_rank))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("MCP search metadata rejected (%s)", type(exc).__name__)
                raise ToolError("Search result contained invalid note metadata") from None

        return SearchNotesResponse(
            query=normalized_query,
            top_k=limit,
            results=results,
        )

    return server


def _to_public_result(
    candidate: Mapping[str, Any], final_rank: int
) -> NoteSearchResult:
    source = _relative_source(candidate["source"])
    text = candidate["text"]
    if not isinstance(text, str):
        raise TypeError("chunk text must be a string")

    return NoteSearchResult(
        text=text,
        source=source,
        chunk_id=candidate["chunk_id"],
        start_line=candidate["start_line"],
        end_line=candidate["end_line"],
        ranking=RankingProvenance(
            final_rank=final_rank,
            rrf_rank=candidate.get("rank"),
            rerank_rank=candidate.get("rerank_rank"),
            vector_rank=candidate.get("vector_rank"),
            vector_score=candidate.get("vector_score"),
            bm25_rank=candidate.get("bm25_rank"),
            bm25_score=candidate.get("bm25_score"),
            fusion_score=candidate.get("fusion_score"),
            reranker_score=candidate.get("reranker_score"),
        ),
    )


def _reject_unexpected_arguments(ctx: Context | None) -> None:
    if ctx is None:
        return
    raw_params = ctx.request_context.params
    if not isinstance(raw_params, Mapping):
        return
    if set(raw_params) - {
        "name",
        "arguments",
        "task",
        "_meta",
        "inputResponses",
        "requestState",
    }:
        raise ToolError("search_notes accepts only query and top_k")
    raw_arguments = raw_params.get("arguments")
    if isinstance(raw_arguments, Mapping):
        if set(raw_arguments) - {"query", "top_k"}:
            raise ToolError("search_notes accepts only query and top_k")


def _relative_source(source: Any) -> str:
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty relative path")
    normalized = source.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(source)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or not posix_path.parts
    ):
        raise ValueError("source must be a non-empty relative path")
    return posix_path.as_posix()


def _matching_jwk(jwks: Mapping[str, Any], key_id: str) -> dict[str, Any] | None:
    for key in jwks.get("keys", []):
        if isinstance(key, dict) and key.get("kid") == key_id:
            return key
    return None


def _fetch_json(url: str) -> dict[str, Any]:
    _validate_metadata_url(url)
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=5) as response:
            final_url = response.geturl()
            _validate_metadata_url(final_url)
            if (
                urlsplit(url).scheme == "https"
                and urlsplit(final_url).scheme != "https"
            ):
                raise ValueError("Issuer metadata redirect downgraded HTTPS")
            body = response.read(_MAX_METADATA_BYTES + 1)
    except (OSError, URLError) as exc:
        raise ValueError("Unable to fetch trusted issuer metadata") from exc
    if len(body) > _MAX_METADATA_BYTES:
        raise ValueError("Issuer metadata response exceeds the size limit")
    document = json.loads(body)
    if not isinstance(document, dict):
        raise ValueError("Issuer metadata must be a JSON object")
    return document


def _validate_metadata_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("Issuer metadata URL must be a valid HTTP URL")
    if parsed.scheme != "https" and not _is_loopback(parsed.hostname):
        raise ValueError("Issuer metadata URL must use HTTPS off loopback")


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", maxsplit=1)[0]).is_loopback
    except ValueError:
        return False


def _create_runtime_server(
    settings: MCPSettings,
) -> tuple[MCPServer, MetaStore]:
    runtime_settings = get_settings()
    rag = rebuild_index()
    meta_store = MetaStore()

    try:
        server = create_mcp_server(
            search=lambda query: retrieve(rag, meta_store, query=query),
            settings=settings,
            max_chunks=runtime_settings.llm.max_chunks,
        )
    except Exception:
        meta_store.close()
        raise
    return server, meta_store


def main() -> None:
    settings = MCPSettings()
    server, meta_store = _create_runtime_server(settings)
    transport_security = (
        TransportSecuritySettings(allowed_hosts=settings.allowed_hosts_list)
        if settings.allowed_hosts_list
        else None
    )
    try:
        server.run(
            transport="streamable-http",
            host=settings.host,
            port=settings.port,
            transport_security=transport_security,
        )
    finally:
        meta_store.close()


if __name__ == "__main__":
    main()

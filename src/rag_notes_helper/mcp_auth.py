from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import ipaddress
import json
import logging
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import jwt
from jwt import PyJWK
from mcp.server.auth.provider import AccessToken, TokenVerifier


logger = logging.getLogger(__name__)
_ALLOWED_JWT_ALGORITHMS = frozenset(
    {"RS256", "RS384", "RS512", "PS256", "ES256", "ES384", "EdDSA"}
)
_DISCOVERY_TTL_SECONDS = 300
_MAX_METADATA_BYTES = 1_000_000


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


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", maxsplit=1)[0]).is_loopback
    except ValueError:
        return False


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
    if parsed.scheme != "https" and not is_loopback_host(parsed.hostname):
        raise ValueError("Issuer metadata URL must use HTTPS off loopback")

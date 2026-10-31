from __future__ import annotations

import json
from pathlib import Path
import tomllib
import time
from types import SimpleNamespace
from typing import Any

import anyio
import httpx2
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.transport_security import TransportSecuritySettings

from rag_notes_helper.mcp_server import (
    MCPSettings,
    OIDCJwksTokenVerifier,
    _create_runtime_server,
    create_mcp_server,
    main,
)


ISSUER = "https://identity.example.test"
RESOURCE = "https://notes.example.test/mcp"


def test_mcp_console_entry_point_is_registered_and_resolvable():
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    assert project["project"]["scripts"]["rag-mcp-server"] == (
        "rag_notes_helper.mcp_server:main"
    )
    assert callable(main)


def _test_signing_keys() -> tuple[Any, dict[str, Any]]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk.update({"kid": "test-key", "alg": "RS256", "use": "sig"})
    return private_key, public_jwk


def _token(private_key: Any, **claims: Any) -> str:
    issued_at = int(time.time())
    return jwt.encode(
        {
            "iss": ISSUER,
            "sub": "user-123",
            "aud": RESOURCE,
            "exp": issued_at + 3_600,
            "iat": issued_at,
            "scope": "notes:read",
            **claims,
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )


def _verifier(public_jwk: dict[str, Any]) -> OIDCJwksTokenVerifier:
    def fetch_json(url: str) -> dict[str, Any]:
        if url == f"{ISSUER}/.well-known/openid-configuration":
            return {"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"}
        if url == f"{ISSUER}/jwks":
            return {"keys": [public_jwk]}
        raise AssertionError(f"unexpected metadata URL: {url}")

    return OIDCJwksTokenVerifier(
        issuer_url=ISSUER,
        resource_server_url=RESOURCE,
        fetch_json=fetch_json,
    )


def _settings() -> MCPSettings:
    return MCPSettings(
        _env_file=None,
        issuer_url=ISSUER,
        resource_server_url=RESOURCE,
    )


def _candidate(**overrides: Any) -> dict[str, Any]:
    return {
        "doc_id": "doc-1",
        "chunk_id": 7,
        "source": "projects/launch-plan.md",
        "text": "The launch is scheduled for November.",
        "start_line": 14,
        "end_line": 16,
        "rank": 2,
        "vector_rank": 3,
        "vector_score": 0.83,
        "bm25_rank": 1,
        "bm25_score": 1.25,
        "fusion_score": 0.032,
        "reranker_score": 0.97,
        "rerank_rank": 1,
        **overrides,
    }


def test_mcp_tool_contract_uses_read_only_hybrid_search_and_bounds_top_k():
    calls: list[str] = []
    candidates = [_candidate(), _candidate(chunk_id=8, rank=1, rerank_rank=2)]

    def search(query: str) -> list[dict[str, Any]]:
        calls.append(query)
        return candidates

    server = create_mcp_server(
        search=search,
        settings=_settings(),
        max_chunks=1,
        token_verifier=_verifier(_test_signing_keys()[1]),
    )

    async def exercise() -> None:
        async with Client(server) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools.tools] == ["search_notes"]
            tool = tools.tools[0]
            assert set(tool.input_schema["properties"]) == {"query", "top_k"}
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is True

            result = await client.call_tool(
                "search_notes",
                {"query": "  launch date?  ", "top_k": 25},
            )

        assert result.is_error is False
        assert calls == ["launch date?"]
        assert result.structured_content == {
            "query": "launch date?",
            "top_k": 1,
            "results": [
                {
                    "text": "The launch is scheduled for November.",
                    "source": "projects/launch-plan.md",
                    "chunk_id": 7,
                    "start_line": 14,
                    "end_line": 16,
                    "ranking": {
                        "final_rank": 1,
                        "rrf_rank": 2,
                        "rerank_rank": 1,
                        "vector_rank": 3,
                        "vector_score": 0.83,
                        "bm25_rank": 1,
                        "bm25_score": 1.25,
                        "fusion_score": 0.032,
                        "reranker_score": 0.97,
                    },
                }
            ],
        }

    anyio.run(exercise)


def test_search_notes_defaults_top_k_and_rejects_paths_or_nonpositive_limits():
    calls: list[str] = []

    def search(query: str) -> list[dict[str, Any]]:
        calls.append(query)
        return [_candidate()]

    server = create_mcp_server(
        search=search,
        settings=_settings(),
        max_chunks=5,
        token_verifier=_verifier(_test_signing_keys()[1]),
    )

    async def exercise() -> None:
        async with Client(server) as client:
            default_result = await client.call_tool(
                "search_notes", {"query": "launch"}
            )
            path_result = await client.call_tool(
                "search_notes",
                {"query": "launch", "path": "/etc/passwd"},
            )
            zero_result = await client.call_tool(
                "search_notes", {"query": "launch", "top_k": 0}
            )

        assert default_result.structured_content["top_k"] == 5
        assert path_result.is_error is True
        assert zero_result.is_error is True
        assert calls == ["launch"]

    anyio.run(exercise)


@pytest.mark.parametrize(
    "claims",
    [
        {"iss": "https://wrong.example.test"},
        {"aud": "https://other.example.test/mcp"},
        {"exp": 1},
    ],
)
def test_jwks_verifier_fails_closed_on_invalid_claims(claims: dict[str, Any]):
    private_key, public_jwk = _test_signing_keys()
    verifier = _verifier(public_jwk)

    async def verify() -> None:
        assert await verifier.verify_token(_token(private_key, **claims)) is None

    anyio.run(verify)


def test_jwks_verifier_rejects_unknown_signing_keys_and_invalid_signatures():
    trusted_private_key, trusted_jwk = _test_signing_keys()
    untrusted_private_key, _ = _test_signing_keys()
    verifier = _verifier(trusted_jwk)

    async def verify() -> None:
        assert await verifier.verify_token(_token(untrusted_private_key)) is None
        assert await verifier.verify_token(_token(trusted_private_key)) is not None

    anyio.run(verify)


def test_mcp_settings_require_remote_https_and_host_allowlist():
    with pytest.raises(ValueError, match="must use HTTPS off loopback"):
        MCPSettings(
            _env_file=None,
            issuer_url="http://identity.example.test",
            resource_server_url=RESOURCE,
        )

    with pytest.raises(ValueError, match="MCP_ALLOWED_HOSTS is required"):
        MCPSettings(
            _env_file=None,
            issuer_url=ISSUER,
            resource_server_url=RESOURCE,
            host="0.0.0.0",
        )

    with pytest.raises(ValueError, match="cannot contain a wildcard"):
        MCPSettings(
            _env_file=None,
            issuer_url=ISSUER,
            resource_server_url=RESOURCE,
            host="0.0.0.0",
            allowed_hosts="notes.example.test,*",
        )


def test_runtime_server_wires_local_index_to_shared_hybrid_retrieval(monkeypatch):
    rag_index = object()

    class FakeMetaStore:
        closed = False

        def close(self) -> None:
            self.closed = True

    meta_store = FakeMetaStore()
    retrieve_calls: list[tuple[Any, Any, str]] = []

    def fake_retrieve(rag: Any, metadata: Any, *, query: str):
        retrieve_calls.append((rag, metadata, query))
        return [_candidate()]

    monkeypatch.setattr(
        "rag_notes_helper.mcp_server.get_settings",
        lambda: SimpleNamespace(llm=SimpleNamespace(max_chunks=3)),
    )
    monkeypatch.setattr(
        "rag_notes_helper.mcp_server.rebuild_index", lambda: rag_index
    )
    monkeypatch.setattr(
        "rag_notes_helper.mcp_server.MetaStore", lambda: meta_store
    )
    monkeypatch.setattr("rag_notes_helper.mcp_server.retrieve", fake_retrieve)

    server, opened_meta_store = _create_runtime_server(_settings())

    async def exercise() -> None:
        async with Client(server) as client:
            result = await client.call_tool("search_notes", {"query": "launch"})
        assert result.is_error is False
        assert result.structured_content["top_k"] == 3

    anyio.run(exercise)
    assert opened_meta_store is meta_store
    assert retrieve_calls == [(rag_index, meta_store, "launch")]
    opened_meta_store.close()
    assert meta_store.closed is True


def test_streamable_http_enforces_bearer_token_and_scope_at_transport_boundary():
    private_key, public_jwk = _test_signing_keys()
    verifier = _verifier(public_jwk)
    search_calls: list[str] = []

    def search(query: str) -> list[dict[str, Any]]:
        search_calls.append(query)
        return [_candidate()]

    server = create_mcp_server(
        search=search,
        settings=_settings(),
        max_chunks=5,
        token_verifier=verifier,
    )
    app = server.streamable_http_app(
        transport_security=TransportSecuritySettings(allowed_hosts=["testserver"])
    )
    endpoint = "http://testserver/mcp"

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                base_url="http://testserver",
            ) as http:
                request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                }
                missing = await http.post("/mcp", json=request)
                invalid = await http.post(
                    "/mcp",
                    json=request,
                    headers={"Authorization": "Bearer not-a-jwt"},
                )
                no_scope = await http.post(
                    "/mcp",
                    json=request,
                    headers={
                        "Authorization": "Bearer "
                        + _token(private_key, scope="notes:write")
                    },
                )
                assert missing.status_code == 401
                assert invalid.status_code == 401
                assert no_scope.status_code == 403
                metadata = await http.get(
                    "/.well-known/oauth-protected-resource/mcp"
                )
                assert metadata.status_code == 200
                assert metadata.json()["authorization_servers"] == [ISSUER]
                assert metadata.json()["scopes_supported"] == ["notes:read"]

                http.headers["Authorization"] = (
                    "Bearer " + _token(private_key)
                )
                transport = streamable_http_client(endpoint, http_client=http)
                async with Client(transport) as client:
                    tools = await client.list_tools()
                    result = await client.call_tool(
                        "search_notes", {"query": "launch"}
                    )
                    path_result = await client.call_tool(
                        "search_notes",
                        {"query": "launch", "path": "/etc/passwd"},
                    )

                assert [tool.name for tool in tools.tools] == ["search_notes"]
                assert result.is_error is False
                assert path_result.is_error is True
                assert search_calls == ["launch"]
                assert result.structured_content["results"][0]["source"] == (
                    "projects/launch-plan.md"
                )

    anyio.run(exercise)


@pytest.mark.parametrize("source", ["/etc/passwd", "../outside.md", "C:\\secret.md"])
def test_search_results_with_non_relative_sources_fail_closed(source: str):
    server = create_mcp_server(
        search=lambda query: [_candidate(source=source)],
        settings=_settings(),
        max_chunks=5,
        token_verifier=_verifier(_test_signing_keys()[1]),
    )

    async def exercise() -> None:
        async with Client(server) as client:
            result = await client.call_tool("search_notes", {"query": "launch"})
        assert result.is_error is True
        assert "etc/passwd" not in str(result.structured_content)

    anyio.run(exercise)

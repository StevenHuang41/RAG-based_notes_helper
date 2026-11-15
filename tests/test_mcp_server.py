from typing import Any

import anyio
from mcp import Client

from rag_notes_helper.mcp_server import create_mcp_server


def candidate(i: int = 1) -> dict[str, Any]:
    return {"text": f"Evidence {i}", "source": "notes/example.md", "chunk_id": i,
            "start_line": 1, "end_line": 2}


def test_local_tool_returns_ranked_evidence_with_default_limit():
    server = create_mcp_server(search=lambda query: [candidate(i) for i in range(1, 8)])

    async def exercise() -> None:
        async with Client(server) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools.tools] == ["search_notes"]
            assert tools.tools[0].annotations.read_only_hint is True
            result = await client.call_tool("search_notes", {"query": "  evidence "})
            assert not result.is_error
            assert result.structured_content["query"] == "evidence"
            assert result.structured_content["top_k"] == 5
            assert len(result.structured_content["results"]) == 5
            assert result.structured_content["results"][0]["source"] == "notes/example.md"
            assert result.structured_content["results"][0]["ranking"]["final_rank"] == 1

    anyio.run(exercise)


def test_limits_validation_and_empty_results():
    server = create_mcp_server(search=lambda query: [] if query == "empty" else [candidate(i) for i in range(1, 26)])

    async def exercise() -> None:
        async with Client(server) as client:
            for requested, applied in [(3, 3), (20, 20), (99, 20)]:
                result = await client.call_tool("search_notes", {"query": "evidence", "top_k": requested})
                assert not result.is_error
                assert result.structured_content["top_k"] == applied
                assert len(result.structured_content["results"]) == applied
            empty = await client.call_tool("search_notes", {"query": "empty"})
            assert not empty.is_error and empty.structured_content["results"] == []
            for arguments in [{"query": " "}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": True},
                              {"query": "x", "top_k": "3"},
                              {"query": "x", "path": "/etc/passwd"}]:
                assert (await client.call_tool("search_notes", arguments)).is_error

    anyio.run(exercise)


def test_errors_and_unsafe_sources_are_not_empty_results():
    async def exercise() -> None:
        for search in [lambda query: [candidate() | {"source": "../secret"}],
                       lambda query: (_ for _ in ()).throw(RuntimeError("search failed"))]:
            async with Client(create_mcp_server(search=search)) as client:
                result = await client.call_tool("search_notes", {"query": "x"})
                assert result.is_error

    anyio.run(exercise)


def test_stdio_process_initializes_search_and_closes_resources(tmp_path):
    import os
    import sys
    from mcp.client.stdio import StdioServerParameters

    marker = tmp_path / "closed"
    script = tmp_path / "server.py"
    script.write_text('''\
from pathlib import Path
import sys
import rag_notes_helper.mcp_server as server
class Meta:
    def close(self):
        Path(sys.argv[1]).write_text("closed")
def rebuild():
    print("index refreshed")
    return object()
server.rebuild_index = rebuild
server.MetaStore = Meta
server.retrieve = lambda rag, metadata, query: [{"text": "Result", "source": "note.md", "chunk_id": 1, "start_line": 1, "end_line": 2}]
server.main()
''')
    params = StdioServerParameters(command=sys.executable, args=[str(script), str(marker)], env={**os.environ, "PYTHONPATH": os.path.abspath("src")})

    async def exercise() -> None:
        async with Client(params) as client:
            assert [tool.name for tool in (await client.list_tools()).tools] == ["search_notes"]
            result = await client.call_tool("search_notes", {"query": "find"})
            assert not result.is_error
            assert result.structured_content["results"][0]["text"] == "Result"

    anyio.run(exercise)
    assert marker.read_text() == "closed"


def test_search_returns_complete_chunk_and_ranking_provenance():
    hit = {
        "text": "Deployment is on Friday.", "source": "project/release.md",
        "chunk_id": 42, "start_line": 12, "end_line": 14,
        "rank": 3, "rerank_rank": 1, "vector_rank": 2, "vector_score": 0.8,
        "bm25_rank": 4, "bm25_score": 1.3, "fusion_score": 0.03,
        "reranker_score": 0.9,
    }
    server = create_mcp_server(search=lambda query: [hit])

    async def exercise() -> None:
        async with Client(server) as client:
            response = await client.call_tool("search_notes", {"query": "deployment"})
        assert not response.is_error
        assert response.structured_content["results"] == [{
            "text": "Deployment is on Friday.", "source": "project/release.md",
            "chunk_id": 42, "start_line": 12, "end_line": 14,
            "ranking": {"final_rank": 1, "rrf_rank": 3, "rerank_rank": 1,
                        "vector_rank": 2, "vector_score": 0.8, "bm25_rank": 4,
                        "bm25_score": 1.3, "fusion_score": 0.03,
                        "reranker_score": 0.9},
        }]
    anyio.run(exercise)


def test_search_respects_upstream_fusion_limit(monkeypatch):
    from types import SimpleNamespace
    import faiss
    import numpy as np
    from rag_notes_helper.rag.retrieval import retrieve

    records = [candidate(i) | {"doc_id": str(i)} for i in range(4)]
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0], [0.7, 0.0]], dtype="float32"))
    rag = SimpleNamespace(index=index, embed_model=SimpleNamespace(
        encode=lambda *args, **kwargs: np.array([[1.0, 0.0]], dtype="float32")))
    metadata = SimpleNamespace(get=lambda i: records[i])
    encoder = SimpleNamespace(predict=lambda pairs: [1.0] * len(pairs))
    monkeypatch.setattr("rag_notes_helper.rag.retrieval.get_settings", lambda: SimpleNamespace(
        vector_top_k=4, bm25_top_k=4, fusion_top_k=2, min_retrieval_score=0.0,
        bm25_min_score_ratio=0.0, llm=SimpleNamespace(max_chunks=1)))
    server = create_mcp_server(search=lambda query: retrieve(rag, metadata, query=query, cross_encoder=encoder))

    async def exercise() -> None:
        async with Client(server) as client:
            response = await client.call_tool("search_notes", {"query": "Evidence", "top_k": 4})
        assert not response.is_error
        assert response.structured_content["top_k"] == 4
        assert len(response.structured_content["results"]) == 2
    anyio.run(exercise)

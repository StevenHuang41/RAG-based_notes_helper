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

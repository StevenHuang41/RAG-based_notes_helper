from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
import sys
from contextlib import redirect_stdout
import threading
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, model_validator

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.index import rebuild_index
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.rag.retrieval import retrieve


logger = logging.getLogger(__name__)


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


def create_mcp_server(
    *,
    search: Callable[[str], Sequence[Mapping[str, Any]]],
    max_chunks: int,
) -> MCPServer:
    """Build a read-only local note search server."""
    if max_chunks < 1:
        raise ValueError("max_chunks must be positive")
    server = MCPServer(
        "RAG Notes Helper",
        version="0.2.0",
        instructions=(
            "Search the configured local note corpus and return ranked evidence. "
            "This server does not generate answers or expose filesystem access."
        ),
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
        top_k: Annotated[int, Field(strict=True)] | None = None,
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


def _create_runtime_server() -> tuple[MCPServer, MetaStore]:
    max_chunks = get_settings().llm.max_chunks
    # Index rebuilding reports progress for the CLI; stdio is reserved for MCP.
    with redirect_stdout(sys.stderr):
        rag = rebuild_index()
    meta_store = MetaStore()
    try:
        server = create_mcp_server(
            search=lambda query: retrieve(rag, meta_store, query=query),
            max_chunks=max_chunks,
        )
    except Exception:
        meta_store.close()
        raise
    return server, meta_store


def main() -> None:
    server, meta_store = _create_runtime_server()
    try:
        server.run(transport="stdio")
    finally:
        meta_store.close()


if __name__ == "__main__":
    main()

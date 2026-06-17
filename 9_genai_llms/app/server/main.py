from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

from databricks.ai_search.client import VectorSearchClient
from databricks.sdk.core import Config
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

ASGIMessage = dict[str, Any]
ASGIScope = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]


CATALOG_NAME = os.getenv("CATALOG_NAME", "ai_ml_in_practice")
SCHEMA_NAME = os.getenv("SCHEMA_NAME", "genai_workshop")
VECTOR_SEARCH_ENDPOINT_NAME = os.getenv("VECTOR_SEARCH_ENDPOINT_NAME", "genai-workshop-search")
VECTOR_SEARCH_INDEX_NAME = os.getenv(
    "VECTOR_SEARCH_INDEX_NAME",
    f"{CATALOG_NAME}.{SCHEMA_NAME}.document_chunks_rag_index",
)

SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "where",
    "which",
    "who",
    "why",
    "with",
    "workshop",
    "show",
    "showcase",
    "showcases",
}

WORKSHOP_MODULES: list[dict[str, str]] = [
    {
        "module_id": "1",
        "title": "AI/ML Architecture - How It All Fits Together",
        "goal": "Understand the full ML stack from data to production.",
    },
    {
        "module_id": "2",
        "title": "Data Preparation - Practical Foundations",
        "goal": "Learn data preparation techniques for ML models.",
    },
    {
        "module_id": "3",
        "title": "Feature Engineering - The Art of Extracting Value from Data",
        "goal": "Create high-quality features and avoid pitfalls.",
    },
    {
        "module_id": "4",
        "title": "ML Algorithms - The Classical Approach",
        "goal": "Learn key ML algorithms and when to use them.",
    },
    {
        "module_id": "5",
        "title": "Model Training in Practice",
        "goal": "Learn the model training process with code and metrics.",
    },
    {
        "module_id": "6",
        "title": "Deep Learning - Leveling up",
        "goal": "Understand deep learning concepts and implementation frameworks.",
    },
    {
        "module_id": "7",
        "title": "ML Pipelines - Automation and CI/CD",
        "goal": "Build a repeatable ML workflow.",
    },
    {
        "module_id": "8",
        "title": "MLOps - Manage Your ML Solution",
        "goal": "Get a full view of the ML lifecycle in production.",
    },
    {
        "module_id": "9",
        "title": "Generative AI and LLMs - The New Wave of Technology",
        "goal": "Understand how LLMs reshape ML architecture.",
    },
]

mcp = FastMCP("genai-workshop", stateless_http=True, json_response=True)


def _parse_vector_search_result(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the tabular AI Search SDK response into normal dictionaries.

    Databricks AI Search returns column names separately from row values. MCP
    tools should return simple JSON-like objects, so this helper joins each row
    with the manifest columns before the result leaves the server.
    """
    manifest = response.get("manifest", {})
    columns = [column["name"] for column in manifest.get("columns", [])]
    rows = response.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row, strict=False)) for row in rows]


def _query_terms(query: str) -> set[str]:
    """Extract meaningful terms used for lightweight reranking.

    MCP search is usually called by an agent, so the tool should avoid returning
    generic chunks just because they mention `workshop`. Removing common words
    lets stronger terms such as `classic`, `ml`, and `algorithms` influence the
    final ranking.
    """
    return {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if term not in SEARCH_STOPWORDS and len(term) > 1
    }


def _searchable_text(row: dict[str, Any]) -> str:
    """Join chunk content and metadata into one reranking string."""
    return "\n".join(
        str(row.get(field, ""))
        for field in [
            "module_id",
            "module_title",
            "source_path",
            "section_title",
            "retrieval_text",
            "chunk_text",
        ]
    ).lower()


def _lexical_score(query: str, row: dict[str, Any]) -> float:
    """Score a retrieved chunk against explicit words from the query."""
    terms = _query_terms(query)
    text = _searchable_text(row)
    if not terms:
        return 0.0
    score = sum(2.0 for term in terms if term in str(row.get("module_title", "")).lower())
    score += sum(1.0 for term in terms if term in text)
    if "classic" in terms and "classical" in text:
        score += 2.0
    module_match = re.search(r"module\s+(\d+)", query.lower())
    if module_match and str(row.get("module_id")) == module_match.group(1):
        score += 5.0
    return score


def _rerank_chunks(query: str, chunks: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Deduplicate vector-search candidates and keep the strongest chunks."""
    by_id = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id not in by_id:
            by_id[chunk_id] = dict(chunk)
    reranked = sorted(by_id.values(), key=lambda row: _lexical_score(query, row), reverse=True)[:k]
    for chunk in reranked:
        chunk.pop("retrieval_text", None)
    return reranked


def _clamp_k(k: int) -> int:
    """Keep retrieval result counts in a small, predictable range.

    Tool callers can ask for any integer, but an agent tool should protect the
    backing service from very large requests. The lower bound also avoids
    accidental zero-result searches caused by `k=0`.
    """
    return max(1, min(int(k), 10))


def _vector_search_client() -> VectorSearchClient:
    """Create an AI Search client using Databricks App service-principal auth.

    `Config()` follows Databricks unified authentication and reads the app
    runtime credentials injected by Databricks Apps. The AI Search client does
    not consume `Config` directly, so the resolved host and OAuth client values
    are passed into the client explicitly.
    """
    cfg = Config()
    if cfg.client_id and cfg.client_secret:
        return VectorSearchClient(
            workspace_url=cfg.host,
            service_principal_client_id=cfg.client_id,
            service_principal_client_secret=cfg.client_secret,
            disable_notice=True,
        )
    return VectorSearchClient(
        workspace_url=cfg.host,
        personal_access_token=cfg.token,
        disable_notice=True,
    )


def _databricks_auth_mode() -> str:
    """Describe available Databricks auth configuration without exposing secrets."""
    cfg = Config()
    if cfg.client_id and cfg.client_secret:
        return "app_service_principal"
    if cfg.token:
        return "personal_access_token"
    return "unconfigured"


def _health_payload() -> dict[str, str]:
    """Build the shared health response used by HTTP and MCP.

    Returning configuration names alongside `status=ok` makes demos easier to
    debug because the caller can immediately see which catalog, schema, endpoint,
    and index the app is wired to use.
    """
    return {
        "status": "ok",
        "catalog_name": CATALOG_NAME,
        "schema_name": SCHEMA_NAME,
        "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,
        "vector_search_index_name": VECTOR_SEARCH_INDEX_NAME,
        "databricks_auth_mode": _databricks_auth_mode(),
    }


@mcp.tool()
def health() -> dict[str, str]:
    """Return app and configuration status as an MCP tool.

    Agents can call this first to verify that the server is reachable and to
    learn which retrieval resources are configured before attempting search.
    """
    return _health_payload()


@mcp.tool()
def list_workshop_modules() -> list[dict[str, str]]:
    """Return stable metadata for all workshop modules.

    This is a deterministic tool: it does not ask an LLM to remember the module
    list. Agent orchestration frameworks use tools like this when the answer
    should come from a governed system rather than from model memory.
    """
    return WORKSHOP_MODULES


@mcp.tool()
def get_module_summary(module_id: int) -> dict[str, str]:
    """Return one module summary by id.

    The function validates the requested id and raises a clear error when it is
    unknown. That error becomes useful feedback for an agent loop because the
    model can retry with one of the valid ids instead of hallucinating metadata.
    """
    module_id_text = str(module_id)
    for module in WORKSHOP_MODULES:
        if module["module_id"] == module_id_text:
            return module
    valid_ids = ", ".join(module["module_id"] for module in WORKSHOP_MODULES)
    raise ValueError(f"Unknown module_id={module_id}. Valid module ids: {valid_ids}")


@mcp.tool()
def search_workshop_docs(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve relevant workshop chunks through Databricks AI Search.

    This tool is the MCP version of the RAG retriever. The caller provides a
    natural-language query, the vector index finds semantically similar chunks,
    and the tool returns text plus citation metadata that an agent can use before
    drafting an answer.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    client = _vector_search_client()
    index = client.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=VECTOR_SEARCH_INDEX_NAME,
    )
    response = index.similarity_search(
        query_text=query,
        columns=[
            "chunk_id",
            "module_id",
            "module_title",
            "source_path",
            "section_title",
            "chunk_text",
            "retrieval_text",
        ],
        num_results=max(_clamp_k(k) * 5, 20),
    )
    return _rerank_chunks(query, _parse_vector_search_result(response), _clamp_k(k))


async def http_health(_: Request) -> JSONResponse:
    """Expose the same health payload for a browser or load balancer check."""
    return JSONResponse(_health_payload())


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    """Start and stop the MCP session manager with the FastAPI app.

    The MCP server owns session state for streamable HTTP requests. Binding it to
    the FastAPI lifespan ensures sessions are ready before requests are served
    and cleaned up when the app stops.
    """
    async with mcp.session_manager.run():
        yield


class NormalizeMCPPath:
    """Accept both `/mcp` and the double `/mcp/mcp` path seen in Playground."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        if scope["type"] == "http" and scope.get("path", "").startswith("/mcp/mcp"):
            scope = dict(scope)
            scope["path"] = scope["path"].replace("/mcp/mcp", "/mcp", 1)
            scope["raw_path"] = scope["path"].encode("ascii")
        await self.app(scope, receive, send)


app = FastAPI(
    title="GenAI Workshop MCP Server",
    description="MCP tools for workshop module metadata and document retrieval.",
    lifespan=lifespan,
)

app.get("/", include_in_schema=False)(http_health)
app.mount("/", mcp.streamable_http_app())

app = NormalizeMCPPath(app)

app = CORSMiddleware(
    app,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)

"""FastAPI server exposing the agent_guard safety pipeline and frontend.

Run with:
    uvicorn server:app --reload

Endpoints:
    GET  /           → pipeline visualizer UI
    GET  /health     → liveness check
    POST /guard      → run the safety pipeline only (no RAG)
    POST /ask        → guard + forward enhanced query to an upstream RAG service
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agent_guard.graph as _graph_module
from agent_guard import run_pipeline
from agent_guard.logging_config import configure_logging

load_dotenv()
configure_logging()

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Force a fresh compiled graph on every server start so code changes
    # (e.g. adding new nodes) are always picked up without stale caches.
    _graph_module._compiled = None
    _graph_module._compiled = _graph_module.build_graph()
    yield


app = FastAPI(
    title="agent_guard",
    description="Multi-language safety pipeline: PII anonymization, prompt-injection detection, and query enhancement.",
    version="1.0.0",
    lifespan=lifespan,
)


class GuardRequest(BaseModel):
    message: str = Field(..., description="The raw user message to evaluate.")


class AskRequest(BaseModel):
    message: str = Field(..., description="The raw user message.")
    upstream_url: str = Field(
        ...,
        description=(
            "URL of the upstream RAG service. agent_guard will POST "
            '{"query": "<enhanced_message>"} to this URL if the message is safe.'
        ),
    )
    upstream_field: str = Field(
        "query",
        description="JSON field name the upstream service expects for the query.",
    )
    answer_field: str = Field(
        "answer",
        description="JSON field name to extract from the upstream response as the answer.",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/guard")
def guard(req: GuardRequest) -> dict:
    """Run the safety pipeline only — no RAG. Returns the full GraphState."""
    return dict(run_pipeline(req.message))


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Guard the message, then forward the enhanced query to an upstream RAG service."""
    state = dict(run_pipeline(req.message))

    if state.get("status") == "blocked":
        return state

    enhanced_query = state.get("context_enhanced_message") or state.get("pii_masked_message", "")

    try:
        with httpx.Client(timeout=30.0) as client:
            upstream_resp = client.post(
                req.upstream_url,
                json={req.upstream_field: enhanced_query},
            )
            upstream_resp.raise_for_status()
            upstream_data = upstream_resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Upstream RAG service returned {e.response.status_code}") from e
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach upstream RAG service: {e}") from e

    state["rag_response"] = upstream_data.get(req.answer_field) or str(upstream_data)
    state["rag_sources"] = upstream_data.get("sources") or upstream_data.get("documents") or []
    return state


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

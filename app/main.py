"""FastAPI application — the HTTP layer (Free edition).

This file wires the pieces together and exposes two endpoints:
    GET  /health  -> liveness check
    POST /chat    -> talk to the agent (with per-session memory)

Heavy objects (the compiled graph, the memory store) are created ONCE at
startup in the `lifespan` handler and stored on `app.state`, so every request
reuses them instead of rebuilding them.

> Looking for RAG (document ingest + semantic search), Docker, and a deployment
> guide? Those ship in the Pro edition — see the README.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.agents.graph import build_agent_graph, run_agent
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.memory.store import BaseMemoryStore, InMemoryMemoryStore
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook. Build shared resources before serving requests."""
    setup_logging()
    settings = get_settings()
    logger.info("Starting %s (env=%s)", settings.app_name, settings.app_env)

    # Build once, reuse for every request.
    app.state.agent_graph = build_agent_graph()
    app.state.memory = InMemoryMemoryStore()  # swap for Redis/Postgres later

    yield  # ---- application runs here ----

    logger.info("Shutting down.")


app = FastAPI(
    title="AI Agent Starter Kit (Free)",
    description="FastAPI + LangGraph agent with tool calling and memory.",
    version="1.0.0",
    lifespan=lifespan,
)


def _memory() -> BaseMemoryStore:
    return app.state.memory


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness probe. Returns 200 if the server is up."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["agent"])
def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the agent and get a reply.

    Flow:
      1. Load this session's history from memory.
      2. Run the agent (it may call tools).
      3. Save both the user message and the agent's reply back to memory.
    """
    memory = _memory()
    history = memory.get_messages(request.session_id)

    try:
        result = run_agent(app.state.agent_graph, history, request.message)
    except Exception as exc:  # surface a clean 500 instead of a stack trace
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    # Persist the turn so the next request remembers it.
    memory.add_message(request.session_id, "user", request.message)
    memory.add_message(request.session_id, "assistant", result["response"])

    return ChatResponse(
        session_id=request.session_id,
        response=result["response"],
        tool_calls=result["tool_calls"],
    )

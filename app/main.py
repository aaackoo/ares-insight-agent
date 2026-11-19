"""FastAPI app: POST /ask, POST /ask/stream (Server-Sent Events), GET /health."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from app.agent import AgentGraph, Answer, build_graph, create_chat_model, initial_state, run_agent
from app.ares import AresClient
from app.config import get_settings
from app.tools import build_tools

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build long-lived objects once at startup and close them at shutdown.

    One shared httpx.AsyncClient reuses connections to ARES across requests.
    """
    settings = get_settings()
    async with httpx.AsyncClient(timeout=settings.ares_timeout_seconds) as http:
        ares = AresClient(http, settings.ares_base_url)
        llm = create_chat_model(settings)
        app.state.graph = build_graph(llm, build_tools(ares), settings.agent_max_steps)
        yield


app = FastAPI(title="ARES insight agent", lifespan=lifespan)


async def get_graph(request: Request) -> AgentGraph:
    """Dependency, so tests can swap in a graph with a fake LLM.

    async def: FastAPI runs plain-def dependencies in a thread pool, needless here."""
    graph: AgentGraph = request.app.state.graph
    return graph


Graph = Annotated[AgentGraph, Depends(get_graph)]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
async def ask(body: AskRequest, graph: Graph) -> Answer:
    return await run_agent(graph, body.question)


@app.post("/ask/stream", response_class=EventSourceResponse)
async def ask_stream(body: AskRequest, graph: Graph) -> AsyncIterator[ServerSentEvent]:
    """Stream progress while the agent works, so the client sees something immediately
    instead of waiting for the whole run.

    Events: tool_call -> tool_result -> ... -> answer   (or error)
    """
    try:
        # stream_mode="updates" yields {node_name: what_that_node_returned} after each node.
        async for update in graph.astream(initial_state(body.question), stream_mode="updates"):
            for node, node_output in update.items():
                for event in to_events(node, node_output):
                    yield event
    except Exception:
        # The 200 status and headers are already sent, so errors are reported as an event.
        # Details go to the log, not to the client.
        logger.exception("Agent run failed")
        yield ServerSentEvent(event="error", data={"message": "The agent failed, try again."})


def to_events(node: str, node_output: dict[str, Any]) -> list[ServerSentEvent]:
    """Translate one graph step into client-facing events."""
    events = []
    for message in node_output.get("messages", []):
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                events.append(
                    ServerSentEvent(
                        event="tool_call", data={"tool": call["name"], "args": call["args"]}
                    )
                )
        elif isinstance(message, ToolMessage):
            events.append(
                ServerSentEvent(
                    event="tool_result",
                    data={"tool": message.name, "preview": str(message.content)[:200]},
                )
            )
    if node == "respond" and node_output.get("answer") is not None:
        events.append(ServerSentEvent(event="answer", data=node_output["answer"]))
    return events

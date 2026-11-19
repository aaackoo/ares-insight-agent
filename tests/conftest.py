"""Shared test helpers.

- ARES: a real AresClient whose HTTP traffic is intercepted by respx (recorded fixtures).
- LLM: a fake chat model that replays scripted messages, so no API key is needed.
"""

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage

from app.agent import AgentGraph, build_graph
from app.ares import AresClient
from app.tools import build_tools

ARES_BASE = "https://ares.test/rest"
COMPANY_URL = f"{ARES_BASE}/ekonomicke-subjekty/27082440"
VR_URL = f"{ARES_BASE}/ekonomicke-subjekty-vr/27082440"
FIXTURES = Path(__file__).parent / "fixtures"


class FakeToolCallingModel(FakeMessagesListChatModel):
    """Returns the scripted responses in order. bind_tools() is a no-op because the
    responses already contain the tool calls; with_structured_output() then works
    through LangChain's default implementation, which reads an 'Answer' tool call."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolCallingModel":
        return self


def tool_call(tool_name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": call_id}])


def final_answer(answer: str, facts: list[dict[str, str]]) -> AIMessage:
    """What the model emits for with_structured_output(Answer)."""
    return tool_call("Answer", "call_answer", answer=answer, facts=facts)


def make_graph(
    responses: list[BaseMessage], ares_client: AresClient, max_steps: int = 8
) -> AgentGraph:
    llm = FakeToolCallingModel(responses=responses)
    return build_graph(llm, build_tools(ares_client), max_steps=max_steps)


def _load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
async def ares_client() -> AsyncIterator[AresClient]:
    async with httpx.AsyncClient() as http:
        yield AresClient(http, ARES_BASE)


@pytest.fixture
def mock_ares() -> Iterator[respx.MockRouter]:
    """Serve the recorded ARES responses for Alza.cz a.s. (IČO 27082440).

    Any request without a matching route fails the test, so no real network calls happen.
    Tests can add their own routes to the returned router.
    """
    # assert_all_called=False: not every test uses every route.
    with respx.mock(base_url=ARES_BASE, assert_all_called=False) as router:
        router.get("/ekonomicke-subjekty/27082440", name="company").respond(
            json=_load("company_27082440.json")
        )
        router.post("/ekonomicke-subjekty/vyhledat", name="search").respond(
            json=_load("search_alza.json")
        )
        router.get("/ekonomicke-subjekty-vr/27082440", name="vr").respond(
            json=_load("vr_27082440.json")
        )
        yield router

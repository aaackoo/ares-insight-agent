"""API tests: the graph dependency is overridden with one that uses the fake LLM.

httpx.ASGITransport calls the app in-process. It does not run the lifespan, so the real
OpenAI client is never created and no API key is needed.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult

from app.agent import AgentGraph, build_graph
from app.ares import AresClient
from app.main import app, get_graph, summarize_tool_output
from app.tools import build_tools
from tests.conftest import COMPANY_URL, FakeToolCallingModel, final_answer, make_graph, tool_call


@pytest.fixture
async def api() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


def use_graph(graph: AgentGraph) -> None:
    app.dependency_overrides[get_graph] = lambda: graph


def scripted_graph(ares_client: AresClient) -> AgentGraph:
    return make_graph(
        [
            tool_call("get_company_by_ico", "c1", ico="27082440"),
            AIMessage(content="Done."),
            final_answer(
                "Yes, Alza.cz a.s. is active.",
                [{"statement": "Status: active", "source_url": COMPANY_URL}],
            ),
        ],
        ares_client,
    )


def parse_sse(body: str) -> list[tuple[str, Any]]:
    """Turn 'event: x\\ndata: {...}\\n\\n' blocks into (event, data) pairs."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


async def test_health(api: httpx.AsyncClient) -> None:
    response = await api.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ask_returns_structured_answer(
    api: httpx.AsyncClient, ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    use_graph(scripted_graph(ares_client))

    response = await api.post("/ask", json={"question": "Is IČO 27082440 active?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Yes, Alza.cz a.s. is active.",
        "facts": [{"statement": "Status: active", "source_url": COMPANY_URL}],
    }


async def test_ask_validates_question(
    api: httpx.AsyncClient, ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    use_graph(scripted_graph(ares_client))

    response = await api.post("/ask", json={"question": ""})

    assert response.status_code == 422
    assert mock_ares.calls.call_count == 0  # the agent never ran


async def test_ask_stream_emits_each_step(
    api: httpx.AsyncClient, ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    use_graph(scripted_graph(ares_client))

    response = await api.post("/ask/stream", json={"question": "Is IČO 27082440 active?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["tool_call", "tool_result", "answer"]
    assert events[0][1] == {"tool": "get_company_by_ico", "args": {"ico": "27082440"}}
    # The preview is a real JSON object, not a JSON string inside JSON.
    tool_result = events[1][1]
    assert tool_result["tool"] == "get_company_by_ico"
    assert tool_result["preview"]["ico"] == "27082440"
    assert tool_result["preview"]["legal_form"] == "Akciová společnost"
    assert events[2][1]["answer"] == "Yes, Alza.cz a.s. is active."
    # Czech characters are sent as-is in every event, not as \u escapes.
    assert "Holešovice" in response.text
    assert "\\u" not in response.text


class BrokenModel(FakeToolCallingModel):
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("LLM provider is down")


async def test_ask_stream_reports_errors_as_event(
    api: httpx.AsyncClient, ares_client: AresClient
) -> None:
    use_graph(build_graph(BrokenModel(responses=[]), build_tools(ares_client), max_steps=8))

    response = await api.post("/ask/stream", json={"question": "Is IČO 27082440 active?"})

    events = parse_sse(response.text)
    assert events == [("error", {"message": "The agent failed, try again."})]
    assert "LLM provider is down" not in response.text  # internals are not leaked


def test_preview_of_list_shows_first_three_items_and_total() -> None:
    companies = [{"ico": str(i), "name": f"Firma {i}"} for i in range(5)]

    preview = summarize_tool_output(json.dumps(companies))

    assert preview == {"total": 5, "items": companies[:3]}


def test_preview_of_object_shortens_long_fields() -> None:
    record = {
        "ico": "27082440",
        "company_name": "Alza.cz a.s.",
        "status": "AKTIVNI",
        "bodies": [{"members": [{"name": f"Člen {i}"} for i in range(5)]}],
        "note": "x" * 100,
        "source_url": "https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty-vr/27082440",
    }

    preview = summarize_tool_output(json.dumps(record))

    assert preview["ico"] == "27082440"
    assert preview["status"] == "AKTIVNI"
    assert preview["note"] == "x" * 80 + "..."
    assert preview["source_url"] == record["source_url"]  # links are never cut
    members = preview["bodies"][0]["members"]
    assert members == [{"name": "Člen 0"}, {"name": "Člen 1"}, {"name": "Člen 2"}, "... 2 more"]


def test_preview_of_plain_text_is_a_short_string() -> None:
    message = "No active company found in ARES with IČO 12345678. " * 10

    preview = summarize_tool_output(message)

    assert isinstance(preview, str)
    assert preview == message[:200] + "..."

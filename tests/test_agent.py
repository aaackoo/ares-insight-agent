"""Graph tests: a fake chat model replays scripted messages, ARES is mocked by respx.

This tests our graph wiring (routing, tool execution, step limit, structured answer and
the source guardrail) deterministically and without an API key. It does not test whether
a real LLM makes good decisions; that is what evals/ is for.
"""

import respx
from langchain_core.messages import AIMessage, ToolMessage

from app.agent import initial_state
from app.ares import AresClient
from tests.conftest import COMPANY_URL, VR_URL, final_answer, make_graph, tool_call


async def test_agent_calls_tools_and_builds_sourced_answer(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    graph = make_graph(
        [
            # Step 1: the model asks for two tools at once.
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_company_by_ico", "args": {"ico": "27082440"}, "id": "c1"},
                    {"name": "get_statutory_bodies", "args": {"ico": "27082440"}, "id": "c2"},
                ],
            ),
            # Step 2: no more tool calls -> route to respond.
            AIMessage(content="I have everything I need."),
            # respond node: the structured answer.
            final_answer(
                "Alza.cz a.s. is active, it is a joint-stock company.",
                [
                    {"statement": "Status: active", "source_url": COMPANY_URL},
                    {"statement": "Legal form: Akciová společnost", "source_url": COMPANY_URL},
                    {"statement": "Chairman: ALEŠ ZAVORAL", "source_url": VR_URL},
                ],
            ),
        ],
        ares_client,
    )

    state = await graph.ainvoke(initial_state("Is IČO 27082440 active and who runs it?"))

    # The right ARES endpoints were called, once each.
    assert mock_ares.routes["company"].call_count == 1
    assert mock_ares.routes["vr"].call_count == 1
    # The tool results went back into the conversation, matched to their calls.
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert [(m.name, m.tool_call_id) for m in tool_messages] == [
        ("get_company_by_ico", "c1"),
        ("get_statutory_bodies", "c2"),
    ]
    assert "Akciová společnost" in str(tool_messages[0].content)
    assert "ZAVORAL" in str(tool_messages[1].content)
    # The structured answer kept all facts: every source URL was returned by a tool.
    answer = state["answer"]
    assert answer is not None
    assert answer.answer.startswith("Alza.cz a.s. is active")
    assert [f.source_url for f in answer.facts] == [COMPANY_URL, COMPANY_URL, VR_URL]
    assert state["steps"] == 2


async def test_search_by_name_then_lookup_by_ico(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    graph = make_graph(
        [
            tool_call("search_companies_by_name", "c1", name="Alza"),
            tool_call("get_company_by_ico", "c2", ico="27082440"),
            AIMessage(content="Done."),
            final_answer(
                "IČO of Alza.cz a.s. is 27082440.",
                [
                    {"statement": "IČO: 27082440", "source_url": COMPANY_URL},
                ],
            ),
        ],
        ares_client,
    )

    state = await graph.ainvoke(initial_state("What is the IČO of Alza?"))

    assert mock_ares.routes["search"].call_count == 1
    assert mock_ares.routes["company"].call_count == 1
    assert state["answer"] is not None
    assert len(state["answer"].facts) == 1


async def test_facts_without_a_tool_source_are_dropped(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    graph = make_graph(
        [
            tool_call("get_company_by_ico", "c1", ico="27082440"),
            AIMessage(content="Done."),
            final_answer(
                "Alza.cz a.s. is active.",
                [
                    {"statement": "Status: active", "source_url": COMPANY_URL},
                    # Hallucinated: never returned by any tool.
                    {"statement": "Revenue: 70 bn CZK", "source_url": "https://example.com/x"},
                    # Real-looking URL, but the VR tool was never called in this run.
                    {"statement": "Chairman: someone", "source_url": VR_URL},
                ],
            ),
        ],
        ares_client,
    )

    state = await graph.ainvoke(initial_state("Is 27082440 active?"))

    assert state["answer"] is not None
    assert [f.statement for f in state["answer"].facts] == ["Status: active"]


async def test_step_limit_stops_the_loop_and_still_answers(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    # A model that would call tools forever.
    graph = make_graph(
        [
            tool_call("get_company_by_ico", "c1", ico="27082440"),
            tool_call("get_company_by_ico", "c2", ico="27082440"),
            final_answer("Partial answer.", []),
        ],
        ares_client,
        max_steps=2,
    )

    state = await graph.ainvoke(initial_state("Loop forever"))

    # Step 1's tool call ran; step 2's was not executed because the limit was reached.
    assert mock_ares.routes["company"].call_count == 1
    assert state["steps"] == 2
    assert state["answer"] is not None
    assert state["answer"].answer == "Partial answer."


async def test_unknown_tool_is_reported_back_to_the_model(
    ares_client: AresClient, mock_ares: respx.MockRouter
) -> None:
    graph = make_graph(
        [
            tool_call("delete_company", "c1", ico="27082440"),
            AIMessage(content="Sorry."),
            final_answer("I cannot do that.", []),
        ],
        ares_client,
    )

    state = await graph.ainvoke(initial_state("Delete Alza"))

    tool_message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert "unknown tool" in str(tool_message.content)

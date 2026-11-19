"""Run the eval dataset against the real LLM and the real ARES API.

    uv run python -m evals.run_eval

Needs a key in .env and costs a little money, so it is run manually, not in CI.

Metrics:
- tool selection accuracy: share of cases where every expected tool was called
  (extra calls are allowed: e.g. confirming a search result by IČO is reasonable).
- fact accuracy: share of expected key facts found in the answer (case-insensitive;
  each fact lists accepted spellings, e.g. English and Czech).
- unsupported claims: claims in the answer that the tool results don't support,
  judged by the LLM ("LLM as a judge"). Target: 0.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from app.agent import AgentGraph, build_graph, create_chat_model, initial_state
from app.ares import AresClient
from app.config import get_settings
from app.tools import build_tools

DATASET = Path(__file__).with_name("dataset.jsonl")

JUDGE_PROMPT = """\
You check an AI assistant's answer against the tool results it was given.
List every factual claim in the ANSWER (its text and its facts) that is NOT supported by
the TOOL RESULTS. Translating, reformatting or summarizing a supported value is fine.
Saying that information is not available is not a claim.
Judge only whether the tool results contain the claim, not whether it seems plausible:
the tool results are current register data and may contain recent dates.
If everything is supported, return an empty list.
"""


class Verdict(BaseModel):
    unsupported_claims: list[str]


@dataclass
class CaseResult:
    case_id: str
    tools_called: set[str]
    tools_ok: bool
    facts_found: int
    facts_expected: int
    unsupported_claims: list[str]


async def run_case(graph: AgentGraph, judge: Any, case: dict[str, Any]) -> CaseResult:
    state = await graph.ainvoke(initial_state(case["question"]))
    messages = state["messages"]
    answer_json = state["answer"].model_dump_json()

    tools_called = {
        call["name"] for m in messages if isinstance(m, AIMessage) for call in m.tool_calls
    }
    facts_found = sum(
        any(spelling.lower() in answer_json.lower() for spelling in spellings)
        for spellings in case["expected_facts"]
    )

    tool_results = "\n\n".join(str(m.content) for m in messages if isinstance(m, ToolMessage))
    verdict = Verdict.model_validate(
        await judge.ainvoke(
            [
                SystemMessage(JUDGE_PROMPT),
                HumanMessage(
                    f"TOOL RESULTS:\n{tool_results or '(none)'}\n\nANSWER:\n{answer_json}"
                ),
            ]
        )
    )
    return CaseResult(
        case_id=case["id"],
        tools_called=tools_called,
        tools_ok=set(case["expected_tools"]) <= tools_called,
        facts_found=facts_found,
        facts_expected=len(case["expected_facts"]),
        unsupported_claims=verdict.unsupported_claims,
    )


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)  # shows guardrail warnings (dropped facts)
    settings = get_settings()
    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    llm = create_chat_model(settings)
    judge = llm.with_structured_output(Verdict)

    results = []
    async with httpx.AsyncClient(timeout=settings.ares_timeout_seconds) as http:
        graph = build_graph(
            llm, build_tools(AresClient(http, settings.ares_base_url)), settings.agent_max_steps
        )
        # Sequential on purpose: easy-to-read output and gentle on the public ARES API.
        for case in cases:
            result = await run_case(graph, judge, case)
            results.append(result)
            print(
                f"{result.case_id:<22} tools {'OK  ' if result.tools_ok else 'MISS'} "
                f"facts {result.facts_found}/{result.facts_expected}  "
                f"unsupported {len(result.unsupported_claims)}  "
                f"called={sorted(result.tools_called)}"
            )
            for claim in result.unsupported_claims:
                print(f"    unsupported: {claim}")

    facts_found = sum(r.facts_found for r in results)
    facts_expected = sum(r.facts_expected for r in results)
    unsupported = sum(len(r.unsupported_claims) for r in results)
    model = settings.openai_model if settings.llm_provider == "openai" else "azure deployment"
    print()
    print(f"Model:                    {model}")
    print(f"Cases:                    {len(results)}")
    print(f"Tool selection accuracy:  {sum(r.tools_ok for r in results) / len(results):.0%}")
    fact_accuracy = facts_found / facts_expected
    print(f"Fact accuracy:            {fact_accuracy:.0%} ({facts_found}/{facts_expected})")
    print(f"Unsupported claims:       {unsupported}")


if __name__ == "__main__":
    asyncio.run(main())

"""The agent, built explicitly with LangGraph's StateGraph.

    START ──> agent ──(tool calls and steps left)──> tools ──> agent  (loop)
                 └──(no tool calls, or step limit hit)──> respond ──> END

- agent:   the LLM decides which tools to call (or that it has enough information)
- tools:   runs the requested tools and adds their results to the conversation
- respond: turns the conversation into a structured Answer, every fact with a source URL
"""

import asyncio
import logging
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You answer questions about Czech companies using the public ARES business register.

Rules:
- Get information only by calling the tools. Never state a fact that was not returned
  by a tool, not even facts you believe are well known.
- If the user gives a company name but no IČO, search by name first.
- A company not found by IČO may be dissolved: check the commercial register before
  saying it does not exist.
- If the tools do not return the requested information, say it is not available.
- When you have enough information, stop calling tools and reply briefly.
"""

RESPOND_PROMPT = """\
Write the final answer to the user's question, using ONLY the tool results in this
conversation.
- answer: a short plain-language answer, in the language of the question.
- facts: each fact the answer relies on, as one short statement, with source_url copied
  exactly from the tool result that contains it.
- Never include a fact that is not in a tool result. If information is missing, say so
  in the answer instead.
"""


class Fact(BaseModel):
    statement: str = Field(description="One verifiable fact, e.g. 'Legal form: Akciová společnost'")
    source_url: str = Field(description="URL of the ARES record the fact comes from")


class Answer(BaseModel):
    """Final answer to the user's question, with a source for every fact."""

    answer: str
    facts: list[Fact]


class AgentState(TypedDict):
    # add_messages is a "reducer": nodes return new messages and LangGraph appends them.
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int  # how many times the agent node (LLM) has run
    answer: Answer | None


AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def initial_state(question: str) -> AgentState:
    return {"messages": [HumanMessage(question)], "steps": 0, "answer": None}


def build_graph(llm: BaseChatModel, tools: list[BaseTool], max_steps: int) -> AgentGraph:
    llm_with_tools = llm.bind_tools(tools)
    llm_for_answer = llm.with_structured_output(Answer)
    tools_by_name = {t.name: t for t in tools}

    async def agent(state: AgentState) -> dict[str, Any]:
        response = await llm_with_tools.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response], "steps": state["steps"] + 1}

    async def run_tool(call: ToolCall) -> ToolMessage:
        tool = tools_by_name.get(call["name"])
        if tool is None:
            content = f"Error: unknown tool {call['name']!r}."
        else:
            content = await tool.ainvoke(call["args"])
        return ToolMessage(content=content, tool_call_id=call["id"] or "", name=call["name"])

    async def tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        # The model may request several tools at once; they are independent, so run them
        # concurrently. gather() keeps the order, so results line up with the calls.
        results = await asyncio.gather(*(run_tool(call) for call in last.tool_calls))
        return {"messages": list(results)}

    def route_after_agent(state: AgentState) -> Literal["tools", "respond"]:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        if wants_tools and state["steps"] < max_steps:
            return "tools"
        if wants_tools:
            logger.warning("Agent hit the step limit (%d), answering with what it has", max_steps)
        return "respond"

    async def respond(state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        # If we stopped because of the step limit, the last message holds tool calls that
        # were never executed. OpenAI rejects such a history, so leave that message out.
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            messages = messages[:-1]
        result = await llm_for_answer.ainvoke([SystemMessage(RESPOND_PROMPT), *messages])
        answer = Answer.model_validate(result)
        return {"answer": drop_unsourced_facts(answer, messages)}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools_node)
    graph.add_node("respond", respond)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "respond"])
    graph.add_edge("tools", "agent")
    graph.add_edge("respond", END)
    return graph.compile()


async def run_agent(graph: AgentGraph, question: str) -> Answer:
    final_state = await graph.ainvoke(initial_state(question))
    answer: Answer | None = final_state["answer"]
    assert answer is not None  # the respond node always sets it
    return answer


_SOURCE_URL = re.compile(r'"source_url":\s*"([^"]+)"')


def drop_unsourced_facts(answer: Answer, messages: list[AnyMessage]) -> Answer:
    """Guardrail in code, not just in the prompt: keep only facts whose source URL was
    actually returned by a tool. (It checks the source exists, not that the statement
    matches it; the evals measure that.)"""
    known_urls = {
        url
        for m in messages
        if isinstance(m, ToolMessage)
        for url in _SOURCE_URL.findall(str(m.content))
    }
    kept = [f for f in answer.facts if f.source_url in known_urls]
    for fact in answer.facts:
        if fact not in kept:
            logger.warning(
                "Dropped fact with unknown source %r: %r", fact.source_url, fact.statement
            )
    return answer.model_copy(update={"facts": kept})


def create_chat_model(settings: Settings) -> BaseChatModel:
    """OpenAI by default; Azure OpenAI when LLM_PROVIDER=azure."""
    if settings.llm_provider == "azure":
        if not settings.azure_openai_api_key:  # None or empty string
            raise ValueError("AZURE_OPENAI_API_KEY is not set")
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.azure_openai_deployment,
            api_version=settings.openai_api_version,
            api_key=settings.azure_openai_api_key,
            temperature=0,
        )
    if not settings.openai_api_key:  # None or empty string
        raise ValueError("OPENAI_API_KEY is not set")
    return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0)

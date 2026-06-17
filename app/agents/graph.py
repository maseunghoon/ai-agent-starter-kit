"""The LangGraph agent: a small ReAct-style loop.

The flow is intentionally explicit so you can see exactly how tool calling works:

        ┌─────────┐  tool calls?  ┌────────┐
   ▶───▶│  agent  │──────yes─────▶│  tools │───┐
        └─────────┘               └────────┘   │
             ▲                                  │
             └──────────────────────────────────┘
             │
             └── no tool calls ──▶  END

1. `agent` node: the LLM looks at the conversation and either answers directly
   or decides to call one or more tools.
2. If it asked for tools, the `tools` node runs them and feeds the results back.
3. The loop repeats until the LLM answers with no further tool calls.
"""

import logging

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.state import AgentState
from app.agents.tools import get_tools
from app.core.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant. You have access to tools. "
        "When a user asks for math, use the `calculator` tool. "
        "When a user asks for the time or date, use the `current_time` tool. "
        "Always base factual answers on tool results rather than guessing."
    )
)


def build_agent_graph():
    """Build and compile the agent graph. Call this once at startup.

    The OpenAI client is created lazily (on the first request) rather than here,
    so the server can start — and `/health` can respond — even before an API key
    is configured. The key is only required the moment you actually chat.
    """
    tools = get_tools()
    llm_with_tools_cache: dict = {}

    def _get_llm():
        if "llm" not in llm_with_tools_cache:
            settings = get_settings()
            # `temperature=0` makes the agent deterministic — good for tool use.
            llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0,
            )
            # `bind_tools` tells the model which tools exist and their schemas.
            llm_with_tools_cache["llm"] = llm.bind_tools(tools)
        return llm_with_tools_cache["llm"]

    def agent_node(state: AgentState) -> dict:
        """Ask the LLM what to do next given the conversation so far."""
        response = _get_llm().invoke([SYSTEM_PROMPT, *state["messages"]])
        # Returning {"messages": [...]} appends via the add_messages reducer.
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))  # runs whatever tools were called

    graph.set_entry_point("agent")
    # `tools_condition` is a prebuilt router: if the last AI message contains
    # tool calls it routes to "tools", otherwise to END.
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")  # after running tools, loop back

    return graph.compile()


def _to_lc_message(message: dict) -> BaseMessage:
    """Convert a stored {role, content} dict into a LangChain message."""
    if message["role"] == "assistant":
        return AIMessage(content=message["content"])
    return HumanMessage(content=message["content"])


def run_agent(graph, history: list[dict], user_message: str) -> dict:
    """Run one turn of the agent.

    Args:
        graph: the compiled graph from `build_agent_graph()`.
        history: prior turns as [{"role": ..., "content": ...}, ...].
        user_message: the new user message.

    Returns:
        {"response": str, "tool_calls": [tool names that were used]}
    """
    input_messages: list[BaseMessage] = [_to_lc_message(m) for m in history]
    input_messages.append(HumanMessage(content=user_message))

    result = graph.invoke({"messages": input_messages})
    final_messages = result["messages"]

    # Collect the names of any tools the agent decided to call this turn.
    tool_calls_used: list[str] = []
    for msg in final_messages:
        for call in getattr(msg, "tool_calls", []) or []:
            tool_calls_used.append(call["name"])

    # The final assistant answer is the content of the last message.
    answer = final_messages[-1].content
    logger.info("Agent answered using tools: %s", tool_calls_used or "none")
    return {"response": answer, "tool_calls": tool_calls_used}

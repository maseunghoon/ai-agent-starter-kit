"""The state object passed between nodes in the LangGraph agent.

LangGraph is a state machine: every node receives the current state and returns
a partial update. Our state only needs one thing — the running list of chat
messages.

`add_messages` is a special "reducer": instead of *replacing* the messages list
on each update, it *appends* to it. That's how the conversation accumulates as
the graph loops between the model and the tools.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state for the agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]

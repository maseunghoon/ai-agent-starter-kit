"""Request/response models for the chat endpoint.

Pydantic models give us automatic validation and the interactive API docs at
`/docs`. The `Field(..., examples=[...])` values show up there as defaults.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="Stable id used to group a conversation's history.",
        examples=["demo"],
    )
    message: str = Field(
        ...,
        description="The user's message to the agent.",
        examples=["What time is it and calculate 15 * 23?"],
    )


class ChatResponse(BaseModel):
    session_id: str
    response: str = Field(..., description="The agent's final answer.")
    tool_calls: list[str] = Field(
        default_factory=list,
        description="Names of tools the agent used to produce this answer.",
    )

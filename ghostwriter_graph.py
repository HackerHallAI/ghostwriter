import os
import asyncio
from qdrant_client import QdrantClient
from pydantic import BaseModel
from typing import TypedDict, Annotated, List
from qdrant_utils import get_qdrant_client
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from ai_writer import (
    list_skool_pages_helper,
    PydanticAIDeps,
    pydantic_ai_writer,
)
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import MemorySaver

# Initialize the Ollama model
# ollama_model = OpenAIModel(model_name="llama3.2", base_url="http://localhost:11434/v1")
base_url = "http://localhost:11434/v1"
reasoner_llm_model = "deepseek-r1:14b"
api_key = "fakeshit!"

reasoner = Agent(
    OpenAIModel(reasoner_llm_model, base_url=base_url, api_key=api_key),
    system_prompt="You are an expert at writing viral content for social media.",
)


primary_llm_model = "llama3.1:latest"
router_agent = Agent(
    OpenAIModel(primary_llm_model, base_url=base_url, api_key=api_key),
    system_prompt="Your job is to route the user message either to the end of the conversation or to continue coding the AI agent.",
)

end_conversation_agent = Agent(
    OpenAIModel(primary_llm_model, base_url=base_url, api_key=api_key),
    system_prompt="Your job is to end a conversation for creating an AI agent by giving instructions for how to execute the agent and they saying a nice goodbye to the user.",
)


qdrant_client = get_qdrant_client(mode="docker")


class AgentState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]
    scope: str


# Scope Definition Node with Reasoner LLM
async def define_scope_with_reasoner(state: AgentState):
    # First, get the useful sections of the skool pages

    skool_pages = await list_skool_pages_helper(qdrant_client)
    skool_pages_str = "\n".join(skool_pages)

    prompt = f"""
    User Topic & Platform: {state['latest_user_message']}

    Create a detailed scope document for the AI agent including:
        - Structure of the post
        - What makes a post go viral
        - Structure of the hook
        - Structure of the body
        - Structure of the call to action
        - Any other rules the agent should follow when writing the post

    Do not write the post yourself, only write the scope.

    Base your scope on these documentation pages available:

    {skool_pages_str}

    Include a list of documentation pages that are relevant to creating this agent for the user in the scope document.
    """

    result = await reasoner.run(prompt)
    scope = result.data

    # Save the scope to a file
    scope_path = os.path.join("workbench", "scope.md")
    os.makedirs("workbench", exist_ok=True)

    with open(scope_path, "w", encoding="utf-8") as f:
        f.write(scope)

    return {"scope": scope}


async def writer_agent(state: AgentState, writer):
    # deps = PydanticAIDeps(qdrant_client=qdrant_client, reasoner_output=state["scope"])

    deps = PydanticAIDeps(qdrant_client=qdrant_client, reasoner_output="")

    message_history: list[ModelMessage] = []
    for message_row in state["messages"]:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    writer = get_stream_writer()
    result = await pydantic_ai_writer.run(
        state["latest_user_message"], deps=deps, message_history=message_history
    )

    writer(result.data)

    return {"messages": [result.new_messages_json()]}


def get_next_user_message(state: AgentState):
    value = interrupt({})

    # Set the user's latest message for the LLM to continue the conversation
    return {"latest_user_message": value}


# Determine if the user is finished creating their AI agent or not
async def route_user_message(state: AgentState):
    prompt = f"""
    The user has sent a message: 
    
    {state['latest_user_message']}

    If the user wants to end the conversation, respond with just the text "finish_conversation".
    If the user wants to continue coding the AI agent, respond with just the text "coder_agent".
    """

    result = await router_agent.run(prompt)
    next_action = result.data

    if next_action == "finish_conversation":
        return "finish_conversation"
    else:
        return "writer_agent"


# End of conversation agent to give instructions for executing the agent
async def finish_conversation(state: AgentState, writer):
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state["messages"]:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    writer = get_stream_writer()
    result = await end_conversation_agent.run(
        state["latest_user_message"], message_history=message_history
    )
    writer(result.data)

    return {"messages": [result.new_messages_json()]}


# Please don't delete this code.
# builder = StateGraph(AgentState)

# builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
# builder.add_node("writer_agent", writer_agent)
# builder.add_node("get_next_user_message", get_next_user_message)
# builder.add_node("finish_conversation", finish_conversation)

# builder.add_edge(START, "define_scope_with_reasoner")
# builder.add_edge("define_scope_with_reasoner", "writer_agent")
# builder.add_edge("writer_agent", "get_next_user_message")
# builder.add_conditional_edges(
#     "get_next_user_message",
#     route_user_message,
#     {"writer_agent": "writer_agent", "finish_conversation": "finish_conversation"},
# )
# builder.add_edge("finish_conversation", END)


# # Testing the writer agent
# builder = StateGraph(AgentState)

# builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
# builder.add_node("writer_agent", writer_agent)

# builder.add_edge(START, "define_scope_with_reasoner")
# builder.add_edge("define_scope_with_reasoner", "writer_agent")
# builder.add_edge("writer_agent", END)


# Testing the writer agent
builder = StateGraph(AgentState)

# builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
builder.add_node("writer_agent", writer_agent)

# builder.add_edge(START, "define_scope_with_reasoner")
builder.add_edge(START, "writer_agent")
builder.add_edge("writer_agent", END)


# Configure persistence
memory = MemorySaver()
agentic_flow = builder.compile(checkpointer=memory)

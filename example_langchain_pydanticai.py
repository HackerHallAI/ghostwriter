import os
import asyncio
from qdrant_client import QdrantClient
from langgraph import Graph, Node
from pydantic import BaseModel
from typing import TypedDict, Annotated, List
from ghostwriter.qdrant_utils import get_qdrant_client
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

# Initialize the Ollama model
ollama_model = OpenAIModel(model_name="llama3.2", base_url="http://localhost:11434/v1")
reasoner_llm_model = "deepseek-r1:14b"

reasoner = Agent(
    OpenAIModel(reasoner_llm_model, base_url="http://localhost:11434/v1"),
    system_prompt="You are an expert at writing viral content for social media.",
)


# Define a Pydantic model for the response
class QdrantResponse(BaseModel):
    title: str
    url: str
    content: str


class AgentState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]
    scope: str


def query_qdrant(collection_name: str, limit: int = 5) -> List[QdrantResponse]:
    """Query Qdrant and return a list of responses."""
    # TODO would change this when going to docker
    client = get_qdrant_client(mode="local")
    response, _ = client.scroll(collection_name=collection_name, limit=limit)
    return [QdrantResponse(**point.payload) for point in response]


def process_data_with_langgraph(data: List[QdrantResponse]) -> List[str]:
    """Process data using LangGraph."""
    graph = Graph()

    # Define a simple node that processes each QdrantResponse
    class ProcessNode(Node):
        def run(self, input_data: QdrantResponse) -> str:
            # Example processing: concatenate title and content
            return f"Title: {input_data.title}\nContent: {input_data.content[:100]}..."  # Limit content for display

    # Add nodes to the graph
    for item in data:
        node = ProcessNode(input_data=item)
        graph.add_node(node)

    # Execute the graph
    results = graph.execute()

    return results


async def generate_response_with_ollama(data: List[QdrantResponse]) -> List[str]:
    """Generate responses using the Ollama model."""
    agent = Agent(ollama_model, system_prompt="You are processing Qdrant data.")
    responses = []

    for item in data:
        context = RunContext(input_data=item.content)
        result = await agent.run(context)
        responses.append(result.data)

    return responses


async def chat_with_ollama():
    """Interactively chat with the Ollama model."""
    agent = Agent(ollama_model, system_prompt="You are a helpful assistant.")

    print("Chat with the Ollama model. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        context = RunContext(input_data=user_input)
        result = await agent.run(context)
        print(f"Ollama: {result.data}")


# Scope Definition Node with Reasoner LLM
async def define_scope_with_reasoner(state: AgentState):
    # First, get the useful sections of the skool pages

    skool_pages = await query_qdrant(collection_name="web_crawled_data")
    skool_pages_str = "\n".join(skool_pages)

    # Then, use the reasoner to define the scope
    prompt = f"""
    User AI Agent Request: {state['latest_user_message']}
    
    Create detailed scope document for the AI agent including:
    - Architecture diagram
    - Core components
    - External dependencies
    - Testing strategy

    Also based on these documentation pages available:

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


def get_next_user_message(state: AgentState):
    value = interrupt({})

    # Set the user's latest message for the LLM to continue the conversation
    return {"latest_user_message": value}


# Coding Node with Feedback Handling
async def coder_agent(state: AgentState, writer):
    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase, openai_client=openai_client, reasoner_output=state["scope"]
    )

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state["messages"]:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    writer = get_stream_writer()
    result = await pydantic_ai_coder.run(
        state["latest_user_message"], deps=deps, message_history=message_history
    )
    writer(result.data)

    # print(ModelMessagesTypeAdapter.validate_json(result.new_messages_json()))

    return {"messages": [result.new_messages_json()]}


def main():
    builder = StateGraph(AgentState)

    builder.add_node("get_next_user_message", get_next_user_message)

    # Initialize Qdrant client
    qdrant_client = get_qdrant_client(mode="local")

    # Define the collection name
    collection_name = "web_crawled_data"

    # Query Qdrant
    results = query_qdrant(qdrant_client, collection_name)

    # Process and respond with the data using LangGraph
    processed_results = process_data_with_langgraph(results)

    # Print the processed results
    for result in processed_results:
        print(f"Processed Response:\n{result}\n")

    # Start chat with Ollama model
    asyncio.run(chat_with_ollama())


if __name__ == "__main__":
    main()

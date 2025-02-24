from __future__ import annotations
from dotenv import load_dotenv
from typing import List
import asyncio
import logfire
import httpx
import os
from qdrant_client import QdrantClient
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from ai_writer import pydantic_ai_writer, PydanticAIDeps
from qdrant_utils import get_qdrant_client, get_ollama_flat_embedding_vector

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings
logfire.configure(send_to_logfire="never")


class CLI:
    def __init__(self):
        self.messages: List[ModelMessage] = []
        self.deps = PydanticAIDeps(
            qdrant_client=get_qdrant_client(mode="docker"),
            reasoner_output="",
        )

    async def chat(self):
        print("Ghostwriter Agent CLI (type 'quit' to exit)")
        print("What topic should we write about:")

        try:
            while True:
                user_input = input("> ").strip()
                if user_input.lower() == "quit":
                    break

                # Run the agent with streaming
                result = await pydantic_ai_writer.run(
                    user_input, deps=self.deps, message_history=self.messages
                )

                # Store the user message
                self.messages.append(
                    ModelRequest(parts=[UserPromptPart(content=user_input)])
                )

                # Store intermediary messages like tool calls and responses
                filtered_messages = [
                    msg
                    for msg in result.new_messages()
                    if not (
                        hasattr(msg, "parts")
                        and any(
                            part.part_kind == "user-prompt" or part.part_kind == "text"
                            for part in msg.parts
                        )
                    )
                ]
                self.messages.extend(filtered_messages)

                # Optional if you want to print out tool calls and responses
                # print(filtered_messages + "\n\n")

                print(result.data)

                # Add the final response from the agent
                self.messages.append(
                    ModelResponse(parts=[TextPart(content=result.data)])
                )
        finally:
            # Close the Qdrant client if necessary
            # await self.deps.qdrant_client.close()
            # If the Qdrant client does not need to be closed, remove this line
            # If it does, use the appropriate method (e.g., self.deps.qdrant_client.close())
            pass


async def main():
    cli = CLI()
    await cli.chat()


async def tmp():
    qdrant_client = get_qdrant_client(mode="docker")
    collection = qdrant_client.get_collection(collection_name="ship30")
    query_vector = await get_ollama_flat_embedding_vector(
        "What is the best way to write an essay?"
    )
    result = qdrant_client.search(collection_name="ship30", query_vector=query_vector)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(tmp())

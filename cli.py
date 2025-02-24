from __future__ import annotations

import asyncio
import uuid
from typing import List

from dotenv import load_dotenv
from langgraph.types import Command
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from qdrant_client import QdrantClient

from ai_writer import PydanticAIDeps, pydantic_ai_writer
from ghostwriter_graph import agentic_flow
from qdrant_utils import get_qdrant_client

# Load environment variables
load_dotenv()


class CLI:
    def __init__(self):
        self.messages: List[ModelMessage] = []
        self.deps = PydanticAIDeps(
            qdrant_client=get_qdrant_client(mode="docker"),
            reasoner_output="",
        )

    async def run_agent_with_stream(self, user_input: str):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # Ensure the input includes the required keys
        input_data = {
            "latest_user_message": user_input,
            "messages": self.messages,
            "scope": "",  # Initialize scope if needed
        }

        async for msg in agentic_flow.astream(
            {"latest_user_message": user_input}, config, stream_mode="custom"
        ):
            yield msg

    async def chat(self):
        print("Ghostwriter Agent CLI (type 'quit' to exit)")
        print("Describe to me a topic and I'll write a post about it.")
        print("Example: Write a post about the latest trends in AI.")

        try:
            while True:
                user_input = input("> ").strip()
                if user_input.lower() == "quit":
                    break

                # We append a new request to the conversation explicitly
                self.messages.append({"type": "human", "content": user_input})

                # Display user prompt in the CLI
                print(f"User: {user_input}")

                # Display assistant response
                response_content = ""
                async for chunk in self.run_agent_with_stream(user_input):
                    response_content += chunk
                    print(f"Assistant: {response_content}")

                self.messages.append({"type": "ai", "content": response_content})

        finally:
            # Close the Qdrant client if necessary
            # await self.deps.qdrant_client.close()
            pass


async def main():
    cli = CLI()
    await cli.chat()


if __name__ == "__main__":
    asyncio.run(main())

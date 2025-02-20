from __future__ import annotations
from typing import List
import streamlit as st
import asyncio
import uuid
import os

from langgraph.types import Command

from dotenv import load_dotenv
from ai_writer import pydantic_ai_writer, PydanticAIDeps
from qdrant_client import QdrantClient
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)

from ghostwriter_graph import agentic_flow

# Load environment variables
load_dotenv()

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333")

# Configure Streamlit
st.set_page_config(page_title="Ghostwriter", layout="wide")


@st.cache_resource
def get_thread_id() -> str:
    return str(uuid.uuid4())


thread_id = get_thread_id()


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # First message from user
    if len(st.session_state.messages) == 1:
        async for msg in agentic_flow.astream(
            {"latest_user_message": user_input}, config, stream_mode="custom"
        ):
            yield msg
    # Continue the conversation
    else:
        async for msg in agentic_flow.astream(
            Command(resume=user_input), config, stream_mode="custom"
        ):
            yield msg


async def main():
    st.title("Ghostwriter - AI Content Creator")
    st.write(
        "Describe the topic and platform for your viral post, and Ghostwriter will help you craft it."
    )

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_type = message["type"]
        if message_type in ["human", "ai"]:
            with st.chat_message(message_type):
                st.markdown(message["content"])

    # Chat input for the user
    user_input = st.chat_input("What topic should we write about today?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append({"type": "human", "content": user_input})

        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response in chat message container
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            # Run the async generator to fetch responses
            async for chunk in run_agent_with_streaming(user_input):
                response_content += chunk
                # Update the placeholder with the current response content
                message_placeholder.markdown(response_content)

        st.session_state.messages.append({"type": "ai", "content": response_content})


if __name__ == "__main__":
    asyncio.run(main())

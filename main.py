from __future__ import annotations
from typing import Literal, TypedDict
from langgraph.types import Command
from openai import AsyncOpenAI
import streamlit as st
import logfire
import asyncio
import uuid

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
)


from ghostwriter_graph import agentic_flow

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire="never")


@st.cache_resource
def get_thread_id():
    return str(uuid.uuid4())


thread_id = get_thread_id()


async def run_agent_with_stream(user_input: str):

    config = {"configurable": {"thread_id": thread_id}}

    # TODO Add in config if you want to be able to reference the session state.
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
    st.title("Ghostwriter - Agent Content Writer")
    st.write("Describe to me a topic and I'll write a post about it.")
    st.write("Example: Write a post about the latest trends in AI.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_type = message["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message["content"])

    # Chat input for the user
    user_input = st.chat_input("What do you want to build today?")

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
            async for chunk in run_agent_with_stream(user_input):
                response_content += chunk
                # Update the placeholder with the current response content
                message_placeholder.markdown(response_content)

        st.session_state.messages.append({"type": "ai", "content": response_content})


if __name__ == "__main__":
    asyncio.run(main())

from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from qdrant_client import QdrantClient
from typing import List
from qdrant_utils import get_ollama_flat_embedding_vector

load_dotenv()


# TODO Make sure to add doc strings to your agent tools because that is how it knows when to use them!


llm = os.getenv("PRIMARY_MODEL", "llama3.1:latest")
base_url = os.getenv("BASE_URL", "http://localhost:11434/v1")
model = OpenAIModel(model_name=llm, base_url=base_url, api_key="fakeshit!")


@dataclass
class PydanticAIDeps:
    qdrant_client: QdrantClient
    reasoner_output: str


# TODO: improve this but it is a fine starting point for testing.
system_prompt = """
~~ CONTEXT: ~~
You are an expert at writing viral content for social media. You are to take all of the following context and use it to 
write a viral post for the given topic on a given social media platform.

~~ GOAL: ~~
You will be given a topic and a list of documents that you can use to write a viral post for the given topic on a given social media platform.
The user will describe the topic and the social media platform.
You will take their requirements, and then search through the list of documents to find the most relevant information to write a viral post.

It's important for you to search through multiple pages to get all the information you need.
Almost never stick to just one page - use RAG and the other documentation tools multiple times when you are creating
a viral post for the user.

~~ STRUCTURE: ~~

When you write a viral post, should include the following components at a minimum:
- Title
- Hook
- Body
- Call to Action

~~ INSTRUCTIONS: ~~
- Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before writing the post.
- When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.
- Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.

"""


pydantic_ai_writer = Agent(
    model, system_prompt=system_prompt, deps_type=PydanticAIDeps, retries=2
)


@pydantic_ai_writer.system_prompt
def add_reasoner_output(ctx: RunContext[str]) -> str:
    """
    Add the reasoner output to the system prompt.

    Args:
        ctx (RunContext[str]): The context of the run.

    Returns:
        str: The system prompt with the reasoner output.
    """
    return f"""
    \n\nAdditional thoughts/instructions from the reasoner LLM. 
    This scope includes documentation pages for you to search as well: 
    {ctx.deps.reasoner_output}
    """


# TODO: This should not use the user query to get the vector.
# It was useful to figure out how to do this effectively but
# we actually just need to get all of the relevent information from the system prompt
# and create the basis of a post.
@pydantic_ai_writer.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[PydanticAIDeps], user_query: str
) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.

    Args:
        ctx (RunContext[PydanticAIDeps]): The context of the run.
        user_query (str): The query to retrieve relevant documentation for.

    Returns:
        str: The relevant documentation chunks.
    """
    qdrant_client = ctx.deps.qdrant_client
    query_vector = await get_ollama_flat_embedding_vector(user_query)
    result = qdrant_client.search(collection_name="ship30", query_vector=query_vector)

    if not result:
        return "No relevant documentation found."

    print(f"Results: {result}")

    formatted_chunks = []
    for doc in result:
        chunk_text = f"""
        # {doc.payload['title']}

        {doc.payload['content']}
        """
        formatted_chunks.append(chunk_text)

    output = "\n\n---\n\n".join(formatted_chunks)

    # Save the scope to a file
    scope_path = os.path.join("workbench", "rag_results.md")
    os.makedirs("workbench", exist_ok=True)

    with open(scope_path, "w", encoding="utf-8") as f:
        f.write(output)

    return output


async def list_skool_pages_helper(qdrant_client: QdrantClient) -> List[str]:
    """
    List all the skool pages from the vector database.

    Args:
        qdrant_client (QdrantClient): The qdrant client.

    Returns:
        List[str]: The urls of the skool pages.
    """
    try:
        result, _ = qdrant_client.scroll(collection_name="ship30")
        urls = [point.payload["url"] for point in result if "url" in point.payload]
        if not urls:
            print("No URLs found in the collection.")
        return urls
    except Exception as e:
        print(f"Error retrieving skool pages: {e}")
        return []


# TODO: query is not correct for qdrant fix this if you move this to a tool
# TODO I think that we need to reference the url chunks or title or something to get piece together larger chunks of information
# @pydantic_ai_writer.tool
# async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
#     """
#     Get the content of a page from the vector database.

#     Args:
#         ctx (RunContext[PydanticAIDeps]): The context of the run.
#         url (str): The url of the page to get the content of.

#     Returns:
#         str: The content of the page.
#     """
#     qdrant_client = ctx.deps.qdrant_client
#     result = qdrant_client.query(collection_name="ship30", query=url)
#     return result


# @pydantic_ai_writer.tool
# async def write_post(ctx: RunContext[PydanticAIDeps], post_type: str) -> str:
#     """
#     Write a post for the given social media platform aka post type.

#     Args:
#         ctx (RunContext[PydanticAIDeps]): The context of the run.
#         post_type (str): The type of post to write.

#     Returns:
#         str: The post.
#     """
#     return "Hello, world!"

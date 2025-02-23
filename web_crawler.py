"""
This is just going to crawl the list of urls and then we will turn that into the RAG data.
"""

import os
import asyncio
import uuid
from dotenv import load_dotenv
from typing import List, Dict, Any
from dataclasses import dataclass
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from playwright.async_api import Page, BrowserContext
from qdrant_client.http.models import PointStruct
from qdrant_utils import get_qdrant_client, ensure_collection_exists
from qdrant_client import QdrantClient
import datetime as dt
from datetime import timezone
from urllib.parse import urlparse
import ollama
import json
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
import re

load_dotenv()

model = OpenAIModel(
    model_name="deepseek-r1:14b",
    base_url="http://localhost:11434/v1",
    api_key="fakeshit!",
)

system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative.
    
    Example Response:
    ```json
    {
        "title": "How to Write Atomic Essays in 30 Days",
        "summary": "A guide on mastering Atomic Essay writing in 30 days, including tips on overcoming writer's block and applying effective writing techniques to any project."
    }
    ```
    """


ollama_client = ollama.AsyncClient()


@dataclass
class TitleAndSummary:
    title: str
    summary: str


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


title_and_summary_agent = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=TitleAndSummary,
    retries=3,
)


def read_markdown_file(file_path: str) -> str:
    """Read the content of a markdown file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


async def login_and_crawl(
    urls: List[str], output_dir: str, max_concurrent_requests: int = 5
):
    """Login to Skool and crawl multiple URLs, saving the content as markdown files."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        # TODO This is useful but removed images that were content. So we need to have AI
        # determine if they are content or just images of logos or people.
        # markdown_generator=DefaultMarkdownGenerator(options=dict(ignore_images=True)),
    )

    crawler = AsyncWebCrawler(config=browser_config)

    failed_urls = []

    async def on_page_context_created(page: Page, context: BrowserContext, **kwargs):
        # Perform login
        try:
            await page.goto(
                "https://www.skool.com/ship30for30/about",
                wait_until="networkidle",
                timeout=10000,
            )
            await page.click(
                "button[class='styled__ButtonWrapper-sc-dscagy-1 kkQuiY styled__SignUpButtonDesktop-sc-1y5fz1y-0 clPGTu']"
            )
            await page.fill("input[id='email']", os.getenv("SKOOL_EMAIL"))
            await page.fill("input[id='password']", os.getenv("SKOOL_PASSWORD"))
            await page.click("button[type='submit']")
            await page.wait_for_timeout(2000)
        except Exception as e:
            print(f"Error logging in: {e}")

    try:
        crawler.crawler_strategy.set_hook(
            "on_page_context_created", on_page_context_created
        )
    except Exception as e:
        print(f"Error setting hook: {e}")
        crawler.crawler_strategy.set_hook(
            "on_page_context_created", on_page_context_created
        )

    await crawler.start()

    try:

        semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def process_url(url: str):
            # this was testing the crawler
            # markdown_file_path = "data/30for30_md/og_test_url_1.md"
            # # Read the content from the markdown file
            # content = read_markdown_file(markdown_file_path)
            # await process_and_store_document(url, content)
            async with semaphore:
                result = await crawler.arun(
                    url, config=crawl_config, session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    content = result.markdown_v2.raw_markdown
                    await process_and_store_document(url, content)
                else:
                    print(f"Failed to crawl {url} - Error: {result.error_message}")
                    failed_urls.append(url)

        # Process all URLs concurrently
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

    if failed_urls:
        print(f"Failed to crawl {len(failed_urls)} URLs: {failed_urls}")
        with open("data/failed_urls.txt", "w") as file:
            file.write("\n".join(failed_urls))


def extract_title(content: str) -> str:
    """Extract the title from the content."""
    # Simple extraction logic: assume the first line is the title
    first_line = content.splitlines()[0]
    # Clean up the title to make it a valid filename
    title = first_line.strip().replace(" ", "_").replace("/", "_")
    return title


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Chunk the text into smaller chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end > text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind("```")

        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        elif "\n\n" in chunk:
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size * 0.3:
                end = start + last_break

        elif ". " in chunk:
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, end)

    return chunks


async def get_title_and_summary(chunk: str, url: str) -> Dict[str, Any]:
    """Get the title and summary using LLM."""
    # system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    # Return a JSON object with 'title' and 'summary' keys.
    # For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    # For the summary: Create a concise summary of the main points in this chunk.
    # Keep both title and summary concise but informative."""

    # response = await ollama_client.chat(
    #     model="deepseek-r1:14b",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {
    #             "role": "user",
    #             "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}...",
    #         },
    #     ],
    #     format="json",
    # )

    # print(f"Response: {response.message.content}")

    # # Access the response content directly if 'choices' is not available
    # response_content = json.loads(response.message.content)

    # print(f"Response content: {response_content}")

    # return response_content

    # TODO: worst case is we give this to the chat portion above and tell it to return the right format. Giving it 0 tempature and 0 top_p.

    response = await title_and_summary_agent.run(
        f"URL: {url}\n\nContent:\n{chunk[:1000]}..."
    )

    # Use a regular expression to extract the JSON part
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response.data, re.DOTALL)

    if json_match:
        json_part = json_match.group(1)
        try:
            results = json.loads(json_part)
            return results
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {"title": "Unknown", "summary": "Failed to extract summary"}
    else:
        print("No valid JSON found in the response data.")
        return {"title": "Unknown", "summary": "Failed to extract summary"}


async def get_embedding(text: str) -> List[float]:
    """Get the embedding for a text using LLM."""
    # try:
    response = await ollama_client.embed(
        model="nomic-embed-text",
        input=text,
    )

    # TODO: if something is broken it is probably this!
    return response["embeddings"]
    # except Exception as e:
    #     print(f"Error getting embedding: {e}")
    #     return [0.0] * 768


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    extracted = await get_title_and_summary(chunk, url)
    # extracted = {"title": "test", "summary": "test"}
    embedding = await get_embedding(chunk)

    # TODO: Update the source when we start to have more than one source!
    metadata = {
        "source": "30for30_skool",
        "chunk_size": len(chunk),
        "crawled_at": dt.datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path,
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted["title"],
        summary=extracted["summary"],
        content=chunk,
        metadata=metadata,
        embedding=embedding,
    )


async def insert_chunk(chunk: ProcessedChunk):
    """Insert a chunk into Qdrant."""
    qdrant_client = get_qdrant_client(mode="docker")

    def flatten_embedding(embedding: List[List[float]]) -> List[float]:
        return [item for sublist in embedding for item in sublist]

    # Use this function to flatten your embedding before inserting it into Qdrant
    flattened_embedding = flatten_embedding(chunk.embedding)

    point = PointStruct(
        id=str(uuid.uuid4()),  # Use a unique identifier
        vector=flattened_embedding,  # Use the flattened embedding as the vector
        payload={
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
        },
    )

    qdrant_client.upsert(collection_name="ship30", points=[point])


async def process_and_store_document(url: str, markdown: str):
    """Process and store a document in XXX."""
    chunks = chunk_text(markdown)

    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]

    processed_chunks = await asyncio.gather(*tasks)

    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]

    await asyncio.gather(*insert_tasks)


async def main():
    # Read URLs from file
    with open("data/skool_urls.txt", "r") as file:
        urls = file.read().splitlines()

    # Create output directory if it doesn't exist
    output_dir = "data/30for30_md"
    os.makedirs(output_dir, exist_ok=True)

    urls = urls[0:1]

    # Initialize Qdrant client with local persistence
    qdrant_client = get_qdrant_client(
        mode="docker"
    )  # Change to "docker" for Docker mode or "local" for local mode

    # Ensure the collection exists
    vector_size = 768  # Set to the correct vector size
    ensure_collection_exists(qdrant_client, "ship30", vector_size)

    # Crawl all URLs
    if urls:
        await login_and_crawl(urls, output_dir)
    else:
        print("No URLs found to crawl.")


if __name__ == "__main__":
    asyncio.run(main())

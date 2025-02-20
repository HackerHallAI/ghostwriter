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

load_dotenv()


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


async def login_and_crawl(
    urls: List[str], output_dir: str, qdrant_client: QdrantClient
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

    async def on_page_context_created(page: Page, context: BrowserContext, **kwargs):
        # Perform login
        await page.goto(
            "https://www.skool.com/ship30for30/about",
            wait_until="networkidle",
            timeout=60000,
        )
        await page.click(
            "button[class='styled__ButtonWrapper-sc-dscagy-1 kkQuiY styled__SignUpButtonDesktop-sc-1y5fz1y-0 clPGTu']"
        )
        await page.fill("input[id='email']", os.getenv("SKOOL_EMAIL"))
        await page.fill("input[id='password']", os.getenv("SKOOL_PASSWORD"))
        await page.click("button[type='submit']")
        await page.wait_for_timeout(2000)

    crawler.crawler_strategy.set_hook(
        "on_page_context_created", on_page_context_created
    )

    await crawler.start()

    try:

        async def process_url(url: str):
            result = await crawler.arun(url, config=crawl_config, session_id="session1")
            if result.success:
                content = result.markdown_v2.raw_markdown

                # Extract title from content
                title = extract_title(content)

                # Create a CrawledData instance
                crawled_data = ProcessedChunk(title=title, url=url, content=content)

                # Save content to markdown file using title as filename
                file_path = os.path.join(output_dir, f"{crawled_data.title}.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(crawled_data.content)

                print(f"Successfully crawled and saved: {url}")

                # Generate a UUID for the point
                point_id = str(uuid.uuid4())

                # Use a placeholder vector of the correct size
                vector_size = 4  # Set to the correct vector size
                placeholder_vector = [0.0] * vector_size

                # Insert data into Qdrant
                point = PointStruct(
                    id=point_id,  # Use generated UUID as a unique identifier
                    vector=placeholder_vector,  # Use placeholder vector
                    payload={
                        "title": crawled_data.title,
                        "url": crawled_data.url,
                        "content": crawled_data.content,
                    },
                )
                qdrant_client.upsert(collection_name="web_crawled_data", points=[point])

            else:
                print(f"Failed to crawl {url} - Error: {result.error_message}")

        # Process all URLs concurrently
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()


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
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    try:
        response = await ollama.chat(
            model="deepseek-r1:14b",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}...",
                },
            ],
            format="json",
            stream=False,
        )

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {
            "title": "Error processing title",
            "summary": "Error processing summary",
        }


async def get_embedding(text: str) -> List[float]:
    """Get the embedding for a text using LLM."""
    try:
        response = await ollama.embed(
            model="nomic-embed-text",
            input=text,
        )
        # TODO: if something is broken it is probably this!
        return response["embeddings"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0.0] * 1536


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    extracted = await get_title_and_summary(chunk, url)

    embedding = await get_embedding(chunk)

    # TODO: Update the source when we start to have more than one source!
    metadata = {
        "source": "30for30_skool",
        "chunk_size": len(chunk),
        "crawled_at": dt.now(timezone.utc).isoformat(),
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
    pass


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
    vector_size = 4  # Set to the correct vector size
    ensure_collection_exists(qdrant_client, "web_crawled_data", vector_size)

    # Crawl all URLs
    if urls:
        await login_and_crawl(urls, output_dir, qdrant_client)
    else:
        print("No URLs found to crawl.")


if __name__ == "__main__":
    asyncio.run(main())

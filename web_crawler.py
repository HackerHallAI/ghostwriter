"""
This is just going to crawl the list of urls and then we will turn that into the RAG data.
"""

import os
import asyncio
from dotenv import load_dotenv
from typing import List
from dataclasses import dataclass
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from playwright.async_api import Page, BrowserContext
from qdrant_client.http.models import PointStruct
from ghostwriter.qdrant_utils import get_qdrant_client, ensure_collection_exists
from qdrant_client import QdrantClient

load_dotenv()


@dataclass
class CrawledData:
    title: str
    url: str
    content: str


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
                crawled_data = CrawledData(title=title, url=url, content=content)

                # Save content to markdown file using title as filename
                file_path = os.path.join(output_dir, f"{crawled_data.title}.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(crawled_data.content)

                print(f"Successfully crawled and saved: {url}")

                # Insert data into Qdrant
                point = PointStruct(
                    id=url,  # Use URL as a unique identifier
                    vector=[],  # Placeholder for vector data
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
        mode="local"
    )  # Change to "docker" for Docker mode

    # Ensure the collection exists
    ensure_collection_exists(qdrant_client, "web_crawled_data")

    # Crawl all URLs
    if urls:
        await login_and_crawl(urls, output_dir, qdrant_client)
    else:
        print("No URLs found to crawl.")


if __name__ == "__main__":
    asyncio.run(main())

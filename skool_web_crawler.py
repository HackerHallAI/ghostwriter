"""
Currently this is not actually crawling the different pages of the classroom.
I do not want to hold up progress of the rest of the project for this.

In the future I may fix this because I think it might be useful for someone in the open source community. 

For now I am going to manually grab all of the urls.
"""

import os
from dotenv import load_dotenv
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from playwright.async_api import Page, BrowserContext, ElementHandle

load_dotenv()


async def main():
    browser_config = BrowserConfig(headless=False, verbose=True)

    crawler_run_config = CrawlerRunConfig(
        js_code="window.scrollTo(0, document.body.scrollHeight);",
        wait_for="body",
        cache_mode=CacheMode.BYPASS,
        # css_selector=["div[role='button']"],  # added this to try and click them all.
    )

    crawler = AsyncWebCrawler(config=browser_config)

    async def on_page_context_created(page: Page, context: BrowserContext, **kwargs):
        # Called right after a new page + context are created (ideal for auth or route config).
        print("[HOOK] on_page_context_created - Setting up page & context.")

        await page.goto("https://www.skool.com/ship30for30/about")
        await page.click(
            "button[class='styled__ButtonWrapper-sc-dscagy-1 kkQuiY styled__SignUpButtonDesktop-sc-1y5fz1y-0 clPGTu']"
        )
        print("Filling in username  ")
        await page.fill("input[id='email']", os.getenv("SKOOL_EMAIL"))
        await page.fill("input[id='password']", os.getenv("SKOOL_PASSWORD"))
        print("Clicking submit")
        await page.click("button[type='submit']")
        print("Waiting")
        await page.wait_for_timeout(2000)
        print("Navigating to classroom")
        await page.goto("https://www.skool.com/ship30for30/classroom")
        print("URL found")

        # Extract all divs with role='button'
        buttons: list[ElementHandle] = await page.query_selector_all(
            "div[role='button']"
        )
        print(f"Found {len(buttons)} buttons with role='button'.")

        urls = []

        # Iterate over each button and perform actions
        for button in buttons:
            # Ensure the button is visible and clickable
            await page.wait_for_selector("div[role='button']", state="visible")

            # Use JavaScript to click the button if normal click doesn't work
            await page.evaluate("(button) => button.click()", button)

            # Wait for potential navigation or async operation
            await page.wait_for_load_state("networkidle")

            # Check if the URL has changed
            current_url = page.url
            print(f"Current URL after click: {current_url}")

            # Optionally, you can extract the URL or perform other actions
            urls.append(page.url)
            print(f"Navigated to: {page.url}")

            await page.wait_for_timeout(10000)

            # Navigate back if needed
            await page.go_back()
            break

        print(f"Found {len(urls)} urls")
        print(urls)

        return page

    crawler.crawler_strategy.set_hook(
        "on_page_context_created", on_page_context_created
    )

    await crawler.start()

    url = "https://www.skool.com/ship30for30/classroom/"
    result = await crawler.arun(url, config=crawler_run_config)

    if result.success:
        internal_links = result.links.get("internal", [])
        print(f"Found {len(internal_links)} internal links.")
        if internal_links:
            print("Sample Internal Link:", internal_links[0])

        print("\nCrawled URL:", result.url)
        print("HTML length:", len(result.html))
        print(f"\n{result.markdown}\n")
    else:
        print("Error:", result.error_message)

    await crawler.close()


if __name__ == "__main__":
    asyncio.run(main())

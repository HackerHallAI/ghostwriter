import os
import uuid
from qdrant_client.http.models import PointStruct
from ghostwriter.qdrant_utils import get_qdrant_client, ensure_collection_exists


def read_markdown_file(file_path: str) -> str:
    """Read the content of a markdown file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_title(content: str) -> str:
    """Extract the title from the content."""
    # Simple extraction logic: assume the first line is the title
    first_line = content.splitlines()[0]
    # Clean up the title to make it a valid filename
    title = first_line.strip().replace(" ", "_").replace("/", "_")
    return title


def main():
    # Path to the markdown file
    markdown_file_path = "data/30for30_md/og_test_url_1.md"

    # Read the content from the markdown file
    content = read_markdown_file(markdown_file_path)

    # Extract title from content
    title = extract_title(content)

    # Initialize Qdrant client with local persistence
    qdrant_client = get_qdrant_client(mode="local")

    # Ensure the collection exists
    ensure_collection_exists(qdrant_client, "web_crawled_data", vector_size=0)

    # Generate a UUID for the point ID
    point_id = str(uuid.uuid4())

    # Insert data into Qdrant
    point = PointStruct(
        id=point_id,  # Use a valid UUID
        vector=[],  # Placeholder for vector data
        payload={
            "title": title,
            "url": "https://www.skool.com/ship30for30/classroom/d5bbd4ff?md=6e5b391606c545eb9d8c7f2347421aa9",
            "content": content,
        },
    )
    qdrant_client.upsert(collection_name="web_crawled_data", points=[point])

    print(f"Successfully inserted data for: {title} with ID: {point_id}")


if __name__ == "__main__":
    main()

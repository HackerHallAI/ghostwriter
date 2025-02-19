from ghostwriter.qdrant_utils import get_qdrant_client


def main():
    # Initialize Qdrant client with local persistence
    qdrant_client = get_qdrant_client(mode="local")

    # Define the collection name
    collection_name = "web_crawled_data"

    # Query the collection to retrieve all points
    response, _ = qdrant_client.scroll(collection_name=collection_name, limit=10)

    # Define the specific URL to look for
    specific_url = "https://www.skool.com/ship30for30/classroom/d5bbd4ff?md=6e5b391606c545eb9d8c7f2347421aa9"

    # Print the retrieved data for the specific URL
    for point in response:
        if point.payload["url"] == specific_url:
            print(f"ID: {point.id}")
            print(f"Title: {point.payload['title']}")
            print(f"URL: {point.payload['url']}")
            print(f"Content:\n{point.payload['content']}")
            print("-" * 40)


if __name__ == "__main__":
    main()

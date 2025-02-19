
# Initial Design Document / Roadmap

Just thinking out loud here. Ghostwriter is a project that I want to build a web app for.

First thing we want to do is crawl the web and get all of the data for the project.

We must save the templates and data to the vector database. (qdrant or supabase)

Then I think we want to build the AI agent that will help write our articles.

This will be part pydantic AI and part langchain.

At first we are going to have more human interaction because I think that is important to people writing articles, correctly.

We will also have a research agent that looks up information about the topic of the article.
    Within this I think there is the search the internet portion.
    Then there is the crawl the web portion.
    Finally there is a search our vector database portion.
    So possibly 3 agents here, just for this section.

We will need an agent that can write the article.
    This will be walking through the steps to building an article.
    Starting with type of article. (blog post, whitepaper, linkedin post, etc.)
    The initial thing will ask us for an idea for an article.
    Then it will help us create a title. 
    We will get more in the weeds here based on Cole's writing rules so this is unfinished.


We will need an agent that fact checks the article.
    This will use the search and crawler agents to make sure our posts are factually correct.

We will need an agent that edits the article into our style.
    That part will be part of the fine tuning process because this is the part that changes depending on the user.

Each step I believe the human will give the approval, but we can make a version that is all agentic if we want to, as well.

Probably have that be a toggle in the UI.

The UI will be built with streamlit.


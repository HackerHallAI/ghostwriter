
# Initial Design Document / Roadmap

Just thinking out loud here. Ghostwriter is a project that I want to build a web app for.

First thing we want to do is crawl the web and get all of the data for the project.

We must save the templates and data to the vector database. (qdrant or supabase)

Then I think we want to build the AI agent that will help write our articles.

---
*** Currently here ***
Start with just a simple agent that can read the data from the vector database.
And use that as context to answer a question about the data.

Then move on to having it write an article. 
** Save the first article **
This is to show it's growth.

When you get that done add the ability to use streamlit to chat with the agent.
When you get that part done I would add a research agent.
---

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




Knowledge base
- Rag Reader (ship30for30)
- Internet Search
- my writings
- my notes
- questions about me or the topic I am interested in


I think those are all ai agents that feed the reasoning agent.


draw a structure tho.


So we are prompted like this:

1. Do you have an idea for an article?
2. Do you want to research a topic?
3. Do you want to write an article?
4. Do you want to fact check an article?
5. Do you want to edit an article?


but actually

1. Do you want to co-write an article?
2. Do you want me to write an article?


if 1 is selected
    1. Do you want to use Idea Generator?
    2. Do you already have an idea?


Flow

Do you want to co-write an article?
    - Yes
    - No

Do you have an idea for an article?
    - Yes
    - No

What medium do you want to write the article for?
    - Long Form Article
    - Short Form Article
    - Newsletter
    - LinkedIn Post
    - Twitter Post

Which of the 4 styles do you want to write in? (Get these from the RAG information from ship30for30)
    - Style 1
    - Style 2
    - Style 3
    - Style 4
    - Let AI choose for you





(Agent 1)
Idea Generator
    - suggests 3 topics
    - when you pick one it asks you which of the styles of articles do you want to write. Which it gets from the RAG information from ship30for30.
    - Has access to the articles we have already written to prevent duplicates without changing styles.


(Agent 2)
Notes collector
    - Looks through my history of notes and thoughts to feed my bias and opinions to the writer agent.
    - So this needs to watch my notes and trigger the database to update if I add a new note or thought.
    - I am going to start with my Obsidian notes because they live locally on my computer. 
        - Eventually we would want to connect APis for major note taking apps like Notion, Roam, etc.


(Agent 3)
Internet Search
    - Searches the internet for information about the topic.


(Agent 4)
Writer Agent
    - Goal is to write the article based on the notes and thoughts of the Notes collector and the internet search of the Internet Search agent.
    - It will also use the RAG information from ship30for30 to help formulate the article.
    - It will need to know the platform and style of the article.
    - If it is co-write then all of the steps in here will be done together asking for approval at each step.


(Agent 5)
Editor Agent
    - Edits the article to make sure it is grammatically correct and flows well.
    - It will also make sure the article is written in the style of the Notes collector.


(Agent 6)
Fact Checker Agent
    - Looks through the article and checks for factual correctness.
    - It will use the internet search and the RAG information from ship30for30 to help check the facts.
    - It will cite the sources of the information to be checked against the internet search and the RAG information from ship30for30.


(Agent 7)
Reasoning Agent
    - This is the main agent.
    - It will use the Notes collector, Internet Search, Writer Agent, Editor Agent, and Fact Checker Agent to help it reason about the article.
    - It will also use the Base Knowledge to help it reason about the article.
    - Only when it is satisfied that the article is factually correct and written in the style of the Notes collector will it be given to the human to review.
    - If you are unsure about anything you will ask the human for approval. (This is co-write only)


(Agent 8)
Fine Tuner Agent
    - This agent will take our previous writings and styles to help fine tune the Writer Agent, Editor Agent, & Reasoning Agent.
    - This takes my articles and feeds them into a model to get the prompts that would have been used to write them. 
    - It then uses that to fine tune the Writer Agent, Editor Agent, & Reasoning Agent.


(Fine Tuning Model)
Base Knowledge
    - My ideal customer
    - My mission
    - My values
    - My goals
    - My tone
    - My style
    - My audience
    - Brand breif
    - Brand voice


(This part will be last to be built)
if 2 is selected all of this is automated. AI looks at recent notes and thoughts and suggests a topic. Then writes the article for you.





## Run Qdrant

docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant


# Notes

retrieve_relevant_documentation
TODO: This should not use the user query to get the vector. 
It was useful to figure out how to do this effectively but
we actually just need to get all of the relevent information from the system prompt
and create the basis of a post. 
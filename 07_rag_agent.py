"""
07_rag_agent.py: RAG agent: retriever as a tool
RUN:
    uv run 07_rag_agent.py

WHAT IT DOES:
    - Wraps the vector store from 06 into a tool the agent can call.
    - The agent decides what to search for; you don't hardcode the query.
    - The agent calls the tool, reads the results, and writes a natural-language answer.

EXPECTED OUTPUT:
    Based on the stored notes, you enjoy a variety of fruits ...

DIFFERENCE FROM 06:
    In 06 the query was hardcoded: similarity_search("What fruits do I like?", k=3)
    Here the LLM figures out what to search for.
    If the user asks "Do I like exotic fruits?", the agent constructs an appropriate query.

MODEL NOTE:
    We use Qwen/Qwen3-32B here.
    Llama 3.3 sometimes generates 2 tool calls in a single turn for multi-topic questions,
    which Nebius rejects ("This model only supports single tool-calls at once").
    Qwen3-32B avoids this issue.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent

load_dotenv()

# Build the vector store (same setup as 06_rag_basic.py).
embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL"),
    check_embedding_ctx_length=False,
)

vector_store = InMemoryVectorStore(embedding=embeddings)

documents = [
    Document(page_content="I love apples."),
    Document(page_content="Bananas are my favourite fruit."),
    Document(page_content="I really dislike oranges."),
    Document(page_content="I enjoy eating grapes."),
    Document(page_content="Strawberries are amazing."),
    Document(page_content="I hate durian."),
]

vector_store.add_documents(documents)

# Wrap the retriever as a tool.
# as_retriever() wraps the vector store in a "take query, return documents" interface.
# create_retriever_tool() makes that retriever callable by the LLM.
# description teaches the model when and how to call this tool.
retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    name="fruit_preferences",
    description="Search stored notes about the user's fruit preferences. Input: a plain English question.",
)

# Qwen3-32B handles multi-topic questions without generating parallel tool calls.
llm = ChatOpenAI(
    model="Qwen/Qwen3-32B",
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL"),
)

agent = create_agent(
    model=llm,
    tools=[retriever_tool],
    system_prompt="You are a helpful assistant. Use the fruit_preferences tool to answer questions about the user's tastes.",
)

# The agent builds its own query, calls the tool, and writes the answer.
response = agent.invoke({
    "messages": [{"role": "user", "content": "What fruits do I like and which ones do I dislike?"}]
})
print(response["messages"][-1].content)

"""
06_rag_basic.py: RAG fundamentals: vector store and semantic search
RUN:
    uv run 06_rag_basic.py

WHAT IT DOES:
    - Stores a few sentences in a vector store.
    - Runs a query and retrieves the semantically closest sentences.
    - This is semantic search, not keyword search.

EXPECTED OUTPUT:
    Semantic search: 'What fruits do I like?'
     * I love apples.
     * I enjoy eating grapes.
     * Bananas are my favourite fruit.

    Semantic search: 'technology companies'
     * Anthropic is an AI safety company.
     ...

KEY CONCEPTS:

    WHAT IS AN EMBEDDING?
        Converting text into a list of numbers (a vector).
        "I love apples" -> [0.12, -0.34, 0.87, ...]  (thousands of numbers)
        Sentences with similar meaning produce vectors that are close together.

    WHAT IS A VECTOR STORE?
        A store that holds those vectors and can answer "find the closest one."
        Here we use InMemoryVectorStore (RAM only, gone when the process exits).
        In production: Chroma, Pinecone, pgvector, Weaviate.

    WHY SEMANTIC SEARCH?
        Keyword search for "apple" won't match "fruit preference."
        Semantic search for "What fruits do I like?" does match "I love apples,"
        because they're close in meaning space even without shared words.

    WHY check_embedding_ctx_length=False?
        By default, langchain_openai tokenizes text and sends integer arrays.
        Nebius only accepts raw strings. This flag disables tokenization.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

load_dotenv()

# Embedding model: converts text to a numeric vector.
# Qwen3-Embedding-8B is available on Nebius.
embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL"),
    # Nebius expects raw strings; tokenized integer arrays are rejected.
    check_embedding_ctx_length=False,
)

# InMemoryVectorStore stores vectors in RAM. Good for learning, not for production.
vector_store = InMemoryVectorStore(embedding=embeddings)

# Document is the basic data type: page content plus optional metadata.
documents = [
    Document(page_content="I love apples."),
    Document(page_content="Bananas are my favourite fruit."),
    Document(page_content="I dislike oranges."),
    Document(page_content="I enjoy eating grapes."),
    Document(page_content="Anthropic is an AI safety company."),
]

# add_documents embeds each document and stores the result.
vector_store.add_documents(documents)

# similarity_search embeds the query, then returns the k closest documents.
print("Semantic search: 'What fruits do I like?'")
results = vector_store.similarity_search("What fruits do I like?", k=3)
for r in results:
    print(" *", r.page_content)

# "Anthropic" scores low on the fruit query: even though it shares no bad words,
# it sits far from "fruit preference" in the meaning space.
print("\nSemantic search: 'technology companies'")
results = vector_store.similarity_search("technology companies", k=3)
for r in results:
    print(" *", r.page_content)

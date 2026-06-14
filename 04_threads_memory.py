"""
04_threads_memory.py: Automatic memory with MemorySaver and thread IDs
RUN:
    uv run 04_threads_memory.py

WHAT IT DOES:
    - Provides persistent memory without manually managing history like in 02.
    - Separates conversations by thread_id.
    - Same thread_id: the model remembers previous turns.
    - Different thread_id: fresh start, no memory.

EXPECTED OUTPUT:
    T1 -> Nice to meet you, Mesut! ...
    T1 recall -> Your name is Mesut.
    T2 (fresh thread) -> I don't know your name ...

HOW IT WORKS:
    MemorySaver saves the agent's full state (including all messages) after each invoke().
    thread_id is the key used to store and retrieve that state.

    On the next call with the same thread_id, the saved state is loaded automatically,
    so previous messages are already in context.

    For production use, replace MemorySaver with:
    - SqliteSaver    -> writes to a SQLite file (survives process restarts)
    - PostgresSaver  -> writes to PostgreSQL (for multi-user systems)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL"),
)

# MemorySaver keeps everything in RAM. It's gone when the process exits.
# Ideal for development and testing.
memory = MemorySaver()

# checkpointer=memory: after each step, the agent saves its state.
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant.",
    checkpointer=memory,
)

# The thread_id inside config determines which saved conversation to load or write.
thread_1 = {"configurable": {"thread_id": "thread_1"}}
thread_2 = {"configurable": {"thread_id": "thread_2"}}

# Thread 1: introduce a name.
# This message is saved under thread_1.
r = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Mesut."}]},
    config=thread_1,
)
print("T1 ->", r["messages"][-1].content)

# Thread 1: ask again. The model still remembers.
# MemorySaver reloads thread_1's state; previous messages are in context.
r = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config=thread_1,
)
print("T1 recall ->", r["messages"][-1].content)

# Thread 2: different thread_id, no history.
# Nothing has been saved for thread_2, so the model has no idea.
r = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config=thread_2,
)
print("T2 (fresh thread) ->", r["messages"][-1].content)

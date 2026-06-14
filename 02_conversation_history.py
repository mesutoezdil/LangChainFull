"""
02_conversation_history.py: Manual conversation history
RUN:
    uv run 02_conversation_history.py
    (interactive: type a question, press Enter. Type "exit" to quit.)

WHAT IT DOES:
    - Solves a core LLM problem: each call is stateless, the model forgets previous messages.
    - Fix: send the full conversation history on every call.
    - This file does that manually: append each message to a list, send the whole list.

HOW IT WORKS:
    history = [SystemMessage]       <- starts with only the system message
    user types     -> append HumanMessage
    invoke(history) -> returns AIMessage
    append AIMessage -> next call will include it in context

NOTE:
    This approach is simple but has limits:
    - Cost grows as history grows (you send everything every time).
    - History is lost when the process exits.
    For a persistent, automatic solution, see 04_threads_memory.py.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL"),
)

# history holds the full conversation. SystemMessage is always first.
history = [
    SystemMessage(content="You are a helpful assistant."),
]

print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    # 1. Add the user's message to history.
    history.append(HumanMessage(content=user_input))

    # 2. Send the full history. The model sees all previous turns this way.
    response = llm.invoke(history)

    # 3. Add the model's reply to history so it appears in the next turn's context.
    history.append(AIMessage(content=response.content))

    print(f"AI: {response.content}\n")

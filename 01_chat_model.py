"""
01_chat_model.py: Calling the LLM directly and swapping models
RUN:
    uv run 01_chat_model.py

WHAT IT DOES:
    - Calls the LLM directly: no agent, no tools.
    - Shows how to swap models without changing any other code.
    - Shows how to shape model behavior with SystemMessage.

EXPECTED OUTPUT:
    Basic: The capital of France is Paris.
    With system: The capital of France is Paris.
    Smaller model: The capital of Germany is Berlin.

KEY IDEA:
    LangChain gives every model the same interface.
    Swapping models means changing one string. Everything else stays the same.
    ChatOpenAI connects to OpenAI, Nebius, and other providers equally.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


def get_llm(model="meta-llama/Llama-3.3-70B-Instruct"):
    # base_url redirects OpenAI-format requests to Nebius.
    # Change only the model string; everything else is identical.
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("NEBIUS_API_KEY"),
        base_url=os.getenv("NEBIUS_BASE_URL"),
    )


llm = get_llm()

# 1. Simplest usage: send a string, get a string back.
# invoke() accepts a string or a list of messages; it always returns an AIMessage.
response = llm.invoke("What is the capital of France?")
print("Basic:", response.content)

# 2. Control behavior with SystemMessage.
# SystemMessage tells the model who it is and how to behave.
# HumanMessage is the user's question.
# Order matters: system first, then user messages.
messages = [
    SystemMessage(content="You are a helpful assistant that responds in exactly one sentence."),
    HumanMessage(content="What is the capital of France?"),
]
response = llm.invoke(messages)
print("With system:", response.content)

# 3. Different model, identical code.
# Only the model name changed. LangChain handles the rest.
llm_small = get_llm(model="google/gemma-3-27b-it")
response = llm_small.invoke("What is the capital of Germany?")
print("Smaller model:", response.content)

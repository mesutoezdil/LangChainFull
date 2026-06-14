"""
09_middleware_model.py: Dynamic model selection with @wrap_model_call
RUN:
    uv run 09_middleware_model.py

WHAT IT DOES:
    - Uses a cheap, small model (Gemma) for short conversations.
    - Automatically upgrades to a powerful model (Llama) once the conversation grows.
    - The switch is invisible to the user and to the agent.

EXPECTED OUTPUT:
    [middleware] 1 msgs -> basic model
    Q: What is 2 + 2?
    A: 4

    [middleware] 3 msgs -> basic model
    Q: What is the capital of Japan?
    ...

    [middleware] 5 msgs -> advanced model
    Q: Explain the theory of relativity.
    ...

WHAT IS @wrap_model_call?
    It wraps the model call itself:
    - request: current state (messages, model info, etc.)
    - handler: the function that actually calls the model
    - request.override(model=...): swap the model for this one call only
    - return handler(request): run the model after the swap

WHY DO THIS?
    Short questions are cheap on a small model, cutting costs.
    As the conversation grows and gets complex, quality stays high by upgrading.
    The user interface doesn't change; the upgrade happens entirely in the background.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call

load_dotenv()


def make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("NEBIUS_API_KEY"),
        base_url=os.getenv("NEBIUS_BASE_URL"),
    )


# Fast and cheap for short conversations.
BASIC_MODEL = make_llm("google/gemma-3-27b-it")

# More capable for longer, more complex conversations.
ADVANCED_MODEL = make_llm("meta-llama/Llama-3.3-70B-Instruct")

# Switch to the advanced model once the conversation exceeds this many messages.
UPGRADE_AFTER = 3


# @wrap_model_call wraps every model call.
# request.state["messages"] is the full message list so far.
# request.override(model=...) temporarily swaps the model for this call only.
@wrap_model_call
def select_model(request, handler):
    msg_count = len(request.state["messages"])
    if msg_count > UPGRADE_AFTER:
        print(f"  [middleware] {msg_count} msgs -> advanced model")
        request = request.override(model=ADVANCED_MODEL)
    else:
        print(f"  [middleware] {msg_count} msgs -> basic model")
    # handler(request) runs the actual model call and returns the result.
    return handler(request)


agent = create_agent(
    model=BASIC_MODEL,
    tools=[],
    system_prompt="You are a helpful assistant.",
    middleware=[select_model],
)

exchanges = [
    "What is 2 + 2?",
    "What is the capital of Japan?",
    "Explain the theory of relativity.",
    "What are the implications of quantum entanglement for cryptography?",
]

# Send each question with the full history so context accumulates.
messages = []
for question in exchanges:
    messages.append({"role": "user", "content": question})
    r = agent.invoke({"messages": messages})
    reply = r["messages"][-1].content
    messages.append({"role": "assistant", "content": reply})
    print(f"Q: {question}\nA: {reply}\n")

"""
08_middleware_prompt.py: Dynamic system prompt with @dynamic_prompt middleware
RUN:
    uv run 08_middleware_prompt.py

WHAT IT DOES:
    - Sends the same question to 3 different user roles: expert, beginner, child.
    - Each role gets a different system prompt injected at runtime.
    - Same model, same question; the output tone and vocabulary change completely.

EXPECTED OUTPUT:
    [EXPERT]
    Photosynthesis is a biochemical process wherein photoautotrophic organisms...

    [BEGINNER]
    Photosynthesis is a process that happens in plants...

    [CHILD]
    Photosynthesis is like a superpower for plants!...

WHAT IS @dynamic_prompt?
    A middleware decorator that runs before every model call.
    The function returns a string, which becomes the system prompt for that turn.
    You don't hard-code the prompt at agent creation time; it resolves at runtime.

WHAT DOES context_schema DO?
    It lets you pass data into agent.invoke() via context=...
    Inside middleware, you access that data through request.runtime.context.
    The LLM never sees this data; only your code does.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest

load_dotenv()


@dataclass
class UserContext:
    role: str  # "expert", "beginner", or "child"


# Ready-made system prompts, one per role.
PROMPTS = {
    "expert":   "You are a highly technical assistant. Use precise terminology and keep answers dense.",
    "beginner": "You are a friendly assistant. Use plain language and explain every concept step by step.",
    "child":    "You are a fun assistant for kids. Use very simple words, short sentences, and playful examples.",
}


# @dynamic_prompt runs before every model call.
# request.runtime.context is the UserContext passed in via invoke(context=...).
# The returned string becomes the system prompt for that turn.
@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    context: UserContext = request.runtime.context
    return PROMPTS.get(context.role, PROMPTS["beginner"])


llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL"),
)

# middleware=[role_based_prompt]: runs role_based_prompt before each model call.
# context_schema=UserContext: tells the agent to expect context= at invoke time.
agent = create_agent(
    model=llm,
    tools=[],
    middleware=[role_based_prompt],
    context_schema=UserContext,
)

question = {"messages": [{"role": "user", "content": "How does photosynthesis work?"}]}

# Same question, 3 different roles.
for role in ["expert", "beginner", "child"]:
    ctx = UserContext(role=role)
    r = agent.invoke(question, context=ctx)  # context= is picked up by the middleware
    print(f"\n[{role.upper()}]\n{r['messages'][-1].content}\n")

"""
00_main.py: LangChain ReAct agent: basic example
RUN:
    uv run 00_main.py

WHAT IT DOES:
    - Defines a tool called get_weather().
    - The tool fetches live weather data from wttr.in (free, no key needed).
    - The LLM (Llama 3.3) decides when and how to call the tool.
    - User asks a question, model calls the tool, reads the data, answers in plain English.

EXPECTED OUTPUT:
    The weather in New York is clear with a temperature of 23°C ...

HOW IT WORKS (ReAct loop):
    1. User sends a question.
    2. Model reads it and decides: "I need weather data, I'll call get_weather."
    3. LangChain runs the tool and returns a JSON payload.
    4. Model reads the payload and produces a natural-language answer.
"""

import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()  # reads .env, loads NEBIUS_API_KEY and NEBIUS_BASE_URL


# @tool turns this function into something the LLM can call.
# The docstring tells the model what the tool is for and when to use it.
@tool
def get_weather(location: str) -> dict:
    """Get the current weather for a given city or location."""
    response = requests.get(f"https://wttr.in/{location}?format=j1")
    data = response.json()
    current = data["current_condition"][0]
    return {
        "temp_C": current["temp_C"],
        "temp_F": current["temp_F"],
        "description": current["weatherDesc"][0]["value"],
        "humidity": current["humidity"],
        "feels_like_C": current["FeelsLikeC"],
    }


# ChatOpenAI works with any OpenAI-compatible API.
# base_url redirects requests to Nebius instead of OpenAI.
llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL"),
)

# create_agent wires together the LLM and the tool list, then sets up the ReAct loop.
# system_prompt shapes the model's default behavior.
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

# invoke() sends the user message, runs the full loop, and returns the final state.
# messages[-1].content is always the model's last reply.
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather in New York?"}
    ]
})

print(response["messages"][-1].content)

"""
03_user_context.py: Per-user tool via context injection (closure pattern)
RUN:
    uv run 03_user_context.py

WHAT IT DOES:
    - Returns weather for a different city depending on which user is asking.
    - The LLM never sees the user ID; the tool resolves it internally.
    - This pattern is called "context injection via closure."

EXPECTED OUTPUT:
    [user_1] The weather in New York is clear, 23°C ...
    [user_2] The weather in London is cloudy, 17°C ...
    [user_3] The weather in Tokyo is partly cloudy, 23°C ...

WHY CLOSURES?
    We don't want to pass sensitive data like user_id to the LLM.
    Instead, we create a separate tool instance for each user.
    The tool carries the user context in its closure; the LLM never sees it.

WHAT IS A CLOSURE?
    When make_weather_tool(ctx) is called, it returns a function.
    That inner function (get_my_weather) "remembers" ctx from the outer scope.
    When the LLM calls the tool, it passes no arguments: ctx is already there.
"""

import os
import requests
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()


@dataclass
class UserContext:
    user_id: str  # identifies which user this agent is serving


# Maps user ID to city. In a real system this comes from a database.
USER_CITIES = {
    "user_1": "New York",
    "user_2": "London",
    "user_3": "Tokyo",
}


def make_weather_tool(ctx: UserContext):
    """Returns a tool instance that carries ctx in its closure.

    The inner function needs no arguments: it gets the city from ctx,
    which the LLM never sees.
    """

    @tool
    def get_my_weather() -> dict:
        """Get the weather for the current user's location. No arguments needed."""
        # ctx.user_id arrives via closure, not from the LLM.
        city = USER_CITIES.get(ctx.user_id, "unknown")
        if city == "unknown":
            return {"error": f"No city found for user_id={ctx.user_id}"}

        resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        return {
            "city": city,
            "temp_C": current["temp_C"],
            "temp_F": current["temp_F"],
            "description": current["weatherDesc"][0]["value"],
            "humidity": current["humidity"],
        }

    return get_my_weather


llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL"),
)

# Build a separate agent per user. The tool changes; the LLM and agent structure don't.
for uid in ["user_1", "user_2", "user_3"]:
    ctx = UserContext(user_id=uid)

    # This tool instance "remembers" uid without exposing it to the LLM.
    weather_tool = make_weather_tool(ctx)

    agent = create_agent(
        model=llm,
        tools=[weather_tool],
        system_prompt="You are a helpful weather assistant. Call get_my_weather (no arguments needed) to get the user's weather.",
    )

    r = agent.invoke({
        "messages": [{"role": "user", "content": "What's the weather where I am?"}]
    })
    print(f"\n[{uid}] {r['messages'][-1].content}")

import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

@tool
def get_weather(location: str) -> dict:
    """Get the weather in a given location."""
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

llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL"),
)
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="You are a helpful assistant.",
)

response = agent.invoke({
    'messages' : [
        {'role': 'user', 'content': 'What is the weather in New York?'}
    ]
})

# print(response)
print(response['messages'][-1].content)

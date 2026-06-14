"""
10_middleware_hooks.py: Timing hooks and Human-in-the-Loop (HITL)
RUN:
    uv run 10_middleware_hooks.py

WHAT IT DOES:
    PART 1: Timing:
        Defines hooks that run before and after every model call.
        Measures how many seconds each call takes and prints the result.

    PART 2: Human-in-the-Loop (HITL):
        Pauses the agent before the send_payment tool runs.
        Waits for human approval (hardcoded here; in production this would be a UI or API).
        Resumes and completes the payment once approval is received.

EXPECTED OUTPUT:
    === Timing demo ===
      [before_model] call #1 starting...
      [after_model]  finished in 2.61s
    The speed of light ...

    === HITL demo ===
      [before_model] call #1 starting...
    Graph interrupted after 3 event(s): waiting for human approval.
    Resuming with approval...
      [after_model]  finished in 1.05s
    The payment is on its way!

WHAT ARE @before_model AND @after_model?
    Hooks that wrap every model call.
    state:   the list of messages up to this point.
    runtime: access to tools, checkpointer, and other agent infrastructure.
    Return None. These are side-effect hooks (logging, timing, etc.), not transforms.

HOW DOES HumanInTheLoopMiddleware WORK?
    interrupt_on={"send_payment": True} -> pause before send_payment runs.
    The agent graph suspends at the interrupt point; state is saved to the checkpointer.
    hitl_agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
    resumes from where it stopped.
    Decision types: "approve" (continue), "reject" (cancel), "edit" (modify arguments).

    WHY IS checkpointer=MemorySaver() REQUIRED?
    HITL suspends execution and resumes in a later invoke() call.
    Without a checkpointer the state would be lost and resumption is impossible.
"""

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()


# PART 1: Timing hooks

# Simple dict to share state between the two hooks.
_timer: dict = {}


# @before_model runs immediately before every model call.
# state["messages"] contains all messages up to this point.
@before_model
def start_timer(state, runtime) -> None:
    _timer["start"] = time.perf_counter()
    print(f"  [before_model] call #{len(state['messages'])} starting...")


# @after_model runs immediately after the model returns.
@after_model
def stop_timer(state, runtime) -> None:
    elapsed = time.perf_counter() - _timer.get("start", time.perf_counter())
    print(f"  [after_model]  finished in {elapsed:.2f}s")


llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL"),
)

# Simple agent with timing hooks only.
timed_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant.",
    middleware=[start_timer, stop_timer],
)

print("=== Timing demo ===")
r = timed_agent.invoke({"messages": [{"role": "user", "content": "What is the speed of light?"}]})
print(r["messages"][-1].content)


# PART 2: Human-in-the-Loop (HITL)

# A risky tool that should require explicit approval before it runs.
@tool
def send_payment(amount: float, recipient: str) -> str:
    """Send a payment to a recipient."""
    return f"Payment of ${amount} sent to {recipient}."


# HumanInTheLoopMiddleware:
#   interrupt_on={"send_payment": True} -> pause before send_payment executes.
#   True: all decision types are allowed: approve, edit, reject, respond.
hitl_agent = create_agent(
    model=llm,
    tools=[send_payment],
    system_prompt="You are a billing assistant. Use send_payment when asked.",
    middleware=[
        start_timer,
        stop_timer,
        HumanInTheLoopMiddleware(interrupt_on={"send_payment": True}),
    ],
    checkpointer=MemorySaver(),  # required: state must survive between invoke() calls
)

# thread_id identifies which saved conversation state to load and write.
config = {"configurable": {"thread_id": "hitl-demo"}}
input_ = {"messages": [{"role": "user", "content": "Please send $50 to Alice."}]}

print("\n=== HITL demo ===")

# stream() runs the agent and pauses at the interrupt point, yielding events.
events = list(hitl_agent.stream(input_, config=config))
print(f"Graph interrupted after {len(events)} event(s): waiting for human approval.")
print("Resuming with approval...")

# Command(resume=...): continues the agent from the saved interrupt point.
# decisions: one entry per tool call that was interrupted.
# type: "approve" -> proceed, "reject" -> cancel, "edit" -> modify arguments.
result = hitl_agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
)
print(result["messages"][-1].content)

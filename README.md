# LangChain crash course

Standalone, runnable examples covering core LangChain concepts from scratch. Each file teaches one idea, runs without modification, and prints a clear result.

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/): fast Python package manager (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A [Nebius AI Studio](https://studio.nebius.ai) account (free tier works)

## Setup

**1. Clone and enter the repo:**
```bash
git clone https://github.com/mesutoezdil/LangChainFull.git
cd LangChainFull
```

**2. Install dependencies:**
```bash
uv sync
```

**3. Create a `.env` file in the project root:**
```
NEBIUS_API_KEY=your_key_here
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
```

Get your API key at [studio.nebius.ai](https://studio.nebius.ai) under API Keys. It's free to sign up.

**4. Run any example:**
```bash
uv run 00_main.py
```

That's it. No further configuration needed.

## What is LangChain and why does it exist?

When you work with AI models (like Llama, GPT, Gemini), you normally deal with raw HTTP requests, different API formats per provider, manual JSON parsing, and wiring everything together yourself.

LangChain is an abstraction layer on top of all that. It gives you a unified interface so you can swap models, add tools, and build pipelines without rewriting everything from scratch.

**Without LangChain:**
```python
import requests

response = requests.post(
    "https://api.nebius.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": "meta-llama/Llama-3.3-70B-Instruct", "messages": [{"role": "user", "content": "Hello"}]}
)
result = response.json()["choices"][0]["message"]["content"]
```

**With LangChain:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="meta-llama/Llama-3.3-70B-Instruct", api_key=..., base_url=...)
result = llm.invoke("Hello")
```

Same result. To switch to a different model or provider, you change one string. LangChain handles the rest.

| Without LangChain | With LangChain |
|---|---|
| Different API format per provider | One interface for all models |
| Manually wire tool calls into the model loop | `@tool` decorator, loop handled automatically |
| Parse JSON, handle retries, manage errors | Built in |
| Manage prompt strings manually | Structured `PromptTemplate` |

**The core idea:** you define tools (functions the model can call), connect them to an LLM, and LangChain runs the loop: user asks, model thinks, calls a tool, gets the result, produces a response. This pattern is called a **ReAct agent**.

## Examples

Each file is standalone and can be run independently. They build on each other conceptually, so reading in order is recommended.

### `00_main.py`: Basic ReAct agent

The starting point. A `@tool`-decorated function fetches live weather data. The agent decides when and how to call it based on the user's question.

```bash
uv run 00_main.py
```

**Key concepts:** `@tool`, `create_agent`, `ChatOpenAI` with Nebius backend.

### `01_chat_model.py`: Direct LLM usage and model swap

Calls the model directly with no agent, no tools. Shows that swapping models is a one-line change: the same code works for Llama, Gemma, Qwen, etc.

```bash
uv run 01_chat_model.py
```

**Key concepts:** `ChatOpenAI.invoke()`, `SystemMessage`, `HumanMessage`, model swap via string.

### `02_conversation_history.py`: Manual conversation history

> **Interactive.** Type a question and press Enter. Type `exit` to quit.

LLMs are stateless: they forget everything between calls. To simulate memory, you send the full conversation history on every request. This file shows the manual approach.

```bash
uv run 02_conversation_history.py
```

**Key concepts:** `HumanMessage`, `AIMessage`, `SystemMessage`, building history as a list.

### `03_user_context.py`: Per-user tool via closure

Shows how to build a tool that knows who the current user is without the LLM having to pass that information. Context is captured in a closure and injected at the tool level.

```bash
uv run 03_user_context.py
```

**Key concepts:** closure-based context injection, `@tool` with no LLM-visible arguments, per-user tool instantiation.

### `04_threads_memory.py`: Automatic memory with MemorySaver

Instead of managing history yourself (like in `02`), `MemorySaver` saves the full agent state after every step. Same `thread_id` restores that state on the next call. Different `thread_id` means a fresh conversation.

```bash
uv run 04_threads_memory.py
```

**Key concepts:** `MemorySaver`, `checkpointer`, `thread_id`, persistent state across calls.

### `05_multimodal.py`: Image + text input

> **Requires `image.jpg`** in the project root before running.

Vision-language models can read images alongside text. This file reads a `.jpg` from disk, Base64-encodes it, and sends it with a text question in a single message.

```bash
# Place any .jpg in the project root named image.jpg, then:
uv run 05_multimodal.py
```

**Key concepts:** `HumanMessage` with mixed content, Base64 encoding, data URI format, vision models.

### `06_rag_basic.py`: Vector store and semantic search

The foundation of RAG. Text is converted to vectors by an embedding model. Similar text produces similar vectors. Searching the store returns semantically related documents, not keyword matches.

```bash
uv run 06_rag_basic.py
```

**Key concepts:** `OpenAIEmbeddings`, `InMemoryVectorStore`, `similarity_search`, semantic vs keyword search.

### `07_rag_agent.py`: RAG agent with retriever as a tool

Combines the vector store from `06` with the agent pattern. The retriever becomes a tool the agent calls on its own: it decides what to search for, reads the results, and writes a natural-language answer.

```bash
uv run 07_rag_agent.py
```

**Key concepts:** `create_retriever_tool`, retriever as agent tool, agent-driven RAG.

### `08_middleware_prompt.py`: Dynamic system prompt

`@dynamic_prompt` runs before every model call and returns the system prompt for that turn. The same question produces completely different answers depending on the user's role (expert, beginner, child).

```bash
uv run 08_middleware_prompt.py
```

**Key concepts:** `@dynamic_prompt`, `ModelRequest`, runtime context access, per-turn system prompt.

### `09_middleware_model.py`: Dynamic model selection

`@wrap_model_call` intercepts the model invocation. `request.override(model=...)` swaps the model for that single call. Short conversations use a cheaper model; longer ones upgrade automatically.

```bash
uv run 09_middleware_model.py
```

**Key concepts:** `@wrap_model_call`, `request.override()`, runtime model selection.

### `10_middleware_hooks.py`: Timing hooks and human-in-the-loop

**Part 1 (timing):** `@before_model` and `@after_model` run before and after each model call. Used here for elapsed-time logging.

**Part 2 (HITL):** `HumanInTheLoopMiddleware` pauses the agent before a specified tool fires, waits for a human decision (`approve` / `edit` / `reject`), then continues. Requires `MemorySaver` so the graph state survives the pause.

```bash
uv run 10_middleware_hooks.py
```

**Key concepts:** `@before_model`, `@after_model`, `HumanInTheLoopMiddleware`, graph interrupts, `Command(resume=...)`.

## Models used

All examples run against [Nebius AI Studio](https://studio.nebius.ai), which provides an OpenAI-compatible API for open-source models.

| Model | Used in |
|---|---|
| `meta-llama/Llama-3.3-70B-Instruct` | Main chat and agent examples |
| `google/gemma-3-27b-it` | Smaller model swap demo (`01`, `09`) |
| `Qwen/Qwen2.5-VL-72B-Instruct` | Vision input (`05`) |
| `Qwen/Qwen3-Embedding-8B` | Text embeddings (`06`, `07`) |
| `Qwen/Qwen3-32B` | RAG agent (`07`) |

**Known limitation:** Nebius Llama 3.3 rejects responses with more than 1 tool call per turn. `07_rag_agent.py` uses Qwen3-32B instead, which handles multi-topic questions without hitting this error.

## Interview preparation

[`interview.md`](interview.md) covers 76 questions at senior AI/ML engineer level. Topics: LCEL, LangGraph, agents, RAG, memory, middleware, multi-agent patterns, error handling, security, cost optimization, and evaluation. Answers are grounded in the code in this repo.

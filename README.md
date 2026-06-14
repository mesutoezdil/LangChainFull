# LangChain

Personal repo for following a LangChain full crash course and running experiments.

---

## What is LangChain and Why Does It Exist?

When you work with AI models (like Llama, GPT, Gemini), you normally have to deal with raw HTTP requests, different API formats for each provider, manual JSON parsing, and wiring everything together yourself.

LangChain is an abstraction layer that sits on top of all that. It gives you a unified interface so you can swap models, add tools, and build complex pipelines without rewriting everything from scratch.

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

Same result. But now if you want to switch to a different model or provider, you change one line. LangChain handles the rest.

**What it actually gives you:**

| Without LangChain | With LangChain |
|---|---|
| Learn a different API format per provider | One interface for all models |
| Manually wire tool calls into the model loop | `@tool` decorator, LangChain handles the rest |
| Parse JSON, handle retries, manage errors | Built in |
| Manage prompt strings manually | Structured `PromptTemplate` |

**The core idea:** You define tools (functions the model can call), connect them to an LLM, and LangChain runs the loop: user asks > model thinks > calls a tool > gets result > produces a response. This pattern is called a **ReAct agent**.

---

## Examples

Each file is a standalone, runnable example. They build on each other conceptually but can be run independently.

### `00_main.py`: Basic ReAct Agent
The starting point. A `@tool`-decorated function that fetches weather from a free API. The agent decides when and how to call it based on the user's question.

```bash
uv run 00_main.py
```

**Key concepts:** `@tool`, `create_agent`, `ChatOpenAI` with Nebius backend.

---

### `01_chat_model.py`: Direct LLM Usage & Model Swap
Shows how to call a model directly (no agent, no tools). Demonstrates that swapping between models is a one-line change: same code works for Llama, Gemma, Qwen, etc.

```bash
uv run 01_chat_model.py
```

**Key concepts:** `ChatOpenAI.invoke()`, `SystemMessage`, `HumanMessage`, model swap via string.

---

### `02_conversation_history.py`: Conversation History
LLMs are stateless: they forget everything between calls. To simulate memory you must send the full conversation history on every request. This file shows the manual approach with a live interactive loop.

```bash
uv run 02_conversation_history.py
```

**Key concepts:** `HumanMessage`, `AIMessage`, `SystemMessage`, building history as a list.

---

### `03_user_context.py`: User Context & Tool Injection
Shows how to build a tool that knows *who the current user is* without the LLM having to pass that information. Context is captured in a closure and injected at the tool level: the model calls the tool with no user-specific arguments.

```bash
uv run 03_user_context.py
```

**Key concepts:** closure-based context injection, `@tool` with no LLM-visible arguments, per-user tool instantiation.

---

### `04_threads_memory.py`: Thread-Based Memory
Instead of managing conversation history yourself (like in `02`), LangGraph's `MemorySaver` saves the full agent state after every step. On the next call with the same `thread_id`, it restores that state: the agent picks up exactly where it left off. Different `thread_id` = fresh conversation.

```bash
uv run 04_threads_memory.py
```

**Key concepts:** `MemorySaver`, `checkpointer`, `thread_id`, persistent state across calls.

---

### `05_multimodal.py`: Image + Text Input
Vision-language models (VL) can read images in addition to text. This file reads a `.jpg` from disk, Base64-encodes it, and sends it alongside a text question in a single message.

Place any `.jpg` named `image.jpg` in the project root, then:
```bash
uv run 05_multimodal.py
```

**Key concepts:** `HumanMessage` with mixed content, Base64 encoding, data URI format, vision models.

---

### `06_rag_basic.py`: Vector Store & Semantic Search
The foundation of RAG. Text is converted to vectors (numbers that capture meaning) by an embedding model. Similar text → similar vectors. Searching a vector store returns semantically related documents, not just keyword matches.

```bash
uv run 06_rag_basic.py
```

**Key concepts:** `OpenAIEmbeddings`, `InMemoryVectorStore`, `similarity_search`, semantic vs keyword search.

---

### `07_rag_agent.py`: RAG Agent: Retriever as a Tool
Combines the vector store from `06` with the agent pattern. The retriever becomes a tool the agent can call: the agent decides what to search for, reads the results, and writes a natural-language answer.

```bash
uv run 07_rag_agent.py
```

**Key concepts:** `create_retriever_tool`, retriever as agent tool, agent-driven RAG.

---

### `08_middleware_prompt.py`: Dynamic Prompt Middleware
`@dynamic_prompt` runs before every model call and returns the system prompt for that turn. Here, the prompt changes based on the user's role (expert / beginner / child), producing completely different response styles for the same question.

```bash
uv run 08_middleware_prompt.py
```

**Key concepts:** `@dynamic_prompt`, `ModelRequest`, runtime context access, per-turn system prompt.

---

### `09_middleware_model.py`: Dynamic Model Selection
`@wrap_model_call` intercepts the model invocation itself. `request.override(model=...)` swaps the model for that single call. Here, short conversations use a smaller model; longer ones automatically upgrade to a more capable one.

```bash
uv run 09_middleware_model.py
```

**Key concepts:** `@wrap_model_call`, `request.override()`, runtime model selection.

---

### `10_middleware_hooks.py`: Timing Hooks + Human-in-the-Loop
**Part 1 (timing):** `@before_model` and `@after_model` run before/after each model call. Used here for elapsed-time logging: useful for tracing, rate-limit counting, and debugging latency.

**Part 2 (HITL):** `HumanInTheLoopMiddleware` pauses the agent graph before a specified tool fires, waits for a human decision (`approve` / `edit` / `reject`), then continues. Requires a `MemorySaver` checkpointer so the graph state survives the pause.

```bash
uv run 10_middleware_hooks.py
```

**Key concepts:** `@before_model`, `@after_model`, `HumanInTheLoopMiddleware`, graph interrupts, `Command(resume=...)`.

---

## Setup

### 1. Initialize project
```bash
uv init
```

### 2. Install dependencies
```bash
uv add langchain langchain-openai langchain-anthropic python-dotenv numpy
```

### 3. Create `.env` file
```
NEBIUS_API_KEY=your_key_here
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
```
> Get your API key at [studio.nebius.ai](https://studio.nebius.ai) → API Keys

### 4. Run any example
```bash
uv run 00_main.py
uv run 01_chat_model.py
# etc.
```

---

## Nebius Notes

All examples run against [Nebius AI Studio](https://studio.nebius.ai), which provides an OpenAI-compatible API for open-source models.

| Model used | Purpose |
|---|---|
| `meta-llama/Llama-3.3-70B-Instruct` | Main chat/agent model |
| `google/gemma-3-27b-it` | Smaller model swap demo (`01`) |
| `Qwen/Qwen2.5-VL-72B-Instruct` | Vision / multimodal (`05`) |
| `Qwen/Qwen3-Embedding-8B` | Text embeddings for RAG (`06`, `07`) |
| `Qwen/Qwen3-32B` | RAG agent (`07`): better parallel tool call handling |

**Known limitation:** Nebius Llama 3.3 rejects responses that contain more than one tool call per turn. `07_rag_agent.py` uses Qwen3-32B instead, which handles multi-call questions without triggering this error.

---

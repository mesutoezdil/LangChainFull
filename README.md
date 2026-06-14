# LangChainFull

Personal repo for following a LangChain full crash course and running experiments.

## Setup Steps

### 1. Initialize project
```bash
uv init
```

### 2. Install LangChain
```bash
uv add langchain
```

### 3. Install Anthropic integration
```bash
uv add langchain-anthropic
```

### 4. Install OpenAI integration (for Nebius compatibility)
```bash
uv add langchain-openai
```

### 5. Install python-dotenv
```bash
uv add python-dotenv
```

### 6. Create .env file
Create a `.env` file in the project root:
```
NEBIUS_API_KEY=your_key_here
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
```
> Get your API key at [studio.nebius.ai](https://studio.nebius.ai) → API Keys

### 7. Run
```bash
uv run main.py
```

## Roadmap

- [x] First LangChain agent call via Nebius endpoint using `ChatOpenAI` + `create_react_agent`
- [x] Custom tool with `@tool` decorator (`get_weather`)
- [ ] Prompt templates
- [ ] Chains
- [ ] Memory
- [ ] Agents (advanced)
- [ ] RAG (Retrieval-Augmented Generation)

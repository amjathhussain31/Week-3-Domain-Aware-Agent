# 🤖 Week 3 — Domain-Aware ReAct Agent

> A conversational AI agent built with LangChain, Mistral (via Ollama), and Streamlit — featuring dual-layer memory, live tool tracing, and a clean chat UI.

---

## Overview

This project is Week 3 of the Presidio AI internship series. It implements a **ReAct (Reasoning + Acting)** agent that reasons step-by-step and selects from a suite of tools to answer user queries — all running **100% locally** using Mistral 7B through Ollama.

The agent maintains two types of memory:

- **Short-term memory** — a sliding window of the last 6 conversation turns, kept in-session
- **Long-term memory** — user preferences and session summaries persisted to disk (`memory_store.json`) and reloaded on every run

A Streamlit web UI provides a side-by-side chat and live tool trace panel, showing exactly what the agent is doing at every step.

---

## Features

- **ReAct reasoning loop** — the agent thinks, selects a tool, observes the result, and repeats until it has a final answer
- **6 integrated tools** — web search, Wikipedia, weather, calculator, and memory read/write
- **Dual-layer memory** — in-session buffer memory + cross-session JSON persistence
- **Dynamic prompt injection** — long-term memory is injected into the system prompt at each turn so the agent always knows what it remembers
- **Session summarisation** — on exit, the LLM generates a structured summary of the conversation and saves it to long-term memory
- **Streamlit UI** — native chat interface with a live tool trace panel and token usage metrics
- **Fully local** — no OpenAI required; runs on Mistral 7B via Ollama

---

## Project Structure

```
Week-3-Domain-Aware-Agent/
├── agent.py            # Core agent: tools, memory, prompt, ReAct executor
├── app.py              # Streamlit UI with chat + live tool trace panel
├── memory_store.json   # Persisted long-term memory (auto-created on first run)
├── .env                # API keys (not committed)
├── .gitignore
└── README.md
```

---

## Tools

| Tool | Description |
|---|---|
| `duckduckgo_search` | Live web search via DuckDuckGo |
| `wikipedia` | Fetches Wikipedia article summaries |
| `get_weather` | Current weather for any city via OpenWeatherMap API |
| `calculator` | Safe evaluation of math expressions |
| `save_preference` | Stores user facts (name, age, location…) to long-term memory |
| `recall_memory` | Retrieves saved preferences or past session summaries |

---

## Memory Architecture

```
Short-term (in-session)
└── ConversationBufferWindowMemory  →  last 6 turns  →  injected into {chat_history}

Long-term (cross-session)
└── memory_store.json
    ├── preferences   →  key-value facts saved by save_preference tool
    └── past_summaries  →  LLM-generated summaries of past sessions (last 10)
```

On every agent turn, `build_prompt_template()` rebuilds the system prompt with the latest long-term memory baked in — so the agent always has full context without relying on the context window alone.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- Mistral model pulled: `ollama pull mistral`
- OpenWeatherMap API key (free tier)

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/amjathhussain31/Week-3-Domain-Aware-Agent.git
cd Week-3-Domain-Aware-Agent
```

**2. Install dependencies**

```bash
pip install langchain langchain-ollama langchain-community streamlit python-dotenv requests wikipedia
```

**3. Create a `.env` file**

```
OPENWEATHER_API_KEY=your_api_key_here
```

Get a free key at [openweathermap.org](https://openweathermap.org/api).

**4. Start Ollama with Mistral**

```bash
ollama serve
ollama pull mistral   # only needed once
```

---

## Running the App

**Streamlit UI (recommended)**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Terminal / CLI mode**

```bash
python agent.py
```

---

## Example Interactions

```
You: What is the weather in Chennai?
Agent: 🌦 Weather in Chennai: 32.9°C, few clouds

You: My name is Amjath
Agent: ✓ Remembered — name: Amjath

You: What is 680000 * 890000?
Agent: 680000 × 890000 = 605,200,000,000

You: What do you know about me?
Agent: Here's what I remember — name: Amjath
```

---

## How It Works

1. User sends a message via the Streamlit UI or terminal
2. `build_agent_executor()` rebuilds the agent with the freshest long-term memory in the prompt
3. The ReAct loop begins: `Thought → Action → Action Input → Observation → ...`
4. Tools are called as needed; results feed back into the reasoning chain
5. The agent outputs a `Final Answer`
6. On session end, `save_session_summary()` generates an LLM summary and appends it to `memory_store.json`

---

## Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `k` (window size) | `agent.py` | `6` | Number of turns kept in short-term memory |
| `max_iterations` | `agent.py` | `5` | Max ReAct loop steps per query |
| `temperature` | `agent.py` | `0` | LLM temperature (0 = deterministic) |
| `model` | `agent.py` | `mistral` | Ollama model name |
| Past summaries kept | `agent.py` | `10` | Max session summaries in long-term memory |

---

## Known Limitations

- Mistral 7B occasionally misformats the ReAct output (missing `Action:` after `Thought:`). `handle_parsing_errors=True` catches and retries these automatically.
- Token counts in the UI are estimated (word-count-based), since Ollama does not expose token counts in its LangChain integration.
- The `memory_store.json` file is committed by default. Add it to `.gitignore` if your preferences contain personal data.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Mistral 7B via Ollama |
| Agent framework | LangChain (ReAct, AgentExecutor) |
| UI | Streamlit |
| Web search | DuckDuckGo (no API key needed) |
| Knowledge | Wikipedia API |
| Weather | OpenWeatherMap API |
| Memory | LangChain `ConversationBufferWindowMemory` + JSON |

---

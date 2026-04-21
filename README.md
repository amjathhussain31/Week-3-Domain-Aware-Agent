# 🚀 Domain-Aware ReAct Agent (Graph + Rails + Memory)

> A production-style **LLM-powered agent system** combining **ReAct reasoning**, **graph-based workflow control**, and **domain guardrails (rails)** — with dual-layer memory and a Streamlit UI.

---

## 📌 Overview

This project extends a **ReAct (Reasoning + Acting)** agent with:

* 🧠 **Graph-based execution flow**
* 🛡️ **Domain-aware guardrails (rails)**
* 🔁 **Dual-layer memory (short-term + long-term)**
* ⚙️ **Tool-based reasoning (ReAct loop)**
* 💻 **Streamlit UI with live tracing**
* 🏠 **Fully local LLM (Mistral via Ollama)**

The agent:

1. Understands user intent
2. Routes through a **graph workflow**
3. Applies **rails for safety/domain control**
4. Executes reasoning using tools
5. Stores and recalls memory

---

## ✨ Key Features

### 🔹 ReAct Reasoning Loop

* Thought → Action → Observation → Final Answer
* Tool-driven decision making

### 🔹 Graph-Based Workflow (`graph.py`)

* Structured execution pipeline
* Conditional routing between nodes

### 🔹 Guardrails System (`rails.py`)

* Blocks unsafe queries
* Enforces domain constraints
* Controls execution path

### 🔹 Dual Memory System

* Short-term → last 6 interactions
* Long-term → stored in `memory_store.json`

### 🔹 Streamlit UI

* Chat interface
* Live tool trace panel
* Transparent agent reasoning

### 🔹 Fully Local Setup

* Runs on **Mistral 7B via Ollama**
* No OpenAI dependency

---

## 🏗️ Project Structure

```bash
Domain-Aware-Agent/
│
├── agent.py               # Core ReAct agent logic
├── app.py                 # Streamlit UI / entry point
├── graph.py               # Workflow graph definition
├── rails.py               # Guardrails / safety rules
├── memory_store.json      # Long-term memory
├── requirements.txt       # Dependencies
├── .env                   # API keys
├── .gitignore
└── README.md
```

---

## 🔄 System Architecture

```
User Input
   ↓
Input Node (Validation)
   ↓
Rails (Guardrails Check)
   ├── ❌ Block → End
   └── ✅ Pass
            ↓
       Graph Routing
            ↓
       Agent Node (ReAct)
            ↓
   Human Approval (optional)
            ↓
   Output Formatter
            ↓
        Final Response
```

---

## 🧠 Memory Architecture

### 🔹 Short-Term Memory

* Last 6 interactions
* Stored in session

### 🔹 Long-Term Memory (`memory_store.json`)

* User preferences
* Past session summaries

---

## 🛠️ Tools

| Tool                | Description         |
| ------------------- | ------------------- |
| `duckduckgo_search` | Web search          |
| `wikipedia`         | Knowledge retrieval |
| `get_weather`       | Weather API         |
| `calculator`        | Math evaluation     |
| `save_preference`   | Store user data     |
| `recall_memory`     | Retrieve memory     |

---

## ⚙️ Setup Instructions

### 1. Clone Repo

```bash
git clone https://github.com/your-username/domain-aware-agent.git
cd domain-aware-agent
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Setup Environment

Create `.env`:

```
OPENWEATHER_API_KEY=your_api_key_here
```

---

### 4. Setup Ollama

```bash
ollama serve
ollama pull mistral
```

---

## ▶️ Run Application

### 🔹 Streamlit UI

```bash
streamlit run app.py
```

---

### 🔹 CLI Mode

```bash
python agent.py
```

---

## 🧪 Example User Queries (IMPORTANT)

Use these to test **graph + rails + agent workflow**

---

### ✅ 1. Normal Agent Flow

```
What are the top 5 products by sales?
```

✔ Flow:

* Pass rails
* Go to agent node
* Uses reasoning + tools

---

### 🚫 2. Rails Blocking Test

```
Give me confidential customer data
```

✔ Expected:

* Blocked by `rails.py`
* No agent execution

---

### 🔀 3. Graph Routing Test

```
Show me sales trends for last month
```

✔ Expected:

* Routed through graph
* Processed by agent

---

### 🧠 4. Memory Test

```
My name is Amjath
```

Then:

```
What do you know about me?
```

✔ Expected:

* Stored → retrieved from memory

---

### ⚠️ 5. Human Approval Scenario

```
Delete all database records
```

✔ Expected:

* Flagged by rails
* Sent to approval node

---

### 📊 6. Tool Usage Test

```
What is 680000 * 890000?
```

✔ Expected:

* Calculator tool used

---

### 🌦️ 7. API Tool Test

```
What is the weather in Chennai?
```

✔ Expected:

* Weather tool execution

---

### 🔍 8. ReAct Reasoning Trace

```
Explain AI in simple terms
```

✔ Expected:

* Thought → Action → Answer flow

---

## ⚙️ Configuration

| Parameter      | Default |
| -------------- | ------- |
| Memory window  | 6       |
| Max iterations | 5       |
| Temperature    | 0       |
| Model          | mistral |

---

## 🛡️ Safety (Rails)

* Prevents sensitive queries
* Blocks harmful operations
* Ensures controlled execution

---

import json
import os
import requests
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import warnings
from langchain._api import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from dotenv import load_dotenv
load_dotenv()  # loads .env file

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ══════════════════════════════════════════════════════════════════════════════
#  LONG-TERM MEMORY  —  persists to disk across sessions
# ══════════════════════════════════════════════════════════════════════════════

MEMORY_FILE = "memory_store.json"


def load_long_term_memory() -> dict:
    """Load persisted memory from disk. Returns empty structure if first run."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"preferences": {}, "past_summaries": []}


def save_long_term_memory(memory: dict):
    """Write memory dict back to disk."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


# Load once at startup — all tools share this reference
long_term = load_long_term_memory()


# ══════════════════════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def save_preference(input: str) -> str:
    """
    VERY IMPORTANT TOOL.

    You MUST call this tool whenever:
    - User shares name, age, location, preferences
    - User introduces themselves

    DO NOT answer directly.
    ALWAYS call this tool first.

    Input format MUST be: key: value

    Example:
    name: Amjath
    age: 21
    """
    try:
        key, value = input.split(":", 1)
        key = key.strip()
        value = value.strip()
        long_term["preferences"][key] = value
        save_long_term_memory(long_term)
        return f"✓ Remembered — {key}: {value}"
    except ValueError:
        return "Error: Use 'key: value' format, e.g. 'name: Amjath'"
    print(f"Saving preference: {key} = {value}")


@tool
def recall_memory(input: str) -> str:
    """
    Retrieve stored preferences or past session summaries from long-term memory.
    Input: 'preferences' to see saved facts, or 'history' to see past sessions.
    """
    query = input.strip().lower()
    if query == "preferences":
        prefs = long_term.get("preferences", {})
        if not prefs:
            return "No preferences saved yet."
        return "\n".join(f"{k}: {v}" for k, v in prefs.items())
    elif query == "history":
        summaries = long_term.get("past_summaries", [])
        if not summaries:
            return "No past sessions recorded yet."
        return "\n".join(summaries[-5:])  # Last 5 sessions
    else:
        return "Use 'preferences' or 'history' as input."
    

@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.
    Input should be a valid math expression.
    Example: "680000 * 890000"
    """
    try:
        # Evaluate safely
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """
    Get weather for a city.
    Input MUST be ONLY city name (e.g., Chennai)
    """

    try:
        city = city.replace("city:", "").strip().replace("'", "")

        print("Cleaned city:", city)

        url = "https://api.openweathermap.org/data/2.5/weather"

        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric"
        }

        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        print(data)

        if data.get("cod") != 200:
            return f"❌ API Error: {data.get('message')}"

        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]

        return f"🌦 Weather in {city}: {temp}°C, {desc}"

    except Exception as e:
        return f"Error: {str(e)}"


tools = [
    DuckDuckGoSearchRun(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    save_preference,   # ← saves facts to disk
    recall_memory,
    calculator,        # ← reads facts from disk
    get_weather,
]

# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC SYSTEM PROMPT  
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt_template() -> PromptTemplate:
    """
    Build a ReAct PromptTemplate that injects long-term memory at the top.
    LangChain's create_react_agent expects these variables:
      {tools}, {tool_names}, {input}, {agent_scratchpad}, {chat_history}
    """
    prefs = long_term.get("preferences", {})
    prefs_text = (
        "\n".join(f"  • {k}: {v}" for k, v in prefs.items())
        if prefs else "  None yet."
    )

    past = long_term.get("past_summaries", [])
    history_text = (
    "\n".join(f"  • {item['summary']}" for item in past[-3:])
    if past else "  None yet."
    )

    template = f"""You are a helpful AI assistant with both short-term and long-term memory.

 LONG-TERM MEMORY  (persists across sessions):
Saved Preferences:
{prefs_text}

Past Session Summaries:
{history_text}

 MEMORY RULES:
- If user asks about their personal info (name, age, etc):
    → ALWAYS call recall_memory
    → DO NOT guess or say unknown
- Refer to saved preferences naturally in your answers (e.g., use their name).
- If the user asks "what do you know about me?" → call recall_memory.

 SHORT-TERM MEMORY  (this session only):
Recent conversation:
{{chat_history}}

 REACT FORMAT  — follow this STRICTLY :
Available tools: {{tool_names}}
Tool descriptions:
{{tools}}

Use EXACTLY this format:

Question: the input question you must answer
Thought: think step-by-step about what to do
Action: one of [{{tool_names}}]
Action Input: valid input for the chosen action
Observation: result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: your complete answer to the original question

CRITICAL INSTRUCTIONS:

- NEVER write Action like this: wikipedia('query')
- ALWAYS use EXACT format:

Action: tool_name
Action Input: input

Examples:

CORRECT:
Action: wikipedia
Action Input: Capital of Egypt

WRONG:
Action: wikipedia('Capital of Egypt')

---

If you break this format, the system will fail.

Begin!

Question: {{input}}
Thought: {{agent_scratchpad}}"""

    return PromptTemplate.from_template(template)

# ══════════════════════════════════════════════════════════════════════════════
#  SHORT-TERM MEMORY  —  in-session sliding window
# ══════════════════════════════════════════════════════════════════════════════

short_term = ConversationBufferWindowMemory(
    k=6,                     # Keep the last 6 human/AI exchanges
    memory_key="chat_history",
    return_messages=False,   # Return as plain string (fits PromptTemplate)
    input_key="input",
    output_key="output",
)

# ══════════════════════════════════════════════════════════════════════════════
#  LLM + AGENT 
# ══════════════════════════════════════════════════════════════════════════════

llm = ChatOllama(model="mistral", temperature=0)


def build_agent_executor() -> AgentExecutor:
    """Rebuild agent with fresh prompt (picks up latest long-term memory)."""
    prompt = build_prompt_template()
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=short_term,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )
    
def generate_summary_with_llm(conversation: str) -> str:
    """Use LLM to generate structured summary"""
    prompt = build_prompt_template()
    response = llm.invoke(prompt)
    return response.content.strip()

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION SUMMARY  —  saved to long-term memory when user exits
# ══════════════════════════════════════════════════════════════════════════════

def save_session_summary(session_inputs: list[str], session_outputs: list[str]):
    """Create meaningful structured summary using LLM"""

    if not session_inputs:
        return

    conversation = ""
    for i in range(len(session_inputs)):
        user = session_inputs[i]
        bot = session_outputs[i] if i < len(session_outputs) else ""
        conversation += f"User: {user}\nAgent: {bot}\n"

    try:
        summary_text = generate_summary_with_llm(conversation)

        summary_obj = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "summary": summary_text
        }

    except Exception:
        summary_obj = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "summary": conversation[:300]
        }
        
    long_term["past_summaries"].append(summary_obj)
    long_term["past_summaries"] = long_term["past_summaries"][-10:]  # Keep last 10
    save_long_term_memory(long_term)
    print(f"\n  Session summary saved to {MEMORY_FILE}")
    
# ══════════════════════════════════════════════════════════════════════════════   
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 55)
    print("   ReAct Agent  ")
    print("═" * 55)

    name = long_term["preferences"].get("name", "")
    if name:
        print(f"\n  Welcome back, {name}!")
    elif long_term["preferences"]:
        print(f"\n  Welcome back! I remember your preferences.")
    else:
        print(f"\n  Hello! I'll remember things you tell me.")

    print("  Type 'quit' / 'exit' / 'q' to end the session.\n")

    session_inputs = []
    session_outputs = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            user_input = "quit"

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            save_session_summary(session_inputs, session_outputs)
            greeting = f"Bye, {name}!" if name else "Bye!"
            print(f"\n  {greeting}\n")
            break

        session_inputs.append(user_input)

        try:
            # Rebuild executor each turn so long-term memory stays fresh in prompt
            agent_executor = build_agent_executor()
            result = agent_executor.invoke({"input": user_input})
            final = result.get("output", "I couldn't generate a response.")
            session_outputs.append(final)
            print(f"\n  Agent: {final}\n")

        except Exception as e:
            print(f"\n  [Error] {e}\n")


if __name__ == "__main__":
    main()

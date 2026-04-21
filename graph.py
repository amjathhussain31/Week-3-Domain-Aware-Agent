# graph.py
"""
3-Node LangGraph
  Node 1: input_node        — validates & preprocesses input via rails
  Node 2: agent_node        — tool-calling ReAct agent (your existing executor)
  Node 3: output_formatter  — formats final response + output safety rail

Conditional routing:
  input_node  → agent_node        (if input passes rails)
  input_node  → blocked           (if input fails rails)
  agent_node  → human_approval    (if agent used sensitive tools)
  agent_node  → output_formatter  (if no approval needed)
  human_approval → output_formatter (if approved)
  human_approval → blocked        (if rejected)
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from rails import apply_input_rails, check_output_safety
from agent import build_agent_executor, langfuse_handler

# ══════════════════════════════════════════════════════════════════════════════
#  STATE — shared dict passed between all nodes
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    user_input:       str
    processed_input:  str
    agent_output:     str
    final_output:     str
    tools_used:       list[str]
    blocked:          bool
    rejection_reason: str
    needs_approval:   bool
    approved:         bool

# Tools that require human approval before the response is shown
SENSITIVE_TOOLS = {"save_preference"}

# ══════════════════════════════════════════════════════════════════════════════
#  NODE 1 — Input Node
#  Runs both input rails; sets blocked=True if rejected
# ══════════════════════════════════════════════════════════════════════════════

def input_node(state: AgentState) -> AgentState:
    print("\n[Graph] ▶ input_node")
    user_input = state["user_input"].strip()

    passed, rejection = apply_input_rails(user_input)

    if not passed:
        return {
            **state,
            "processed_input":  "",
            "blocked":          True,
            "rejection_reason": rejection,
        }

    return {
        **state,
        "processed_input":  user_input,
        "blocked":          False,
        "rejection_reason": "",
    }

# ══════════════════════════════════════════════════════════════════════════════
#  NODE 2 — Tool-Calling Agent Node
#  Runs your existing ReAct AgentExecutor; tracks which tools were used
# ══════════════════════════════════════════════════════════════════════════════

def agent_node(state: AgentState) -> AgentState:
    print("\n[Graph] ▶ agent_node")

    tools_used = []

    # Lightweight callback to capture tool names
    from langchain.callbacks.base import BaseCallbackHandler
    class ToolTracker(BaseCallbackHandler):
        def on_tool_start(self, serialized, input_str, **kwargs):
            name = serialized.get("name", "")
            tools_used.append(name)
            print(f"  [Tool] {name} ← {str(input_str)[:80]}")
        def on_tool_end(self, output, **kwargs):
            print(f"  [Obs]  {str(output)[:120]}")

    executor = build_agent_executor()
    tracker  = ToolTracker()

    result = executor.invoke(
        {"input": state["processed_input"]},
        config={"callbacks": [tracker, langfuse_handler]},
    )

    raw_output = result.get("output", "")
    needs_approval = bool(SENSITIVE_TOOLS & set(tools_used))

    return {
        **state,
        "agent_output":  raw_output,
        "tools_used":    tools_used,
        "needs_approval": needs_approval,
        "approved":      False,   # default; human_approval_node may flip this
    }

# ══════════════════════════════════════════════════════════════════════════════
#  NODE 2b — Human Approval Node
#  Only reached when a sensitive tool (e.g. save_preference) was used.
#  In CLI mode: prompts the user in the terminal.
#  In Streamlit mode: app.py intercepts and renders its own UI instead.
# ══════════════════════════════════════════════════════════════════════════════

def human_approval_node(state: AgentState) -> AgentState:
    print("\n[Graph] ▶ human_approval_node")
    print(f"\n  ⚠️  The agent wants to save data to memory.")
    print(f"  Agent response preview:\n  {state['agent_output'][:200]}")

    answer = input("\n  Approve this action? (yes/no): ").strip().lower()
    approved = answer in {"yes", "y"}

    if not approved:
        return {
            **state,
            "approved":         False,
            "blocked":          True,
            "rejection_reason": "❌ Action rejected by user.",
        }

    return {**state, "approved": True, "blocked": False}

# ══════════════════════════════════════════════════════════════════════════════
#  NODE 3 — Output Formatter Node
#  Runs output safety rail, then formats the final response
# ══════════════════════════════════════════════════════════════════════════════

def output_formatter_node(state: AgentState) -> AgentState:
    print("\n[Graph] ▶ output_formatter_node")

    is_safe, filtered = check_output_safety(state["agent_output"])

    if not is_safe:
        final = filtered   # SAFE_FALLBACK message from rails.py
    else:
        # Clean up any ReAct artifacts the LLM occasionally leaks
        text = filtered
        for artifact in ("Final Answer:", "Thought:", "Observation:"):
            if text.startswith(artifact):
                text = text[len(artifact):].strip()

        tools_str = ""
        if state.get("tools_used"):
            tools_str = "\n\n🔧 *Tools used: " + ", ".join(state["tools_used"]) + "*"

        final = text + tools_str

    return {**state, "final_output": final}

# ══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL ROUTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def route_after_input(state: AgentState) -> Literal["agent_node", "blocked_end"]:
    """After input_node: blocked inputs go straight to end."""
    return "blocked_end" if state["blocked"] else "agent_node"


def route_after_agent(state: AgentState) -> Literal["human_approval_node", "output_formatter_node"]:
    """After agent_node: sensitive tool use triggers approval step."""
    return "human_approval_node" if state["needs_approval"] else "output_formatter_node"


def route_after_approval(state: AgentState) -> Literal["output_formatter_node", "blocked_end"]:
    """After human_approval_node: rejected → end, approved → format."""
    return "blocked_end" if state["blocked"] else "output_formatter_node"

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("input_node",           input_node)
    graph.add_node("agent_node",           agent_node)
    graph.add_node("human_approval_node",  human_approval_node)
    graph.add_node("output_formatter_node",output_formatter_node)

    # Entry point
    graph.set_entry_point("input_node")

    # Conditional edge after input_node
    graph.add_conditional_edges(
        "input_node",
        route_after_input,
        {
            "agent_node":   "agent_node",
            "blocked_end":  END,
        },
    )

    # Conditional edge after agent_node
    graph.add_conditional_edges(
        "agent_node",
        route_after_agent,
        {
            "human_approval_node":   "human_approval_node",
            "output_formatter_node": "output_formatter_node",
        },
    )

    # Conditional edge after human_approval_node
    graph.add_conditional_edges(
        "human_approval_node",
        route_after_approval,
        {
            "output_formatter_node": "output_formatter_node",
            "blocked_end":           END,
        },
    )

    # output_formatter always ends
    graph.add_edge("output_formatter_node", END)

    return graph.compile()


# Singleton — import this in agent.py and app.py
compiled_graph = build_graph()
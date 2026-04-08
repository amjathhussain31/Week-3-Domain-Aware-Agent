"""
app.py  —  Clean Streamlit UI for ReAct Agent (mistral via Ollama)
Run:  streamlit run app.py
"""

import time
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReAct Agent",
    page_icon="🤖",
    layout="wide",
)

# ── Load agent (cached so Ollama model loads only once) ───────────────────────
@st.cache_resource(show_spinner="Loading agent...")
def load_agent():
    from agent import (
        build_agent_executor,
        long_term,
        save_session_summary,
    )
    return build_agent_executor, long_term, save_session_summary

try:
    build_agent_executor, long_term, save_session_summary = load_agent()
    agent_ready = True
    agent_error = ""
except Exception as exc:
    agent_ready = False
    agent_error = str(exc)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"         not in st.session_state: st.session_state.messages         = []
if "tool_trace"       not in st.session_state: st.session_state.tool_trace       = []
if "session_inputs"   not in st.session_state: st.session_state.session_inputs   = []
if "session_outputs"  not in st.session_state: st.session_state.session_outputs  = []
if "reasoning"        not in st.session_state: st.session_state.reasoning        = "Awaiting query…"


# ── Callback: captures tool calls into session state ──────────────────────────
class TraceCallback(BaseCallbackHandler):
    def __init__(self):
        self._timers = {}

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "tool")
        self._timers[name] = time.time()
        st.session_state.tool_trace.append({
            "name":    name,
            "status":  "running",
            "elapsed": "…",
            "input":   str(input_str)[:80],
            "output":  "",
        })
        st.session_state.reasoning = f"Using **{name}** → `{str(input_str)[:60]}`"

    def on_tool_end(self, output, **kwargs):
        if st.session_state.tool_trace:
            last = st.session_state.tool_trace[-1]
            elapsed = time.time() - self._timers.get(last["name"], time.time())
            last["status"]  = "ok"
            last["elapsed"] = f"{elapsed:.2f}s"
            last["output"]  = str(output)[:100]

    def on_tool_error(self, error, **kwargs):
        if st.session_state.tool_trace:
            st.session_state.tool_trace[-1]["status"] = "error"
            st.session_state.tool_trace[-1]["output"] = str(error)[:100]

    def on_agent_finish(self, finish, **kwargs):
        st.session_state.reasoning = "Done."


# ── Helpers ───────────────────────────────────────────────────────────────────
TOOL_ICONS = {
    "duckduckgo_search": "🔍",
    "wikipedia":         "📖",
    "save_preference":   "💾",
    "recall_memory":     "🧠",
    "calculator":        "🧮",
    "get_weather":       "🌦",
}

def tool_icon(name):
    return TOOL_ICONS.get(name, "🔧")

def status_badge(status):
    return {"ok": "✅", "running": "⏳", "error": "❌"}.get(status, "•")


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🤖 Domain-Aware Agent – Weather Forecasting")
    st.caption("mistral:7b · Ollama · LangChain")
    st.divider()

    if agent_ready:
        st.success("Agent online", icon="✅")
    else:
        st.error(f"Agent offline — {agent_error}", icon="❌")
        st.stop()

    # ── Long-term memory ──
    st.markdown("**Long-term memory**")
    prefs = long_term.get("preferences", {})
    if prefs:
        for k, v in prefs.items():
            st.markdown(f"- `{k}`: {v}")
    else:
        st.caption("No preferences saved yet")

    st.divider()

    # ── Tools list ──
    st.markdown("**Loaded tools**")
    active_tool = st.session_state.tool_trace[-1]["name"] if st.session_state.tool_trace else None
    for name, icon in TOOL_ICONS.items():
        prefix = "→ " if name == active_tool else "\u00a0\u00a0"
        st.markdown(f"{prefix}{icon} `{name}`")

    st.divider()

    # ── Past sessions ──
    summaries = long_term.get("past_summaries", [])
    if summaries:
        st.markdown("**Past sessions**")
        for s in summaries[-3:]:
            ts  = s.get("timestamp", "")
            txt = s.get("summary", "")
            st.caption(f"{ts} — {txt[:50]}…")

    st.divider()

    # ── Save & clear ──
    if st.button("💾 Save & clear session", use_container_width=True):
        if st.session_state.session_inputs:
            save_session_summary(
                st.session_state.session_inputs,
                st.session_state.session_outputs,
            )
            st.success("Session saved!")
        for key in ("messages", "tool_trace", "session_inputs", "session_outputs"):
            st.session_state[key] = []
        st.session_state.reasoning        = "Awaiting query…"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — two columns: chat | trace
# ══════════════════════════════════════════════════════════════════════════════
chat_col, trace_col = st.columns([2, 1], gap="large")

# ── LEFT: Chat ────────────────────────────────────────────────────────────────
with chat_col:
    name = long_term.get("preferences", {}).get("name", "")
    greeting = f"Welcome back, **{name}**!" if name else "Hello! Ask me anything."
    st.markdown(f"#### {greeting}")
    st.caption("I can search the web, check weather, do math, and remember things across sessions.")
    st.divider()

    # Render all existing messages
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("tools_used"):
                chips = " · ".join(
                    f"{tool_icon(t)} `{t}`" for t in msg["tools_used"]
                )
                st.caption(f"Tools used: {chips}")

    # Native chat input — no empty-label warning
    user_input = st.chat_input("Ask me something…")

    if user_input:
        # Show user bubble immediately
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.session_inputs.append(user_input)

        # Reset trace
        st.session_state.tool_trace = []
        suffix = "…" if len(user_input) > 60 else ""
        st.session_state.reasoning = f"Processing: '{user_input[:60]}{suffix}'"

        # Run agent inside assistant bubble
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    cb       = TraceCallback()
                    executor = build_agent_executor()
                    result   = executor.invoke(
                        {"input": user_input},
                        config={"callbacks": [cb]},
                    )
                    final = result.get("output", "I couldn't generate a response.")
                    st.markdown(final)

                    tools_used = [s["name"] for s in st.session_state.tool_trace]
                    if tools_used:
                        chips = " · ".join(f"{tool_icon(t)} `{t}`" for t in tools_used)
                        st.caption(f"Tools used: {chips}")

                    st.session_state.messages.append({
                        "role":       "assistant",
                        "content":    final,
                        "tools_used": tools_used,
                    })
                    st.session_state.session_outputs.append(final)

                    # Rough token estimate (Ollama doesn't expose counts directly)
                    st.session_state.token_prompt     += len(user_input.split()) * 2
                    st.session_state.token_completion += len(final.split())

                except Exception as e:
                    err = f"⚠️ Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

        st.rerun()


# ── RIGHT: Tool trace ─────────────────────────────────────────────────────────
with trace_col:
    st.markdown("#### Tool trace")

    # Reasoning box
    st.info(st.session_state.reasoning, icon="💭")

    if not st.session_state.tool_trace:
        st.caption("No tools called yet this turn.")
    else:
        for step in st.session_state.tool_trace:
            badge = status_badge(step["status"])
            icon  = tool_icon(step["name"])
            label = f"{badge} {icon} **{step['name']}** — {step['elapsed']}"
            with st.expander(label, expanded=True):
                st.markdown(f"**Input:** `{step['input']}`")
                if step["output"]:
                    st.markdown(f"**Output:** {step['output']}")
                if step["status"] == "error":
                    st.error("Tool returned an error")

    st.divider()

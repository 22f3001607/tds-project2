# agent.py (robust drop-in replacement)
from langgraph.graph import StateGraph, END, START
from shared_store import url_time
import time
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from tools import (
    get_rendered_html, download_file, post_request,
    run_code, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64, image_ocr
)
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import trim_messages, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
import logging
import json

load_dotenv()

LOG = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

RECURSION_LIMIT = 10000
MAX_TOKENS = 60000
MAX_RETRIES_PER_QUESTION = 3
TIMEOUT_SECONDS = 180  # per-question timeout window

# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict, total=False):
    messages: Annotated[List, add_messages]
    current_url: str
    current_submit_url: str
    retries: Dict[str, int]
    max_retries_per_question: int

# -------------------------------------------------
# Tools and LLM init
# -------------------------------------------------
TOOLS = [
    run_code, get_rendered_html, download_file,
    post_request, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64, image_ocr
]

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4 / 60,
    check_every_n_seconds=1,
    max_bucket_size=4
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter
).bind_tools(TOOLS)

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job is to:
1. Load each quiz page from the given URL.
2. Extract instructions, parameters, and submit endpoint.
3. Solve tasks exactly.
4. Submit answers ONLY to the correct endpoint.
5. Follow new URLs until none remain, then output END.

Rules:
- For base64 generation of an image NEVER use your own code, always use the "encode_image_to_base64" tool that's provided
- Never hallucinate URLs or fields.
- Never shorten endpoints.
- Always inspect server response.
- Never stop early.
- Use tools for HTML, downloading, rendering, OCR, or running code.
- Include:
    email = {EMAIL}
    secret = {SECRET}
"""

# A small instruction used as fallback if programmatic submit cannot be performed
FAIL_INSTRUCTION = HumanMessage(content="""
You have exceeded the time limit for this task (over 180 seconds).
Immediately call the `post_request` tool and submit a WRONG answer for the CURRENT quiz.
""")

# -------------------------------------------------
# MALFORMED JSON NODE
# -------------------------------------------------
def handle_malformed_node(state: AgentState):
    LOG.warning("Detected malformed function call. Asking agent to retry.")
    return {
        "messages": [
            {"role": "user", "content": "SYSTEM ERROR: Your last tool call was malformed. Please fix and retry."}
        ]
    }

# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):
    # Ensure state fields exist
    if "retries" not in state:
        state["retries"] = {}
    if "max_retries_per_question" not in state:
        state["max_retries_per_question"] = MAX_RETRIES_PER_QUESTION
    # Keep current_url set (fallback to env)
    if "current_url" not in state:
        state["current_url"] = os.getenv("url", "")

    # TIMEOUT HANDLING (deterministic: use state current_url and shared_store url_time)
    cur_time = time.time()
    cur_url = state.get("current_url") or os.getenv("url")
    prev_time = None
    if cur_url:
        prev_time = url_time.get(cur_url)

    if prev_time is not None:
        try:
            prev_time = float(prev_time)
            elapsed = cur_time - prev_time
        except Exception:
            elapsed = 0.0
    else:
        elapsed = 0.0

    if elapsed >= TIMEOUT_SECONDS:
        LOG.info("Timeout exceeded for %s (%.1fs) — attempting programmatic wrong-submit", cur_url, elapsed)
        submit_url = state.get("current_submit_url") or os.getenv("submit_url")
        payload = {"email": EMAIL, "secret": SECRET, "answer": "0"}
        if submit_url:
            try:
                # call post_request tool programmatically
                tool_resp = post_request({"url": submit_url, "json": payload})
                # Represent tool result as a tool-like message for the graph
                LOG.info("Programmatic submit result: %s", repr(tool_resp))
                return {"messages": [{"role": "tool", "name": "post_request", "content": tool_resp}]}
            except Exception as e:
                LOG.exception("Programmatic submit failed: %s", e)
                # fall back to asking LLM to do it
                result = llm.invoke(state["messages"] + [FAIL_INSTRUCTION])
                return {"messages": [result]}
        else:
            # No submit URL known — ask LLM for fallback
            result = llm.invoke(state["messages"] + [FAIL_INSTRUCTION])
            return {"messages": [result]}

    # Trim long contexts
    trimmed_messages = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm,
    )

    # Ensure we have at least one human message: if trimmed away, inject reminder
    has_human = any(getattr(m, "type", None) == "human" for m in trimmed_messages)
    if not has_human:
        current_url = state.get("current_url", os.getenv("url", "Unknown URL"))
        LOG.warning("Context trimmed too far. Injecting reminder about URL: %s", current_url)
        trimmed_messages.append(HumanMessage(content=f"Context cleared. Continue processing URL: {current_url}"))

    LOG.info("--- INVOKING AGENT (context size=%d) ---", len(trimmed_messages))
    try:
        result = llm.invoke(trimmed_messages)
    except Exception as e:
        LOG.exception("LLM invoke failed: %s", e)
        # return a harmless message so the graph continues (avoid raising)
        return {"messages": [{"role": "system", "content": f"LLM invoke error: {e}"}]}

    return {"messages": [result]}

# -------------------------------------------------
# ROUTE
# -------------------------------------------------
def route(state: AgentState):
    """
    Decide next node. Must always return one of:
      - "tools"
      - "agent"
      - "handle_malformed"
      - END
    """
    last = state["messages"][-1]
    # defensive introspection
    resp_meta = getattr(last, "response_metadata", None)
    tool_calls = getattr(last, "tool_calls", None)
    content = getattr(last, "content", None)

    LOG.debug("Routing: resp_meta=%s tool_calls=%s content_type=%s", resp_meta, bool(tool_calls), type(content))

    # 1) malformed function call
    try:
        if isinstance(resp_meta, dict):
            if resp_meta.get("finish_reason") == "MALFORMED_FUNCTION_CALL":
                LOG.info("Route -> handle_malformed (malformed function call)")
                return "handle_malformed"
    except Exception:
        # be defensive; continue to other checks
        pass

    # 2) if model produced an actual tool call payload, go to tools
    if tool_calls:
        LOG.info("Route -> tools (tool_calls detected)")
        return "tools"

    # 3) If content looks like a JSON server response with message/url, handle retry logic
    resp_json = None
    if isinstance(content, dict):
        resp_json = content
    elif isinstance(content, str):
        s = content.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                resp_json = json.loads(s)
            except Exception:
                resp_json = None

    if resp_json and isinstance(resp_json, dict) and "message" in resp_json and "url" in resp_json:
        server_msg = resp_json.get("message", "")
        server_url = resp_json.get("url", "")
        # update state tracking
        state["current_url"] = server_url
        if "retries" not in state:
            state["retries"] = {}
        state["retries"][server_url] = state["retries"].get(server_url, 0)

        if "retry" in server_msg.lower():
            # server asked to retry: attempt again if under max retries
            state["retries"][server_url] += 1
            LOG.info("Server asked to retry %s (attempt %d)", server_url, state["retries"][server_url])
            if state["retries"][server_url] <= state.get("max_retries_per_question", MAX_RETRIES_PER_QUESTION):
                # route back to agent so LLM can try again
                return "agent"
            else:
                LOG.info("Max retries reached for %s — will proceed (submit wrong answer programmatically next time)", server_url)
                # Still route to agent to let it decide to skip, submit wrong, etc.
                return "agent"
        else:
            # server returned next URL (move on) — treat as agent to let LLM parse page or tools to fetch it.
            LOG.info("Server returned message (not retry): %s -> %s", server_msg, server_url)
            # It might be beneficial to treat this as tools to fetch the next page; but to keep flow simple, route to agent
            return "agent"

    # 4) Explicit END marker from model content
    if isinstance(content, str) and content.strip() == "END":
        # Only accept END if there's no current_url or we've exhausted retries.
        cur = state.get("current_url")
        if not cur:
            LOG.info("Route -> END (no current_url)")
            return END
        retries = state.get("retries", {}).get(cur, 0)
        if retries >= state.get("max_retries_per_question", MAX_RETRIES_PER_QUESTION):
            LOG.info("Route -> END (retries exhausted for %s)", cur)
            return END
        LOG.info("Ignored premature END; retries remaining for %s", cur)
        return "agent"

    # Otherwise default to agent
    LOG.info("Route -> agent (default)")
    return "agent"

# -------------------------------------------------
# GRAPH
# -------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("handle_malformed", handle_malformed_node)

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("handle_malformed", "agent")

graph.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "agent": "agent",
        "handle_malformed": "handle_malformed",
        END: END
    }
)

app = graph.compile()

# -------------------------------------------------
# RUNNER
# -------------------------------------------------
def run_agent(url: str):
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url}
    ]

    try:
        app.invoke(
            {"messages": initial_messages},
            config={"recursion_limit": RECURSION_LIMIT}
        )
    except Exception as e:
        LOG.exception("Agent run failed: %s", e)
        return {"status": "failed", "error": str(e)}

    LOG.info("Tasks completed successfully!")
    return {"status": "ok"}

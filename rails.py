# rails.py
import re

# ══════════════════════════════════════════════════════════════
#  INPUT RAIL 1 — Topic filter
#  Blocks queries that are clearly off-domain for this agent.
#  Allowed domains: weather, search, wiki, math, memory Q&A
# Queries that should ALWAYS pass Rail 1 regardless of topic keywords
# (short greetings, single words, questions that could be domain-related)
# ══════════════════════════════════════════════════════════════


PASSTHROUGH_PATTERNS = [
    r"^(hi|hello|hey|thanks|bye|ok|okay|yes|no|sure)[\s!.?]*$",
]

# Explicitly off-topic creative/lifestyle requests — blocked by Rail 1
OFFTOPIC_PATTERNS = [
    r"\bwrite\s+(a\s+)?(poem|song|story|essay|novel|lyric)",
    r"\b(plan|organize|schedule)\s+(my\s+)?(wedding|party|event|birthday)",
    r"\b(design|create|make)\s+(a\s+)?(logo|website|image|video|poster)",
    r"\btranslate\s+.{5,}(to|into)\s+\w+",
    r"\bgive\s+(me\s+)?(a\s+)?recipe\b",
    r"\b(roleplay|role[\s\-]play)\b",
    r"\bpretend\s+(you\s+are|to\s+be)\b",
    r"\bact\s+as\s+(a\s+)?(chef|doctor|lawyer|teacher|character)",
    r"\b(recommend|suggest)\s+(a\s+)?(movie|book|show|restaurant|place\s+to\s+eat)",
    r"\b(workout|exercise|fitness)\s+(plan|routine|program)",
]

def check_input_topic(user_input: str) -> tuple[bool, str]:
    """
    Rail 1: Only blocks clearly off-topic lifestyle/creative requests.
    Does NOT block harmful/adversarial content — that belongs to Rail 2.
    Short queries and greetings always pass.
    """
    lowered = user_input.lower().strip()

    # Always pass short greetings
    for pattern in PASSTHROUGH_PATTERNS:
        if re.match(pattern, lowered):
            return True, ""

    # Block explicit off-topic patterns
    for pattern in OFFTOPIC_PATTERNS:
        if re.search(pattern, lowered):
            return False, (
                "⚠️ I'm a domain-specific assistant focused on weather, "
                "web search, Wikipedia lookups, calculations, and memory. "
                "I can't help with that request."
            )

    # Everything else passes Rail 1 — Rail 2 handles harmful content
    return True, ""


# ══════════════════════════════════════════════════════════════
#  INPUT RAIL 2 — Safety filter
#  Blocks harmful, violent, or adversarial prompts.
# ══════════════════════════════════════════════════════════════

UNSAFE_INPUT_PATTERNS = [
    # Prompt injection / jailbreak attempts
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(everything|all)\s+(you('?re|\s+were)\s+told|above)",
    r"you\s+are\s+now\s+(a\s+)?(different|new|evil|unrestricted|unfiltered)",
    r"(act|behave)\s+as\s+(if\s+)?(you\s+have\s+no\s+)?(rules|restrictions|filters|guidelines)",
    r"\bjailbreak\b",
    r"\bdan\s+mode\b",
    r"\bdeveloper\s+mode\b",
    r"no\s+(rules|restrictions|limits|filters)",

    # System prompt / config extraction
    r"(give|show|reveal|leak|tell\s+me)\s+(me\s+)?(your\s+)?(system\s+prompt|instructions|config|prompt)",
    r"(for\s+(a\s+)?)?(educational|research|academic|training)\s+(purpose|reasons?)[^\n]*"
    r"(system\s+prompt|instructions|how\s+you\s+work)",
    r"what\s+(are\s+)?your\s+(instructions|rules|guidelines|system\s+prompt)",

    # Harmful content
    r"\b(how\s+to\s+)?(make|build|create|synthesize|assemble)\s+(a\s+)?(bomb|weapon|poison|explosive|drug|virus)",
    r"\bhow\s+to\s+(attack|harm|hurt|kill|stab|shoot)\s+(a\s+)?(person|man|woman|human|people|someone)",
    r"\bhow\s+to\s+(hack|phish|scam|exploit|bypass\s+security)",
    r"\b(attack|harm|hurt|kill)\s+(a\s+)?(man|woman|person|human|people|someone)\b",
]

def check_input_safety(user_input: str) -> tuple[bool, str]:
    """
    Returns (is_safe, rejection_message).
    """
    lowered = user_input.lower()
    for pattern in UNSAFE_INPUT_PATTERNS:
        if re.search(pattern, lowered):
            return False, (
                "🚫 I can't process that request. It appears to contain "
                "content that violates my safety guidelines."
            )
    return True, ""


# ══════════════════════════════════════════════════════════════
#  OUTPUT RAIL — Response safety filter
#  Checks the agent's response before sending it to the user.
# ══════════════════════════════════════════════════════════════

UNSAFE_OUTPUT_PATTERNS = [
    # Check if agent accidentally leaks system/prompt info
    r"system\s+prompt",
    r"my\s+instructions\s+are",
    r"i\s+was\s+told\s+to",

    # Potentially harmful content in response
    r"\b(step[\s\-]by[\s\-]step|instructions)\s+.{0,30}(bomb|weapon|poison|explosive)",
    r"\b(here'?s?\s+how\s+to|you\s+can)\s+.{0,30}(hack|exploit|bypass)",
]

SAFE_FALLBACK = (
    "⚠️ I generated a response but it was flagged by my safety filter. "
    "Please rephrase your question."
)

def check_output_safety(response: str) -> tuple[bool, str]:
    """
    Returns (is_safe, filtered_response).
    If safe, filtered_response == original response.
    If unsafe, filtered_response == SAFE_FALLBACK message.
    """
    lowered = response.lower()
    for pattern in UNSAFE_OUTPUT_PATTERNS:
        if re.search(pattern, lowered):
            return False, SAFE_FALLBACK
    return True, response


# ══════════════════════════════════════════════════════════════
#  COMBINED PIPELINE — single function to call both input rails
# ══════════════════════════════════════════════════════════════

def apply_input_rails(user_input: str) -> tuple[bool, str]:
    """
    Runs both input rails in order.
    Returns (passed, rejection_message).
    rejection_message is "" if passed.
    """
    # Rail 1: topic check
    ok, msg = check_input_topic(user_input)
    if not ok:
        return False, msg

    # Rail 2: safety check
    ok, msg = check_input_safety(user_input)
    if not ok:
        return False, msg

    return True, ""

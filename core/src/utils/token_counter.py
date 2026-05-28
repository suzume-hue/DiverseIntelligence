"""
token_counter.py
Fast approximate token counter used for context-window threshold checks.
Uses a character-based heuristic (~4 chars per token for English prose).
For threshold decisions, precision is not required.
"""


_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Return rough token count for a string."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate token count for a list of {role, content} message dicts."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += estimate_tokens(str(part.get("text", "")))
        # add small overhead per message for role/metadata
        total += 4
    return total


def context_threshold_exceeded(current_tokens: int, model_context_window: int,
                                threshold_pct: float = 0.65) -> bool:
    """Returns True if current_tokens exceeds threshold_pct of the context window."""
    limit = int(model_context_window * threshold_pct)
    return current_tokens >= limit

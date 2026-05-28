"""
core/src/agents/meta_communicator.py
The MetaCommunicator — translates messages between domain groups.

FIX (2026-05-09): Gemma-4-31b was leaking its chain-of-thought reasoning
into translation output. The prompt now requires <translation> XML tags,
and _extract_translation() robustly strips all reasoning before delivery.

FIX (2026-05-09): Added unclosed-tag fallback to _extract_translation so
truncated outputs (opening <translation> tag present but closing tag cut off
by token limit) degrade gracefully instead of leaking the raw tag into the
broadcast via the weak heuristic fallbacks.
"""

import re

from core.src.api.client import LLMClient

META_COMM_TASK_TEMPLATE = """
=== TRANSLATION TASK ===

Sender   : {sender_name} ({sender_domain})
Receiver : {receiver_name} ({receiver_domain})

=== ORIGINAL MESSAGE ===
{message}

=== YOUR TASK ===
Translate the message above so that {receiver_name} can genuinely engage with its content.

Apply your full translation protocol internally:
  1. Strip the domain vocabulary to its structural core.
  2. Identify {receiver_name}'s conceptual vocabulary (domain: {receiver_domain}).
  3. Find the structural analogue in what {receiver_name} already knows.
  4. Build the bridge explicitly — name the connection, don't assert it.
  5. Flag where the analogy breaks down.
  6. Return to the original idea with the new framing intact.

Your output MUST be wrapped in <translation> tags like this:

<translation>
[The translated message and nothing else. This is delivered directly to {receiver_name}.]
</translation>

Everything outside <translation> tags is discarded. Do not include reasoning,
analysis, protocol steps, or meta-commentary inside or outside the tags.
"""


def translate_message(
    message: str,
    sender_id: str,
    sender_name: str,
    sender_domain: str,
    receiver_id: str,
    receiver_name: str,
    receiver_domain: str,
    meta_comm_prompt: str,
    client: LLMClient,
    model_cfg: dict,
    temperature: float,
) -> dict:
    """
    Translate one message for one specific receiver.
    Returns: {from, for, original, translated}
    """
    task = META_COMM_TASK_TEMPLATE.format(
        sender_name=sender_name,
        sender_domain=sender_domain,
        receiver_name=receiver_name,
        receiver_domain=receiver_domain,
        message=message,
    )

    raw = client.chat(
        provider=model_cfg["provider"],
        model_id=model_cfg["model_id"],
        messages=[{"role": "user", "content": task}],
        temperature=temperature,
        max_tokens=model_cfg.get("max_tokens", 1000),
        system=meta_comm_prompt,
    )

    cleaned = _extract_translation(raw)

    return {
        "from": sender_id,
        "sender_name": sender_name,
        "for": receiver_id,
        "original": message,
        "translated": cleaned,
    }


def _extract_translation(raw: str) -> str:
    """
    Extract only the translated message from raw LLM output.

    Strategy (in order):
    1. Extract from <translation>...</translation> tags  (primary path)
    1b. Unclosed tag fallback — opening tag present but closing tag missing
        (model was truncated by token limit); return whatever content exists
    2. Strip <thought>...</thought> blocks, return remainder if clean
    3. Find last quoted block of substantial length
    4. Take last paragraph that doesn't look like analysis notes
    5. Return stripped raw as last resort
    """
    text = raw.strip()

    # 1. Primary: complete <translation>...</translation> tags
    match = re.search(r"<translation>\s*(.*?)\s*</translation>", text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if len(extracted) > 30:
            return extracted

    # 1b. Unclosed tag fallback: opening tag present but output was truncated
    # before the closing tag arrived. Deliver what content exists rather than
    # falling through to weak heuristics that would return the raw tag string.
    open_match = re.search(r"<translation>\s*(.*)", text, re.DOTALL)
    if open_match:
        extracted = open_match.group(1).strip()
        if len(extracted) > 30:
            return extracted

    # 2. Strip <thought>...</thought> blocks
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()

    # 3. If text is now clean (doesn't start with a domain-annotation header),
    #    return it directly
    domain_header_pattern = re.compile(
        r"^[A-Z][^\n]{5,50}\([^)]{3,40}\)\.\n", re.MULTILINE
    )
    if text and not domain_header_pattern.match(text):
        # Looks clean — but still strip any trailing analysis sections
        return _strip_trailing_analysis(text)

    # 4. Find last quoted block (model sometimes puts final translation in quotes)
    quotes = re.findall(
        r'["\u201c\u2018]((?:[^"\u201d\u2019\n].{20,}?\n?){1,15})["\u201d\u2019]',
        text,
        re.DOTALL,
    )
    if quotes:
        candidate = quotes[-1].strip()
        if len(candidate) > 60:
            return candidate

    # 5. Take last substantive paragraph that doesn't look like protocol analysis
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    clean_paragraphs = [p for p in paragraphs if not _looks_like_analysis(p)]
    if clean_paragraphs:
        return clean_paragraphs[-1]

    # 6. Last resort
    return text or raw.strip()


def _looks_like_analysis(paragraph: str) -> bool:
    """Returns True if the paragraph looks like chain-of-thought analysis."""
    analysis_signals = [
        r"^\s*\*+\s",  # lines starting with asterisks (bullet points)
        r"\$\\(?:approx|rightarrow|to)",  # LaTeX math symbols
        r"^\s*\d+\.\s+\*",  # numbered list with bold
        r"Step \d+",  # protocol step headers
        r"\*\*?(?:Domain|Drafting|Refining|Check|Final Polish|Structural)",  # protocol headers
        r"→.*→",  # arrow chains
        r"\\rightarrow",
    ]
    for pattern in analysis_signals:
        if re.search(pattern, paragraph, re.MULTILINE | re.IGNORECASE):
            return True
    # Lines with more than 40% starting-asterisk or bullet density
    lines = paragraph.split("\n")
    if lines:
        bullet_lines = sum(1 for l in lines if re.match(r"\s*[\*\-]", l))
        if bullet_lines / len(lines) > 0.3:
            return True
    return False


def _strip_trailing_analysis(text: str) -> str:
    """Remove trailing analysis blocks that sometimes follow the actual translation."""
    section_markers = [
        r"\*\*?(?:The Bridge|Flag|Where does|Return to|Checking|Drafting|Wait,)",
        r"---+\s*$",
    ]
    for marker in section_markers:
        parts = re.split(marker, text, flags=re.MULTILINE | re.IGNORECASE)
        if len(parts) > 1 and len(parts[0].strip()) > 60:
            return parts[0].strip()
    return text


def run_translations(
    agent_outputs: list[dict],
    agent_registry: dict,
    translation_groups: list[list[str]],
    meta_comm_prompt: str,
    client: LLMClient,
    model_cfg: dict,
    temperature: float,
) -> list[dict]:
    """
    For each speaking agent, translate their message for every agent
    in a DIFFERENT domain group. Works for any number of groups.
    """
    domain_to_group: dict[str, int] = {}
    for idx, group in enumerate(translation_groups):
        for domain in group:
            domain_to_group[domain.lower()] = idx

    translations = []
    speaking = [o for o in agent_outputs if not o["passed"] and o["message"]]

    for output in speaking:
        sender_id = output["agent_id"]
        message = output["message"]
        sender_info = agent_registry[sender_id]
        sender_group = domain_to_group.get(sender_info["domain"].lower(), -1)

        for receiver_id, recv_info in agent_registry.items():
            if receiver_id == sender_id:
                continue
            receiver_group = domain_to_group.get(recv_info["domain"].lower(), -1)
            if sender_group == -1 or receiver_group == -1:
                continue
            if sender_group == receiver_group:
                continue

            t = translate_message(
                message=message,
                sender_id=sender_id,
                sender_name=sender_info["display_name"],
                sender_domain=sender_info["domain"],
                receiver_id=receiver_id,
                receiver_name=recv_info["display_name"],
                receiver_domain=recv_info["domain"],
                meta_comm_prompt=meta_comm_prompt,
                client=client,
                model_cfg=model_cfg,
                temperature=temperature,
            )
            translations.append(t)

    return translations

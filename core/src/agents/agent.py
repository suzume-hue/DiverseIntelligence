"""
core/src/agents/agent.py
Handles one agent's turn in a round.

FIX (2026-05-09): Added judge observation feedback and self-awareness
(agent's own last message) to create belief-update pressure. Without
these, agents defaulted to their persona's opening position every round.

FIX (2026-05-09): Added problem_anchor block so agents can detect when
the conversation has drifted from the original problem and self-correct.
The anchor is framed as a private self-check, not an external instruction.

FIX (2026-05-10): Added critic pass between LLM call and XML parse.
When use_critic=True, an internal critic evaluates the draft for evasion
of direct challenges. If evasion is detected, a revision pass is triggered
before the message is finalised. The critic is advisory — parse failures
or API errors fall back to the original draft silently.

FIX (2026-05-10): Added belief context block. When a BeliefStore is
provided, the agent's current active beliefs are injected as a private
self-check so they can track their own intellectual commitments across rounds.

FIX (2026-05-10): Added outer-wrapper parse fallback. When the
<agent>...</agent> wrapper is missing, direct <think>/<message> extraction
is attempted before defaulting to PASS.
"""

import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Optional

from core.src.api.client import LLMClient
from core.src.memory.retriever import retrieve_top_k
from core.src.memory.store import MemoryStore

if TYPE_CHECKING:
    from core.src.memory.belief_store import BeliefStore


AGENT_INSTRUCTION = """
You participate in a structured multi-agent discussion following the MACP protocol.

FORMAT — respond with ONLY this XML block (nothing outside it):

If you have something to contribute:
<agent id="{agent_id}" round="{round_num}">
  <think>
    [Your private reasoning. Not shown to other agents.]
  </think>
  <message>
    [Your public statement. This is what others see. Must stand alone.]
  </message>
</agent>

If you have nothing to add this round:
<agent id="{agent_id}" round="{round_num}">
  <think>
    [Optional private note on why you are passing.]
  </think>
  <pass />
</agent>

CRITICAL — on intellectual honesty:
You only speak when you have something GENUINELY NEW to contribute.
Before you write your message, ask yourself: am I saying something I haven't
already said? Am I actually responding to what was just argued, or restating
my prior position in different words?

If you are repeating yourself without engaging with the strongest challenge
directed at you, that is worse than silence. Pass instead.

If you are genuinely moved, changed, or updated by what someone said —
show it explicitly. Intellectual movement is the point of this conversation.
"""


def run_agent_turn(
    agent_id: str,
    persona_prompt: str,
    broadcast: str,
    round_num: int,
    store: MemoryStore,
    client: LLMClient,
    model_cfg: dict,
    temperature: float,
    memory_cfg: dict,
    prior_judge_observation: Optional[str] = None,
    agent_last_message: Optional[str] = None,
    problem_anchor: Optional[str] = None,
    use_critic: bool = False,
    belief_store: Optional["BeliefStore"] = None,
) -> dict:
    """
    Execute one agent turn. Returns a dict with:
      {agent_id, round, think, message (or None if pass), passed (bool),
       critic_result (or None)}

    Optional params:
      use_critic    — if True, run the internal critic pass before finalising
      belief_store  — if provided, inject agent's current beliefs as context
    """
    # 1. Retrieve relevant reflections
    top_k = memory_cfg.get("retrieval_top_k", 3)
    emb_model = memory_cfg.get("embedding_model", "all-MiniLM-L6-v2")
    retrieved = retrieve_top_k(broadcast, store, top_k=top_k, model_name=emb_model)

    # 2. Build system prompt
    instruction = AGENT_INSTRUCTION.format(agent_id=agent_id, round_num=round_num)
    system = persona_prompt + "\n\n" + instruction

    # 3. Build user message
    user_parts = []

    # Memory block (reflections)
    memory_block = _format_memory(retrieved)
    if memory_block:
        user_parts.append(f"=== Your Relevant Memory (private) ===\n{memory_block}")

    # Belief context block — what this agent currently holds to be true
    if belief_store is not None:
        beliefs_summary = belief_store.get_summary_for_context(max_beliefs=5)
        if beliefs_summary:
            user_parts.append(
                f"=== Your Current Beliefs (private self-check) ===\n"
                f"{beliefs_summary}\n\n"
                f"These are the positions you have publicly committed to so far. "
                f"If you are about to retract or revise one, do so explicitly in "
                f"your message rather than quietly shifting ground."
            )

    # Self-awareness: what you said last round
    if agent_last_message:
        user_parts.append(
            f"=== What You Said Last Round (private) ===\n{agent_last_message}\n\n"
            f"Have you genuinely moved from this position? If not, what specifically "
            f"are you responding to that you haven't yet addressed?"
        )

    # Judge feedback from prior round
    if prior_judge_observation:
        user_parts.append(
            f"=== Observer Note on Prior Round (private) ===\n{prior_judge_observation}\n\n"
            f"This is how the prior round was assessed. Consider whether there is "
            f"unresolved tension you should address, concede, or reframe."
        )

    # Problem anchor — agent-driven drift correction
    if problem_anchor:
        user_parts.append(
            f"=== Original Problem (private self-check) ===\n{problem_anchor}\n\n"
            f"Your response must contain at least one sentence that directly engages "
            f"this problem — not just the current conversational thread. Ask yourself: "
            f"does what I am about to say connect to the problem above, or have I drifted "
            f"into adjacent territory that no longer addresses it? If you cannot honestly "
            f"connect your response to the problem, pass this round rather than "
            f"continuing the drift."
        )

    # The actual broadcast
    user_parts.append(f"=== Round {round_num} Broadcast ===\n{broadcast}")

    user_content = "\n\n".join(user_parts)

    # 4. Call LLM (draft)
    raw = client.chat(
        provider=model_cfg["provider"],
        model_id=model_cfg["model_id"],
        messages=[{"role": "user", "content": user_content}],
        temperature=temperature,
        max_tokens=model_cfg.get("max_tokens", 1500),
        system=system,
    )

    # 5. Critic pass — only when enabled and agent has a prior message
    critic_result = None
    if use_critic and agent_last_message:
        draft_parsed = _parse_agent_xml(raw, agent_id, round_num)
        draft_message = draft_parsed.get("message")

        if draft_message:  # Only critique speaking agents, not passers
            from core.src.agents.critic import run_critic_pass

            agent_name = _extract_agent_name(persona_prompt, agent_id)
            raw, critic_result = run_critic_pass(
                agent_name=agent_name,
                persona_prompt=persona_prompt,
                last_message=agent_last_message,
                broadcast=broadcast,
                draft_raw=raw,
                draft_message=draft_message,
                client=client,
                model_cfg=model_cfg,
                temperature=temperature,
            )

    # 6. Parse final XML output
    result = _parse_agent_xml(raw, agent_id, round_num)
    result["critic_result"] = critic_result
    return result


# ── helpers ────────────────────────────────────────────────────────────────────


def _extract_agent_name(persona_prompt: str, fallback: str) -> str:
    first_line = persona_prompt.strip().split("\n")[0]
    match = re.search(r"You are ([^,\.]+)", first_line)
    if match:
        return match.group(1).strip()
    return fallback


def _format_memory(retrieved: list[dict]) -> str:
    if not retrieved:
        return ""
    lines = []
    for r in retrieved:
        lines.append(f"- [{', '.join(r.get('speakers', []))}] {r['content']}")
    return "\n".join(lines)


def _parse_agent_xml(raw: str, agent_id: str, round_num: int) -> dict:
    """
    Parse MACP agent XML. Falls back to regex on malformed XML.

    FIX (2026-05-10): When the outer <agent> wrapper is missing or malformed,
    attempt direct <think>/<message> extraction before defaulting to PASS.
    """
    result = {
        "agent_id": agent_id,
        "round": round_num,
        "think": None,
        "message": None,
        "passed": True,
        "raw": raw,
    }

    text = raw.strip()
    match = re.search(r"<agent\b[^>]*>(.*?)</agent>", text, re.DOTALL)
    if not match:
        # Outer wrapper missing — try direct extraction before giving up
        think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        msg_m = re.search(r"<message>(.*?)</message>", text, re.DOTALL)

        if think_m:
            result["think"] = think_m.group(1).strip()

        if msg_m and msg_m.group(1).strip():
            result["message"] = msg_m.group(1).strip()
            result["passed"] = False
        else:
            result["think"] = (
                result["think"] or "[XML parse failed — raw output preserved]"
            )

        return result

    inner = f"<agent>{match.group(1)}</agent>"

    try:
        root = ET.fromstring(inner)
    except ET.ParseError:
        root = None

    if root is not None:
        think_el = root.find("think")
        msg_el = root.find("message")
        pass_el = root.find("pass")

        result["think"] = (
            think_el.text.strip() if think_el is not None and think_el.text else None
        )

        if msg_el is not None and msg_el.text and msg_el.text.strip():
            result["message"] = msg_el.text.strip()
            result["passed"] = False
        elif pass_el is not None:
            result["passed"] = True
        else:
            result["passed"] = True
    else:
        think_m = re.search(r"<think>(.*?)</think>", match.group(1), re.DOTALL)
        msg_m = re.search(r"<message>(.*?)</message>", match.group(1), re.DOTALL)

        if think_m:
            result["think"] = think_m.group(1).strip()
        if msg_m and msg_m.group(1).strip():
            result["message"] = msg_m.group(1).strip()
            result["passed"] = False

    return result

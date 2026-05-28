"""
core/src/memory/belief_revision.py
Post-round belief revision pass.

After each round, each agent reviews what was said and updates their explicit
belief state: asserting new beliefs, revising confidence on existing ones,
retracting beliefs that were conceded, suspending beliefs that are now uncertain.

This is separate from reflection generation. Reflections are private notes
(what happened, what I noticed). Beliefs are public commitments (what I
currently hold to be true, with what confidence).

The revision prompt uses the agent's persona so the belief extraction is
voiced authentically — Feynman's beliefs sound like Feynman's beliefs.

FIX (2026-05-10): Introduced alongside belief_store.py and argument_graph.py.
"""

import json
import re
from typing import Optional

from core.src.api.client import LLMClient
from core.src.memory.belief_store import BeliefStore

# ── Prompts ────────────────────────────────────────────────────────────────────

_REVISION_SYSTEM_SUFFIX = """

--- BELIEF REVISION INSTRUCTION ---
You are reviewing the round just completed and updating your explicit belief state.

Your current beliefs:
{current_beliefs}

Argument structure from this round (external):
{argument_graph_summary}

If any of your beliefs appear under "Unanswered attacks" above, you have received
a direct challenge you have not yet answered. Lower confidence on those beliefs
rather than holding them firm — suspension or a confidence drop below 0.6 is
appropriate until you have answered the challenge.

A belief is a proposition you have publicly committed to in this discussion.
Not everything you say is a belief — only the substantive intellectual claims
you are actually asserting.

Review the round broadcast and your response. Then output ONLY a valid JSON
object in exactly this format — no preamble, no markdown fences:

{{
  "belief_updates": [
    {{
      "action": "assert",
      "proposition": "A clear, specific claim in one sentence.",
      "confidence": 0.8,
      "evidence": "Brief reason — what in this round supports this belief."
    }},
    {{
      "action": "revise",
      "belief_id": "bel_existing_id",
      "new_confidence": 0.5,
      "reason": "What changed this round that shifts my confidence."
    }},
    {{
      "action": "retract",
      "belief_id": "bel_existing_id",
      "reason": "Why I am abandoning this position."
    }},
    {{
      "action": "suspend",
      "belief_id": "bel_existing_id",
      "reason": "Why I am holding this uncertain — what challenge I have not yet answered."
    }}
  ]
}}

Rules:
- confidence is a float 0.0–1.0. Use 0.9+ for firm commitments, 0.5–0.7 for
  tentative positions, below 0.5 for claims you are considering but not asserting.
- Only assert beliefs that you actually stated in your message this round.
  Do not assert private suspicions that you did not voice.
- Retract honestly: if you conceded a point, retract the contradicted belief.
- Suspend if you received a challenge you have not answered but are not ready
  to retract. Suspension is intellectually honest holding.
- Zero updates is valid if nothing changed.
- belief_id for revise/retract/suspend must match an ID from your current beliefs list.
"""


def generate_belief_revision(
    agent_id: str,
    persona_prompt: str,
    round_broadcast: str,
    agent_message: str | None,
    round_num: int,
    store: BeliefStore,
    client: LLMClient,
    model_cfg: dict,
    temperature: float,
    argument_graph_summary: str | None = None,
) -> list[dict]:
    """
    Run belief revision for one agent after a round.
    Returns list of applied update dicts.

    If the agent passed this round (no message), beliefs are not revised —
    passing is not an intellectual event, it is an absence.
    """
    if not agent_message:
        return []

    current_beliefs = store.load_index()
    beliefs_summary = _format_beliefs(current_beliefs)

    system = persona_prompt + _REVISION_SYSTEM_SUFFIX.format(
        current_beliefs=beliefs_summary
        or "(none yet — this may be your first assertions)",
        argument_graph_summary=argument_graph_summary or "(none available)",
    )

    user_msg = (
        f"=== Round {round_num} Broadcast (what everyone said) ===\n\n"
        f"{round_broadcast}\n\n"
        f"=== Your Message This Round ===\n\n"
        f"{agent_message}\n\n"
        f"Update your belief state now."
    )

    try:
        raw = client.chat(
            provider=model_cfg["provider"],
            model_id=model_cfg["model_id"],
            messages=[{"role": "user", "content": user_msg}],
            temperature=temperature,
            max_tokens=model_cfg.get("max_tokens", 800),
            system=system,
        )
    except Exception:
        return []

    updates = _parse_revision_json(raw)
    if not updates:
        return []

    results = []
    for update in updates:
        action = update.get("action", "")

        if action == "assert":
            proposition = update.get("proposition", "").strip()
            confidence = float(update.get("confidence", 0.7))
            evidence = update.get("evidence", "")
            if proposition:
                bid = store.assert_belief(proposition, confidence, round_num, evidence)
                results.append(
                    {
                        "action": "assert",
                        "id": bid,
                        "proposition": proposition,
                        "confidence": confidence,
                    }
                )

        elif action == "revise":
            bid = update.get("belief_id", "")
            new_confidence = float(update.get("new_confidence", 0.5))
            reason = update.get("reason", "")
            if bid and store.revise_belief(bid, new_confidence, round_num, reason):
                results.append(
                    {"action": "revise", "id": bid, "new_confidence": new_confidence}
                )

        elif action == "retract":
            bid = update.get("belief_id", "")
            reason = update.get("reason", "")
            if bid and store.retract_belief(bid, round_num, reason):
                results.append({"action": "retract", "id": bid})

        elif action == "suspend":
            bid = update.get("belief_id", "")
            reason = update.get("reason", "")
            if bid and store.suspend_belief(bid, round_num, reason):
                results.append({"action": "suspend", "id": bid})

    return results


# ── Helpers ────────────────────────────────────────────────────────────────────


def _format_beliefs(index: list[dict]) -> str:
    if not index:
        return ""
    lines = []
    for b in index:
        conf = b.get("confidence", 0)
        status = b.get("status", "held")
        tag = f"  [{b['id']}] (conf={conf:.2f}, {status}) {b['proposition']}"
        lines.append(tag)
    return "\n".join(lines)


def _parse_revision_json(raw: str) -> list[dict]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
        return data.get("belief_updates", [])
    except json.JSONDecodeError:
        match = re.search(r'\{.*?"belief_updates"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("belief_updates", [])
            except json.JSONDecodeError:
                pass
    return []

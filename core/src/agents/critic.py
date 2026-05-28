"""
core/src/agents/critic.py
Internal critic pass — runs against an agent's draft message before it is
finalised. Checks whether the agent has genuinely addressed the strongest
challenge directed at them this round, or is restating a prior position.

Called from run_agent_turn() in agent.py between the draft LLM call and
the final XML parse. Only fires when:
  - The agent has spoken (not passed)
  - The agent has a prior message (last_message is not None)
  - use_critic=True is passed into run_agent_turn

If evasion is detected, a revision pass is triggered using the same model
and temperature as the original agent call. The revision prompt preserves
the agent's position but forces direct engagement with the challenge.

If the critic call itself fails (parse error, API error), the original
draft is returned unchanged — the critic is advisory, not blocking.

FIX (2026-05-10): Introduced to address the cycling pattern where agents
(especially those with single-move philosophical repertoires) repeatedly
restate their position without engaging the strongest counter-argument
directed at them.
"""

import json
import re

from core.src.api.client import LLMClient


# ── Prompts ────────────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = (
    "You are the internal critic for {agent_name}. "
    "Your sole function is intellectual integrity: determine whether a draft "
    "response genuinely engages the specific challenge directed at this agent, "
    "or evades it by restating a prior position. "
    "You are precise and unsentimental. You do not reward performance of engagement."
)

_CRITIC_PROMPT = """=== This Agent's Last Message ===
{last_message}

=== What This Agent Received This Round (challenges directed at them) ===
{broadcast}

=== This Agent's Draft Response ===
{draft_message}

Evaluate with precision:

1. What is the STRONGEST specific challenge directed at this agent in the broadcast?
   (A challenge is a direct question, a counter-argument, or a demand for
   something — a mechanism, a prediction, a concession — the agent has not provided.)

2. Does the draft response actually address that challenge? A response addresses
   a challenge if it:
   - Concedes the challenged point (wholly or partially), OR
   - Directly refutes the challenge with a new argument, OR
   - Identifies a genuine paradox that contains both positions.

   A response does NOT address a challenge if it:
   - Reasserts the same position with different rhetoric
   - Changes the subject to a different aspect of the debate
   - Issues a counter-challenge without first acknowledging the incoming one

Respond ONLY with valid JSON — no preamble, no markdown fences:

{{"strongest_challenge": "One sentence: what specific challenge must this agent answer?", "verdict": "approved", "evasion": ""}}

OR if evasion is detected:

{{"strongest_challenge": "One sentence: what specific challenge must this agent answer?", "verdict": "needs_revision", "evasion": "Specific description of what was evaded and how.", "revision_instruction": "A single concrete instruction: what the agent must address before reasserting their position. Do not tell them to change their conclusion — only to engage the challenge first."}}

Only flag needs_revision for CLEAR evasion of a DIRECT challenge.
Do not flag for legitimate disagreement, different framing, or opening new territory."""

_REVISION_PROMPT = """Your internal critic has reviewed your draft and flagged a specific evasion.

=== Critic's Finding ===
Challenge you must address: {strongest_challenge}
What you evaded: {evasion}
Instruction: {revision_instruction}

=== Your Draft (to revise) ===
{draft_raw}

Revise your response. Address the challenge identified above directly before
reasserting your position. You may still hold your position — but you must
first engage the strongest version of the challenge honestly.

Do not add new XML wrapper elements. Output ONLY the revised XML block in
exactly the same format as your draft (same <agent id=...> wrapper)."""


# ── Public interface ───────────────────────────────────────────────────────────

def run_critic_pass(
    agent_name:     str,
    persona_prompt: str,
    last_message:   str,
    broadcast:      str,
    draft_raw:      str,
    draft_message:  str,
    client:         LLMClient,
    model_cfg:      dict,
    temperature:    float,
) -> tuple[str, dict | None]:
    """
    Evaluate a draft message and optionally trigger a revision.

    Returns:
        (final_raw_xml, critic_result)
        final_raw_xml  — original draft if approved/error, revised XML if evasion detected
        critic_result  — None if not triggered or parse error, else the parsed verdict dict

    Caller is responsible for re-parsing the returned XML with _parse_agent_xml.
    """
    # Step 1: Evaluate
    critic_system = _CRITIC_SYSTEM.format(agent_name=agent_name)
    critic_user   = _CRITIC_PROMPT.format(
        last_message=  last_message,
        broadcast=     broadcast,
        draft_message= draft_message,
    )

    try:
        critic_raw = client.chat(
            provider=    model_cfg["provider"],
            model_id=    model_cfg["model_id"],
            messages=    [{"role": "user", "content": critic_user}],
            temperature= 0.1,   # Evaluation should be near-deterministic
            max_tokens=  500,
            system=      critic_system,
        )
    except Exception:
        # Critic call failed — return original, don't block the agent
        return draft_raw, None

    verdict = _parse_critic_json(critic_raw)
    if verdict is None or verdict.get("verdict") != "needs_revision":
        return draft_raw, verdict

    # Step 2: Revision pass (only when evasion confirmed)
    revision_user = _REVISION_PROMPT.format(
        strongest_challenge= verdict.get("strongest_challenge", ""),
        evasion=             verdict.get("evasion", ""),
        revision_instruction=verdict.get("revision_instruction", ""),
        draft_raw=           draft_raw,
    )

    try:
        revised_raw = client.chat(
            provider=    model_cfg["provider"],
            model_id=    model_cfg["model_id"],
            messages=    [{"role": "user", "content": revision_user}],
            temperature= temperature,
            max_tokens=  model_cfg.get("max_tokens", 1500),
            system=      persona_prompt,
        )
    except Exception:
        return draft_raw, verdict

    final_raw = revised_raw.strip() if revised_raw.strip() else draft_raw
    return final_raw, verdict


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_critic_json(raw: str) -> dict | None:
    """Parse critic JSON. Returns None on failure (treat as approved)."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",           "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*?"verdict".*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None

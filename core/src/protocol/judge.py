"""
protocol/judge.py
The Judge evaluates each round's broadcast and decides CONTINUE or CLOSE.
Receives:
  - The current round's canonical broadcast (what was actually said)
  - Its own running observation log (prior evaluations, concise)
  - Agent summaries (reflections as context proxies) — not full transcripts

FIX (2026-05-09): _build_judge_prompt now explicitly requires <judge> XML
output. Previously the format constraint existed only in the parser, not
in the prompt — so the model had no instruction to produce tags at all.
"""

import re
import xml.etree.ElementTree as ET

from core.src.api.client import LLMClient

JUDGE_FORMAT_INSTRUCTION = """
=== REQUIRED OUTPUT FORMAT ===

Respond with ONLY this XML block — nothing outside it:

<judge round="{round_num}">
  <observation>
    [2-4 sentences. What actually happened this round? Did agents engage with
    each other's arguments or restate prior positions? Was there genuine movement,
    regression, or cycling? Be specific — name agents and claims.]
  </observation>
  <decision>CONTINUE</decision>
</judge>

Replace CONTINUE with CLOSE if any of the following are true:
- The conversation has reached genuine convergence (all positions settled, no remaining tension)
- The exchange is irreversibly cycling (same attack, same defence, no movement for 2+ rounds)
- The conversation has drifted from the original problem for 2+ consecutive rounds with no
  agent attempting to re-anchor it — continued drift is not generative, it is evasion

Do not include any text, reasoning, or commentary outside the <judge> tags.
"""


def evaluate_round(
    round_num: int,
    canonical_broadcast: str,
    observation_log: list[dict],
    agent_reflection_snapshots: dict,
    judge_config: dict,
    client: LLMClient,
    model_cfg: dict,
    temperature: float,
    argument_graph_summary: str | None = None,
) -> dict:
    """
    Returns: {round, observation, decision, raw}
    decision is "CONTINUE" or "CLOSE"

    argument_graph_summary — optional structural summary from ArgumentGraph.
    When provided, injected into the judge prompt so the judge has precise
    information about unanswered attacks, commitment violations, and cycling.
    """
    system = judge_config["system_prompt"].replace("{ROUND}", str(round_num))
    user_msg = _build_judge_prompt(
        round_num,
        canonical_broadcast,
        observation_log,
        agent_reflection_snapshots,
        argument_graph_summary=argument_graph_summary,
    )

    raw = client.chat(
        provider=model_cfg["provider"],
        model_id=model_cfg["model_id"],
        messages=[{"role": "user", "content": user_msg}],
        temperature=temperature,
        max_tokens=model_cfg.get("max_tokens", 1200),
        system=system,
    )

    result = _parse_judge_xml(raw, round_num)
    return result


def _build_judge_prompt(
    round_num: int,
    broadcast: str,
    obs_log: list[dict],
    snapshots: dict,
    argument_graph_summary: str | None = None,
) -> str:
    parts = []

    parts.append(f"=== Round {round_num} — What Was Said ===\n{broadcast}")

    if snapshots:
        parts.append("\n=== Agent Memory Snapshots (most recent reflections) ===")
        for agent_id, refls in snapshots.items():
            if refls:
                parts.append(f"\n[{agent_id}]")
                for r in refls[-3:]:  # last 3 reflections per agent
                    parts.append(f"  • {r['content'][:200]}")

    if obs_log:
        parts.append("\n=== Your Prior Observations ===")
        for entry in obs_log[-6:]:  # last 6 observations
            parts.append(f"  Round {entry['round']}: {entry['observation']}")

    # Argument graph summary — structural view of what has and hasn't been answered
    if argument_graph_summary:
        parts.append(f"\n{argument_graph_summary}")

    # Explicit format instruction — appended last so it is closest to the
    # generation boundary and least likely to be overridden by context.
    parts.append(JUDGE_FORMAT_INSTRUCTION.format(round_num=round_num))

    return "\n".join(parts)


def _parse_judge_xml(raw: str, round_num: int) -> dict:
    result = {
        "round": round_num,
        "observation": "",
        "decision": "CONTINUE",
        "raw": raw,
    }

    text = raw.strip()
    match = re.search(r"<judge\b[^>]*>(.*?)</judge>", text, re.DOTALL)
    if not match:
        result["observation"] = "[Judge XML parse failed — defaulting to CONTINUE]"
        return result

    inner = f"<judge>{match.group(1)}</judge>"

    try:
        root = ET.fromstring(inner)
        obs_el = root.find("observation")
        dec_el = root.find("decision")

        if obs_el is not None and obs_el.text:
            result["observation"] = obs_el.text.strip()
        if dec_el is not None and dec_el.text:
            decision = dec_el.text.strip().upper()
            if decision in ("CONTINUE", "CLOSE"):
                result["decision"] = decision

    except ET.ParseError:
        # Regex fallback
        obs_m = re.search(
            r"<observation>(.*?)</observation>", match.group(1), re.DOTALL
        )
        dec_m = re.search(
            r"<decision>\s*(CONTINUE|CLOSE)\s*</decision>",
            match.group(1),
            re.IGNORECASE,
        )
        if obs_m:
            result["observation"] = obs_m.group(1).strip()
        if dec_m:
            result["decision"] = dec_m.group(1).upper()

    return result

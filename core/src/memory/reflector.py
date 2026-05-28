"""
memory/reflector.py
After each round, calls the actor LLM (using the agent's own persona prompt)
to generate structured reflections from what was said.

The reflector produces JSON output: a list of reflection entries.
Each entry is either "new" or "extend" (referencing an existing reflection ID).
"""

import json
import re
from typing import Optional

from core.src.api.client import LLMClient
from core.src.memory.store import MemoryStore
from core.src.memory.retriever import embed


REFLECTION_SYSTEM_SUFFIX = """

--- MEMORY INSTRUCTION ---
After reading the round above, generate structured reflections capturing the semantically significant moments. You are building your own private memory index.

Your existing reflections (for reference, to decide whether to extend or create new):
{existing_reflections}

Respond ONLY with a valid JSON object in this exact format — no preamble, no explanation, nothing outside the JSON:

{{
  "reflections": [
    {{
      "content": "A concise but semantically rich insight or observation from this round. Written in first person as your private note.",
      "speakers_referenced": ["agent_id_1", "agent_id_2"],
      "action": "new"
    }},
    {{
      "content": "Additional insight to append to an existing reflection.",
      "speakers_referenced": ["agent_id_1"],
      "existing_reflection_id": "refl_abc12345",
      "action": "extend"
    }}
  ]
}}

Rules:
- You decide how many reflections to generate. Zero is valid if nothing meaningful happened.
- Reference speakers by their agent IDs (e.g. "richard_feynman", "soren_kierkegaard").
- For "extend", existing_reflection_id must match an ID from your existing reflections list.
- If existing reflections list is empty, all actions must be "new".
- Do not reproduce lengthy quotes. Write distilled insights in your own internal voice.
"""


def generate_reflections(
    agent_id:       str,
    persona_prompt: str,
    round_broadcast: str,
    round_num:      int,
    store:          MemoryStore,
    client:         LLMClient,
    model_cfg:      dict,
    temperature:    float,
    max_reflections: Optional[int] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """
    Generate and persist reflections for an agent after a round.
    Returns list of created/extended reflection dicts.
    """
    existing = store.get_all_reflection_contents()
    existing_summary = _format_existing(store.load_index())

    system = persona_prompt + REFLECTION_SYSTEM_SUFFIX.format(
        existing_reflections=existing_summary or "(none yet)"
    )

    user_msg = (
        f"=== Round {round_num} Broadcast ===\n\n"
        f"{round_broadcast}\n\n"
        f"Generate your reflections now."
    )

    raw = client.chat(
        provider=    model_cfg["provider"],
        model_id=    model_cfg["model_id"],
        messages=    [{"role": "user", "content": user_msg}],
        temperature= temperature,
        max_tokens=  model_cfg.get("max_tokens", 800),
        system=      system,
    )

    entries = _parse_reflection_json(raw)

    if max_reflections is not None:
        entries = entries[:max_reflections]

    results = []
    for entry in entries:
        content   = entry.get("content", "").strip()
        speakers  = entry.get("speakers_referenced", [])
        action    = entry.get("action", "new")
        exist_id  = entry.get("existing_reflection_id")

        if not content:
            continue

        emb = embed(content, embedding_model)

        if action == "extend" and exist_id:
            store.extend_reflection(exist_id, content, round_num, speakers, emb)
            results.append({"action": "extend", "id": exist_id, "content": content})
        else:
            new_id = store.add_reflection(content, round_num, speakers, emb)
            results.append({"action": "new", "id": new_id, "content": content})

    return results


# ── helpers ────────────────────────────────────────────────────────────────────

def _format_existing(index: list[dict]) -> str:
    if not index:
        return ""
    lines = []
    for entry in index:
        lines.append(f"  [{entry['id']}] {entry['content'][:120]}")
    return "\n".join(lines)


def _parse_reflection_json(raw: str) -> list[dict]:
    """Parse JSON from LLM output. Strips markdown fences if present."""
    text = raw.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",           "", text)
    text = text.strip()

    try:
        data = json.loads(text)
        return data.get("reflections", [])
    except json.JSONDecodeError:
        # Try to extract JSON object with regex as fallback
        match = re.search(r'\{.*"reflections"\s*:\s*\[.*\]\s*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("reflections", [])
            except json.JSONDecodeError:
                pass
    return []

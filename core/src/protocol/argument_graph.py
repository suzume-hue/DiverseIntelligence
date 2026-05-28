"""
core/src/protocol/argument_graph.py
Tracks argument attack/support relationships and commitment violations.

After each round, the argument graph is updated with:
  - New claims made by agents
  - Which prior claims each new claim attacks or supports
  - Whether any agent has contradicted their prior commitments

This is a lightweight implementation of Dung-style argumentation, adapted
for the MACP protocol. The graph is not used to decide correctness — it is
used to give the judge a precise structural picture of what is actually
happening in the dialogue rather than just rhetorical surface.

The judge receives a graph summary that names:
  - Which claims have been attacked and whether the attack was answered
  - Which commitments have been violated (agent asserted X, then effectively denied X)
  - Whether cycling is structural (same attack, same defence, no movement)

Graph is persisted as a single JSON file per session.

FIX (2026-05-10): Introduced alongside belief_store.py and belief_revision.py.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Optional


class ArgumentGraph:
    def __init__(self, sess_dir: str):
        self.sess_dir  = sess_dir
        self._path     = os.path.join(sess_dir, "argument_graph.json")
        self._graph    = self._load()

    # ── write ──────────────────────────────────────────────────────────────────

    def add_claim(
        self,
        agent_id:   str,
        claim_text: str,
        round_num:  int,
        attacks:    list[str] | None = None,   # claim IDs this attacks
        supports:   list[str] | None = None,   # claim IDs this supports
    ) -> str:
        """
        Register a new claim. Returns claim ID.
        attacks/supports are IDs of prior claims in the graph.
        """
        claim_id = f"cl_{agent_id[:8]}_{uuid.uuid4().hex[:6]}"
        claim = {
            "id":       claim_id,
            "agent_id": agent_id,
            "text":     claim_text[:300],   # truncated for storage
            "round":    round_num,
            "attacks":  attacks  or [],
            "supports": supports or [],
        }
        self._graph["claims"].append(claim)

        # Mark attacked claims as challenged
        for attacked_id in (attacks or []):
            self._mark_challenged(attacked_id, claim_id, round_num)

        self._save()
        return claim_id

    def record_commitment_violation(
        self,
        agent_id:      str,
        prior_claim_id: str,
        round_num:     int,
        description:   str,
    ) -> None:
        """
        Record that an agent has contradicted a prior public commitment.
        """
        self._graph["commitment_violations"].append({
            "id":            f"cv_{uuid.uuid4().hex[:6]}",
            "agent_id":      agent_id,
            "prior_claim_id":prior_claim_id,
            "round":         round_num,
            "description":   description,
        })
        self._save()

    def update_from_round(
        self,
        round_num:      int,
        agent_outputs:  list[dict],
        belief_stores:  dict,   # agent_id -> BeliefStore
        client,
        model_cfg:      dict,
        temperature:    float,
    ) -> dict:
        """
        Run a lightweight LLM extraction pass to identify new claims and their
        relationships from this round's agent outputs.

        Returns summary dict for this round's updates.
        """
        if not agent_outputs:
            return {}

        # Build context of existing high-level claims for the extraction prompt
        existing_summary = self._build_existing_summary(max_claims=15)

        # Collect speaking outputs
        speaking = [o for o in agent_outputs if not o.get("passed") and o.get("message")]
        if not speaking:
            return {}

        messages_text = "\n\n".join(
            f"[{o['agent_id']}]: {o['message'][:600]}"
            for o in speaking
        )

        extraction_prompt = _CLAIM_EXTRACTION_PROMPT.format(
            existing_claims=existing_summary or "(none yet)",
            messages=messages_text,
            round_num=round_num,
        )

        try:
            raw = client.chat(
                provider=    model_cfg["provider"],
                model_id=    model_cfg["model_id"],
                messages=    [{"role": "user", "content": extraction_prompt}],
                temperature= 0.1,
                max_tokens=  1000,
                system=      _EXTRACTION_SYSTEM,
            )
        except Exception:
            return {}

        updates = _parse_extraction_json(raw)
        if not updates:
            return {}

        applied = []
        for item in updates.get("new_claims", []):
            agent_id   = item.get("agent_id", "")
            claim_text = item.get("claim", "").strip()
            attacks    = item.get("attacks", [])
            supports   = item.get("supports", [])
            if agent_id and claim_text:
                cid = self.add_claim(agent_id, claim_text, round_num, attacks, supports)
                applied.append({"id": cid, "agent": agent_id, "claim": claim_text[:80]})

        for viol in updates.get("commitment_violations", []):
            agent_id    = viol.get("agent_id", "")
            prior_id    = viol.get("prior_claim_id", "")
            description = viol.get("description", "")
            if agent_id and description:
                self.record_commitment_violation(agent_id, prior_id, round_num, description)

        return {"claims_added": applied, "round": round_num}

    # ── read ───────────────────────────────────────────────────────────────────

    def get_judge_summary(self, last_n_rounds: int = 3) -> str:
        """
        Build a concise summary for the judge about argument structure.
        Covers: unanswered attacks, commitment violations, cycling detection.
        """
        if not self._graph["claims"]:
            return ""

        lines = ["=== Argument Structure ==="]

        # Unanswered attacks
        unanswered = self._find_unanswered_attacks(last_n_rounds)
        if unanswered:
            lines.append("\nUnanswered attacks (challenge made, no direct response):")
            for item in unanswered[:5]:
                lines.append(
                    f"  [{item['attacker']}] challenged [{item['defender']}]: "
                    f"\"{item['attack_text'][:100]}\" (round {item['round']})"
                )

        # Commitment violations
        violations = self._graph.get("commitment_violations", [])
        recent_v   = [v for v in violations if v["round"] >= max(1, self._current_round() - last_n_rounds)]
        if recent_v:
            lines.append("\nCommitment violations (agent contradicted prior assertion):")
            for v in recent_v[:3]:
                lines.append(f"  [{v['agent_id']}]: {v['description'][:120]} (round {v['round']})")

        # Cycling detection
        cycling = self._detect_cycling(last_n_rounds)
        if cycling:
            lines.append("\nStructural cycling detected:")
            for c in cycling[:3]:
                lines.append(f"  [{c['agent_id']}]: {c['description']}")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ── internals ──────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "claims": [],
            "commitment_violations": [],
            "challenged_status": {},   # claim_id -> {challenged_by: [], answered: bool}
        }

    def _save(self) -> None:
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._graph, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._path)

    def _mark_challenged(self, claim_id: str, challenger_id: str, round_num: int) -> None:
        cs = self._graph.setdefault("challenged_status", {})
        if claim_id not in cs:
            cs[claim_id] = {"challenged_by": [], "answered_round": None}
        cs[claim_id]["challenged_by"].append({
            "claim_id": challenger_id,
            "round":    round_num,
        })

    def _build_existing_summary(self, max_claims: int = 15) -> str:
        claims = self._graph["claims"][-max_claims:]
        if not claims:
            return ""
        lines = []
        for c in claims:
            atk = f"  attacks={c['attacks']}" if c["attacks"] else ""
            lines.append(f"[{c['id']}] ({c['agent_id']}, r{c['round']}): {c['text'][:100]}{atk}")
        return "\n".join(lines)

    def _find_unanswered_attacks(self, last_n_rounds: int) -> list[dict]:
        current_r = self._current_round()
        cutoff    = max(1, current_r - last_n_rounds)
        results   = []

        cs = self._graph.get("challenged_status", {})
        claims_by_id = {c["id"]: c for c in self._graph["claims"]}

        for claim_id, status in cs.items():
            challenged_by = status.get("challenged_by", [])
            recent_challenges = [ch for ch in challenged_by if ch["round"] >= cutoff]
            if not recent_challenges:
                continue
            if status.get("answered_round") and status["answered_round"] >= recent_challenges[-1]["round"]:
                continue

            defended_claim = claims_by_id.get(claim_id, {})
            last_challenge = recent_challenges[-1]
            attacking_claim = claims_by_id.get(last_challenge["claim_id"], {})

            results.append({
                "defender":   defended_claim.get("agent_id", "?"),
                "attacker":   attacking_claim.get("agent_id", "?"),
                "attack_text":attacking_claim.get("text", "")[:120],
                "round":      last_challenge["round"],
                "claim_id":   claim_id,
            })

        return results

    def _detect_cycling(self, last_n_rounds: int) -> list[dict]:
        """
        Simple cycling detection: agent has made 3+ claims in last N rounds
        that all attack the same prior claim without any support relationships
        changing — indicates the same argument being restated.
        """
        current_r = self._current_round()
        cutoff    = max(1, current_r - last_n_rounds)

        recent = [c for c in self._graph["claims"] if c["round"] >= cutoff]
        # Group by agent
        by_agent: dict[str, list] = {}
        for c in recent:
            by_agent.setdefault(c["agent_id"], []).append(c)

        results = []
        for agent_id, claims in by_agent.items():
            if len(claims) < 3:
                continue
            # Check if all attacks target the same small set of claims
            all_attacks = [a for c in claims for a in c.get("attacks", [])]
            if len(all_attacks) >= 3:
                unique_targets = len(set(all_attacks))
                if unique_targets <= 2:
                    results.append({
                        "agent_id":    agent_id,
                        "description": (
                            f"Made {len(claims)} claims over {last_n_rounds} rounds, "
                            f"all attacking the same {unique_targets} target(s) — "
                            f"possible cycling."
                        ),
                    })
        return results

    def _current_round(self) -> int:
        claims = self._graph.get("claims", [])
        if not claims:
            return 0
        return max(c.get("round", 0) for c in claims)


# ── Prompts ────────────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = (
    "You extract argument structure from dialogue. "
    "You identify substantive intellectual claims, which prior claims they attack or support, "
    "and whether any agent has contradicted a prior public commitment. "
    "You are precise and structurally focused — not interpretive."
)

_CLAIM_EXTRACTION_PROMPT = """=== Existing Claims in the Argument Graph ===
{existing_claims}

=== Round {round_num} Agent Messages ===
{messages}

Extract the argument structure. Respond ONLY with valid JSON — no preamble:

{{
  "new_claims": [
    {{
      "agent_id": "richard_feynman",
      "claim": "A specific, non-trivial intellectual claim in one sentence.",
      "attacks": ["cl_abc123"],
      "supports": []
    }}
  ],
  "commitment_violations": [
    {{
      "agent_id": "soren_kierkegaard",
      "prior_claim_id": "cl_xyz789",
      "description": "Agent previously asserted X but now asserts not-X."
    }}
  ]
}}

Rules:
- Only include substantive claims that could be disputed — not pleasantries or meta-comments.
- attacks and supports reference IDs from the existing claims list. Empty arrays if no relationship.
- Only flag commitment_violations for clear direct contradictions, not nuance or development.
- Limit to the 3-5 most significant claims per round. Zero is valid.
- Omit commitment_violations key if none detected.
"""


def _parse_extraction_json(raw: str) -> dict:
    import re
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",           "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*?"new_claims".*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}

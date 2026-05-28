"""
core/src/memory/belief_store.py
Manages explicit belief state for one agent across rounds.

A belief is a proposition the agent has publicly committed to in the dialogue,
tracked with a confidence score and revision history. Unlike reflections (private
notes), beliefs represent what the agent has *asserted* — making commitment
violations detectable by the judge and argument graph.

Structure under sessions/{exp}/run_{N}/memory/{agent_id}/beliefs/:
  bel_{uuid}.json        — one file per belief
  belief_index.json      — [{id, proposition, confidence, status, first_stated_round}]

Status values:
  "held"      — currently asserted
  "suspended" — acknowledged as uncertain or under revision
  "retracted" — explicitly abandoned

Confidence: 0.0–1.0 float. Updated by belief_revision.py after each round.

FIX (2026-05-10): Introduced alongside belief_revision.py and argument_graph.py
to give agents an explicit representation of their own intellectual commitments,
enabling genuine tracking of belief change rather than just rhetorical development.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Optional


class BeliefStore:
    def __init__(self, sess_dir: str, agent_id: str):
        self.sess_dir   = sess_dir
        self.agent_id   = agent_id
        # Compute path directly — mirrors reflection_dir pattern from checkpoint.py
        self._belief_dir  = os.path.join(sess_dir, "memory", agent_id, "beliefs")
        self._index_path  = os.path.join(self._belief_dir, "belief_index.json")
        Path(self._belief_dir).mkdir(parents=True, exist_ok=True)

    # ── write ──────────────────────────────────────────────────────────────────

    def assert_belief(
        self,
        proposition:   str,
        confidence:    float,
        round_num:     int,
        evidence_text: str = "",
    ) -> str:
        """
        Record a new belief. Returns the belief ID.
        confidence: 0.0–1.0. 1.0 = fully committed, 0.5 = tentative.
        """
        belief_id = f"bel_{self.agent_id}_{uuid.uuid4().hex[:8]}"
        belief = {
            "id":                belief_id,
            "agent_id":          self.agent_id,
            "proposition":       proposition,
            "confidence":        round(float(confidence), 3),
            "status":            "held",
            "first_stated_round":round_num,
            "last_revised_round":round_num,
            "revision_history":  [],
            "evidence_text":     evidence_text,
        }
        self._write_belief(belief_id, belief)
        self._upsert_index(belief_id, proposition, confidence, "held", round_num)
        return belief_id

    def revise_belief(
        self,
        belief_id:    str,
        new_confidence: float,
        round_num:    int,
        reason:       str = "",
    ) -> bool:
        """
        Update confidence on an existing belief. Returns True if found.
        """
        belief = self._read_belief(belief_id)
        if belief is None:
            return False

        old_confidence = belief["confidence"]
        belief["confidence"]         = round(float(new_confidence), 3)
        belief["last_revised_round"] = round_num
        belief["revision_history"].append({
            "round":              round_num,
            "confidence_before":  old_confidence,
            "confidence_after":   round(float(new_confidence), 3),
            "reason":             reason,
        })

        self._write_belief(belief_id, belief)
        self._upsert_index(
            belief_id, belief["proposition"], new_confidence,
            belief["status"], round_num
        )
        return True

    def retract_belief(
        self,
        belief_id: str,
        round_num: int,
        reason:    str = "",
    ) -> bool:
        """Mark a belief as retracted. Returns True if found."""
        belief = self._read_belief(belief_id)
        if belief is None:
            return False

        belief["status"]            = "retracted"
        belief["last_revised_round"] = round_num
        belief["revision_history"].append({
            "round":  round_num,
            "action": "retracted",
            "reason": reason,
        })

        self._write_belief(belief_id, belief)
        self._upsert_index(
            belief_id, belief["proposition"], belief["confidence"],
            "retracted", round_num
        )
        return True

    def suspend_belief(
        self,
        belief_id: str,
        round_num: int,
        reason:    str = "",
    ) -> bool:
        """Mark a belief as suspended (uncertain, under revision)."""
        belief = self._read_belief(belief_id)
        if belief is None:
            return False

        belief["status"]             = "suspended"
        belief["last_revised_round"] = round_num
        belief["revision_history"].append({
            "round":  round_num,
            "action": "suspended",
            "reason": reason,
        })

        self._write_belief(belief_id, belief)
        self._upsert_index(
            belief_id, belief["proposition"], belief["confidence"],
            "suspended", round_num
        )
        return True

    # ── read ───────────────────────────────────────────────────────────────────

    def load_index(self) -> list[dict]:
        """Return the full belief index list."""
        if not os.path.exists(self._index_path):
            return []
        with open(self._index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_active_beliefs(self) -> list[dict]:
        """Return held and suspended beliefs (not retracted)."""
        return [
            b for b in self.load_index()
            if b.get("status") in ("held", "suspended")
        ]

    def get_belief_by_id(self, belief_id: str) -> Optional[dict]:
        return self._read_belief(belief_id)

    def get_summary_for_context(self, max_beliefs: int = 5) -> str:
        """
        Return a formatted string of the agent's active beliefs for injection
        into the agent's context. Sorted by confidence descending.
        """
        active = sorted(
            self.get_active_beliefs(),
            key=lambda b: b.get("confidence", 0),
            reverse=True
        )[:max_beliefs]

        if not active:
            return ""

        lines = []
        for b in active:
            conf   = b.get("confidence", 0)
            status = b.get("status", "held")
            stars  = _confidence_bar(conf)
            tag    = "" if status == "held" else f" [{status.upper()}]"
            lines.append(f"  {stars} {b['proposition']}{tag}")
        return "\n".join(lines)

    # ── internals ──────────────────────────────────────────────────────────────

    def _write_belief(self, belief_id: str, belief: dict) -> None:
        path = os.path.join(self._belief_dir, f"{belief_id}.json")
        tmp  = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(belief, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _read_belief(self, belief_id: str) -> Optional[dict]:
        path = os.path.join(self._belief_dir, f"{belief_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _upsert_index(
        self,
        belief_id:    str,
        proposition:  str,
        confidence:   float,
        status:       str,
        round_num:    int,
    ) -> None:
        index = self.load_index()
        for entry in index:
            if entry["id"] == belief_id:
                entry["proposition"] = proposition
                entry["confidence"]  = round(float(confidence), 3)
                entry["status"]      = status
                entry["last_revised_round"] = round_num
                break
        else:
            index.append({
                "id":                belief_id,
                "proposition":       proposition,
                "confidence":        round(float(confidence), 3),
                "status":            status,
                "first_stated_round":round_num,
                "last_revised_round":round_num,
            })
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._index_path)


def _confidence_bar(conf: float) -> str:
    """Visual confidence indicator: ●●●○○ style."""
    filled = round(conf * 5)
    return "●" * filled + "○" * (5 - filled)

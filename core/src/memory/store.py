"""
memory/store.py
Manages per-agent memory storage: reflections and full round data.

Structure under sessions/{exp}/run_{N}/memory/{agent_id}/:
  reflections/
    refl_{uuid}.json   — one file per reflection
  reflection_index.json — list of {id, content, embedding, round_refs, speakers}
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.src.utils.checkpoint import reflection_dir, round_file


class MemoryStore:
    def __init__(self, sess_dir: str, agent_id: str):
        self.sess_dir    = sess_dir
        self.agent_id    = agent_id
        self._refl_dir   = reflection_dir(sess_dir, agent_id)
        self._index_path = os.path.join(self._refl_dir, "reflection_index.json")
        Path(self._refl_dir).mkdir(parents=True, exist_ok=True)

    # ── reflection write ───────────────────────────────────────────────────────

    def add_reflection(self, content: str, round_num: int,
                       speakers: list[str],
                       embedding: list[float]) -> str:
        """Create a new reflection. Returns its ID."""
        refl_id = f"refl_{self.agent_id}_{uuid.uuid4().hex[:8]}"
        refl = {
            "id":            refl_id,
            "agent_id":      self.agent_id,
            "content":       content,
            "created_round": round_num,
            "extended_rounds": [],
            "speakers_referenced": speakers,
            "embedding":     embedding,
        }
        self._write_reflection(refl_id, refl)
        self._upsert_index(refl_id, content, embedding, round_num, speakers)
        return refl_id

    def extend_reflection(self, refl_id: str, additional_content: str,
                          round_num: int, speakers: list[str],
                          new_embedding: list[float]) -> None:
        """Append to an existing reflection and update its embedding."""
        refl = self._read_reflection(refl_id)
        if refl is None:
            # Graceful fallback: treat as new
            self.add_reflection(additional_content, round_num, speakers, new_embedding)
            return

        refl["content"] += f"\n[Round {round_num}] {additional_content}"
        refl["extended_rounds"].append(round_num)
        refl["speakers_referenced"] = list(
            set(refl["speakers_referenced"]) | set(speakers)
        )
        refl["embedding"] = new_embedding  # updated embedding for extended content
        self._write_reflection(refl_id, refl)
        self._upsert_index(refl_id, refl["content"], new_embedding, round_num, speakers)

    # ── reflection read ────────────────────────────────────────────────────────

    def load_index(self) -> list[dict]:
        """Return the list of reflection index entries (id, content, embedding)."""
        if not os.path.exists(self._index_path):
            return []
        with open(self._index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_reflections_by_ids(self, refl_ids: list[str]) -> list[dict]:
        """Load full reflection objects for the given IDs."""
        result = []
        for rid in refl_ids:
            r = self._read_reflection(rid)
            if r:
                result.append(r)
        return result

    def get_all_reflection_contents(self) -> list[str]:
        """Return just the content strings for all reflections (for reflector prompt)."""
        return [entry["content"] for entry in self.load_index()]

    # ── round data access (researcher archive) ─────────────────────────────────

    def load_round_events(self, round_num: int) -> list[dict]:
        """Load all events for a given round from the session's JSONL log."""
        from core.src.utils.checkpoint import load_events
        path = round_file(self.sess_dir, round_num)
        return load_events(path)

    # ── internals ──────────────────────────────────────────────────────────────

    def _write_reflection(self, refl_id: str, refl: dict) -> None:
        path = os.path.join(self._refl_dir, f"{refl_id}.json")
        tmp  = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(refl, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _read_reflection(self, refl_id: str) -> Optional[dict]:
        path = os.path.join(self._refl_dir, f"{refl_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _upsert_index(self, refl_id: str, content: str,
                      embedding: list[float], round_num: int,
                      speakers: list[str]) -> None:
        index = self.load_index()
        # Replace existing entry or append
        for entry in index:
            if entry["id"] == refl_id:
                entry["content"]   = content
                entry["embedding"] = embedding
                break
        else:
            index.append({
                "id":        refl_id,
                "content":   content,
                "embedding": embedding,
                "round":     round_num,
                "speakers":  speakers,
            })
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._index_path)

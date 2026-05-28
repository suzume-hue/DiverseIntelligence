"""
rate_tracker.py
Tracks per-key RPD (requests per day) usage across providers.
Persists state to state/key_usage.json.
Rotates keys on 429. Raises QuotaExhaustedError when ALL keys are exhausted.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo


STATE_FILE = "state/key_usage.json"

# Provider daily reset configuration
RESET_CONFIG = {
    "groq":   {"tz": "UTC",        "hour": 0},
    "google": {"tz": "US/Pacific", "hour": 0},
}


class QuotaExhaustedError(Exception):
    """Raised when all keys for a provider+model are at their daily limit."""
    def __init__(self, provider: str, model_id: str, reset_info: str):
        self.provider   = provider
        self.model_id   = model_id
        self.reset_info = reset_info
        super().__init__(f"All {provider} keys exhausted for {model_id}. {reset_info}")


class RateTracker:
    def __init__(self, base_dir: str = ".", rate_limits: dict = None):
        self.base_dir     = base_dir
        self.rate_limits  = rate_limits or {}
        self._state_path  = os.path.join(base_dir, STATE_FILE)
        self._state       = self._load_state()

    # ── public API ─────────────────────────────────────────────────────────────

    def get_available_key(self, provider: str, model_id: str,
                          api_keys: list[str]) -> tuple[str, int]:
        """
        Returns (api_key_string, key_index) for the first key that still has
        RPD headroom. Raises QuotaExhaustedError if none remain.
        """
        self._maybe_reset_all(provider)
        rpd_limit = self._get_rpd_limit(provider, model_id)

        for idx, key in enumerate(api_keys):
            used = self._get_used(provider, idx, model_id)
            if rpd_limit is None or used < rpd_limit:
                return key, idx

        reset_msg = self._reset_message(provider)
        raise QuotaExhaustedError(provider, model_id, reset_msg)

    def record_request(self, provider: str, key_index: int, model_id: str) -> None:
        """Increment the RPD counter for a specific key after a successful call."""
        self._ensure_keys(provider, key_index, model_id)
        self._state[provider][f"key_{key_index}"][model_id]["rpd_used"] += 1
        self._save_state()

    def handle_429(self, provider: str, key_index: int, model_id: str,
                   api_keys: list[str]) -> tuple[str, int]:
        """
        Called when a 429 is received. Marks the current key as exhausted,
        then returns the next available key. Raises QuotaExhaustedError if
        no more keys remain.
        """
        # Treat this key as fully used
        rpd_limit = self._get_rpd_limit(provider, model_id)
        if rpd_limit is not None:
            self._ensure_keys(provider, key_index, model_id)
            self._state[provider][f"key_{key_index}"][model_id]["rpd_used"] = rpd_limit
            self._save_state()

        # Try next keys
        for idx in range(key_index + 1, len(api_keys)):
            used = self._get_used(provider, idx, model_id)
            if rpd_limit is None or used < rpd_limit:
                return api_keys[idx], idx

        reset_msg = self._reset_message(provider)
        raise QuotaExhaustedError(provider, model_id, reset_msg)

    def remaining(self, provider: str, key_index: int, model_id: str) -> Optional[int]:
        """Returns remaining RPD for a key, or None if unlimited."""
        self._maybe_reset_all(provider)
        rpd_limit = self._get_rpd_limit(provider, model_id)
        if rpd_limit is None:
            return None
        used = self._get_used(provider, key_index, model_id)
        return max(0, rpd_limit - used)

    def status_summary(self, provider: str, model_id: str, api_keys: list[str]) -> str:
        """Human-readable summary of key pool status."""
        lines = []
        rpd_limit = self._get_rpd_limit(provider, model_id)
        for idx, _ in enumerate(api_keys):
            used = self._get_used(provider, idx, model_id)
            cap  = str(rpd_limit) if rpd_limit else "∞"
            lines.append(f"  key_{idx}: {used}/{cap} RPD used")
        return "\n".join(lines)

    # ── internals ──────────────────────────────────────────────────────────────

    def _get_rpd_limit(self, provider: str, model_id: str) -> Optional[int]:
        return (self.rate_limits
                .get(provider, {})
                .get("models", {})
                .get(model_id, {})
                .get("rpd"))

    def _get_used(self, provider: str, key_index: int, model_id: str) -> int:
        self._ensure_keys(provider, key_index, model_id)
        return self._state[provider][f"key_{key_index}"][model_id]["rpd_used"]

    def _ensure_keys(self, provider: str, key_index: int, model_id: str) -> None:
        key_label = f"key_{key_index}"
        self._state.setdefault(provider, {})
        self._state[provider].setdefault(key_label, {})
        self._state[provider][key_label].setdefault(model_id, {
            "rpd_used": 0,
            "last_reset": self._current_reset_ts(provider),
        })

    def _maybe_reset_all(self, provider: str) -> None:
        """Reset counters for any key+model whose last_reset is before the current daily window."""
        if provider not in self._state:
            return
        reset_ts = self._current_reset_ts(provider)
        changed  = False
        for key_label, models in self._state[provider].items():
            for model_id, entry in models.items():
                if entry.get("last_reset", "") < reset_ts:
                    entry["rpd_used"]   = 0
                    entry["last_reset"] = reset_ts
                    changed = True
        if changed:
            self._save_state()

    def _current_reset_ts(self, provider: str) -> str:
        """ISO timestamp of the most recent daily reset for this provider."""
        cfg  = RESET_CONFIG.get(provider, {"tz": "UTC", "hour": 0})
        tz   = ZoneInfo(cfg["tz"])
        now  = datetime.now(tz)
        reset = now.replace(hour=cfg["hour"], minute=0, second=0, microsecond=0)
        if now < reset:
            reset -= timedelta(days=1)
        return reset.isoformat()

    def _reset_message(self, provider: str) -> str:
        cfg  = RESET_CONFIG.get(provider, {"tz": "UTC", "hour": 0})
        tz   = ZoneInfo(cfg["tz"])
        now  = datetime.now(tz)
        reset = now.replace(hour=cfg["hour"], minute=0, second=0, microsecond=0)
        if now >= reset:
            reset += timedelta(days=1)
        local = reset.strftime("%Y-%m-%d %H:%M %Z")
        return f"Quota resets at {local} ({cfg['tz']}). Restart the run after that."

    def _load_state(self) -> dict:
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_state(self) -> None:
        Path(self._state_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._state_path)

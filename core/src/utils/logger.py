import logging
import os
from datetime import datetime
from pathlib import Path


class Logger:
    def __init__(self, log_dir: str = "logs", session_id: str = ""):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"_{session_id}" if session_id else ""
        log_file = os.path.join(log_dir, f"session{label}_{timestamp}.log")

        self._logger = logging.getLogger(f"macp.{session_id or 'main'}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%H:%M:%S")

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        self._logger.addHandler(fh)
        self._logger.addHandler(ch)
        self._logger.propagate = False

    # ── core ──────────────────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    # ── structured events ─────────────────────────────────────────────────────

    def session_start(self, session_id: str, experiment: str, run_num: int) -> None:
        self.info("=" * 65)
        self.info(f"  SESSION START  |  {session_id}  |  Exp-{experiment.upper()}  Run-{run_num}")
        self.info("=" * 65)

    def session_resume(self, session_id: str, from_round: int) -> None:
        self.info(f"  RESUMING  {session_id}  from round {from_round}")
        self.info("=" * 65)

    def round_start(self, round_num: int) -> None:
        self.info(f"\n{'─'*55}")
        self.info(f"  ROUND {round_num}")
        self.info(f"{'─'*55}")

    def agent_output(self, agent_id: str, output_type: str, preview: str = "") -> None:
        snippet = f": {preview[:90]}..." if preview else ""
        self.info(f"    [{agent_id:<22}] → {output_type.upper()}{snippet}")

    def meta_comm(self, for_receiver: str, preview: str = "") -> None:
        snippet = f": {preview[:80]}..." if preview else ""
        self.info(f"    [meta-comm → {for_receiver:<18}] {snippet}")

    def reflection(self, agent_id: str, count: int) -> None:
        self.info(f"    [{agent_id:<22}] ← {count} reflection(s) generated")

    def judge_decision(self, round_num: int, decision: str, observation: str) -> None:
        symbol = "✓" if decision == "CONTINUE" else "✗"
        self.info(f"\n  JUDGE [{symbol}] {decision}  — {observation[:120]}")

    def quota_exhausted(self, provider: str, reset_info: str) -> None:
        self.warning("!" * 65)
        self.warning(f"  QUOTA EXHAUSTED  [{provider.upper()}]")
        self.warning(f"  {reset_info}")
        self.warning("!" * 65)

    def session_complete(self, session_id: str, total_rounds: int) -> None:
        self.info("=" * 65)
        self.info(f"  SESSION COMPLETE  |  {session_id}  |  {total_rounds} rounds")
        self.info("=" * 65)

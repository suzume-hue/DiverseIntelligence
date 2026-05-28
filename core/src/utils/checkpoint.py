"""
core/src/utils/checkpoint.py
Atomic disk I/O for session state, round events, and config snapshots.
All paths are relative to experiment_dir (each experiment is self-contained).
"""

import json, os, shutil, sys
from datetime import datetime, timezone
from pathlib import Path


# ── paths ──────────────────────────────────────────────────────────────────────

def session_dir(experiment_dir: str, run_num: int) -> str:
    return os.path.join(experiment_dir, "sessions", f"run_{run_num}")

def round_file(sess_dir: str, round_num: int) -> str:
    return os.path.join(sess_dir, "rounds", f"round_{round_num}.jsonl")

def state_file(sess_dir: str) -> str:
    return os.path.join(sess_dir, "session_state.json")

def memory_dir(sess_dir: str, agent_id: str) -> str:
    return os.path.join(sess_dir, "memory", agent_id)

def reflection_dir(sess_dir: str, agent_id: str) -> str:
    return os.path.join(memory_dir(sess_dir, agent_id), "reflections")

def snapshot_dir(experiment_dir: str) -> str:
    return os.path.join(experiment_dir, ".di_state")

def snapshot_file(experiment_dir: str, run_num: int) -> str:
    return os.path.join(snapshot_dir(experiment_dir), f"run_{run_num}_config.json")


# ── atomic JSONL append ────────────────────────────────────────────────────────

def append_event(filepath: str, event: dict) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    event["_ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(event, ensure_ascii=False)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_events(filepath: str) -> list[dict]:
    if not os.path.exists(filepath):
        return []
    events = []
    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed event on line {lineno} of {filepath}")
    return events


def events_by_type(filepath: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for ev in load_events(filepath):
        t = ev.get("event_type", "unknown")
        result.setdefault(t, []).append(ev)
    return result


# ── session state ──────────────────────────────────────────────────────────────

def init_session_state(sess_dir: str, experiment: str, run_num: int,
                       agent_ids: list, problem_id: str, temperature: float) -> dict:
    Path(sess_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "experiment": experiment, "run_number": run_num,
        "problem_id": problem_id, "agent_ids": agent_ids,
        "temperature": temperature, "status": "in_progress",
        "current_round": 0, "last_completed_round": -1,
        "started_at": datetime.now(timezone.utc).isoformat(), "completed_at": None,
    }
    _write_state(sess_dir, state)
    return state


def load_session_state(sess_dir: str) -> dict | None:
    path = state_file(sess_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_session_state(sess_dir: str, **kwargs) -> dict:
    state = load_session_state(sess_dir) or {}
    state.update(kwargs)
    _write_state(sess_dir, state)
    return state


def mark_session_complete(sess_dir: str, total_rounds: int) -> None:
    update_session_state(sess_dir, status="complete",
                         last_completed_round=total_rounds,
                         completed_at=datetime.now(timezone.utc).isoformat())


def _write_state(sess_dir: str, state: dict) -> None:
    path = state_file(sess_dir)
    tmp  = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ── config snapshot & conflict ─────────────────────────────────────────────────

def save_config_snapshot(experiment_dir: str, run_num: int, config_path: str) -> None:
    snap_dir = snapshot_dir(experiment_dir)
    Path(snap_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, snapshot_file(experiment_dir, run_num))


def check_config_conflict(experiment_dir: str, run_num: int, config_path: str) -> bool:
    snap = snapshot_file(experiment_dir, run_num)
    if not os.path.exists(snap):
        return False
    with open(config_path, "r", encoding="utf-8") as f:
        current = f.read()
    with open(snap, "r", encoding="utf-8") as f:
        saved = f.read()
    try:
        return json.loads(current) != json.loads(saved)
    except json.JSONDecodeError:
        return current != saved


def prompt_config_conflict(experiment_dir: str, run_num: int,
                           config_path: str, sess_dir: str) -> None:
    snap = snapshot_file(experiment_dir, run_num)
    print("\n" + "!" * 65)
    print(f"  CONFIG CONFLICT — experiment dir: {os.path.basename(experiment_dir)}  Run: {run_num}")
    print(f"  experiment.json changed since this run began.")
    print("!" * 65)
    print("\n  [Y] Override — clear this run and restart with new config")
    print("  [N] Revert   — restore original config and resume as-is\n")
    for _ in range(3):
        choice = input("  Enter Y or N: ").strip().upper()
        if choice == "Y":
            if os.path.exists(sess_dir):
                shutil.rmtree(sess_dir)
            save_config_snapshot(experiment_dir, run_num, config_path)
            return
        elif choice == "N":
            shutil.copy2(snap, config_path)
            print("  Config restored. Resuming.")
            return
        else:
            print("  Please enter Y or N.")
    print("\n  No valid input. Exiting.")
    sys.exit(1)

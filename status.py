#!/usr/bin/env python3
"""
status.py  —  Show status of all experiments at a glance

Usage:
  python status.py              # all experiments
  python status.py --watch      # refresh every 10s
"""

import os
import sys
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
REFRESH_SECS    = 10


def main():
    watch = "--watch" in sys.argv
    while True:
        if watch:
            os.system("clear" if os.name != "nt" else "cls")
        _print_all()
        if not watch:
            break
        print(f"\n  Refreshing every {REFRESH_SECS}s — Ctrl+C to stop")
        time.sleep(REFRESH_SECS)


def _print_all():
    from core.src.utils.checkpoint import session_dir, load_session_state

    experiments = _discover_experiments()
    if not experiments:
        print("\n  No experiments found in experiments/\n")
        return

    print(f"\n{'═'*65}")
    print(f"  DiverseIntelligence — Experiment Status")
    print(f"{'═'*65}")

    for exp_dir in experiments:
        exp_name = os.path.basename(exp_dir)
        exp_json = os.path.join(exp_dir, "experiment.json")
        try:
            with open(exp_json) as f:
                cfg = json.load(f)
            num_runs = cfg.get("num_runs", 5)
            display  = cfg.get("name", exp_name)
        except Exception:
            print(f"\n  [{exp_name}]  ← could not read experiment.json")
            continue

        complete  = 0
        in_prog   = 0
        not_start = 0

        for run_num in range(1, num_runs + 1):
            sd    = session_dir(exp_dir, run_num)
            state = load_session_state(sd)
            if state is None:
                not_start += 1
            elif state.get("status") == "complete":
                complete += 1
            else:
                in_prog += 1

        bar = _progress_bar(complete, in_prog, num_runs)
        print(f"\n  {display}")
        print(f"  {exp_dir.replace(PROJECT_ROOT+'/', '')}")
        print(f"  {bar}  {complete}/{num_runs} complete"
              + (f"  ({in_prog} in progress)" if in_prog else ""))

        # Detail per run
        for run_num in range(1, num_runs + 1):
            sd    = session_dir(exp_dir, run_num)
            state = load_session_state(sd)
            if state is None:
                sym, detail = "·", "not started"
            elif state.get("status") == "complete":
                r = state.get("last_completed_round", "?")
                sym, detail = "✓", f"complete  ({r} rounds)"
            else:
                r = state.get("last_completed_round", -1) + 1
                t = state.get("temperature", "?")
                sym, detail = "⟳", f"round {r}  temp={t}"
            print(f"    Run {run_num}  {sym}  {detail}")

    print(f"\n{'─'*65}\n")


def _discover_experiments():
    if not os.path.isdir(EXPERIMENTS_DIR):
        return []
    found = []
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        if name.startswith("_"):
            continue
        path = os.path.join(EXPERIMENTS_DIR, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "experiment.json")):
            found.append(path)
    return found


def _progress_bar(complete, in_prog, total, width=20):
    done_w  = int(width * complete  / total) if total else 0
    prog_w  = int(width * in_prog   / total) if total else 0
    rest_w  = width - done_w - prog_w
    return "[" + "█" * done_w + "▒" * prog_w + "·" * rest_w + "]"


if __name__ == "__main__":
    main()

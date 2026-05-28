#!/usr/bin/env python3
"""
run.py  —  DiverseIntelligence experiment runner

Usage:
  python run.py <experiment_dir>              # run all pending runs
  python run.py <experiment_dir> --run 3     # run or resume a specific run
  python run.py <experiment_dir> --status    # show status without running

Examples:
  python run.py experiments/quantum_gravity_mixed
  python run.py experiments/quantum_gravity_mixed --run 2
  python run.py experiments/quantum_gravity_control --status
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Run a DiverseIntelligence experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("experiment_dir",
                        help="Path to the experiment directory (e.g. experiments/my_experiment)")
    parser.add_argument("--run",    type=int, metavar="N",
                        help="Run or resume a specific run number (1-based)")
    parser.add_argument("--status", action="store_true",
                        help="Show run status and exit without running")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(exp_dir):
        print(f"[ERROR] Experiment directory not found: {exp_dir}")
        sys.exit(1)
    if not os.path.exists(os.path.join(exp_dir, "experiment.json")):
        print(f"[ERROR] No experiment.json found in: {exp_dir}")
        print("  Is this a valid experiment directory?")
        sys.exit(1)

    from core.src.bootstrap       import bootstrap
    from core.src.session_runner  import run_session
    from core.src.utils.checkpoint import session_dir, load_session_state
    from core.src.api.rate_tracker import QuotaExhaustedError

    loader, client = bootstrap(exp_dir, PROJECT_ROOT)
    num_runs        = loader.num_runs()
    exp_name        = loader.experiment_name()

    # ── Status ─────────────────────────────────────────────────────────────────
    if args.status:
        _print_status(exp_dir, exp_name, num_runs)
        return

    # ── Single run ─────────────────────────────────────────────────────────────
    if args.run:
        if args.run < 1 or args.run > num_runs:
            print(f"[ERROR] --run must be between 1 and {num_runs}.")
            sys.exit(1)
        _execute(args.run, exp_dir, exp_name, num_runs, loader, client)
        return

    # ── All pending runs ───────────────────────────────────────────────────────
    any_run = False
    for run_num in range(1, num_runs + 1):
        sd    = session_dir(exp_dir, run_num)
        state = load_session_state(sd)
        if state and state.get("status") == "complete":
            print(f"[{exp_name}] Run {run_num}/{num_runs} already complete — skipping.")
            continue
        any_run = True
        _execute(run_num, exp_dir, exp_name, num_runs, loader, client)

    if not any_run:
        print(f"\nAll {num_runs} runs complete for: {exp_name}")


def _execute(run_num, exp_dir, exp_name, num_runs, loader, client):
    from core.src.session_runner  import run_session
    from core.src.api.rate_tracker import QuotaExhaustedError

    print(f"\n{'═'*60}")
    print(f"  {exp_name}  |  Run {run_num}/{num_runs}")
    print(f"{'═'*60}\n")

    try:
        run_session(
            experiment_dir= exp_dir,
            run_num=        run_num,
            project_root=   PROJECT_ROOT,
            loader=         loader,
            client=         client,
        )
    except QuotaExhaustedError as e:
        print(f"\n[QUOTA EXHAUSTED] {e}")
        print("Progress saved. Re-run this command tomorrow to continue.")
        sys.exit(0)


def _print_status(exp_dir, exp_name, num_runs):
    from core.src.utils.checkpoint import session_dir, load_session_state
    print(f"\n{'─'*60}")
    print(f"  {exp_name}")
    print(f"  {exp_dir}")
    print(f"{'─'*60}")
    for run_num in range(1, num_runs + 1):
        sd    = session_dir(exp_dir, run_num)
        state = load_session_state(sd)
        if state is None:
            status = "not started"
        elif state.get("status") == "complete":
            rounds = state.get("last_completed_round", "?")
            status = f"✓ complete  ({rounds} rounds)"
        else:
            r = state.get("last_completed_round", -1) + 1
            t = state.get("temperature", "?")
            status = f"⟳ in progress  (next: round {r},  temp: {t})"
        print(f"  Run {run_num:>2}:  {status}")
    print()


if __name__ == "__main__":
    main()

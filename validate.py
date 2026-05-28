#!/usr/bin/env python3
"""
validate.py  —  Pre-run validation for any experiment

Usage:
  python validate.py experiments/my_experiment
  python validate.py experiments/quantum_gravity_mixed
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

OK   = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp_dir = os.path.abspath(sys.argv[1])
    errors  = 0

    print(f"\n{'─'*60}")
    print(f"  Validating: {os.path.basename(exp_dir)}")
    print(f"  Path: {exp_dir}")
    print(f"{'─'*60}")

    errors += _check_global_files()
    errors += _check_api_keys()
    errors += _check_experiment_files(exp_dir)

    if errors == 0:
        from core.src.utils.config_loader import ConfigLoader
        loader = ConfigLoader(exp_dir, PROJECT_ROOT)
        loader.load_all()
        errors += _check_agents(loader)
        errors += _check_problem(loader)
        errors += _check_models(loader)
        errors += _check_meta_comm(loader)
        errors += _check_temperature(loader)

    print()
    if errors == 0:
        print("  ✓ All checks passed. Ready to run:\n")
        print(f"    python run.py {sys.argv[1]}\n")
    else:
        print(f"  {errors} issue(s) found. Fix them before running.\n")
        sys.exit(1)


def _check_global_files():
    print("\n── Global files ─────────────────────────────────────────────")
    errors = 0
    for rel in ["config/api_keys.json", "config/rate_limits.json"]:
        path = os.path.join(PROJECT_ROOT, rel)
        if os.path.exists(path):
            print(f"{OK} {rel}")
        else:
            print(f"{FAIL} {rel}  ← MISSING")
            errors += 1
    return errors


def _check_api_keys():
    print("\n── API keys ─────────────────────────────────────────────────")
    path = os.path.join(PROJECT_ROOT, "config/api_keys.json")
    if not os.path.exists(path):
        return 1
    with open(path) as f:
        raw = json.load(f)
    errors = 0
    for provider, data in raw.items():
        real = [k for k in data.get("keys", [])
                if k and not k.startswith("your_") and not k.startswith("gsk_your")]
        if not real:
            print(f"{FAIL} {provider}: no real keys — update config/api_keys.json")
            errors += 1
        else:
            print(f"{OK} {provider}: {len(real)} key(s)")
    return errors


def _check_experiment_files(exp_dir):
    print("\n── Experiment files ─────────────────────────────────────────")
    errors = 0
    required = ["experiment.json"]
    for fname in required:
        path = os.path.join(exp_dir, fname)
        if os.path.exists(path):
            print(f"{OK} {fname}")
        else:
            print(f"{FAIL} {fname}  ← MISSING")
            errors += 1
    return errors


def _check_agents(loader):
    print("\n── Agents ───────────────────────────────────────────────────")
    errors = 0
    try:
        agents = loader.validate_agents()
        for a in agents:
            print(f"{OK} '{a['id']}' ({a['domain']}) → {a['display_name']}")
    except SystemExit:
        errors += 1
    return errors


def _check_problem(loader):
    print("\n── Problem ──────────────────────────────────────────────────")
    try:
        p = loader.problem
        print(f"{OK} '{p.get('id', '?')}' — {p.get('title', '?')[:55]}")
    except Exception as e:
        print(f"{FAIL} Problem load failed: {e}")
        return 1
    return 0


def _check_models(loader):
    print("\n── Models ───────────────────────────────────────────────────")
    errors = 0
    rl_path = os.path.join(PROJECT_ROOT, "config/rate_limits.json")
    with open(rl_path) as f:
        rl = json.load(f)

    exp = loader.experiment
    models = exp.get("models", {})
    use_mc = loader.is_meta_comm_enabled()

    for role, mcfg in models.items():
        if role == "meta_comm" and not use_mc:
            continue
        provider = mcfg.get("provider", "")
        model_id = mcfg.get("model_id", "")
        known    = rl.get(provider, {}).get("models", {})
        if model_id in known:
            ctx = known[model_id].get("context_window", 0)
            rpd = known[model_id].get("rpd", "∞")
            print(f"{OK} [{role}] {provider}/{model_id}  ctx:{ctx:,}  rpd:{rpd}")
        else:
            print(f"{WARN} [{role}] {provider}/{model_id} not in rate_limits.json "
                  f"— quota tracking will use defaults")
    return errors


def _check_meta_comm(loader):
    print("\n── MetaCommunicator ─────────────────────────────────────────")
    if not loader.is_meta_comm_enabled():
        print(f"  (disabled — skipping)")
        return 0
    errors = 0
    if loader.meta_comm_prompt:
        print(f"{OK} Persona loaded ({len(loader.meta_comm_prompt)} chars)")
    else:
        print(f"{FAIL} meta_comm.persona_file not found or empty")
        errors += 1
    groups = loader.get_translation_groups()
    if groups:
        for i, g in enumerate(groups):
            print(f"{OK} Group {i}: {g}")
    else:
        print(f"{WARN} meta_comm.enabled=true but no groups defined — "
              f"translation will not occur")
    return errors


def _check_temperature(loader):
    print("\n── Temperature schedule ─────────────────────────────────────")
    sched     = loader.experiment.get("temperature_schedule", {})
    num_runs  = loader.num_runs()
    agents_t  = sched.get("agents", [0.7])
    if isinstance(agents_t, list) and len(agents_t) < num_runs:
        print(f"{WARN} agents temperature list has {len(agents_t)} entries "
              f"but num_runs={num_runs}. Last value reused for remaining runs.")
    else:
        print(f"{OK} Temperature schedule looks good")
    return 0


if __name__ == "__main__":
    main()

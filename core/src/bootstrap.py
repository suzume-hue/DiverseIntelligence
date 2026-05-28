"""
core/src/bootstrap.py
Shared initialisation. Loads global config, API keys, builds LLM client.
Call bootstrap(project_root) from any entry point to get (loader, client).
"""

import json
import os
import sys


def bootstrap(experiment_dir: str, project_root: str):
    """
    Returns (ConfigLoader, LLMClient) ready to use.
    experiment_dir : path to the specific experiment folder
    project_root   : path to DiverseIntelligence root
    """
    from core.src.utils.config_loader import ConfigLoader
    from core.src.api.rate_tracker    import RateTracker
    from core.src.api.client          import LLMClient

    loader = ConfigLoader(experiment_dir, project_root)
    loader.load_all()

    # ── API keys ───────────────────────────────────────────────────────────────
    keys_path = os.path.join(project_root, "config", "api_keys.json")
    if not os.path.exists(keys_path):
        print("[ERROR] config/api_keys.json not found.")
        sys.exit(1)

    with open(keys_path) as f:
        raw_keys = json.load(f)

    api_keys: dict[str, list[str]] = {}
    for provider, data in raw_keys.items():
        real = [k for k in data.get("keys", [])
                if k and not k.startswith("your_") and not k.startswith("gsk_your")]
        if real:
            api_keys[provider] = real
        else:
            print(f"[WARN] No real keys for provider '{provider}'. "
                  f"Edit config/api_keys.json.")

    if not api_keys:
        print("[ERROR] No valid API keys found. Edit config/api_keys.json.")
        sys.exit(1)

    # ── Rate limits ────────────────────────────────────────────────────────────
    rl_path = os.path.join(project_root, "config", "rate_limits.json")
    with open(rl_path) as f:
        rate_limits = json.load(f)

    # State dir is global (project-level)
    state_dir = os.path.join(project_root, "state")
    tracker = RateTracker(base_dir=project_root, rate_limits=rate_limits)
    client  = LLMClient(api_keys=api_keys, rate_tracker=tracker)

    return loader, client

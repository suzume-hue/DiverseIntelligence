"""
core/src/utils/config_loader.py
Loads and validates all config for a given experiment directory.

An experiment directory contains:
  experiment.json   - master settings (agents, runs, models, temperatures, groups)
  personas.json     - persona definitions for this experiment
                      OR per-agent persona_file paths pointing anywhere
  problem.json      - the problem statement
  judge.json        - judge persona
  meta_comm.json    - MetaCommunicator persona (if meta_comm.enabled = true)

All paths in experiment.json are resolved relative to the experiment directory.
Shared resources (../../shared/personas/...) are resolved relative to project root.
"""

import json
import os
import sys
from typing import Any


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[ERROR] Required file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ConfigLoader:
    def __init__(self, experiment_dir: str, project_root: str):
        self.exp_dir      = os.path.abspath(experiment_dir)
        self.project_root = os.path.abspath(project_root)
        self._exp         = None
        self._problem     = None
        self._judge       = None
        self._meta_comm_prompt = None
        self._persona_cache: dict[str, dict] = {}  # file_path -> {domain: {name: persona}}
        self._rate_limits = None

    # ── load ──────────────────────────────────────────────────────────────────

    def load_all(self) -> None:
        exp_path = self._exp_path("experiment.json")
        self._exp = _load_json(exp_path)

        problem_file  = self._exp.get("problem_file", "problem.json")
        judge_file    = self._exp.get("judge_file",   "judge.json")

        self._problem = _load_json(self._resolve(problem_file))
        self._judge   = _load_json(self._resolve(judge_file))

        mc_cfg = self._exp.get("meta_comm", {})
        if mc_cfg.get("enabled"):
            mc_file = mc_cfg.get("persona_file", "meta_comm.json")
            mc_data = _load_json(self._resolve(mc_file))
            # Accept either a flat {system_prompt: ...} or a nested {personas: {the_meta_communicator: {...}}}
            if "system_prompt" in mc_data:
                self._meta_comm_prompt = mc_data["system_prompt"]
            else:
                personas = mc_data.get("personas", {})
                first = next(iter(personas.values()), {})
                self._meta_comm_prompt = first.get("system_prompt", "")

        rl_path = os.path.join(self.project_root, "config", "rate_limits.json")
        self._rate_limits = _load_json(rl_path)

    # ── agent validation ───────────────────────────────────────────────────────

    def validate_agents(self) -> list[dict]:
        """
        Resolve and validate all agents listed in experiment.json.
        Returns list of resolved agent dicts on success. Exits on error.
        """
        agent_specs = self._exp.get("agents", [])
        errors  = []
        resolved = []

        for spec in agent_specs:
            name        = spec.get("name", "").strip()
            domain      = spec.get("domain", "").strip().lower()
            persona_file = spec.get("personas_file", self._exp.get("personas_file", "personas.json"))

            if not name or not domain:
                errors.append(f"  ✗ Agent entry missing 'name' or 'domain': {spec}")
                continue

            # Load (and cache) the persona file
            abs_file = self._resolve(persona_file)
            if abs_file not in self._persona_cache:
                if not os.path.exists(abs_file):
                    errors.append(f"  ✗ personas_file not found: {abs_file}  (agent: '{name}')")
                    continue
                raw = _load_json(abs_file)
                # Normalise: accept {domain: {name: persona}} or {personas: {name: persona}} (single-domain file)
                if "personas" in raw and "domain" in raw:
                    # Single-domain file: {domain: "...", personas: {name: {...}}}
                    self._persona_cache[abs_file] = {raw["domain"]: raw["personas"]}
                elif all(isinstance(v, dict) for v in raw.values()):
                    # Multi-domain: {domain_key: {name: persona}, ...}
                    self._persona_cache[abs_file] = raw
                else:
                    errors.append(f"  ✗ Unrecognised personas file format: {abs_file}")
                    continue

            persona_map = self._persona_cache[abs_file]

            domain_data = persona_map.get(domain)
            if domain_data is None:
                errors.append(
                    f"  ✗ Domain '{domain}' not found in {persona_file}.\n"
                    f"    Available: {list(persona_map.keys())}"
                )
                continue

            persona_data = domain_data.get(name)
            if persona_data is None:
                errors.append(
                    f"  ✗ Agent '{name}' not found in domain '{domain}' in {persona_file}.\n"
                    f"    Available: {list(domain_data.keys())}"
                )
                continue

            resolved.append({
                "id":            name,
                "display_name":  persona_data.get("name", name),
                "domain":        domain,
                "system_prompt": persona_data["system_prompt"],
                "lifespan":      persona_data.get("lifespan", "N/A"),
            })

        if errors:
            print(f"\n[CONFIG ERROR] Agent validation failed for: {self.exp_dir}")
            for e in errors:
                print(e)
            print("\nFix experiment.json and/or personas.json, then retry.")
            sys.exit(1)

        return resolved

    # ── group resolution ───────────────────────────────────────────────────────

    def get_translation_groups(self) -> list[list[str]]:
        """
        Returns list of domain-lists defining translation groups.
        Meta-comm translates between groups (not within them).
        Example: [["theoretical_physics"], ["philosopher", "cognitive_science"]]
        """
        mc_cfg = self._exp.get("meta_comm", {})
        groups = mc_cfg.get("groups", [])
        if not groups:
            return []
        return [g.get("domains", []) for g in groups]

    def get_group_label_for_domain(self, domain: str) -> int:
        """Returns the group index (0, 1, ...) that a domain belongs to. -1 if unassigned."""
        for idx, group in enumerate(self.get_translation_groups()):
            if domain.lower() in [d.lower() for d in group]:
                return idx
        return -1

    # ── accessors ─────────────────────────────────────────────────────────────

    @property
    def experiment(self) -> dict:
        return self._exp

    @property
    def problem(self) -> dict:
        return self._problem

    @property
    def judge_config(self) -> dict:
        return self._judge

    @property
    def meta_comm_prompt(self) -> str | None:
        return self._meta_comm_prompt

    def is_meta_comm_enabled(self) -> bool:
        return bool(self._exp.get("meta_comm", {}).get("enabled", False))

    def get_agent_temperature(self, run_index: int) -> float:
        schedule = self._exp.get("temperature_schedule", {}).get("agents", [0.7])
        if isinstance(schedule, list):
            idx = min(run_index, len(schedule) - 1)
            return schedule[idx]
        return float(schedule)

    def get_role_temperature(self, role: str) -> float:
        schedule = self._exp.get("temperature_schedule", {})
        val = schedule.get(role, 0.3)
        if isinstance(val, list):
            return val[0]
        return float(val)

    def get_model(self, role: str) -> dict:
        return self._exp.get("models", {}).get(role, {})

    def get_rate_limits(self, provider: str, model_id: str) -> dict:
        models = self._rate_limits.get(provider, {}).get("models", {})
        if model_id not in models:
            return {"context_window": 32000, "rpm": 10, "tpm": None, "rpd": 100}
        return models[model_id]

    def memory_cfg(self) -> dict:
        return self._exp.get("memory", {
            "embedding_model": "all-MiniLM-L6-v2",
            "retrieval_top_k": 3,
            "context_window_threshold_pct": 0.65,
            "max_reflections_per_round": None,
        })

    def session_cfg(self) -> dict:
        return self._exp.get("session", {"max_rounds": 30})

    def num_runs(self) -> int:
        return self._exp.get("num_runs", 5)

    def experiment_name(self) -> str:
        return self._exp.get("name", os.path.basename(self.exp_dir))

    # ── path helpers ───────────────────────────────────────────────────────────

    def _exp_path(self, filename: str) -> str:
        return os.path.join(self.exp_dir, filename)

    def _resolve(self, rel_path: str) -> str:
        """Resolve a path relative to the experiment directory."""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(self.exp_dir, rel_path))

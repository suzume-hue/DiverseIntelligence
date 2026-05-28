# Creating a New Experiment — Guide

Copy this entire `_template/` directory and rename it.  
Edit the JSON files. Run `python validate.py experiments/your_name`.  
That's it — no code changes needed.

---

## The Files

```
your_experiment/
  experiment.json    ← hyperparameters, agent list, models, temperatures
  personas.json      ← who the agents are (or reference shared personas)
  problem.json       ← what they discuss
  judge.json         ← who evaluates the conversation
  meta_comm.json     ← the translator (only if meta_comm.enabled = true)
```

---

## experiment.json — Field by Field

### `agents`
List every agent you want in this session. Each entry needs:
- `name` — must exactly match a key inside the domain in `personas.json`
- `domain` — must exactly match a top-level key in `personas.json`

```json
"agents": [
  {"name": "richard_feynman",   "domain": "theoretical_physics"},
  {"name": "soren_kierkegaard", "domain": "philosopher"},
  {"name": "william_james",     "domain": "cognitive_science"}
]
```

Add as many agents as you want. There is no hard limit.

You can also point an agent to a **shared persona file**:
```json
{
  "name": "richard_feynman",
  "domain": "theoretical_physics",
  "personas_file": "../../shared/personas/theoretical_physics.json"
}
```

If `personas_file` is not set per-agent, the experiment-level `personas_file` is used (defaults to `personas.json` in this directory).

---

### `meta_comm`

```json
"meta_comm": {
  "enabled": true,
  "persona_file": "meta_comm.json",
  "groups": [
    {"label": "physics",     "domains": ["theoretical_physics"]},
    {"label": "humanities",  "domains": ["philosopher", "cognitive_science"]}
  ]
}
```

- `enabled`: `true` activates the MetaCommunicator translation layer
- `groups`: defines the domain groups. Translation happens **between** groups, not within them
- Agents in the same group talk to each other verbatim
- Agents in different groups receive translated versions of each other's messages
- You can have 2, 3, or more groups — the system handles any number
- Agents not assigned to any group talk to everyone verbatim

**Examples of group configurations:**

Two groups (physics vs humanities):
```json
"groups": [
  {"label": "physics",     "domains": ["theoretical_physics"]},
  {"label": "humanities",  "domains": ["philosopher", "theology"]}
]
```

Three groups (physicists, philosophers, engineers — all bridges):
```json
"groups": [
  {"label": "physics",      "domains": ["theoretical_physics"]},
  {"label": "philosophy",   "domains": ["philosopher"]},
  {"label": "engineering",  "domains": ["systems_engineering"]}
]
```

No meta_comm (everyone talks to everyone directly — control experiment):
```json
"meta_comm": {"enabled": false}
```

---

### `models`

Assign a model to each role:
```json
"models": {
  "agent":     {"provider": "groq",   "model_id": "llama-3.3-70b-versatile", "max_tokens": 1500},
  "meta_comm": {"provider": "google", "model_id": "gemma-4-31b-it",          "max_tokens": 1000},
  "judge":     {"provider": "google", "model_id": "gemini-3.1-flash-lite",   "max_tokens": 400},
  "reflector": {"provider": "groq",   "model_id": "llama-3.3-70b-versatile", "max_tokens": 800}
}
```

Available providers: `groq`, `google`  
Model IDs must match a key in `config/rate_limits.json`.

---

### `temperature_schedule`

```json
"temperature_schedule": {
  "agents":    [0.70, 0.75, 0.80, 0.85, 0.90],
  "meta_comm": 0.35,
  "judge":     0.10,
  "reflector": 0.40
}
```

- `agents`: list of one temperature per run (run 1 gets index 0, run 2 gets index 1, etc.)  
  If the list is shorter than `num_runs`, the last value repeats.  
  Or use a single scalar to apply the same temperature to all runs.
- Other roles: single scalar, applies to all runs.

**Guidelines:**
- Agents: 0.7–0.9 (personality, variety)
- MetaCommunicator: 0.3–0.5 (faithful translation)
- Judge: 0.1–0.2 (consistent evaluation)
- Reflector: 0.3–0.5 (faithful memory compression)

---

### `memory`

```json
"memory": {
  "embedding_model":             "all-MiniLM-L6-v2",
  "retrieval_top_k":             3,
  "context_window_threshold_pct": 0.65,
  "max_reflections_per_round":   null
}
```

- `retrieval_top_k`: how many past reflections each agent sees per round (default 3)
- `max_reflections_per_round`: null = agent decides freely, integer = hard cap
- `context_window_threshold_pct`: fraction of model context at which to trigger memory compression (not yet in use — architecture is already efficient)

---

### `session`

```json
"session": {
  "max_rounds": 30
}
```

Safety cap. The Judge closes the session before this in normal operation.

---

## personas.json — Structure

```json
{
  "domain_name": {
    "persona_id": {
      "name":          "Display Name",
      "lifespan":      "YYYY–YYYY",
      "domain":        "domain_name",
      "system_prompt": "Full character prompt. This defines everything about how this agent thinks, speaks, and engages. Make it rich and specific."
    }
  }
}
```

Or reference a shared persona file directly (see experiment.json `personas_file` field).

---

## problem.json — Structure

```json
{
  "id":            "unique_id",
  "title":         "Short title",
  "core_question": "The central question.",
  "background": {
    "summary":       "Context paragraph(s).",
    "core_tensions": {
      "tension_1": "Description.",
      "tension_2": "Description."
    }
  }
}
```

---

## Using Shared Resources

Instead of duplicating persona files, reference the shared library:

In `experiment.json`:
```json
"agents": [
  {
    "name": "richard_feynman",
    "domain": "theoretical_physics",
    "personas_file": "../../shared/personas/theoretical_physics.json"
  }
]
```

Or set a default for all agents:
```json
"personas_file": "../../shared/personas/theoretical_physics.json"
```

Shared personas are in `shared/personas/`. Shared problems in `shared/problems/`.  
Point `problem_file` at a shared problem:
```json
"problem_file": "../../shared/problems/quantum_gravity_reconciliation.json"
```

---

## Checklist Before Running

- [ ] All agent `name` + `domain` pairs exist in the referenced personas file
- [ ] `problem_file` resolves and has required fields
- [ ] If `meta_comm.enabled = true`, `meta_comm.persona_file` exists
- [ ] Model IDs exist in `config/rate_limits.json`
- [ ] API keys are in `config/api_keys.json`

Run: `python validate.py experiments/your_experiment`

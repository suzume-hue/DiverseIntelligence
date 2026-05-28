# DiverseIntelligence

Multi-agent conversation experiment engine. LLM-powered agents embodying historical intellectual personas discuss unsolved problems in structured rounds.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API keys
# Edit config/api_keys.json — replace placeholder values with real keys

# 3. Validate an experiment
python validate.py experiments/quantum_gravity_mixed

# 4. Run
python run.py experiments/quantum_gravity_mixed

# 5. Optional: Terminal UI
python ui.py
```

---

## Running Experiments

```bash
# Run all pending runs for an experiment
python run.py experiments/quantum_gravity_mixed

# Run or resume a specific run number
python run.py experiments/quantum_gravity_mixed --run 3

# Check status without running
python run.py experiments/quantum_gravity_mixed --status

# Overview of all experiments
python status.py

# Watch mode (refreshes every 10s)
python status.py --watch
```

If a run stops for any reason (quota, crash, Ctrl+C) — re-run the same command. It resumes from exactly where it stopped.

---

## Creating a New Experiment

```bash
cp -r experiments/_template experiments/my_experiment
```

Edit the JSON files in `experiments/my_experiment/`:

| File | What to edit |
|---|---|
| `experiment.json` | Agents, runs, models, temperatures, MetaComm groups |
| `personas.json` | Who the agents are (or reference `shared/personas/`) |
| `problem.json` | What they discuss |
| `judge.json` | Who evaluates (default semiotician works for most cases) |
| `meta_comm.json` | The translator (only needed if MetaComm is enabled) |

Then validate and run:
```bash
python validate.py experiments/my_experiment
python run.py experiments/my_experiment
```

See `experiments/_template/GUIDE.md` for a complete field-by-field explanation.

---

## Experiments Included

| Experiment | Description |
|---|---|
| `quantum_gravity_mixed` | Feynman + Kierkegaard + James — physics meets philosophy/cognitive science, MetaComm enabled |
| `quantum_gravity_control` | Feynman + Einstein + Dirac — physics only, no MetaComm |

---

## Adding API Keys

Edit `config/api_keys.json`. Add as many keys per provider as you have — they are rotated automatically when one hits its daily quota.

```json
{
  "groq":   { "keys": ["gsk_key1", "gsk_key2", "..."] },
  "google": { "keys": ["AIzaSy_key1", "AIzaSy_key2", "..."] }
}
```

When all keys for a provider are exhausted, the system stops cleanly, reports the reset time, and saves its checkpoint so you can continue the next day.

---

## Project Structure

```
DiverseIntelligence/
├── run.py                  ← run any experiment
├── validate.py             ← validate before running
├── status.py               ← all experiments overview
├── ui.py                   ← terminal UI (textual)
│
├── config/
│   ├── api_keys.json       ← your keys (fill in)
│   └── rate_limits.json    ← per-model quotas
│
├── core/                   ← engine (no need to touch)
├── experiments/
│   ├── _template/          ← copy to create new experiments
│   ├── quantum_gravity_mixed/
│   └── quantum_gravity_control/
│
├── shared/
│   ├── personas/           ← reusable persona libraries
│   └── problems/           ← reusable problem definitions
│
├── PROJECT_MAP.json        ← dependency map (for Claude handoff)
├── HANDOFF.md              ← continuation guide (for Claude handoff)
└── SYSTEM.md               ← full architecture documentation
```

---

## Output Data

Session data is written to `experiments/<name>/sessions/run_N/rounds/`.  
Each round is a `.jsonl` file — one JSON event per line:

```python
import json
events = [json.loads(line) for line in open("round_3.jsonl")]

# Get agent messages
for e in events:
    if e["event_type"] == "agent_output":
        output = e["output"]
        if not output["passed"]:
            print(f"{output['agent_id']}: {output['message'][:100]}")

# Get judge decision
judge = next(e for e in events if e["event_type"] == "judge_evaluated")
print(judge["judge"]["decision"], "—", judge["judge"]["observation"])
```

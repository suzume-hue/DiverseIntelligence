# System Architecture
## DiverseIntelligence

---

## 1. What This Is

A general-purpose multi-agent conversation experiment engine. Any number of LLM-powered agents, each embodying a persona, discuss a structured problem across synchronous rounds. The system is entirely config-driven — new experiments require zero code.

The central research capability: testing whether cross-domain collaboration produces qualitatively different discourse than domain-internal groups, and whether a translation layer (the MetaCommunicator) enables genuine intellectual exchange between agents from different knowledge domains.

---

## 2. The MACP Protocol

**MACP (Multi-Agent Communication Protocol)** governs every session.

### Roles

| Role | Description |
|---|---|
| **Agent** | Active participant. Has a persona, private memory, speaks or passes each round. |
| **MetaCommunicator** | Translates messages between domain groups. Audience-aware. Only active when enabled in config. |
| **Hub** | Passive relay. Assembles broadcasts from agent outputs and translations. No voice. |
| **Judge** | External observer. Reads each broadcast, decides CONTINUE or CLOSE. Agents are unaware. |

### Round Flow

```
Round 0  →  Problem delivery (raw to physics group, translated for others)
Round N  →  Each agent: RECEIVE → THINK → SPEAK or PASS
         →  MetaCommunicator translates cross-group messages
         →  Hub assembles per-agent broadcasts
         →  Each agent generates reflections
         →  Judge evaluates: CONTINUE or CLOSE
```

All agents run in sequence per round (not truly parallel — API calls are sequential). The system is designed to be extended to async if needed.

### Session Termination

The Judge closes the session when it detects that meaning is no longer being made: positions restating, agents passing, exchanges cycling without movement. A hard `max_rounds` cap (default 30) prevents runaway loops.

---

## 3. The MetaCommunicator

The MetaCommunicator performs **audience-aware translation** — not simplification to a neutral register, but dynamic bridging built from the receiver's own vocabulary.

For each message from sender X to receiver Y, it:
1. Strips domain vocabulary to structural core
2. Identifies Y's existing conceptual vocabulary (their domain)
3. Finds the structural analogue
4. Builds the bridge explicitly, naming the connection
5. Flags where the analogy breaks down
6. Returns to the original with the new framing intact

**This is called once per sender/receiver pair per round** — each receiver gets a translation tailored specifically to them, not a generic paraphrase.

**Groups** define who needs translation. Agents in the same group talk verbatim. Agents across groups are translated. Any number of groups is supported.

**Round 0** also goes through the MetaCommunicator: agents not in group 0 receive a translated version of the problem statement before the discussion begins.

---

## 4. Agent Memory

Two-layer architecture — no summarisation.

### Layer 1: Reflections (working memory)

After each round, each agent generates structured reflections from what was said. A reflection is a concise semantic insight written in the agent's own voice, created by the actor LLM using the agent's persona prompt.

Reflections are:
- Embedded locally using `all-MiniLM-L6-v2` (~80MB, CPU, no API calls)
- Stored in `sessions/run_N/memory/{agent_id}/reflections/`
- Indexed in `reflection_index.json` with embeddings for fast retrieval

Before each round, the incoming broadcast is embedded and compared via cosine similarity against all existing reflections. Top-K most relevant are retrieved and included in the agent's context — the agent never attends to all prior content, only what is semantically relevant to the current moment.

### Layer 2: Full round archive (researcher memory)

Every round's complete verbatim data — agent outputs, translations, broadcasts, judge evaluations — is preserved in JSONL files. This is never fed back to agents during the session. It is the researcher's data for post-experiment analysis.

### Why No Summarisation

Summarisation is lossy. Reflections replace it by keeping the full archive intact while providing a semantically indexed working memory layer. The agent never sees more tokens than necessary, but nothing is discarded.

---

## 5. The Judge

A structural semiotician and discourse analyst (by default). Evaluates whether meaning is still being made.

**Receives per round:**
- Current broadcast (what was said this round)
- Running observation log (its own prior brief evaluations)
- Agent reflection snapshots (recent reflections, not full transcripts)

**Closes when it detects:**
- Positions restated without development
- Agreement without substance
- Most agents passing silently
- Same exchange cycling multiple rounds without movement

The Judge never sees full conversation history — lean context, consistent evaluation.

---

## 6. Checkpoint and Resumption

Every phase of every round is saved atomically before the next API call:

```
Phase 1: agent_output       → appended to round_N.jsonl
Phase 2: meta_comm_translation → appended (one per sender/receiver pair)
Phase 3: broadcasts_assembled  → appended
Phase 4: reflection_generated  → appended (one per agent)
Phase 5: judge_evaluated       → appended
         session_state.json updated
```

On restart: reads `session_state.json` + `round_N.jsonl` → skips completed events → resumes from first missing phase. At most one API response is ever lost (the one in flight when the crash happened). No round is ever re-run.

**Config conflict detection:** a snapshot of `experiment.json` is saved when each run begins (in `.di_state/`). On resume, if the config has changed, the user is prompted: Override (restart) or Revert (restore snapshot).

---

## 7. API and Rate Management

### Providers

| Provider | Used for | Reset |
|---|---|---|
| Groq | Agents, reflectors | Midnight UTC |
| Google AI Studio | MetaCommunicator, Judge | Midnight Pacific |

Both expose an OpenAI-compatible `/chat/completions` endpoint.

### Key Rotation

Multiple keys per provider are supported. When a key returns 429, the tracker marks it as exhausted and rotates to the next key. If all keys are exhausted, `QuotaExhaustedError` is raised — the system stops cleanly with checkpoint intact and tells the user when to resume.

### No Cross-Provider Fallback

If Google keys are exhausted, the system stops rather than substituting Groq for MetaCommunicator calls. Substituting a different model mid-experiment would contaminate the data.

---

## 8. Experiment Configuration

Each experiment lives in its own directory with five JSON files:

```
experiments/my_experiment/
  experiment.json   ← hyperparameters (runs, models, temperatures, agents, groups)
  personas.json     ← agent character definitions
  problem.json      ← the problem statement
  judge.json        ← judge persona prompt
  meta_comm.json    ← MetaCommunicator persona (when enabled)
```

All paths in `experiment.json` are resolved relative to the experiment directory. Shared resources (`shared/personas/`, `shared/problems/`) are referenced with `../../shared/...` paths.

### Agent Resolution

Each agent spec in `experiment.json` has `name`, `domain`, and optionally `personas_file`. The loader resolves the persona by:
1. Loading the referenced personas file (or the experiment-default `personas.json`)
2. Looking up `[domain][name]` in the nested structure
3. Extracting `system_prompt`, `display_name`, `lifespan`

Domain names must be lowercase and consistent across the config and persona files.

---

## 9. Temperature Scheduling

Agents vary temperature across runs (producing diverse conversation data). Other roles use fixed temperatures appropriate to their function.

Default schedule across 5 runs:
- Run 1: 0.70 · Run 2: 0.75 · Run 3: 0.80 · Run 4: 0.85 · Run 5: 0.90

MetaCommunicator: 0.35 (faithful translation requires consistency)  
Judge: 0.10 (structural evaluation must be deterministic)  
Reflector: 0.40 (faithful memory, slight variation acceptable)

All configurable per experiment.

---

## 10. Terminal UI

Built with Textual. Four tabs per experiment:

- **Overview** — agents, run statuses, progress bar
- **Hyperparameters** — live editing of runs, temperatures, models, MetaComm toggle, memory settings. Changes saved directly to `experiment.json`.
- **Run** — launch control. Choose which run(s), whether to keep UI open.
  - *Headless mode*: UI closes, experiment runs in terminal
  - *Live mode*: UI stays open, logs stream in real time to the Logs tab
- **Live Logs** — colour-coded event stream while experiment runs

---

## 11. Output Data Structure

```
experiments/<name>/
  sessions/
    run_1/
      session_state.json          ← checkpoint (current status, round, agents)
      rounds/
        round_0.jsonl             ← problem delivery events
        round_1.jsonl             ← all phases, one event per line
        round_2.jsonl
        ...
      memory/
        <agent_id>/
          reflections/
            refl_<id>.json        ← individual reflection objects
          reflection_index.json   ← all embeddings + content, indexed
  logs/
    run1_<timestamp>.log
  .di_state/
    run_1_config.json             ← config snapshot for conflict detection
```

### Reading Session Data

```python
import json

def load_round(path):
    return [json.loads(line) for line in open(path)]

events = load_round("sessions/run_1/rounds/round_3.jsonl")

# Agent messages
for e in [e for e in events if e["event_type"] == "agent_output"]:
    o = e["output"]
    status = "PASS" if o["passed"] else f"SPEAK: {o['message'][:100]}"
    print(f"[{o['agent_id']}] {status}")

# MetaCommunicator translations
for e in [e for e in events if e["event_type"] == "meta_comm_translation"]:
    print(f"{e['from']} → {e['for']}")
    print(f"  Original:   {e['original'][:80]}")
    print(f"  Translated: {e['translated'][:80]}")

# Reflections
for e in [e for e in events if e["event_type"] == "reflection_generated"]:
    print(f"[{e['agent_id']}] {len(e['reflections'])} reflection(s)")

# Judge
judge = next(e for e in events if e["event_type"] == "judge_evaluated")
print(f"Judge: {judge['judge']['decision']} — {judge['judge']['observation']}")
```

---

## 12. Extending the System

### New experiment
Copy `_template/`, edit JSON files, validate, run. Zero code.

### New persona
Add to `shared/personas/<domain>.json` or the experiment's `personas.json`. No code.

### New domain group configuration
Edit `meta_comm.groups` in `experiment.json`. The engine handles any number of groups.

### New provider
1. Add base URL to `PROVIDER_BASE_URLS` in `core/src/api/client.py`
2. Add key pool to `config/api_keys.json`
3. Add models to `config/rate_limits.json`
4. Add reset schedule to `RESET_CONFIG` in `core/src/api/rate_tracker.py`

### New model
Add entry to `config/rate_limits.json`. No code change.

### Analysis tooling
All data is in `experiments/*/sessions/run_N/rounds/*.jsonl`. The structure is stable and documented above. Analysis scripts can be written independently of the running engine.

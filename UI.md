# Terminal UI Guide
## DiverseIntelligence

## Starting the UI
pip install -r requirements.txt
python ui.py

## Three Ways to Run

**1. Full UI** — open, configure, launch, watch logs inside the interface.

**2. UI → Headless** — configure visually, then toggle "Keep UI open" OFF before hitting Start. UI closes, experiment runs in terminal.

**3. Pure CLI** — skip the UI entirely:
  python run.py experiments/my_experiment
  python run.py experiments/my_experiment --run 3
  python run.py experiments/my_experiment --status

The engine has zero dependency on the UI. Works on any terminal, over SSH, headless servers.

## Navigation
Arrow keys / Tab — navigate
Enter — select
Escape — go back
Ctrl+Q — quit

## Experiment Detail Tabs

**Overview** — agents, run statuses, MetaComm status

**Hyperparameters** — live editor: runs, temperatures, models, MetaComm toggle, memory settings. Hit Save to write to experiment.json.

**Run** — pick which run, toggle keep-UI-open, hit Start/Resume

**Live Logs** — colour-coded stream (only when UI stays open):
  Green   — session start/complete
  Cyan    — round start
  White   — agent speaks
  Purple  — MetaCommunicator translation
  Pink    — reflection generated
  Yellow  — judge decision
  Red     — error / quota exhausted

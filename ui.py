#!/usr/bin/env python3
"""
ui.py  —  DiverseIntelligence Terminal UI

Usage:
  python ui.py

Controls:
  Arrow keys / Tab  — navigate
  Enter             — select
  Q / Escape        — quit / back
"""

import asyncio
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Textual imports ────────────────────────────────────────────────────────────
try:
    from textual.app        import App, ComposeResult
    from textual.binding    import Binding
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.screen     import Screen
    from textual.widgets    import (
        Button, DataTable, Footer, Header, Input, Label,
        Log, Select, Static, Switch, TabbedContent, TabPane,
    )
    from textual.reactive   import reactive
    from textual import work
    from rich.text          import Text
except ImportError:
    print("\n[ERROR] Textual is not installed.")
    print("Install with:  pip install textual rich\n")
    sys.exit(1)

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# Thread-safe log queue: experiment engine writes here, UI reads it
_log_queue: queue.Queue = queue.Queue()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_experiments() -> list[dict]:
    results = []
    if not os.path.isdir(EXPERIMENTS_DIR):
        return results
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        if name.startswith("_"):
            continue
        path = os.path.join(EXPERIMENTS_DIR, name)
        cfg_path = os.path.join(path, "experiment.json")
        if os.path.isdir(path) and os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                results.append({
                    "dir":       path,
                    "id":        name,
                    "name":      cfg.get("name", name),
                    "num_runs":  cfg.get("num_runs", 5),
                    "meta_comm": cfg.get("meta_comm", {}).get("enabled", False),
                    "agents":    [a["name"] for a in cfg.get("agents", [])],
                    "cfg":       cfg,
                })
            except Exception:
                pass
    return results


def get_run_statuses(exp_dir: str, num_runs: int) -> list[dict]:
    from core.src.utils.checkpoint import session_dir, load_session_state
    statuses = []
    for run_num in range(1, num_runs + 1):
        sd    = session_dir(exp_dir, run_num)
        state = load_session_state(sd)
        if state is None:
            statuses.append({"run": run_num, "status": "pending", "detail": "—"})
        elif state.get("status") == "complete":
            r = state.get("last_completed_round", "?")
            statuses.append({"run": run_num, "status": "complete",
                              "detail": f"{r} rounds"})
        else:
            r = state.get("last_completed_round", -1) + 1
            t = state.get("temperature", "?")
            statuses.append({"run": run_num, "status": "running",
                              "detail": f"round {r} • temp {t}"})
    return statuses


# ══════════════════════════════════════════════════════════════════════════════
# LOG BRIDGE — replaces Logger for UI mode
# ══════════════════════════════════════════════════════════════════════════════

class UILogger:
    """Drop-in Logger replacement that pushes to the shared queue."""

    def __init__(self, log_dir: str = "", session_id: str = ""):
        import logging, os
        Path(log_dir).mkdir(parents=True, exist_ok=True) if log_dir else None
        self._session_id = session_id

    def _push(self, level: str, msg: str):
        _log_queue.put({"level": level, "msg": msg,
                        "ts": datetime.now().strftime("%H:%M:%S")})

    def info(self, msg):    self._push("INFO",  msg)
    def debug(self, msg):   self._push("DEBUG", msg)
    def warning(self, msg): self._push("WARN",  msg)
    def error(self, msg):   self._push("ERROR", msg)

    def session_start(self, sid, exp, run):
        self._push("START", f"▶ SESSION  {sid}  Exp:{exp}  Run:{run}")
    def session_resume(self, sid, from_round):
        self._push("START", f"⟳ RESUME   {sid}  from round {from_round}")
    def round_start(self, n):
        self._push("ROUND", f"── Round {n} " + "─" * 40)
    def agent_output(self, aid, otype, preview=""):
        sym = "💬" if otype == "SPEAK" else "·"
        snippet = f" {preview[:70]}…" if preview else ""
        self._push("AGENT", f"  {sym} [{aid}]{snippet}")
    def meta_comm(self, receiver, preview=""):
        self._push("META",  f"  ⟷ → {receiver}: {preview[:60]}…")
    def reflection(self, aid, count):
        self._push("MEM",   f"  💭 [{aid}]  {count} reflection(s)")
    def judge_decision(self, n, decision, obs):
        sym = "✓" if decision == "CONTINUE" else "✗ CLOSE"
        self._push("JUDGE", f"  Judge [{sym}]  {obs[:100]}")
    def quota_exhausted(self, provider, reset_info):
        self._push("ERROR", f"⚠ QUOTA EXHAUSTED [{provider}]  {reset_info}")
    def session_complete(self, sid, total_rounds):
        self._push("DONE",  f"✓ COMPLETE  {sid}  ({total_rounds} rounds)")


# ══════════════════════════════════════════════════════════════════════════════
# SCREENS
# ══════════════════════════════════════════════════════════════════════════════

# ── CSS ────────────────────────────────────────────────────────────────────────
APP_CSS = """
Screen {
    background: #0d0d0d;
}

Header {
    background: #1a1a2e;
    color: #00d4ff;
    text-style: bold;
}

Footer {
    background: #1a1a2e;
    color: #888888;
}

.panel {
    border: solid #2a2a4a;
    padding: 0 1;
    margin: 0 1;
}

.panel-title {
    color: #00d4ff;
    text-style: bold;
    padding: 0 1;
}

.experiment-card {
    border: solid #2a2a4a;
    padding: 1 2;
    margin: 0 1 1 1;
    background: #111122;
}

.experiment-card:hover {
    border: solid #00d4ff;
    background: #1a1a3a;
}

.experiment-name {
    color: #ffffff;
    text-style: bold;
}

.experiment-meta {
    color: #888888;
}

.status-complete { color: #00ff88; }
.status-running  { color: #ffaa00; }
.status-pending  { color: #555555; }

.run-table {
    height: 12;
    border: solid #2a2a4a;
    margin: 0 1;
}

Button {
    background: #1a1a3a;
    border: solid #00d4ff;
    color: #00d4ff;
    margin: 0 1;
}

Button:hover {
    background: #00d4ff;
    color: #000000;
}

Button.-primary {
    background: #00aa44;
    border: solid #00ff88;
    color: #ffffff;
}

Button.-primary:hover {
    background: #00ff88;
    color: #000000;
}

Button.-danger {
    background: #3a1a1a;
    border: solid #ff4444;
    color: #ff4444;
}

Input {
    background: #111122;
    border: solid #2a2a4a;
    color: #ffffff;
}

Input:focus {
    border: solid #00d4ff;
}

Select {
    background: #111122;
    border: solid #2a2a4a;
}

.log-panel {
    background: #050510;
    border: solid #2a2a4a;
    height: 1fr;
}

.log-level-START { color: #00ff88; text-style: bold; }
.log-level-ROUND { color: #00d4ff; text-style: bold; }
.log-level-AGENT { color: #ffffff; }
.log-level-META  { color: #aa88ff; }
.log-level-MEM   { color: #ff88aa; }
.log-level-JUDGE { color: #ffaa00; text-style: bold; }
.log-level-DONE  { color: #00ff88; text-style: bold; }
.log-level-ERROR { color: #ff4444; text-style: bold; }
.log-level-WARN  { color: #ffaa00; }
.log-level-INFO  { color: #888888; }
.log-level-DEBUG { color: #444444; }

Switch {
    margin: 0 1;
}

Label {
    padding: 0 1;
}

.section-header {
    color: #00d4ff;
    text-style: bold;
    margin: 1 0 0 1;
}

.value-label {
    color: #aaaaaa;
}

.highlight {
    color: #00d4ff;
}
"""


# ── Home Screen ────────────────────────────────────────────────────────────────
class HomeScreen(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(" DiverseIntelligence", classes="section-header")
        yield Static("  Multi-agent conversation experiments", classes="value-label")
        with ScrollableContainer():
            yield ExperimentList(id="exp-list")
        yield Footer()

    def on_mount(self):
        self.title = "DiverseIntelligence"
        self.sub_title = "Experiment Runner"

    def action_refresh(self):
        self.query_one("#exp-list", ExperimentList).refresh_list()

    def on_experiment_selected(self, event: "ExperimentSelected"):
        self.app.push_screen(ExperimentScreen(event.experiment))


class ExperimentSelected(App.message_class if hasattr(App, "message_class") else object):
    pass


# Message class
from textual.message import Message as _Message
class ExperimentSelected(_Message):
    def __init__(self, experiment: dict):
        super().__init__()
        self.experiment = experiment


# ── Experiment List Widget ─────────────────────────────────────────────────────
class ExperimentList(Static):
    def compose(self) -> ComposeResult:
        experiments = discover_experiments()
        if not experiments:
            yield Static("  No experiments found in experiments/\n"
                         "  Create one by copying experiments/_template/",
                         classes="value-label")
            return
        for exp in experiments:
            yield ExperimentCard(exp)

    def refresh_list(self):
        self.remove_children()
        self.mount(*self._build_children())

    def _build_children(self):
        experiments = discover_experiments()
        return [ExperimentCard(exp) for exp in experiments] if experiments else [
            Static("  No experiments found.", classes="value-label")
        ]


class ExperimentCard(Static):
    def __init__(self, experiment: dict):
        super().__init__()
        self.experiment = experiment

    def compose(self) -> ComposeResult:
        exp = self.experiment
        statuses  = get_run_statuses(exp["dir"], exp["num_runs"])
        complete  = sum(1 for s in statuses if s["status"] == "complete")
        in_prog   = sum(1 for s in statuses if s["status"] == "running")
        mc_label  = "MetaComm ON" if exp["meta_comm"] else "no MetaComm"
        agents    = ", ".join(exp["agents"][:4])
        if len(exp["agents"]) > 4:
            agents += f" +{len(exp['agents'])-4}"

        with Container(classes="experiment-card"):
            yield Label(f"  {exp['name']}", classes="experiment-name")
            yield Label(f"  {mc_label}  ·  agents: {agents}", classes="experiment-meta")
            bar = _progress_bar(complete, in_prog, exp["num_runs"])
            yield Label(f"  {bar}  {complete}/{exp['num_runs']} runs complete",
                        classes="experiment-meta")
            with Horizontal():
                yield Button("▶ Open", id=f"open-{exp['id']}", variant="default")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id and event.button.id.startswith("open-"):
            self.post_message(ExperimentSelected(self.experiment))


# ── Experiment Detail Screen ───────────────────────────────────────────────────
class ExperimentScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q",      "quit", "Quit"),
    ]

    def __init__(self, experiment: dict):
        super().__init__()
        self.experiment = experiment
        self._running   = False

    def compose(self) -> ComposeResult:
        exp = self.experiment
        yield Header(show_clock=True)

        with TabbedContent():
            with TabPane("Overview", id="tab-overview"):
                yield self._build_overview()

            with TabPane("Hyperparameters", id="tab-hyper"):
                yield self._build_hyperparams()

            with TabPane("Run", id="tab-run"):
                yield self._build_run_tab()

            with TabPane("Live Logs", id="tab-logs"):
                yield Log(id="live-log", classes="log-panel", highlight=True)

        yield Footer()

    def on_mount(self):
        self.title    = "DiverseIntelligence"
        self.sub_title = self.experiment["name"]

    def action_back(self):
        self.app.pop_screen()

    # ── Tab builders ────────────────────────────────────────────────────────────

    def _build_overview(self) -> Static:
        exp = self.experiment
        cfg = exp["cfg"]
        statuses = get_run_statuses(exp["dir"], exp["num_runs"])

        lines = [
            f"  Name:      {exp['name']}",
            f"  Directory: {exp['dir'].replace(PROJECT_ROOT+'/', '')}",
            f"  Runs:      {exp['num_runs']}",
            f"  MetaComm:  {'enabled' if exp['meta_comm'] else 'disabled'}",
            f"  Agents ({len(exp['agents'])}):",
        ]
        for a in cfg.get("agents", []):
            lines.append(f"    · {a['name']}  ({a['domain']})")

        lines.append("")
        lines.append("  Run Status:")
        for s in statuses:
            sym = {"complete": "✓", "running": "⟳", "pending": "·"}[s["status"]]
            lines.append(f"    {sym} Run {s['run']}  {s['detail']}")

        return Static("\n".join(lines))

    def _build_hyperparams(self) -> ScrollableContainer:
        exp = self.experiment
        cfg = exp["cfg"]
        mc  = cfg.get("meta_comm", {})
        mem = cfg.get("memory", {})
        ses = cfg.get("session", {})
        sched = cfg.get("temperature_schedule", {})

        container = ScrollableContainer()

        widgets = [
            Static("  ── Session ─────────────────────────────────", classes="section-header"),
            Horizontal(
                Label("  Number of runs:", classes="value-label"),
                Input(str(cfg.get("num_runs", 5)), id="hp-num-runs", placeholder="5"),
            ),
            Horizontal(
                Label("  Max rounds per run:", classes="value-label"),
                Input(str(ses.get("max_rounds", 30)), id="hp-max-rounds", placeholder="30"),
            ),
            Static("  ── MetaCommunicator ─────────────────────────", classes="section-header"),
            Horizontal(
                Label("  Enabled:", classes="value-label"),
                Switch(mc.get("enabled", False), id="hp-meta-comm"),
            ),
            Static("  ── Memory ───────────────────────────────────", classes="section-header"),
            Horizontal(
                Label("  Reflections retrieved (top-K):", classes="value-label"),
                Input(str(mem.get("retrieval_top_k", 3)), id="hp-top-k", placeholder="3"),
            ),
            Horizontal(
                Label("  Max reflections per round (blank=unlimited):", classes="value-label"),
                Input(str(mem.get("max_reflections_per_round") or ""), id="hp-max-refl",
                      placeholder="unlimited"),
            ),
            Static("  ── Agent Temperature Schedule ───────────────", classes="section-header"),
            Static("  One value per run (comma-separated), or a single value for all runs:",
                   classes="value-label"),
        ]

        agent_t = sched.get("agents", [0.7])
        if isinstance(agent_t, list):
            agent_t_str = ", ".join(str(t) for t in agent_t)
        else:
            agent_t_str = str(agent_t)

        widgets += [
            Input(agent_t_str, id="hp-temp-agents", placeholder="0.70, 0.75, 0.80, 0.85, 0.90"),
            Static("  Fixed temperatures for other roles:", classes="value-label"),
            Horizontal(
                Label("  MetaComm:", classes="value-label"),
                Input(str(sched.get("meta_comm", 0.35)), id="hp-temp-mc",   placeholder="0.35"),
                Label("  Judge:", classes="value-label"),
                Input(str(sched.get("judge",     0.10)), id="hp-temp-judge", placeholder="0.10"),
                Label("  Reflector:", classes="value-label"),
                Input(str(sched.get("reflector", 0.40)), id="hp-temp-refl",  placeholder="0.40"),
            ),
            Static("  ── Models ───────────────────────────────────", classes="section-header"),
        ]

        models = cfg.get("models", {})
        for role, mcfg in models.items():
            widgets.append(
                Horizontal(
                    Label(f"  {role}:", classes="value-label"),
                    Input(mcfg.get("provider", ""), id=f"hp-model-prov-{role}",
                          placeholder="groq / google"),
                    Input(mcfg.get("model_id", ""), id=f"hp-model-id-{role}",
                          placeholder="model id"),
                )
            )

        widgets += [
            Static(""),
            Horizontal(
                Button("💾 Save Changes", id="btn-save-hyper", variant="success"),
                Button("↺ Reset",        id="btn-reset-hyper"),
            ),
            Static(""),
        ]

        return ScrollableContainer(*widgets)

    def _build_run_tab(self) -> Container:
        statuses = get_run_statuses(self.experiment["dir"], self.experiment["num_runs"])
        pending  = [s for s in statuses if s["status"] != "complete"]
        num_runs = self.experiment["num_runs"]

        options = [(f"Run {i}", str(i)) for i in range(1, num_runs + 1)]
        options.insert(0, ("All pending runs", "all"))

        return Container(
            Static("  ── Launch Experiment ────────────────────────", classes="section-header"),
            Static(""),
            Horizontal(
                Label("  Which run?", classes="value-label"),
                Select(options, id="select-run", value="all"),
            ),
            Horizontal(
                Label("  Keep UI open during run?", classes="value-label"),
                Switch(True, id="switch-keep-ui"),
            ),
            Static("  (If off, the UI closes and the experiment runs in the terminal.)",
                   classes="value-label"),
            Static(""),
            Horizontal(
                Button("▶ Start / Resume", id="btn-start", variant="primary"),
            ),
            Static(""),
            Static(f"  {len(pending)} run(s) pending  ·  "
                   f"{num_runs - len(pending)} complete",
                   classes="value-label"),
        )

    # ── Button handlers ────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed):
        btn = event.button.id

        if btn == "btn-save-hyper":
            self._save_hyperparams()

        elif btn == "btn-reset-hyper":
            self.notify("Reload the screen to reset (changes not yet saved are discarded).")

        elif btn == "btn-start":
            keep_ui = self.query_one("#switch-keep-ui", Switch).value
            run_val = str(self.query_one("#select-run", Select).value)

            if not keep_ui:
                self._launch_headless(run_val)
            else:
                self._launch_with_ui(run_val)

    def _save_hyperparams(self):
        cfg_path = os.path.join(self.experiment["dir"], "experiment.json")
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)

            # Session
            cfg["num_runs"]                          = int(self.query_one("#hp-num-runs", Input).value or 5)
            cfg.setdefault("session", {})["max_rounds"] = int(self.query_one("#hp-max-rounds", Input).value or 30)

            # MetaComm toggle
            cfg.setdefault("meta_comm", {})["enabled"] = self.query_one("#hp-meta-comm", Switch).value

            # Memory
            cfg.setdefault("memory", {})["retrieval_top_k"] = int(self.query_one("#hp-top-k", Input).value or 3)
            max_r = self.query_one("#hp-max-refl", Input).value.strip()
            cfg["memory"]["max_reflections_per_round"] = int(max_r) if max_r else None

            # Temperature
            agents_raw = self.query_one("#hp-temp-agents", Input).value.strip()
            if "," in agents_raw:
                cfg.setdefault("temperature_schedule", {})["agents"] = [
                    float(x.strip()) for x in agents_raw.split(",") if x.strip()
                ]
            else:
                cfg.setdefault("temperature_schedule", {})["agents"] = float(agents_raw or 0.7)

            cfg["temperature_schedule"]["meta_comm"]  = float(self.query_one("#hp-temp-mc",    Input).value or 0.35)
            cfg["temperature_schedule"]["judge"]      = float(self.query_one("#hp-temp-judge", Input).value or 0.10)
            cfg["temperature_schedule"]["reflector"]  = float(self.query_one("#hp-temp-refl",  Input).value or 0.40)

            # Models
            for role in list(cfg.get("models", {}).keys()):
                prov = self.query_one(f"#hp-model-prov-{role}", Input).value.strip()
                mid  = self.query_one(f"#hp-model-id-{role}",   Input).value.strip()
                if prov:
                    cfg["models"][role]["provider"] = prov
                if mid:
                    cfg["models"][role]["model_id"] = mid

            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)

            self.experiment["cfg"]      = cfg
            self.experiment["num_runs"] = cfg["num_runs"]
            self.experiment["meta_comm"] = cfg.get("meta_comm", {}).get("enabled", False)
            self.notify("✓ Hyperparameters saved to experiment.json")

        except Exception as e:
            self.notify(f"Save failed: {e}", severity="error")

    def _launch_headless(self, run_val: str):
        """Exit UI and run experiment in terminal."""
        exp_dir = self.experiment["dir"]
        rel_dir = exp_dir.replace(PROJECT_ROOT + "/", "")
        run_arg = f"--run {run_val}" if run_val != "all" else ""
        self.app.exit(f"python run.py {rel_dir} {run_arg}".strip())

    def _launch_with_ui(self, run_val: str):
        """Run in background thread; stream logs to Live Logs tab."""
        if self._running:
            self.notify("Already running.", severity="warning")
            return
        self._running = True
        self.notify("▶ Experiment started — watch Live Logs tab")
        self.run_worker(_run_experiment_async(
            self.experiment["dir"], run_val, PROJECT_ROOT
        ), exclusive=True)

    @work(thread=True)
    def _poll_log_queue(self):
        """Background thread: drain _log_queue into the Log widget."""
        log_widget = self.query_one("#live-log", Log)
        while True:
            try:
                while True:
                    entry = _log_queue.get_nowait()
                    level = entry.get("level", "INFO")
                    ts    = entry.get("ts", "")
                    msg   = entry.get("msg", "")
                    line  = f"{ts}  {msg}"
                    self.call_from_thread(log_widget.write_line, line)
            except queue.Empty:
                pass
            time.sleep(0.1)

    def on_mount(self):
        self.title    = "DiverseIntelligence"
        self.sub_title = self.experiment["name"]
        self._poll_log_queue()


# ── Async experiment runner ────────────────────────────────────────────────────

async def _run_experiment_async(exp_dir: str, run_val: str, project_root: str):
    """Run experiment in executor so the event loop stays free."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _run_experiment_sync, exp_dir, run_val, project_root
    )


def _run_experiment_sync(exp_dir: str, run_val: str, project_root: str):
    """Runs in a thread. Uses UILogger so output goes to the log queue."""
    import importlib
    import core.src.utils.logger as logger_module

    # Monkey-patch Logger in the module so session_runner uses UILogger
    original_logger = logger_module.Logger
    logger_module.Logger = UILogger

    try:
        from core.src.bootstrap      import bootstrap
        from core.src.session_runner import run_session
        from core.src.utils.checkpoint import session_dir, load_session_state
        from core.src.api.rate_tracker import QuotaExhaustedError

        loader, client = bootstrap(exp_dir, project_root)
        num_runs        = loader.num_runs()

        runs_to_do = []
        if run_val == "all":
            for n in range(1, num_runs + 1):
                sd    = session_dir(exp_dir, n)
                state = load_session_state(sd)
                if not (state and state.get("status") == "complete"):
                    runs_to_do.append(n)
        else:
            runs_to_do = [int(run_val)]

        for run_num in runs_to_do:
            try:
                run_session(exp_dir, run_num, project_root, loader, client)
            except QuotaExhaustedError as e:
                _log_queue.put({"level": "ERROR", "msg": f"QUOTA EXHAUSTED: {e}",
                                "ts": datetime.now().strftime("%H:%M:%S")})
                break
    finally:
        logger_module.Logger = original_logger


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

class DiverseIntelligenceApp(App):
    CSS = APP_CSS
    TITLE = "DiverseIntelligence"
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def on_mount(self):
        self.push_screen(HomeScreen())

    def on_experiment_selected(self, event: ExperimentSelected):
        self.push_screen(ExperimentScreen(event.experiment))

    def on_exit(self, result=None):
        # If headless launch, print the command to run
        if isinstance(result, str):
            print(f"\nRun this command:\n  {result}\n")


# ── helpers ────────────────────────────────────────────────────────────────────

def _progress_bar(complete, in_prog, total, width=16):
    done_w = int(width * complete / total) if total else 0
    prog_w = int(width * in_prog  / total) if total else 0
    rest_w = width - done_w - prog_w
    return "[" + "█" * done_w + "▒" * prog_w + "·" * rest_w + "]"


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = DiverseIntelligenceApp()
    result = app.run()
    # Headless launch: UI printed a command to run
    if result and isinstance(result, str):
        os.system(result)

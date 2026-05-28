"""
core/src/session_runner.py
Orchestrates one full MACP session for any experiment directory.

FIX (2026-05-10): Stage 2 — Initialises BeliefStore instances alongside
MemoryStore instances when agent_architecture.use_beliefs is True.
Reconstructs belief state on resume from persisted belief index files.
Passes belief_stores into run_round_n.

FIX (2026-05-10): Stage 3 — Initialises ArgumentGraph when
agent_architecture.use_argument_graph is True. Passes argument_graph
into run_round_n. Graph is persisted automatically by ArgumentGraph._save().
"""

import os

from core.src.utils.checkpoint import (
    session_dir, load_session_state, init_session_state,
    mark_session_complete, save_config_snapshot,
    check_config_conflict, prompt_config_conflict,
    load_events, round_file,
)
from core.src.utils.logger        import Logger
from core.src.utils.config_loader import ConfigLoader
from core.src.memory.store        import MemoryStore
from core.src.protocol.round_runner import run_round_0, run_round_n


def run_session(
    experiment_dir: str,
    run_num:        int,
    project_root:   str,
    loader:         ConfigLoader,
    client,
) -> None:
    exp_name   = os.path.basename(experiment_dir)
    cfg_path   = os.path.join(experiment_dir, "experiment.json")
    sess_dir   = session_dir(experiment_dir, run_num)

    # ── Config conflict check ──────────────────────────────────────────────────
    state     = load_session_state(sess_dir)
    is_resume = state is not None and state.get("status") == "in_progress"

    if is_resume:
        if check_config_conflict(experiment_dir, run_num, cfg_path):
            prompt_config_conflict(experiment_dir, run_num, cfg_path, sess_dir)
            loader.load_all()
            state     = load_session_state(sess_dir)
            is_resume = state is not None and state.get("status") == "in_progress"
    else:
        save_config_snapshot(experiment_dir, run_num, cfg_path)

    if state and state.get("status") == "complete":
        print(f"[{exp_name} run {run_num}] Already complete — skipping.")
        return

    # ── Resolve config ─────────────────────────────────────────────────────────
    run_index        = run_num - 1
    agents           = loader.validate_agents()
    problem          = loader.problem
    judge_cfg        = loader.judge_config
    memory_cfg       = loader.memory_cfg()
    session_cfg      = loader.session_cfg()
    use_meta_comm    = loader.is_meta_comm_enabled()
    trans_groups     = loader.get_translation_groups()
    meta_comm_prompt = loader.meta_comm_prompt

    agent_temp     = loader.get_agent_temperature(run_index)
    meta_temp      = loader.get_role_temperature("meta_comm")
    judge_temp     = loader.get_role_temperature("judge")
    reflector_temp = loader.get_role_temperature("reflector")

    agent_registry = {a["id"]: a for a in agents}
    agent_ids      = list(agent_registry.keys())

    # Architecture flags — all default False so existing experiments are unaffected
    arch               = loader.experiment.get("agent_architecture", {})
    use_beliefs        = bool(arch.get("use_beliefs",        False))
    use_argument_graph = bool(arch.get("use_argument_graph", False))

    # Extract the core question from the problem for drift-correction anchor.
    problem_core_question: str | None = (
        problem.get("core_question")
        or problem.get("Core Question")
        or None
    )

    # ── Logger ─────────────────────────────────────────────────────────────────
    log_dir = os.path.join(experiment_dir, "logs")
    log     = Logger(log_dir, f"run{run_num}")

    # ── Memory stores ──────────────────────────────────────────────────────────
    stores = {aid: MemoryStore(sess_dir, aid) for aid in agent_ids}

    # ── Belief stores (Stage 2) ────────────────────────────────────────────────
    # Initialised for all agents when use_beliefs=True.
    # BeliefStore handles its own directory creation and is safe to re-init on
    # resume — it reads from persisted JSON files, not from in-memory state.
    belief_stores = None
    if use_beliefs:
        from core.src.memory.belief_store import BeliefStore
        belief_stores = {aid: BeliefStore(sess_dir, aid) for aid in agent_ids}

    # ── Argument graph (Stage 3) ───────────────────────────────────────────────
    # ArgumentGraph is a single session-level object. On resume it loads the
    # existing graph from argument_graph.json automatically.
    argument_graph = None
    if use_argument_graph:
        from core.src.protocol.argument_graph import ArgumentGraph
        argument_graph = ArgumentGraph(sess_dir)

    # ── Agent last messages ────────────────────────────────────────────────────
    agent_last_messages: dict[str, str] = {}

    # ── Init or resume ─────────────────────────────────────────────────────────
    if not is_resume:
        state = init_session_state(
            sess_dir=    sess_dir,
            experiment=  exp_name,
            run_num=     run_num,
            agent_ids=   agent_ids,
            problem_id=  problem.get("id", "unknown"),
            temperature= agent_temp,
        )
        log.session_start(f"{exp_name}_run{run_num}", exp_name, run_num)
    else:
        log.session_resume(f"{exp_name}_run{run_num}", state["last_completed_round"])

        # Reconstruct agent_last_messages from completed round JSONL files.
        last_done_round = state.get("last_completed_round", -1)
        for r in range(1, last_done_round + 1):
            try:
                r_events = load_events(round_file(sess_dir, r))
                for ev in r_events:
                    if ev.get("event_type") == "agent_output":
                        aid = ev.get("agent_id")
                        msg = ev.get("output", {}).get("message", "")
                        if aid and msg:
                            agent_last_messages[aid] = msg
            except FileNotFoundError:
                pass
        # Belief stores on resume: BeliefStore re-reads from disk above.
        # ArgumentGraph on resume: ArgumentGraph re-reads from disk above.

    last_done  = state.get("last_completed_round", -1)
    max_rounds = session_cfg.get("max_rounds", 30)
    observation_log: list[dict] = []

    # ── Round 0 ────────────────────────────────────────────────────────────────
    if last_done < 0:
        log.round_start(0)
        broadcasts = run_round_0(
            sess_dir=           sess_dir,
            problem=            problem,
            agent_registry=     agent_registry,
            meta_comm_prompt=   meta_comm_prompt,
            client=             client,
            meta_comm_cfg=      loader.get_model("meta_comm"),
            meta_comm_temp=     meta_temp,
            translation_groups= trans_groups,
            use_meta_comm=      use_meta_comm,
        )
    else:
        r0_events = load_events(round_file(sess_dir, 0))
        broadcasts = {
            ev["agent_id"]: ev["broadcast"]
            for ev in r0_events if ev.get("event_type") == "round0_delivery"
        }

    # ── Discussion rounds ───────────────────────────────────────────────────────
    round_num = max(1, last_done + 1)

    while round_num <= max_rounds:
        log.round_start(round_num)

        result = run_round_n(
            round_num=              round_num,
            sess_dir=               sess_dir,
            broadcasts_in=          broadcasts,
            agent_registry=         agent_registry,
            stores=                 stores,
            meta_comm_prompt=       meta_comm_prompt,
            judge_config=           judge_cfg,
            observation_log=        observation_log,
            client=                 client,
            exp_cfg=                loader.experiment,
            use_meta_comm=          use_meta_comm,
            translation_groups=     trans_groups,
            agent_temp=             agent_temp,
            meta_comm_temp=         meta_temp,
            judge_temp=             judge_temp,
            reflector_temp=         reflector_temp,
            memory_cfg=             memory_cfg,
            agent_last_messages=    agent_last_messages,
            problem_core_question=  problem_core_question,
            belief_stores=          belief_stores,
            argument_graph=         argument_graph,
        )

        for o in result["agent_outputs"]:
            log.agent_output(
                o["agent_id"],
                "PASS" if o["passed"] else "SPEAK",
                (o.get("message", "") or "")[:80],
            )
            aid = o["agent_id"]
            msg = o.get("message", "")
            if msg:
                agent_last_messages[aid] = msg

        judge = result["judge"]
        log.judge_decision(round_num, judge["decision"], judge["observation"])
        observation_log.append({
            "round":       round_num,
            "observation": judge["observation"],
            "decision":    judge["decision"],
        })

        broadcasts = result["broadcasts_out"]

        if judge["decision"] == "CLOSE":
            break

        round_num += 1

    mark_session_complete(sess_dir, round_num)
    log.session_complete(f"{exp_name}_run{run_num}", round_num)

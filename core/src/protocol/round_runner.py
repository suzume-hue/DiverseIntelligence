"""
core/src/protocol/round_runner.py
Executes one MACP round phase by phase.

FIX (2026-05-09): Passes prior judge observation and each agent's last
message into agent turns to create belief-update pressure.

FIX (2026-05-09): round_0 now uses META_COMM_TASK_TEMPLATE (same as live
rounds) so _extract_translation's <translation>-tag path fires correctly.

FIX (2026-05-09): Passes problem_core_question into run_agent_turn so
agents have a private self-check anchor and can detect + correct drift
from the original problem without external instruction.

FIX (2026-05-10): Stage 1 — Critic pass. When exp_cfg has
  {"agent_architecture": {"use_critic": true}}
the critic pass fires in run_agent_turn between the draft LLM call and
the final XML parse. Critic results are persisted in the agent_output event.

FIX (2026-05-10): Stage 2 — Belief tracking. When exp_cfg has
  {"agent_architecture": {"use_beliefs": true}}
BeliefStore instances are initialised per agent and passed into run_agent_turn
(for context injection) and generate_belief_revision (Phase 4b, after
reflections). Belief revision events are persisted to the round JSONL.

FIX (2026-05-10): Stage 3 — Argument graph. When exp_cfg has
  {"agent_architecture": {"use_argument_graph": true}}
ArgumentGraph is updated after agent outputs (Phase 1b). The judge receives
the graph summary as additional context in its evaluation prompt.

All three flags default to False — existing runs are completely unaffected.
"""

from core.src.agents.agent import run_agent_turn
from core.src.agents.meta_communicator import (
    META_COMM_TASK_TEMPLATE,
    _extract_translation,
    translate_message,
)
from core.src.memory.reflector import generate_reflections
from core.src.memory.store import MemoryStore
from core.src.protocol.hub import assemble_broadcasts, build_round0_broadcast
from core.src.protocol.judge import evaluate_round
from core.src.utils.checkpoint import (
    append_event,
    events_by_type,
    round_file,
    update_session_state,
)


def run_round_0(
    sess_dir: str,
    problem: dict,
    agent_registry: dict,
    meta_comm_prompt: str | None,
    client,
    meta_comm_cfg: dict,
    meta_comm_temp: float,
    translation_groups: list[list[str]],
    use_meta_comm: bool,
) -> dict:
    """
    Deliver the problem to all agents.
    Group-0 agents get the raw problem.
    All other agents get a MetaCommunicator translation.
    Returns: {agent_id: broadcast_string}
    """
    r_file = round_file(sess_dir, 0)
    done = events_by_type(r_file)
    result = {}

    domain_to_group: dict[str, int] = {}
    for idx, group in enumerate(translation_groups):
        for domain in group:
            domain_to_group[domain.lower()] = idx

    for agent_id, agent in agent_registry.items():
        existing = [
            e for e in done.get("round0_delivery", []) if e.get("agent_id") == agent_id
        ]
        if existing:
            result[agent_id] = existing[0]["broadcast"]
            continue

        agent_group = domain_to_group.get(agent["domain"].lower(), 0)

        translated_intro = None
        if use_meta_comm and meta_comm_prompt and agent_group != 0:
            from core.src.protocol.hub import _format_raw_problem

            raw_problem_text = _format_raw_problem(problem)

            task = META_COMM_TASK_TEMPLATE.format(
                sender_name="the problem statement",
                sender_domain="Physics",
                receiver_name=agent["display_name"],
                receiver_domain=agent["domain"],
                message=raw_problem_text,
            )

            raw_translation = client.chat(
                provider=meta_comm_cfg["provider"],
                model_id=meta_comm_cfg["model_id"],
                messages=[{"role": "user", "content": task}],
                temperature=meta_comm_temp,
                max_tokens=meta_comm_cfg.get("max_tokens", 1000),
                system=meta_comm_prompt,
            )

            translated_intro = _extract_translation(raw_translation)

        broadcast = build_round0_broadcast(
            problem, agent_id, agent_group, translated_intro
        )

        append_event(
            r_file,
            {
                "event_type": "round0_delivery",
                "agent_id": agent_id,
                "agent_group": agent_group,
                "broadcast": broadcast,
            },
        )
        result[agent_id] = broadcast

    return result


def run_round_n(
    round_num: int,
    sess_dir: str,
    broadcasts_in: dict,
    agent_registry: dict,
    stores: dict,
    meta_comm_prompt: str | None,
    judge_config: dict,
    observation_log: list,
    client,
    exp_cfg: dict,
    use_meta_comm: bool,
    translation_groups: list[list[str]],
    agent_temp: float,
    meta_comm_temp: float,
    judge_temp: float,
    reflector_temp: float,
    memory_cfg: dict,
    agent_last_messages: dict | None = None,
    problem_core_question: str | None = None,
    # Stage 2: belief stores (populated by session_runner when use_beliefs=True)
    belief_stores: dict | None = None,
    # Stage 3: argument graph (populated by session_runner when use_argument_graph=True)
    argument_graph=None,
) -> dict:
    """
    Execute one full round. Returns:
      {agent_outputs, translations, broadcasts_out, judge}

    New optional params:
      belief_stores    — dict[agent_id -> BeliefStore], passed through to agent turns
                         and belief revision phase.
      argument_graph   — ArgumentGraph instance, updated after agent outputs and
                         queried by the judge.
    """
    r_file = round_file(sess_dir, round_num)
    done = events_by_type(r_file)
    models = exp_cfg.get("models", {})

    if agent_last_messages is None:
        agent_last_messages = {}

    # Read architecture flags from experiment config
    arch = exp_cfg.get("agent_architecture", {})
    use_critic = bool(arch.get("use_critic", False))
    use_beliefs = bool(arch.get("use_beliefs", False)) and belief_stores is not None
    use_argument_graph = (
        bool(arch.get("use_argument_graph", False)) and argument_graph is not None
    )

    # Prior judge observation for belief-update pressure
    prior_judge_obs = observation_log[-1]["observation"] if observation_log else None

    domain_to_group: dict[str, int] = {}
    for idx, group in enumerate(translation_groups):
        for domain in group:
            domain_to_group[domain.lower()] = idx

    # ── Phase 1: Agent outputs ─────────────────────────────────────────────────
    agent_outputs = []
    for agent_id, agent in agent_registry.items():
        existing = [
            e for e in done.get("agent_output", []) if e.get("agent_id") == agent_id
        ]
        if existing:
            agent_outputs.append(existing[0]["output"])
            continue

        output = run_agent_turn(
            agent_id=agent_id,
            persona_prompt=agent["system_prompt"],
            broadcast=broadcasts_in.get(agent_id, ""),
            round_num=round_num,
            store=stores[agent_id],
            client=client,
            model_cfg=models.get("agent", {}),
            temperature=agent_temp,
            memory_cfg=memory_cfg,
            prior_judge_observation=prior_judge_obs,
            agent_last_message=agent_last_messages.get(agent_id),
            problem_anchor=problem_core_question,
            use_critic=use_critic,
            belief_store=belief_stores.get(agent_id) if use_beliefs else None,
        )
        append_event(
            r_file,
            {
                "event_type": "agent_output",
                "agent_id": agent_id,
                "output": output,
            },
        )
        agent_outputs.append(output)

    # ── Phase 1b: Argument graph update ───────────────────────────────────────
    if use_argument_graph:
        if not done.get("argument_graph_updated"):
            graph_update = argument_graph.update_from_round(
                round_num=round_num,
                agent_outputs=agent_outputs,
                belief_stores=belief_stores or {},
                client=client,
                model_cfg=models.get("reflector", models.get("agent", {})),
                temperature=0.1,
            )
            append_event(
                r_file,
                {
                    "event_type": "argument_graph_updated",
                    "round": round_num,
                    "claims_added": graph_update.get("claims_added", []),
                },
            )

    # ── Phase 2: MetaCommunicator translations ─────────────────────────────────
    translations = []
    if use_meta_comm and translation_groups and meta_comm_prompt:
        existing_trans = done.get("meta_comm_translation", [])
        trans_done_keys = {(e["from"], e["for"]) for e in existing_trans}
        translations = list(existing_trans)

        speaking = [o for o in agent_outputs if not o["passed"] and o["message"]]

        for output in speaking:
            sender_id = output["agent_id"]
            message = output["message"]
            sender_info = agent_registry[sender_id]
            sender_group = domain_to_group.get(sender_info["domain"].lower(), -1)

            for receiver_id, recv_info in agent_registry.items():
                if receiver_id == sender_id:
                    continue
                receiver_group = domain_to_group.get(recv_info["domain"].lower(), -1)
                if sender_group == -1 or receiver_group == -1:
                    continue
                if sender_group == receiver_group:
                    continue
                if (sender_id, receiver_id) in trans_done_keys:
                    continue

                t = translate_message(
                    message=message,
                    sender_id=sender_id,
                    sender_name=sender_info["display_name"],
                    sender_domain=sender_info["domain"],
                    receiver_id=receiver_id,
                    receiver_name=recv_info["display_name"],
                    receiver_domain=recv_info["domain"],
                    meta_comm_prompt=meta_comm_prompt,
                    client=client,
                    model_cfg=models.get("meta_comm", {}),
                    temperature=meta_comm_temp,
                )
                append_event(r_file, {"event_type": "meta_comm_translation", **t})
                translations.append(t)

    # ── Phase 3: Assemble broadcasts ───────────────────────────────────────────
    existing_bc = done.get("broadcasts_assembled", [])
    if existing_bc:
        broadcasts_out = existing_bc[0]["broadcasts"]
    else:
        bc_result = assemble_broadcasts(
            round_num=round_num,
            agent_outputs=agent_outputs,
            translations=translations,
            agent_registry=agent_registry,
            translation_groups=translation_groups,
            use_meta_comm=use_meta_comm,
        )
        broadcasts_out = bc_result["broadcasts"]
        append_event(
            r_file,
            {"event_type": "broadcasts_assembled", "broadcasts": broadcasts_out},
        )

    # canonical broadcast — first value from broadcasts_out, used in Phase 4b and Phase 5
    canonical = next(iter(broadcasts_out.values()), "")

    # ── Argument graph summary — computed once, used in Phase 4b and Phase 5 ──
    arg_graph_summary = (
        argument_graph.get_judge_summary(last_n_rounds=3)
        if use_argument_graph
        else None
    )

    # ── Phase 4: Reflections ───────────────────────────────────────────────────
    for agent_id, agent in agent_registry.items():
        if [
            e
            for e in done.get("reflection_generated", [])
            if e.get("agent_id") == agent_id
        ]:
            continue
        results = generate_reflections(
            agent_id=agent_id,
            persona_prompt=agent["system_prompt"],
            round_broadcast=broadcasts_in.get(agent_id, ""),
            round_num=round_num,
            store=stores[agent_id],
            client=client,
            model_cfg=models.get("reflector", models.get("agent", {})),
            temperature=reflector_temp,
            max_reflections=memory_cfg.get("max_reflections_per_round"),
            embedding_model=memory_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        )
        append_event(
            r_file,
            {
                "event_type": "reflection_generated",
                "agent_id": agent_id,
                "reflections": results,
            },
        )

    # ── Phase 4b: Belief revision ──────────────────────────────────────────────
    if use_beliefs:
        from core.src.memory.belief_revision import generate_belief_revision

        for agent_id, agent in agent_registry.items():
            if [
                e
                for e in done.get("belief_revised", [])
                if e.get("agent_id") == agent_id
            ]:
                continue

            # Find this agent's output for the round
            agent_out = next(
                (o for o in agent_outputs if o["agent_id"] == agent_id), None
            )
            agent_message = agent_out.get("message") if agent_out else None

            belief_updates = generate_belief_revision(
                agent_id=agent_id,
                persona_prompt=agent["system_prompt"],
                round_broadcast=broadcasts_out.get(agent_id, canonical),
                agent_message=agent_message,
                round_num=round_num,
                store=belief_stores[agent_id],
                client=client,
                model_cfg=models.get("reflector", models.get("agent", {})),
                temperature=reflector_temp,
                argument_graph_summary=arg_graph_summary,
            )
            if belief_updates:
                append_event(
                    r_file,
                    {
                        "event_type": "belief_revised",
                        "agent_id": agent_id,
                        "belief_updates": belief_updates,
                    },
                )

    # ── Phase 5: Judge ─────────────────────────────────────────────────────────
    existing_j = done.get("judge_evaluated", [])
    if existing_j:
        judge_result = existing_j[0]["judge"]
    else:
        agent_snapshots = {aid: store.load_index() for aid, store in stores.items()}

        judge_result = evaluate_round(
            round_num=round_num,
            canonical_broadcast=canonical,
            observation_log=observation_log,
            agent_reflection_snapshots=agent_snapshots,
            judge_config=judge_config,
            client=client,
            model_cfg=models.get("judge", {}),
            temperature=judge_temp,
            argument_graph_summary=arg_graph_summary,
        )
        append_event(r_file, {"event_type": "judge_evaluated", "judge": judge_result})

    update_session_state(
        sess_dir, last_completed_round=round_num, current_round=round_num + 1
    )

    return {
        "agent_outputs": agent_outputs,
        "translations": translations,
        "broadcasts_out": broadcasts_out,
        "judge": judge_result,
    }

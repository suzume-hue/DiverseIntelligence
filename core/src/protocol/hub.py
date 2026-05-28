"""
core/src/protocol/hub.py
Assembles broadcasts from agent outputs and translations.

When meta_comm is enabled:
  Each agent receives verbatim messages from agents in their OWN group,
  plus translated messages from agents in OTHER groups.
  Each agent gets their own tailored broadcast.

When meta_comm is disabled:
  All agents receive a single unified broadcast.

Works for any number of agents and any number of domain groups.
"""

from xml.sax.saxutils import escape


def assemble_broadcasts(
    round_num:          int,
    agent_outputs:      list[dict],
    translations:       list[dict],
    agent_registry:     dict,
    translation_groups: list[list[str]],
    use_meta_comm:      bool,
) -> dict:
    """
    Returns: {"broadcasts": {agent_id: broadcast_xml_string}}
    """
    speaking = [o for o in agent_outputs if not o["passed"] and o["message"]]

    if not use_meta_comm or not translation_groups:
        xml = _unified_broadcast(round_num, speaking, agent_registry)
        return {"broadcasts": {aid: xml for aid in agent_registry}}

    # Build domain -> group index map
    domain_to_group: dict[str, int] = {}
    for idx, group in enumerate(translation_groups):
        for domain in group:
            domain_to_group[domain.lower()] = idx

    # Translation lookup: (sender_id, receiver_id) -> translated text
    trans_map = {(t["from"], t["for"]): t["translated"] for t in translations}

    broadcasts = {}
    for receiver_id, recv_info in agent_registry.items():
        recv_group = domain_to_group.get(recv_info["domain"].lower(), -1)
        messages   = []

        for o in speaking:
            sender_id    = o["agent_id"]
            sender_info  = agent_registry[sender_id]
            sender_group = domain_to_group.get(sender_info["domain"].lower(), -1)

            if sender_id == receiver_id:
                # Own message not echoed back
                continue

            if sender_group == recv_group or sender_group == -1 or recv_group == -1:
                # Same group or unassigned — verbatim
                messages.append((
                    sender_id,
                    sender_info["display_name"],
                    o["message"],
                    False,
                ))
            else:
                # Cross-group — use translation
                translated = trans_map.get((sender_id, receiver_id))
                if translated:
                    messages.append((
                        sender_id,
                        sender_info["display_name"],
                        translated,
                        True,   # is_translated
                    ))

        broadcasts[receiver_id] = _build_xml(round_num, messages)

    return {"broadcasts": broadcasts}


def build_round0_broadcast(
    problem:          dict,
    agent_id:         str,
    agent_group:      int,
    translated_intro: str | None,
) -> str:
    """
    Opening broadcast for one agent.
    Agents in group 0 receive the raw problem.
    Agents in other groups receive the MetaCommunicator's translation.
    If no translation provided, all agents get the raw problem.
    """
    if agent_group == 0 or translated_intro is None:
        content = _format_raw_problem(problem)
    else:
        content = translated_intro

    return (
        f'<broadcast round="0">\n'
        f'  <prompt>\n{content}\n  </prompt>\n'
        f'</broadcast>'
    )


def _format_raw_problem(problem: dict) -> str:
    tensions = problem.get("background", {}).get("core_tensions", {})
    tension_lines = "\n".join(
        f"  - {k}: {v}" for k, v in tensions.items()
    ) if isinstance(tensions, dict) else str(tensions)

    return (
        f"Title: {problem.get('title', '')}\n\n"
        f"Core Question:\n{problem.get('core_question', '')}\n\n"
        f"Background:\n{problem.get('background', {}).get('summary', '')}\n\n"
        f"Key Tensions:\n{tension_lines}"
    )


def _build_xml(round_num: int, messages: list[tuple]) -> str:
    """messages: list of (agent_id, display_name, text, is_translated)"""
    if not messages:
        return f'<broadcast round="{round_num}"><notice>No messages this round.</notice></broadcast>'

    lines = [f'<broadcast round="{round_num}">']
    for agent_id, display_name, text, is_translated in messages:
        label = escape(display_name)
        if is_translated:
            label += " [via MetaCommunicator]"
        lines.append(f'  <message from="{agent_id}" name="{label}">')
        lines.append(f'    {escape(text)}')
        lines.append(f'  </message>')
    lines.append('</broadcast>')
    return "\n".join(lines)


def _unified_broadcast(round_num: int, speaking: list[dict],
                        agent_registry: dict) -> str:
    messages = [
        (o["agent_id"], agent_registry[o["agent_id"]]["display_name"],
         o["message"], False)
        for o in speaking
    ]
    return _build_xml(round_num, messages)

"""
capability_registry.py — Phase 1 agent capability map

This is the placeholder for the real Capability Registry that will
come in a later phase (PostgreSQL + pgvector + performance scoring).

For Phase 1: a simple dict mapping capability → agent.
To add agents: just add entries here.
"""

from typing import Optional

# ── Agent definitions ──────────────────────────────────────────────────────
# Each agent has:
#   id          — unique agent identifier
#   name        — human readable name
#   capabilities — list of capabilities this agent handles
#   description — plain language description (used in CoT context)
#   speed_score  — 1 (slow) to 5 (fast), used for priority=fast routing
#   cost_score   — 1 (cheap) to 5 (expensive), used for priority=cheap routing

AGENTS = [
    {
        "id":           "agent_doc_extractor",
        "name":         "Document Extractor",
        "capabilities": ["document_extraction", "file_parsing", "pdf_extraction", "text_extraction"],
        "description":  "Extracts and parses text from documents, PDFs, Word files, spreadsheets",
        "speed_score":  5,
        "cost_score":   1,
    },
    {
        "id":           "agent_summariser",
        "name":         "Summariser",
        "capabilities": ["summarisation", "text_summarisation", "content_summarisation", "legal_summarisation", "document_summarisation"],
        "description":  "Summarises long documents, contracts, reports into concise outputs",
        "speed_score":  4,
        "cost_score":   2,
    },
    {
        "id":           "agent_risk_analyst",
        "name":         "Risk Analyst",
        "capabilities": ["risk_analysis", "risk_identification", "legal_risk", "compliance_check", "clause_flagging"],
        "description":  "Identifies risk clauses, compliance issues, legal flags in documents",
        "speed_score":  3,
        "cost_score":   3,
    },
    {
        "id":           "agent_data_analyst",
        "name":         "Data Analyst",
        "capabilities": ["data_analysis", "data_extraction", "spreadsheet_analysis", "number_crunching", "reporting"],
        "description":  "Analyses structured data, spreadsheets, performs calculations and reporting",
        "speed_score":  4,
        "cost_score":   2,
    },
    {
        "id":           "agent_researcher",
        "name":         "Researcher",
        "capabilities": ["research", "information_gathering", "web_research", "fact_finding", "knowledge_retrieval"],
        "description":  "Gathers information, performs research tasks, retrieves relevant knowledge",
        "speed_score":  2,
        "cost_score":   3,
    },
    {
        "id":           "agent_writer",
        "name":         "Writer",
        "capabilities": ["writing", "content_generation", "drafting", "email_drafting", "report_writing", "copywriting"],
        "description":  "Drafts documents, emails, reports, and other written content",
        "speed_score":  3,
        "cost_score":   2,
    },
    {
        "id":           "agent_classifier",
        "name":         "Classifier",
        "capabilities": ["classification", "categorisation", "tagging", "labelling", "sorting"],
        "description":  "Classifies and categorises items, documents, or data into defined categories",
        "speed_score":  5,
        "cost_score":   1,
    },
    {
        "id":           "agent_qa",
        "name":         "QA Agent",
        "capabilities": ["question_answering", "qa", "knowledge_base_query", "faq", "lookup"],
        "description":  "Answers questions from a knowledge base or document set",
        "speed_score":  4,
        "cost_score":   2,
    },
    {
        "id":           "agent_general",
        "name":         "General Agent",
        "capabilities": ["general", "fallback", "unknown", "misc"],
        "description":  "Handles general tasks that do not match a specific capability",
        "speed_score":  3,
        "cost_score":   2,
    },
]

# ── Build lookup maps ──────────────────────────────────────────────────────

# capability → list of agents that can handle it
_CAPABILITY_MAP: dict[str, list[dict]] = {}
for agent in AGENTS:
    for cap in agent["capabilities"]:
        _CAPABILITY_MAP.setdefault(cap.lower(), []).append(agent)

# agent_id → agent
_AGENT_MAP: dict[str, dict] = {a["id"]: a for a in AGENTS}


# ── Public interface ───────────────────────────────────────────────────────

def find_agents_for_capability(capability: str) -> list[dict]:
    """
    Returns all agents that can handle the given capability.
    Falls back to general agent if nothing matches.
    """
    cap = capability.lower().strip()
    # direct match
    if cap in _CAPABILITY_MAP:
        return _CAPABILITY_MAP[cap]
    # partial match — check if capability string contains any known cap keyword
    matches = []
    for known_cap, agents in _CAPABILITY_MAP.items():
        if known_cap in cap or cap in known_cap:
            matches.extend(agents)
    # deduplicate by agent id
    seen = set()
    unique = []
    for a in matches:
        if a["id"] not in seen:
            unique.append(a)
            seen.add(a["id"])
    if unique:
        return unique
    # fallback
    return [_AGENT_MAP["agent_general"]]


def pick_best_agent(capability: str, priority: str) -> dict:
    """
    From all agents that can handle a capability, pick the best one
    based on the priority constraint.

    priority=fast   → highest speed_score
    priority=cheap  → lowest cost_score
    priority=balanced → balanced score (speed + (5 - cost)) / 2
    """
    candidates = find_agents_for_capability(capability)

    if priority == "fast":
        return max(candidates, key=lambda a: a["speed_score"])
    elif priority == "cheap":
        return min(candidates, key=lambda a: a["cost_score"])
    else:  # balanced
        return max(candidates, key=lambda a: (a["speed_score"] + (6 - a["cost_score"])) / 2)


def get_all_capabilities() -> list[str]:
    """Returns all known capabilities — used in CoT prompt context."""
    return sorted(_CAPABILITY_MAP.keys())


def get_agent_descriptions_for_prompt() -> str:
    """
    Returns a formatted string of all agents and their capabilities.
    Injected into the Task Interpreter CoT prompt so the LLM knows
    what agents exist when deciding required_capabilities.
    """
    lines = []
    for agent in AGENTS:
        caps = ", ".join(agent["capabilities"])
        lines.append(f"- {agent['name']} ({agent['id']}): {agent['description']} | capabilities: {caps}")
    return "\n".join(lines)

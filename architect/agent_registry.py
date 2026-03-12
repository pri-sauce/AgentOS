"""
agent_registry.py — PostgreSQL-backed Agent Registry

Replaces the old hardcoded capability_registry.py.

Responsibilities:
- Query agents from PostgreSQL by capability
- Pick best agent based on priority + performance score
- Register new agents via API
- Return agent descriptions for CoT prompt context

Selection logic per priority:
  fast      → highest speed_score, tiebreak by performance_score
  cheap     → lowest cost_score, tiebreak by performance_score
  balanced  → weighted score: (speed + accuracy + performance - cost) / 4
  accurate  → highest accuracy_score, tiebreak by performance_score
"""

from typing import Optional
from database import db_connection


# ── Core queries ───────────────────────────────────────────────────────────

def find_agents_for_capability(capability: str) -> list[dict]:
    """
    Returns all active agents that can handle the given capability.
    Uses PostgreSQL GIN index on capabilities array for fast lookup.
    Falls back to partial text match, then to general agent.
    """
    cap = capability.lower().strip()

    with db_connection() as (conn, cur):
        # Exact capability match in array
        cur.execute("""
            SELECT * FROM agents
            WHERE is_active = TRUE
              AND %s = ANY(capabilities)
            ORDER BY performance_score DESC
        """, (cap,))
        results = cur.fetchall()

        if results:
            return [dict(r) for r in results]

        # Partial match — any capability contains the search term
        cur.execute("""
            SELECT * FROM agents
            WHERE is_active = TRUE
              AND EXISTS (
                SELECT 1 FROM unnest(capabilities) AS c
                WHERE c ILIKE %s OR %s ILIKE ('%' || c || '%')
              )
            ORDER BY performance_score DESC
        """, (f"%{cap}%", cap))
        results = cur.fetchall()

        if results:
            return [dict(r) for r in results]

        # Fallback — general agent
        cur.execute("""
            SELECT * FROM agents
            WHERE is_active = TRUE AND agent_id = 'agent_general'
        """)
        fallback = cur.fetchall()
        return [dict(r) for r in fallback]


def pick_best_agent(capability: str, priority: str) -> dict:
    """
    From all capable agents, picks the single best one for the given priority.

    Priority scoring:
      fast     → speed_score DESC, performance_score DESC
      cheap    → cost_score ASC, performance_score DESC
      accurate → accuracy_score DESC, performance_score DESC
      balanced → composite score DESC
    """
    cap      = capability.lower().strip()
    priority = priority.lower().strip()

    with db_connection() as (conn, cur):
        if priority == "fast":
            order_clause = "speed_score DESC, performance_score DESC"
        elif priority == "cheap":
            order_clause = "cost_score ASC, performance_score DESC"
        elif priority == "accurate":
            order_clause = "accuracy_score DESC, performance_score DESC"
        else:  # balanced
            # Composite: speed + accuracy + performance - (cost/5) — all normalised
            order_clause = """
                (speed_score * 0.3 + accuracy_score * 10 * 0.4 +
                 performance_score * 10 * 0.3 - cost_score * 0.2) DESC
            """

        # Try exact match first
        cur.execute(f"""
            SELECT * FROM agents
            WHERE is_active = TRUE
              AND %s = ANY(capabilities)
            ORDER BY {order_clause}
            LIMIT 1
        """, (cap,))
        result = cur.fetchone()

        if result:
            return dict(result)

        # Partial match
        cur.execute(f"""
            SELECT * FROM agents
            WHERE is_active = TRUE
              AND EXISTS (
                SELECT 1 FROM unnest(capabilities) AS c
                WHERE c ILIKE %s
              )
            ORDER BY {order_clause}
            LIMIT 1
        """, (f"%{cap}%",))
        result = cur.fetchone()

        if result:
            return dict(result)

        # Fallback
        cur.execute("""
            SELECT * FROM agents WHERE agent_id = 'agent_general' LIMIT 1
        """)
        result = cur.fetchone()
        return dict(result) if result else _fallback_agent()


def get_all_agents(active_only: bool = True) -> list[dict]:
    """Returns all agents from the registry."""
    with db_connection() as (conn, cur):
        if active_only:
            cur.execute("SELECT * FROM agents WHERE is_active = TRUE ORDER BY name")
        else:
            cur.execute("SELECT * FROM agents ORDER BY name")
        return [dict(r) for r in cur.fetchall()]


def get_agent_by_id(agent_id: str) -> Optional[dict]:
    """Returns a single agent by agent_id."""
    with db_connection() as (conn, cur):
        cur.execute("SELECT * FROM agents WHERE agent_id = %s", (agent_id,))
        result = cur.fetchone()
        return dict(result) if result else None


def register_agent(data: dict) -> dict:
    """
    Inserts a new agent into the registry.
    If agent_id already exists → updates it (upsert).
    Returns the created/updated agent.
    """
    with db_connection() as (conn, cur):
        cur.execute("""
            INSERT INTO agents (
                agent_id, name, description, capabilities,
                speed_score, cost_score, accuracy_score, performance_score,
                trust_level, version
            ) VALUES (
                %(agent_id)s, %(name)s, %(description)s, %(capabilities)s,
                %(speed_score)s, %(cost_score)s, %(accuracy_score)s, %(performance_score)s,
                %(trust_level)s, %(version)s
            )
            ON CONFLICT (agent_id) DO UPDATE SET
                name              = EXCLUDED.name,
                description       = EXCLUDED.description,
                capabilities      = EXCLUDED.capabilities,
                speed_score       = EXCLUDED.speed_score,
                cost_score        = EXCLUDED.cost_score,
                accuracy_score    = EXCLUDED.accuracy_score,
                performance_score = EXCLUDED.performance_score,
                trust_level       = EXCLUDED.trust_level,
                version           = EXCLUDED.version,
                updated_at        = NOW()
            RETURNING *
        """, data)
        result = cur.fetchone()
        return dict(result)


def deactivate_agent(agent_id: str) -> bool:
    """Soft-deletes an agent by setting is_active = FALSE."""
    with db_connection() as (conn, cur):
        cur.execute("""
            UPDATE agents SET is_active = FALSE, updated_at = NOW()
            WHERE agent_id = %s
            RETURNING agent_id
        """, (agent_id,))
        return cur.fetchone() is not None


def get_agent_descriptions_for_prompt() -> str:
    """
    Returns formatted agent descriptions for injection into CoT prompts.
    Called by the Task Interpreter to give the LLM context about available agents.
    """
    agents = get_all_agents(active_only=True)
    lines = []
    for a in agents:
        caps = ", ".join(a["capabilities"])
        lines.append(
            f"- {a['name']} ({a['agent_id']}): {a['description']} "
            f"| capabilities: {caps} "
            f"| speed: {a['speed_score']}/5 | performance: {a['performance_score']:.0%}"
        )
    return "\n".join(lines)


def get_all_capabilities() -> list[str]:
    """Returns all unique capabilities across all active agents."""
    with db_connection() as (conn, cur):
        cur.execute("""
            SELECT DISTINCT unnest(capabilities) AS cap
            FROM agents
            WHERE is_active = TRUE
            ORDER BY cap
        """)
        return [row["cap"] for row in cur.fetchall()]


# ── Fallback ───────────────────────────────────────────────────────────────

def _fallback_agent() -> dict:
    """Last resort fallback if DB has no general agent."""
    return {
        "agent_id":          "agent_general",
        "name":              "General Agent",
        "description":       "Fallback general agent",
        "capabilities":      ["general"],
        "speed_score":       3,
        "cost_score":        2,
        "accuracy_score":    0.80,
        "performance_score": 0.78,
        "trust_level":       "standard",
        "version":           "1.0.0",
    }

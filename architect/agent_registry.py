"""
agent_registry.py — PostgreSQL-backed Agent Registry

v0.4 additions:
  update_heartbeat()      — records last_seen, marks agent active/degraded
  get_agent_health()      — returns health status based on last_seen age
  get_all_health()        — health overview for all agents
  mark_stale_agents()     — background task: marks offline if no heartbeat > threshold
  discover_agents()       — structured discovery query with trust/cert/health filters
  trust_gate()            — enforces minimum trust level for sensitive tasks

Heartbeat thresholds:
  < 60s  → healthy
  60-300s → degraded
  > 300s  → offline
  never seen → unknown
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Optional
from database import db_connection

HEARTBEAT_HEALTHY_SECS  = 60
HEARTBEAT_DEGRADED_SECS = 300

TRUST_RANK = {"standard": 1, "verified": 2, "restricted": 0}
CERT_RANK  = {"uncertified": 0, "standard": 1, "verified": 2, "restricted": -1}


# ── Health helpers ─────────────────────────────────────────────────────────

def _health_from_last_seen(last_seen) -> tuple[str, Optional[float]]:
    """Returns (health_status_string, seconds_since_seen)."""
    if last_seen is None:
        return "unknown", None

    if isinstance(last_seen, str):
        last_seen = datetime.fromisoformat(last_seen)

    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)

    age = (datetime.now(timezone.utc) - last_seen).total_seconds()

    if age < HEARTBEAT_HEALTHY_SECS:
        return "healthy", age
    elif age < HEARTBEAT_DEGRADED_SECS:
        return "degraded", age
    else:
        return "offline", age


# ── Heartbeat ─────────────────────────────────────────────────────────────

def update_heartbeat(agent_id: str, status: str = "healthy", version: str = None) -> Optional[dict]:
    """
    Called when an agent sends a heartbeat.
    Updates last_seen and re-activates the agent if it was marked inactive due to staleness.
    Returns the updated agent record, or None if agent not found.
    """
    with db_connection() as (conn, cur):
        if version:
            cur.execute("""
                UPDATE agents
                SET last_seen = NOW(), is_active = TRUE, version = %s, updated_at = NOW()
                WHERE agent_id = %s
                RETURNING *
            """, (version, agent_id))
        else:
            cur.execute("""
                UPDATE agents
                SET last_seen = NOW(), is_active = TRUE, updated_at = NOW()
                WHERE agent_id = %s
                RETURNING *
            """, (agent_id,))

        result = cur.fetchone()
        if result:
            print(f"[Registry] Heartbeat from {agent_id} | status={status}")
        return dict(result) if result else None


def mark_stale_agents(stale_after_seconds: int = HEARTBEAT_DEGRADED_SECS) -> int:
    """
    Marks agents as inactive if they haven't sent a heartbeat within the threshold.
    Only affects agents that HAVE sent at least one heartbeat (last_seen IS NOT NULL).
    Returns count of agents marked inactive.

    Called periodically by the FastAPI background task.
    """
    with db_connection() as (conn, cur):
        cur.execute("""
            UPDATE agents
            SET is_active = FALSE, updated_at = NOW()
            WHERE last_seen IS NOT NULL
              AND last_seen < NOW() - INTERVAL '%s seconds'
              AND is_active = TRUE
            RETURNING agent_id
        """, (stale_after_seconds,))
        stale = cur.fetchall()
        count = len(stale)
        if count > 0:
            ids = [r["agent_id"] for r in stale]
            print(f"[Registry] Marked {count} agent(s) offline (no heartbeat): {ids}")
        return count


# ── Health queries ─────────────────────────────────────────────────────────

def get_agent_health(agent_id: str) -> Optional[dict]:
    """Returns health status for a single agent."""
    with db_connection() as (conn, cur):
        cur.execute(
            "SELECT agent_id, name, is_active, last_seen, certification_status, trust_level, endpoint_url "
            "FROM agents WHERE agent_id = %s",
            (agent_id,)
        )
        row = cur.fetchone()
        if not row:
            return None

        row = dict(row)
        health_status, seconds = _health_from_last_seen(row["last_seen"])
        return {
            "agent_id":             row["agent_id"],
            "name":                 row["name"],
            "is_active":            row["is_active"],
            "health_status":        health_status,
            "last_seen":            row["last_seen"].isoformat() if row["last_seen"] else None,
            "seconds_since_seen":   round(seconds, 1) if seconds is not None else None,
            "certification_status": row["certification_status"],
            "trust_level":          row["trust_level"],
            "endpoint_url":         row["endpoint_url"],
        }


def get_all_health() -> list[dict]:
    """Returns health status for all agents."""
    with db_connection() as (conn, cur):
        cur.execute(
            "SELECT agent_id, name, is_active, last_seen, certification_status, trust_level, endpoint_url "
            "FROM agents ORDER BY name"
        )
        rows = cur.fetchall()

    result = []
    for row in rows:
        row = dict(row)
        health_status, seconds = _health_from_last_seen(row["last_seen"])
        result.append({
            "agent_id":             row["agent_id"],
            "name":                 row["name"],
            "is_active":            row["is_active"],
            "health_status":        health_status,
            "last_seen":            row["last_seen"].isoformat() if row["last_seen"] else None,
            "seconds_since_seen":   round(seconds, 1) if seconds is not None else None,
            "certification_status": row["certification_status"],
            "trust_level":          row["trust_level"],
            "endpoint_url":         row["endpoint_url"],
        })
    return result


# ── Discovery API ──────────────────────────────────────────────────────────

def discover_agents(
    capability:        Optional[str] = None,
    min_trust_level:   Optional[str] = None,
    min_certification: Optional[str] = None,
    priority:          str  = "balanced",
    exclude_agents:    list = None,
    require_endpoint:  bool = False,
    healthy_only:      bool = True,
    limit:             int  = 5,
) -> list[dict]:
    """
    Structured agent discovery with trust/cert/health filters.
    Used by Workflow Manager and external callers via /registry/agents/discover.

    healthy_only=True:  only returns agents seen in the last 5 minutes
    require_endpoint:   only returns agents with endpoint_url set
    min_trust_level:    standard | verified
    min_certification:  standard | verified
    """
    exclude_agents = exclude_agents or []

    # Build ORDER BY from priority
    if priority == "fast":
        order = "speed_score DESC, performance_score DESC"
    elif priority == "cheap":
        order = "cost_score ASC, performance_score DESC"
    elif priority == "accurate":
        order = "accuracy_score DESC, performance_score DESC"
    else:
        order = "(speed_score * 0.3 + accuracy_score * 10 * 0.4 + performance_score * 10 * 0.3 - cost_score * 0.2) DESC"

    conditions = ["is_active = TRUE"]
    params = []

    if capability:
        conditions.append("(%s = ANY(capabilities) OR EXISTS (SELECT 1 FROM unnest(capabilities) c WHERE c ILIKE %s))")
        params += [capability.lower(), f"%{capability.lower()}%"]

    if healthy_only:
        conditions.append("(last_seen IS NULL OR last_seen > NOW() - INTERVAL '300 seconds')")

    if require_endpoint:
        conditions.append("endpoint_url IS NOT NULL AND endpoint_url != ''")

    if exclude_agents:
        placeholders = ",".join(["%s"] * len(exclude_agents))
        conditions.append(f"agent_id NOT IN ({placeholders})")
        params += exclude_agents

    if min_trust_level and min_trust_level in TRUST_RANK:
        min_rank = TRUST_RANK[min_trust_level]
        allowed  = [t for t, r in TRUST_RANK.items() if r >= min_rank]
        placeholders = ",".join(["%s"] * len(allowed))
        conditions.append(f"trust_level IN ({placeholders})")
        params += allowed

    if min_certification and min_certification in CERT_RANK:
        min_rank = CERT_RANK[min_certification]
        allowed  = [c for c, r in CERT_RANK.items() if r >= min_rank]
        placeholders = ",".join(["%s"] * len(allowed))
        conditions.append(f"certification_status IN ({placeholders})")
        params += allowed

    where = " AND ".join(conditions)
    params.append(limit)

    with db_connection() as (conn, cur):
        cur.execute(f"""
            SELECT * FROM agents
            WHERE {where}
            ORDER BY {order}
            LIMIT %s
        """, params)
        rows = cur.fetchall()

    result = []
    for row in rows:
        row = dict(row)
        health_status, seconds = _health_from_last_seen(row.get("last_seen"))
        row["health_status"] = health_status
        row["seconds_since_seen"] = round(seconds, 1) if seconds is not None else None
        result.append(row)

    return result


# ── Trust gate ────────────────────────────────────────────────────────────

def trust_gate(agent: dict, sensitivity: str) -> bool:
    """
    Returns True if the agent is permitted to handle the given sensitivity level.

    Rules:
      public       → any agent
      internal     → standard or higher trust
      confidential → verified only
      restricted   → verified + certification_status = verified
    """
    trust = agent.get("trust_level", "standard")
    cert  = agent.get("certification_status", "uncertified")

    if sensitivity == "public":
        return True
    elif sensitivity == "internal":
        return TRUST_RANK.get(trust, 0) >= TRUST_RANK["standard"]
    elif sensitivity == "confidential":
        return TRUST_RANK.get(trust, 0) >= TRUST_RANK["verified"]
    elif sensitivity == "restricted":
        return (TRUST_RANK.get(trust, 0) >= TRUST_RANK["verified"]
                and cert == "verified")
    return True


# ── Core queries (unchanged from v0.3, used internally) ────────────────────

def find_agents_for_capability(capability: str) -> list[dict]:
    cap = capability.lower().strip()
    with db_connection() as (conn, cur):
        cur.execute("""
            SELECT * FROM agents
            WHERE is_active = TRUE AND %s = ANY(capabilities)
            ORDER BY performance_score DESC
        """, (cap,))
        results = cur.fetchall()
        if results:
            return [dict(r) for r in results]

        cur.execute("""
            SELECT * FROM agents
            WHERE is_active = TRUE AND EXISTS (
                SELECT 1 FROM unnest(capabilities) AS c
                WHERE c ILIKE %s OR %s ILIKE ('%' || c || '%')
            )
            ORDER BY performance_score DESC
        """, (f"%{cap}%", cap))
        results = cur.fetchall()
        if results:
            return [dict(r) for r in results]

        cur.execute("SELECT * FROM agents WHERE is_active = TRUE AND agent_id = 'agent_general'")
        return [dict(r) for r in cur.fetchall()]


def pick_best_agent(capability: str, priority: str, sensitivity: str = "internal") -> dict:
    """
    Picks the best agent for a capability + priority combination.
    Now also enforces trust gate based on sensitivity.
    """
    cap      = capability.lower().strip()
    priority = priority.lower().strip()

    if priority == "fast":
        order_clause = "speed_score DESC, performance_score DESC"
    elif priority == "cheap":
        order_clause = "cost_score ASC, performance_score DESC"
    elif priority == "accurate":
        order_clause = "accuracy_score DESC, performance_score DESC"
    else:
        order_clause = "(speed_score * 0.3 + accuracy_score * 10 * 0.4 + performance_score * 10 * 0.3 - cost_score * 0.2) DESC"

    with db_connection() as (conn, cur):
        # Try exact capability match
        cur.execute(f"""
            SELECT * FROM agents
            WHERE is_active = TRUE AND %s = ANY(capabilities)
            ORDER BY {order_clause}
        """, (cap,))
        candidates = [dict(r) for r in cur.fetchall()]

        # Partial match fallback
        if not candidates:
            cur.execute(f"""
                SELECT * FROM agents
                WHERE is_active = TRUE AND EXISTS (
                    SELECT 1 FROM unnest(capabilities) AS c WHERE c ILIKE %s
                )
                ORDER BY {order_clause}
            """, (f"%{cap}%",))
            candidates = [dict(r) for r in cur.fetchall()]

    # Apply trust gate — filter out agents that can't handle this sensitivity
    permitted = [a for a in candidates if trust_gate(a, sensitivity)]

    if permitted:
        return permitted[0]

    # If trust gate removed everyone, fall back to general agent (best effort)
    if candidates:
        print(f"[Registry] WARNING: No agent passed trust gate for sensitivity={sensitivity}, using best available")
        return candidates[0]

    # Last resort
    with db_connection() as (conn, cur):
        cur.execute("SELECT * FROM agents WHERE agent_id = 'agent_general' LIMIT 1")
        result = cur.fetchone()
        return dict(result) if result else _fallback_agent()


def get_all_agents(active_only: bool = True) -> list[dict]:
    with db_connection() as (conn, cur):
        if active_only:
            cur.execute("SELECT * FROM agents WHERE is_active = TRUE ORDER BY name")
        else:
            cur.execute("SELECT * FROM agents ORDER BY name")
        return [dict(r) for r in cur.fetchall()]


def get_agent_by_id(agent_id: str) -> Optional[dict]:
    with db_connection() as (conn, cur):
        cur.execute("SELECT * FROM agents WHERE agent_id = %s", (agent_id,))
        result = cur.fetchone()
        return dict(result) if result else None


def register_agent(data: dict) -> dict:
    with db_connection() as (conn, cur):
        cur.execute("""
            INSERT INTO agents (
                agent_id, name, description, capabilities,
                speed_score, cost_score, accuracy_score, performance_score,
                trust_level, certification_status, version, endpoint_url
            ) VALUES (
                %(agent_id)s, %(name)s, %(description)s, %(capabilities)s,
                %(speed_score)s, %(cost_score)s, %(accuracy_score)s, %(performance_score)s,
                %(trust_level)s, %(certification_status)s, %(version)s, %(endpoint_url)s
            )
            ON CONFLICT (agent_id) DO UPDATE SET
                name                 = EXCLUDED.name,
                description          = EXCLUDED.description,
                capabilities         = EXCLUDED.capabilities,
                speed_score          = EXCLUDED.speed_score,
                cost_score           = EXCLUDED.cost_score,
                accuracy_score       = EXCLUDED.accuracy_score,
                performance_score    = EXCLUDED.performance_score,
                trust_level          = EXCLUDED.trust_level,
                certification_status = EXCLUDED.certification_status,
                version              = EXCLUDED.version,
                endpoint_url         = EXCLUDED.endpoint_url,
                updated_at           = NOW()
            RETURNING *
        """, {
            "agent_id":             data.get("agent_id"),
            "name":                 data.get("name"),
            "description":          data.get("description"),
            "capabilities":         data.get("capabilities"),
            "speed_score":          data.get("speed_score", 3),
            "cost_score":           data.get("cost_score", 3),
            "accuracy_score":       data.get("accuracy_score", 0.90),
            "performance_score":    data.get("performance_score", 0.90),
            "trust_level":          data.get("trust_level", "standard"),
            "certification_status": data.get("certification_status", "uncertified"),
            "version":              data.get("version", "1.0.0"),
            "endpoint_url":         data.get("endpoint_url"),
        })
        return dict(cur.fetchone())


def deactivate_agent(agent_id: str) -> bool:
    with db_connection() as (conn, cur):
        cur.execute("""
            UPDATE agents SET is_active = FALSE, updated_at = NOW()
            WHERE agent_id = %s RETURNING agent_id
        """, (agent_id,))
        return cur.fetchone() is not None


def get_agent_descriptions_for_prompt() -> str:
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
    with db_connection() as (conn, cur):
        cur.execute("""
            SELECT DISTINCT unnest(capabilities) AS cap
            FROM agents WHERE is_active = TRUE ORDER BY cap
        """)
        return [row["cap"] for row in cur.fetchall()]


def persist_job_result(result: dict) -> None:
    """Saves a completed job result to PostgreSQL job_results table."""
    import json
    with db_connection() as (conn, cur):
        cur.execute("""
            INSERT INTO job_results (
                job_id, session_id, user_id, raw_task, goal, status,
                steps_run, agents_used, final_output, total_time_ms, completed_at
            ) VALUES (
                %(job_id)s, %(session_id)s, %(user_id)s, %(raw_task)s, %(goal)s, %(status)s,
                %(steps_run)s, %(agents_used)s, %(final_output)s, %(total_time_ms)s, %(completed_at)s
            )
            ON CONFLICT (job_id) DO UPDATE SET
                status        = EXCLUDED.status,
                steps_run     = EXCLUDED.steps_run,
                agents_used   = EXCLUDED.agents_used,
                final_output  = EXCLUDED.final_output,
                total_time_ms = EXCLUDED.total_time_ms,
                completed_at  = EXCLUDED.completed_at
        """, {
            "job_id":        result.get("job_id"),
            "session_id":    result.get("session_id"),
            "user_id":       result.get("user_id"),
            "raw_task":      result.get("raw_task"),
            "goal":          result.get("goal"),
            "status":        result.get("status", "complete"),
            "steps_run":     json.dumps(result.get("steps_run", [])),
            "agents_used":   result.get("agents_used", []),
            "final_output":  json.dumps(result.get("final_output", {})),
            "total_time_ms": result.get("total_time_ms", 0),
            "completed_at":  result.get("completed_at"),
        })


def _fallback_agent() -> dict:
    return {
        "agent_id":             "agent_general",
        "name":                 "General Agent",
        "description":          "Fallback general agent",
        "capabilities":         ["general"],
        "speed_score":          3,
        "cost_score":           2,
        "accuracy_score":       0.80,
        "performance_score":    0.78,
        "trust_level":          "standard",
        "certification_status": "standard",
        "version":              "1.0.0",
        "endpoint_url":         None,
    }

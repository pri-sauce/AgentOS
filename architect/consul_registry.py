"""
consul_registry.py — Consul Integration for Architect

What Consul does here:
  - Active health probing: Consul pings agent endpoint_url on a schedule (every 30s)
    without the agent needing to send heartbeats itself
  - Service catalog: every registered agent gets a Consul service entry
  - Health status sync: Consul health results get synced back into PostgreSQL
    so the existing discovery/routing logic uses live health data

What PostgreSQL still owns:
  - Capabilities, trust_level, certification_status, performance scores
  - Agent metadata and CoT prompt descriptions
  - All routing decisions (agent_registry.py unchanged)

The split:
  PostgreSQL = what an agent CAN do + how good it is
  Consul     = IS the agent alive right now

Consul API endpoints used:
  PUT  /v1/agent/service/register    — register a service + health check
  PUT  /v1/agent/service/deregister/{id}  — deregister
  GET  /v1/health/service/{service}  — get health status for a service
  GET  /v1/agent/services            — list all registered services
  GET  /v1/health/state/any          — all health check results

Health check types supported:
  HTTP  — Consul GETs the agent's health endpoint every 30s (preferred)
  TTL   — Agent must ping Consul every N seconds (fallback if no HTTP endpoint)
"""

import os
import json
import requests
from typing import Optional
from datetime import datetime, timezone

CONSUL_HOST = os.getenv("CONSUL_HOST", "http://localhost:8500")
CONSUL_TIMEOUT = 5  # seconds


# ── Low-level Consul API helpers ───────────────────────────────────────────

def _consul_get(path: str) -> Optional[dict | list]:
    try:
        r = requests.get(f"{CONSUL_HOST}/v1/{path}", timeout=CONSUL_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        print(f"[Consul] Not reachable — skipping: GET {path}")
        return None
    except Exception as e:
        print(f"[Consul] GET {path} failed: {e}")
        return None


def _consul_put(path: str, payload: dict) -> bool:
    try:
        r = requests.put(
            f"{CONSUL_HOST}/v1/{path}",
            json=payload,
            timeout=CONSUL_TIMEOUT,
        )
        r.raise_for_status()
        return True
    except requests.exceptions.ConnectionError:
        print(f"[Consul] Not reachable — skipping: PUT {path}")
        return False
    except Exception as e:
        print(f"[Consul] PUT {path} failed: {e}")
        return False


def _consul_put_raw(path: str) -> bool:
    try:
        r = requests.put(f"{CONSUL_HOST}/v1/{path}", timeout=CONSUL_TIMEOUT)
        r.raise_for_status()
        return True
    except requests.exceptions.ConnectionError:
        print(f"[Consul] Not reachable — skipping: PUT {path}")
        return False
    except Exception as e:
        print(f"[Consul] PUT {path} failed: {e}")
        return False


# ── Is Consul available? ───────────────────────────────────────────────────

def consul_available() -> bool:
    """Check if Consul is reachable. Non-fatal — everything degrades gracefully."""
    try:
        r = requests.get(f"{CONSUL_HOST}/v1/status/leader", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ── Register an agent with Consul ──────────────────────────────────────────

def register_agent_consul(agent: dict) -> bool:
    """
    Registers an agent as a Consul service.

    If the agent has an endpoint_url, Consul will actively probe it via HTTP
    every 30 seconds and automatically mark it healthy/unhealthy.

    If no endpoint_url, registers with a TTL check — the agent must call
    /registry/agents/{id}/heartbeat (which calls update_consul_ttl) to stay healthy.

    Args:
        agent: agent dict from PostgreSQL registry

    Returns:
        True if registered successfully, False if Consul unavailable (non-fatal)
    """
    agent_id     = agent["agent_id"]
    endpoint_url = agent.get("endpoint_url")

    # Build the service registration payload
    service = {
        "ID":   agent_id,
        "Name": "architect-agent",
        "Tags": agent.get("capabilities", []) + [
            f"trust:{agent.get('trust_level', 'standard')}",
            f"cert:{agent.get('certification_status', 'uncertified')}",
        ],
        "Meta": {
            "agent_id":             agent_id,
            "name":                 agent.get("name", ""),
            "trust_level":          agent.get("trust_level", "standard"),
            "certification_status": agent.get("certification_status", "uncertified"),
            "version":              agent.get("version", "1.0.0"),
        },
    }

    if endpoint_url:
        # HTTP health check — Consul actively probes this URL
        # Expects 200 OK from the agent's health endpoint
        service["Checks"] = [{
            "HTTP":                          f"{endpoint_url}/health",
            "Interval":                      "30s",
            "Timeout":                       "5s",
            "DeregisterCriticalServiceAfter": "5m",
            "Notes": f"Active HTTP health check for {agent_id}",
        }]
        print(f"[Consul] Registering {agent_id} with HTTP health check -> {endpoint_url}/health")
    else:
        # TTL check — agent must call update_consul_ttl() within 90s or be marked critical
        service["Checks"] = [{
            "TTL":                           "90s",
            "DeregisterCriticalServiceAfter": "5m",
            "Notes": f"TTL check for {agent_id} — expects heartbeat every 60s",
        }]
        print(f"[Consul] Registering {agent_id} with TTL check (no endpoint_url)")

    success = _consul_put("agent/service/register", service)
    if success:
        # If no endpoint, mark TTL as passing immediately so it starts healthy
        if not endpoint_url:
            _consul_put_raw(f"agent/check/pass/service:{agent_id}")
    return success


def deregister_agent_consul(agent_id: str) -> bool:
    """Removes an agent from Consul service catalog."""
    return _consul_put_raw(f"agent/service/deregister/{agent_id}")


# ── TTL heartbeat update ───────────────────────────────────────────────────

def update_consul_ttl(agent_id: str, healthy: bool = True) -> bool:
    """
    Updates the TTL check for an agent.
    Called from main.py heartbeat endpoint when an agent pings in.

    For agents without endpoint_url, this keeps them marked healthy in Consul.
    For agents with endpoint_url, Consul probes them directly — this is a no-op.
    """
    if healthy:
        return _consul_put_raw(f"agent/check/pass/service:{agent_id}")
    else:
        return _consul_put_raw(f"agent/check/fail/service:{agent_id}")


# ── Read health from Consul ────────────────────────────────────────────────

def get_consul_health(agent_id: str) -> Optional[str]:
    """
    Returns the Consul health status for a single agent.

    Returns: "passing" | "warning" | "critical" | None (if not registered or Consul down)
    """
    data = _consul_get(f"health/service/architect-agent?filter=ID=={agent_id}")
    if not data:
        return None

    for entry in data:
        service = entry.get("Service", {})
        if service.get("ID") == agent_id:
            checks = entry.get("Checks", [])
            # Aggregate: if any check is critical → critical, warning → warning, else passing
            statuses = [c.get("Status", "critical") for c in checks]
            if "critical" in statuses:
                return "critical"
            elif "warning" in statuses:
                return "warning"
            return "passing"
    return None


def get_all_consul_health() -> dict[str, str]:
    """
    Returns a dict of agent_id -> consul_status for all registered agents.
    Used to enrich the /registry/health response.

    Returns {} if Consul is unavailable (graceful degradation).
    """
    data = _consul_get("health/state/any")
    if not data:
        return {}

    result = {}
    for check in data:
        service_id = check.get("ServiceID", "")
        status     = check.get("Status", "critical")
        if service_id:
            # Keep worst status per service
            current = result.get(service_id)
            if current is None or _status_rank(status) > _status_rank(current):
                result[service_id] = status

    return result


def _status_rank(status: str) -> int:
    return {"passing": 0, "warning": 1, "critical": 2}.get(status, 2)


def get_all_consul_services() -> list[dict]:
    """Returns all services registered in Consul with their health."""
    data = _consul_get("health/service/architect-agent")
    if not data:
        return []

    services = []
    for entry in data:
        service = entry.get("Service", {})
        checks  = entry.get("Checks", [])
        statuses = [c.get("Status", "critical") for c in checks]

        if "critical" in statuses:
            health = "critical"
        elif "warning" in statuses:
            health = "warning"
        else:
            health = "passing"

        services.append({
            "agent_id":   service.get("ID"),
            "name":       service.get("Meta", {}).get("name", ""),
            "tags":       service.get("Tags", []),
            "meta":       service.get("Meta", {}),
            "health":     health,
            "checks":     [{"name": c.get("Name"), "status": c.get("Status"), "output": c.get("Output")} for c in checks],
        })

    return services


# ── Sync Consul health → PostgreSQL ───────────────────────────────────────

def sync_consul_to_postgres() -> int:
    """
    Reads Consul health results and updates is_active in PostgreSQL.

    - passing  → is_active = TRUE
    - critical → is_active = FALSE (after DeregisterCriticalServiceAfter threshold)
    - warning  → is_active stays TRUE (degraded but not dead)

    Called periodically by the background health monitor in main.py.
    Returns count of agents whose status changed.
    """
    from database import db_connection

    consul_health = get_all_consul_health()
    if not consul_health:
        return 0

    changed = 0
    for agent_id, status in consul_health.items():
        should_be_active = status != "critical"
        try:
            with db_connection() as (conn, cur):
                cur.execute("""
                    UPDATE agents
                    SET is_active = %s, updated_at = NOW()
                    WHERE agent_id = %s
                      AND is_active != %s
                    RETURNING agent_id
                """, (should_be_active, agent_id, should_be_active))
                row = cur.fetchone()
                if row:
                    action = "reactivated" if should_be_active else "deactivated"
                    print(f"[Consul Sync] {agent_id} {action} (consul={status})")
                    changed += 1
        except Exception as e:
            print(f"[Consul Sync] DB update failed for {agent_id}: {e}")

    return changed


# ── Register all active agents on startup ─────────────────────────────────

def sync_postgres_to_consul() -> int:
    """
    On startup, registers all active PostgreSQL agents with Consul.
    Safe to call multiple times — Consul upserts on re-registration.
    Returns count of agents registered.
    """
    if not consul_available():
        print("[Consul] Not available — skipping initial sync")
        return 0

    from agent_registry import get_all_agents
    agents = get_all_agents(active_only=True)

    registered = 0
    for agent in agents:
        if register_agent_consul(agent):
            registered += 1

    print(f"[Consul] Synced {registered}/{len(agents)} agents to Consul")
    return registered

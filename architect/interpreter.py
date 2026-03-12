"""
interpreter.py — Task Interpreter

Uses Ollama (local LLM) with Few-Shot Chain-of-Thought prompting
to convert a raw task string into a structured TaskObject.

CoT forces the LLM to reason through:
1. What the user is actually asking for
2. What it requires technically (sub-goals + capabilities)
3. What constraints the user expressed
4. Whether anything is ambiguous
5. What the sensitivity level likely is

The reasoning chain is preserved in task_object.cot_reasoning for audit.
"""

import json
import re
import os
from typing import Optional

try:
    from ollama import Client as OllamaClient
except ImportError:
    OllamaClient = None

from models import TaskObject, Constraints, Priority, Sensitivity
from capability_registry import get_agent_descriptions_for_prompt

# ── Config ─────────────────────────────────────────────────────────────────

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


# ── Few-Shot CoT examples ──────────────────────────────────────────────────
# These are baked-in examples for Phase 1.
# In a later phase these will be pulled dynamically from routing memory
# using pgvector similarity search (in-context learning).

FEW_SHOT_EXAMPLES = """
--- EXAMPLE 1 ---
USER TASK: "summarise the Q3 report and pull out the top 3 risks"

REASONING:
Step 1 — What is the user asking for?
Two distinct things: a summary of the report AND extraction of specific items (top 3 risks). These are separable sub-goals.

Step 2 — What does this require technically?
First the report document needs to be extracted/parsed. Then summarisation runs on it. Then risk/finding extraction runs on it. Steps 2 and 3 can run in parallel — both only need step 1's output.

Step 3 — What constraints did the user express?
No speed or cost constraint mentioned. Default to balanced priority. No sensitivity hint but it's a financial report — likely internal.

Step 4 — Is anything ambiguous?
"top 3 risks" — risks to what? Financial? Operational? Assuming business risks from context. Noting assumption.

Step 5 — Sensitivity?
A Q3 report is likely internal company data. Sensitivity = internal.

OUTPUT:
{
  "goal": "summarise Q3 report and extract top 3 business risks",
  "sub_goals": ["extract_report_content", "summarise_report", "identify_top_risks"],
  "required_capabilities": ["document_extraction", "summarisation", "risk_analysis"],
  "constraints": {"priority": "balanced", "cost": "any", "accuracy": "high"},
  "sensitivity": "internal",
  "is_continuation": false,
  "ambiguous": false,
  "assumption_notes": "risks interpreted as business/operational risks"
}

--- EXAMPLE 2 ---
USER TASK: "draft a quick email to the client telling them the project is delayed by 2 weeks"

REASONING:
Step 1 — What is the user asking for?
Draft a specific type of communication (email) with specific content (project delay, 2 weeks). Single clear goal.

Step 2 — What does this require technically?
Just writing/drafting. No document extraction needed. No analysis needed. Single step task.

Step 3 — What constraints did the user express?
"quick" → speed is the priority. The word "quick" also implies keep it concise, not elaborate.

Step 4 — Is anything ambiguous?
Client name not given — will produce a template with [CLIENT NAME] placeholder. Not truly ambiguous enough to halt — noting it.

Step 5 — Sensitivity?
Client communications are typically internal or confidential. Sensitivity = confidential.

OUTPUT:
{
  "goal": "draft email to client about 2-week project delay",
  "sub_goals": ["draft_delay_notification_email"],
  "required_capabilities": ["writing", "email_drafting"],
  "constraints": {"priority": "fast", "cost": "any", "accuracy": "high"},
  "sensitivity": "confidential",
  "is_continuation": false,
  "ambiguous": false,
  "assumption_notes": "client name not provided — template will use [CLIENT NAME] placeholder"
}
"""


# ── System prompt ──────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    agent_descriptions = get_agent_descriptions_for_prompt()
    return f"""You are the Task Interpreter for Architect, an intelligent orchestration system.

Your job is to take a raw task from a user and produce a structured JSON task object.
You must reason through the task step by step BEFORE producing the output (Chain-of-Thought).

AVAILABLE AGENTS AND THEIR CAPABILITIES:
{agent_descriptions}

ALWAYS follow this exact reasoning structure:

Step 1 — What is the user actually asking for?
(identify distinct goals — are there multiple? are they separable?)

Step 2 — What does this require technically?
(what sub-goals are needed? what capabilities are required? which can run in parallel?)

Step 3 — What constraints did the user express?
(speed, cost, accuracy — infer from language like "quick", "cheap", "thorough", "fast")

Step 4 — Is anything ambiguous?
(if truly ambiguous and cannot proceed safely → set ambiguous: true and explain)
(if mildly ambiguous but can make a reasonable assumption → note assumption and proceed)

Step 5 — What is the sensitivity level?
(public / internal / confidential / restricted — infer from context)

AFTER reasoning, produce ONLY a valid JSON object in this exact format:
{{
  "goal": "...",
  "sub_goals": ["...", "..."],
  "required_capabilities": ["...", "..."],
  "constraints": {{
    "priority": "fast|balanced|cheap",
    "cost": "any|low|...",
    "accuracy": "high|medium|..."
  }},
  "sensitivity": "public|internal|confidential|restricted",
  "is_continuation": false,
  "ambiguous": false,
  "ambiguity_note": null,
  "assumption_notes": "..."
}}

Use ONLY capabilities from the available agents listed above.
The required_capabilities list must map to real agent capabilities.
Do not invent capabilities that no agent handles.
"""


# ── Main interpreter function ──────────────────────────────────────────────

def interpret_task(raw_task: str, session_id: Optional[str] = None) -> TaskObject:
    """
    Takes a raw task string, runs it through Ollama with CoT prompting,
    returns a structured TaskObject.

    Raises ValueError if the LLM response cannot be parsed.
    """
    if OllamaClient is None:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")

    client = OllamaClient(host=OLLAMA_HOST)

    user_prompt = f"""Here are two examples of how to reason through a task:

{FEW_SHOT_EXAMPLES}

--- NOW YOUR TASK ---
USER TASK: "{raw_task}"

Reason through this step by step, then produce the JSON output.
"""

    print(f"\n[Interpreter] Sending task to Ollama ({OLLAMA_MODEL})...")

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system",  "content": build_system_prompt()},
            {"role": "user",    "content": user_prompt},
        ],
        options={
            "temperature": 0.2,   # low temp for structured output
            "num_predict": 1500,
        }
    )

    raw_response = response["message"]["content"]
    print(f"[Interpreter] Raw response received ({len(raw_response)} chars)")

    # ── Extract CoT reasoning and JSON separately ──
    cot_reasoning, task_data = _parse_response(raw_response)

    # ── Build TaskObject ──
    constraints_data = task_data.get("constraints", {})
    constraints = Constraints(
        priority  = _safe_enum(Priority,     constraints_data.get("priority", "balanced"), Priority.BALANCED),
        cost      = constraints_data.get("cost", "any"),
        accuracy  = constraints_data.get("accuracy", "high"),
    )

    task_object = TaskObject(
        goal                  = task_data.get("goal", raw_task),
        sub_goals             = task_data.get("sub_goals", []),
        required_capabilities = task_data.get("required_capabilities", ["general"]),
        constraints           = constraints,
        sensitivity           = _safe_enum(Sensitivity, task_data.get("sensitivity", "internal"), Sensitivity.INTERNAL),
        is_continuation       = task_data.get("is_continuation", False),
        ambiguous             = task_data.get("ambiguous", False),
        ambiguity_note        = task_data.get("ambiguity_note"),
        assumption_notes      = task_data.get("assumption_notes"),
        cot_reasoning         = cot_reasoning,
    )

    print(f"[Interpreter] Task object created: goal='{task_object.goal}' ambiguous={task_object.ambiguous}")
    return task_object


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> tuple[str, dict]:
    """
    Splits the LLM response into:
    - cot_reasoning: everything before the JSON
    - task_data: the parsed JSON dict

    Handles cases where the LLM wraps JSON in ```json ... ``` blocks.
    """
    # Try to find JSON block (with or without markdown fences)
    json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if json_match:
        json_str     = json_match.group(1)
        cot_reasoning = raw[:json_match.start()].strip()
    else:
        # Find the last { ... } block in the response
        brace_match = re.search(r"(\{[\s\S]*\})\s*$", raw)
        if brace_match:
            json_str      = brace_match.group(1)
            cot_reasoning = raw[:brace_match.start()].strip()
        else:
            raise ValueError(f"Could not find JSON in LLM response.\nResponse was:\n{raw}")

    try:
        task_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common LLM JSON mistakes
        json_str  = _fix_json(json_str)
        task_data = json.loads(json_str)

    return cot_reasoning, task_data


def _fix_json(s: str) -> str:
    """Basic JSON repair for common LLM mistakes."""
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Replace single quotes with double quotes (basic)
    s = s.replace("'", '"')
    return s


def _safe_enum(enum_class, value: str, default):
    """Safely convert a string to an enum, returning default on failure."""
    try:
        return enum_class(value.lower().strip())
    except (ValueError, AttributeError):
        return default

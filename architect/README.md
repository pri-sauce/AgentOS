# Architect — Phase 1

## What this does
- Accepts a raw task via POST /task
- Interprets it using Ollama (local LLM) with Chain-of-Thought
- Creates a structured job from the interpretation
- Assigns agents to each step
- Appends to jobs.json and agent_tasks.json (never overwrites)

## Project structure
```
architect/
├── main.py                  # FastAPI app — entry point
├── interpreter.py           # Task Interpreter — CoT + Ollama
├── job_creator.py           # Turns task object into a job
├── agent_assigner.py        # Assigns agents to job steps
├── exporter.py              # Appends to JSON files
├── models.py                # Pydantic models for everything
├── capability_registry.py   # Simple agent capability map (Phase 1)
├── output/
│   ├── jobs.json            # All jobs, appended on every task
│   └── agent_tasks.json     # All agent assignments, appended
├── requirements.txt
└── .env.example
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and run Ollama
```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2        # or mistral, phi3, etc.
ollama serve                # runs on localhost:11434
```

### 3. Configure environment
```bash
cp .env.example .env
# edit .env — set OLLAMA_MODEL to whatever model you pulled
```

### 4. Run Architect
```bash
uvicorn main:app --reload --port 8000
```

### 5. Send a task
```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"task": "summarise this contract and flag risk clauses, need it fast"}'
```

## Output files
- `output/jobs.json` — array of all jobs ever created, newest appended at end
- `output/agent_tasks.json` — array of all agent assignments, newest appended at end

## What each file does
- `main.py` — FastAPI routes, wires everything together
- `interpreter.py` — CoT prompt → Ollama → structured task object
- `job_creator.py` — task object → job with ID, steps, timestamps
- `agent_assigner.py` — for each step, picks best agent from capability registry
- `exporter.py` — safely appends to JSON files without overwriting
- `capability_registry.py` — simple map of agent_id → capabilities (Phase 1 placeholder for real registry)
- `models.py` — all Pydantic models shared across files

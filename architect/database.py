"""
database.py — PostgreSQL connection and table setup

Tables:
  agents         — agent registry (capabilities, trust, health)
  routing_memory — past task → agent chain mappings (for cache lookup)
  heartbeat_log  — recent heartbeat events per agent (optional audit)

New in v0.4:
  agents.last_seen            — UTC timestamp of last heartbeat
  agents.certification_status — uncertified / standard / verified / restricted
  agents.endpoint_url         — where Architect sends tasks (Agent Connector URL)
  routing_memory              — task embedding + agent chain for cache hits (pgvector Phase 2)
"""

import os
import psycopg2
import psycopg2.extras
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

# ── Connection config ──────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "architect"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

_pool: pool.SimpleConnectionPool | None = None


def get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.SimpleConnectionPool(1, 10, **DB_CONFIG)
    return _pool


def get_conn():
    return get_pool().getconn()


def release_conn(conn):
    get_pool().putconn(conn)


# ── Context manager ────────────────────────────────────────────────────────

class db_connection:
    """
    Usage:
        with db_connection() as (conn, cur):
            cur.execute(...)
    Auto-commits on success, rolls back on error, always releases.
    """
    def __init__(self):
        self.conn = None
        self.cur  = None

    def __enter__(self):
        self.conn = get_conn()
        self.cur  = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return self.conn, self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.cur.close()
        release_conn(self.conn)


# ── DDL ────────────────────────────────────────────────────────────────────

CREATE_AGENTS_TABLE = """
CREATE TABLE IF NOT EXISTS agents (
    id                   SERIAL PRIMARY KEY,
    agent_id             TEXT UNIQUE NOT NULL,
    name                 TEXT NOT NULL,
    description          TEXT NOT NULL,
    capabilities         TEXT[] NOT NULL,
    speed_score          INTEGER NOT NULL DEFAULT 3  CHECK (speed_score  BETWEEN 1 AND 5),
    cost_score           INTEGER NOT NULL DEFAULT 3  CHECK (cost_score   BETWEEN 1 AND 5),
    accuracy_score       FLOAT   NOT NULL DEFAULT 0.90 CHECK (accuracy_score   BETWEEN 0.0 AND 1.0),
    performance_score    FLOAT   NOT NULL DEFAULT 0.90 CHECK (performance_score BETWEEN 0.0 AND 1.0),
    trust_level          TEXT    NOT NULL DEFAULT 'standard',
    certification_status TEXT    NOT NULL DEFAULT 'uncertified',
    is_active            BOOLEAN NOT NULL DEFAULT TRUE,
    version              TEXT    NOT NULL DEFAULT '1.0.0',
    endpoint_url         TEXT,
    last_seen            TIMESTAMPTZ,
    registered_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Migrate: add new columns if they don't exist yet (safe to run on existing DB)
MIGRATE_AGENTS = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='agents' AND column_name='last_seen'
    ) THEN
        ALTER TABLE agents ADD COLUMN last_seen TIMESTAMPTZ;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='agents' AND column_name='certification_status'
    ) THEN
        ALTER TABLE agents ADD COLUMN certification_status TEXT NOT NULL DEFAULT 'uncertified';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='agents' AND column_name='endpoint_url'
    ) THEN
        ALTER TABLE agents ADD COLUMN endpoint_url TEXT;
    END IF;
END $$;
"""

CREATE_AGENTS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_agents_capabilities
ON agents USING GIN (capabilities);

CREATE INDEX IF NOT EXISTS idx_agents_active
ON agents (is_active, trust_level);

CREATE INDEX IF NOT EXISTS idx_agents_last_seen
ON agents (last_seen);
"""

CREATE_UPDATED_AT_TRIGGER = """
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS agents_updated_at ON agents;
CREATE TRIGGER agents_updated_at
BEFORE UPDATE ON agents
FOR EACH ROW EXECUTE FUNCTION update_updated_at();
"""

# routing_memory: stores past task → agent chain for cache lookup
# Phase 2 will add a vector column (pgvector) for semantic similarity search
# For now: stores task_text + agent_chain as JSON, indexed by task hash
CREATE_ROUTING_MEMORY_TABLE = """
CREATE TABLE IF NOT EXISTS routing_memory (
    id             SERIAL PRIMARY KEY,
    task_hash      TEXT NOT NULL,
    task_text      TEXT NOT NULL,
    goal           TEXT,
    capabilities   TEXT[],
    agent_chain    JSONB NOT NULL,
    outcome        TEXT NOT NULL DEFAULT 'success',
    confidence     FLOAT NOT NULL DEFAULT 1.0,
    used_count     INTEGER NOT NULL DEFAULT 1,
    last_used_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_routing_memory_hash
ON routing_memory (task_hash);

CREATE INDEX IF NOT EXISTS idx_routing_memory_capabilities
ON routing_memory USING GIN (capabilities);
"""

# job_results: persists completed job results to PostgreSQL
# (previously only written to JSON files)
CREATE_JOB_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS job_results (
    id           SERIAL PRIMARY KEY,
    job_id       TEXT UNIQUE NOT NULL,
    session_id   TEXT,
    user_id      TEXT,
    raw_task     TEXT NOT NULL,
    goal         TEXT,
    status       TEXT NOT NULL DEFAULT 'complete',
    steps_run    JSONB,
    agents_used  TEXT[],
    final_output JSONB,
    total_time_ms INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_job_results_session
ON job_results (session_id);

CREATE INDEX IF NOT EXISTS idx_job_results_user
ON job_results (user_id);
"""


def init_db():
    """
    Creates all tables, indexes, and triggers.
    Also runs safe column migrations for existing DBs.
    Called once on app startup.
    """
    print("[DB] Initialising database...")
    with db_connection() as (conn, cur):
        cur.execute(CREATE_AGENTS_TABLE)
        cur.execute(MIGRATE_AGENTS)
        cur.execute(CREATE_AGENTS_INDEXES)
        cur.execute(CREATE_UPDATED_AT_TRIGGER)
        cur.execute(CREATE_ROUTING_MEMORY_TABLE)
        cur.execute(CREATE_JOB_RESULTS_TABLE)
        print("[DB] All tables and indexes ready")

    _seed_dummy_data()
    print("[DB] Database ready")


# ── Dummy seed data ────────────────────────────────────────────────────────

DUMMY_AGENTS = [
    {
        "agent_id":             "agent_doc_extractor",
        "name":                 "Document Extractor",
        "description":          "Extracts and parses text from documents including PDFs, Word files, spreadsheets, and images.",
        "capabilities":         ["document_extraction", "file_parsing", "pdf_extraction", "text_extraction", "ocr"],
        "speed_score":          5,
        "cost_score":           1,
        "accuracy_score":       0.95,
        "performance_score":    0.95,
        "trust_level":          "verified",
        "certification_status": "verified",
        "version":              "1.2.0",
        "endpoint_url":         None,
    },
    {
        "agent_id":             "agent_summariser",
        "name":                 "Summariser",
        "description":          "Summarises long documents, contracts, reports, and articles into concise structured outputs.",
        "capabilities":         ["summarisation", "text_summarisation", "legal_summarisation", "document_summarisation", "content_summarisation"],
        "speed_score":          4,
        "cost_score":           2,
        "accuracy_score":       0.91,
        "performance_score":    0.91,
        "trust_level":          "verified",
        "certification_status": "verified",
        "version":              "2.0.1",
        "endpoint_url":         None,
    },
    {
        "agent_id":             "agent_risk_analyst",
        "name":                 "Risk Analyst",
        "description":          "Identifies risk clauses, compliance issues, legal flags, and liability concerns in documents.",
        "capabilities":         ["risk_analysis", "risk_identification", "legal_risk", "compliance_check", "clause_flagging", "liability_analysis"],
        "speed_score":          3,
        "cost_score":           3,
        "accuracy_score":       0.93,
        "performance_score":    0.88,
        "trust_level":          "verified",
        "certification_status": "verified",
        "version":              "1.5.0",
        "endpoint_url":         None,
    },
    {
        "agent_id":             "agent_data_analyst",
        "name":                 "Data Analyst",
        "description":          "Analyses structured data, spreadsheets, and databases. Performs calculations, trend analysis, and reporting.",
        "capabilities":         ["data_analysis", "data_extraction", "spreadsheet_analysis", "number_crunching", "reporting", "trend_analysis", "aggregation"],
        "speed_score":          4,
        "cost_score":           2,
        "accuracy_score":       0.94,
        "performance_score":    0.92,
        "trust_level":          "verified",
        "certification_status": "verified",
        "version":              "1.3.2",
        "endpoint_url":         None,
    },
    {
        "agent_id":             "agent_researcher",
        "name":                 "Researcher",
        "description":          "Gathers information from knowledge bases, performs deep research, retrieves and cross-references relevant information.",
        "capabilities":         ["research", "information_gathering", "fact_finding", "knowledge_retrieval", "background_research"],
        "speed_score":          2,
        "cost_score":           3,
        "accuracy_score":       0.89,
        "performance_score":    0.85,
        "trust_level":          "standard",
        "certification_status": "standard",
        "version":              "1.1.0",
        "endpoint_url":         None,
    },
    {
        "agent_id":             "agent_writer",
        "name":                 "Writer",
        "description":          "Drafts documents, emails, reports, proposals, and other written content. Adapts tone and style.",
        "capabilities":         ["writing", "content_generation", "drafting", "email_drafting", "report_writing", "copywriting", "proposal_writing"],
        "speed_score":          3,
        "cost_score":           2,
        "accuracy_score":       0.90,
        "performance_score":    0.90,
        "trust_level":          "verified",
        "certification_status": "verified",
        "version":              "1.4.0",
        "endpoint_url":         None,
    },
    {
        "agent_id":             "agent_classifier",
        "name":                 "Classifier",
        "description":          "Classifies and categorises documents, items, or data into predefined or dynamic categories.",
        "capabilities":         ["classification", "categorisation", "tagging", "labelling", "sorting", "routing"],
        "speed_score":          5,
        "cost_score":           1,
        "accuracy_score":       0.96,
        "performance_score":    0.96,
        "trust_level":          "verified",
        "certification_status": "verified",
        "version":              "2.1.0",
        "endpoint_url":         None,
    },
    {
        "agent_id":             "agent_qa",
        "name":                 "QA Agent",
        "description":          "Answers specific questions from a provided document set or knowledge base.",
        "capabilities":         ["question_answering", "qa", "knowledge_base_query", "faq", "information_lookup"],
        "speed_score":          4,
        "cost_score":           2,
        "accuracy_score":       0.92,
        "performance_score":    0.91,
        "trust_level":          "standard",
        "certification_status": "standard",
        "version":              "1.2.0",
        "endpoint_url":         None,
    },
    {
        "agent_id":             "agent_finance",
        "name":                 "Finance Agent",
        "description":          "Handles financial document processing, invoice extraction, expense categorisation, and financial reporting.",
        "capabilities":         ["invoice_processing", "financial_analysis", "expense_categorisation", "budget_analysis", "financial_reporting", "accounting"],
        "speed_score":          3,
        "cost_score":           3,
        "accuracy_score":       0.94,
        "performance_score":    0.93,
        "trust_level":          "verified",
        "certification_status": "verified",
        "version":              "1.0.0",
        "endpoint_url":         None,
    },
    {
        "agent_id":             "agent_general",
        "name":                 "General Agent",
        "description":          "Fallback agent for tasks that do not match any specific capability.",
        "capabilities":         ["general", "fallback", "misc", "unknown"],
        "speed_score":          3,
        "cost_score":           2,
        "accuracy_score":       0.80,
        "performance_score":    0.78,
        "trust_level":          "standard",
        "certification_status": "standard",
        "version":              "1.0.0",
        "endpoint_url":         None,
    },
]


def _seed_dummy_data():
    """Seeds dummy agents if the table is empty. Safe to call every startup."""
    with db_connection() as (conn, cur):
        cur.execute("SELECT COUNT(*) as count FROM agents")
        result = cur.fetchone()
        count  = result["count"] if result else 0

        if count > 0:
            print(f"[DB] Registry already has {count} agents — skipping seed")
            return

        print(f"[DB] Seeding {len(DUMMY_AGENTS)} dummy agents...")
        for agent in DUMMY_AGENTS:
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
                ON CONFLICT (agent_id) DO NOTHING
            """, agent)
        print(f"[DB] Seeded {len(DUMMY_AGENTS)} agents successfully")

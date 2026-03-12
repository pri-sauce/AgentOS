"""
database.py — PostgreSQL connection and table setup

Handles:
- Connection pool via psycopg2
- Table creation on startup (if not exists)
- Dummy data seeding (runs once, skips if data already present)

Table: agents
  id               SERIAL PRIMARY KEY
  agent_id         TEXT UNIQUE NOT NULL       — e.g. "agent_summariser"
  name             TEXT NOT NULL              — e.g. "Summariser"
  description      TEXT NOT NULL              — plain language, used in CoT prompt
  capabilities     TEXT[] NOT NULL            — array of capability strings
  speed_score      INTEGER DEFAULT 3          — 1 (slow) to 5 (fast)
  cost_score       INTEGER DEFAULT 3          — 1 (cheap) to 5 (expensive)
  accuracy_score   FLOAT DEFAULT 0.90         — 0.0 to 1.0, fetched from LLM/Agent connector later
  performance_score FLOAT DEFAULT 0.90        — overall score, updated by connectors later
  trust_level      TEXT DEFAULT 'standard'    — standard / verified / restricted
  is_active        BOOLEAN DEFAULT TRUE
  version          TEXT DEFAULT '1.0.0'
  registered_at    TIMESTAMPTZ DEFAULT NOW()
  updated_at       TIMESTAMPTZ DEFAULT NOW()
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

# Connection pool — min 1, max 10 connections
_pool: pool.SimpleConnectionPool | None = None


def get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.SimpleConnectionPool(1, 10, **DB_CONFIG)
    return _pool


def get_conn():
    """Get a connection from the pool."""
    return get_pool().getconn()


def release_conn(conn):
    """Return a connection to the pool."""
    get_pool().putconn(conn)


# ── Context manager for clean connection handling ──────────────────────────

class db_connection:
    """
    Usage:
        with db_connection() as (conn, cur):
            cur.execute(...)
    Auto-commits on success, rolls back on error, always releases connection.
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


# ── Table creation ─────────────────────────────────────────────────────────

CREATE_AGENTS_TABLE = """
CREATE TABLE IF NOT EXISTS agents (
    id               SERIAL PRIMARY KEY,
    agent_id         TEXT UNIQUE NOT NULL,
    name             TEXT NOT NULL,
    description      TEXT NOT NULL,
    capabilities     TEXT[] NOT NULL,
    speed_score      INTEGER NOT NULL DEFAULT 3 CHECK (speed_score BETWEEN 1 AND 5),
    cost_score       INTEGER NOT NULL DEFAULT 3 CHECK (cost_score  BETWEEN 1 AND 5),
    accuracy_score   FLOAT   NOT NULL DEFAULT 0.90 CHECK (accuracy_score  BETWEEN 0.0 AND 1.0),
    performance_score FLOAT  NOT NULL DEFAULT 0.90 CHECK (performance_score BETWEEN 0.0 AND 1.0),
    trust_level      TEXT    NOT NULL DEFAULT 'standard',
    is_active        BOOLEAN NOT NULL DEFAULT TRUE,
    version          TEXT    NOT NULL DEFAULT '1.0.0',
    registered_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Index on capabilities for faster lookups
CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_agents_capabilities
ON agents USING GIN (capabilities);
"""

# Auto-update updated_at on row change
CREATE_TRIGGER = """
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


def init_db():
    """
    Creates the agents table and indexes if they don't exist.
    Seeds dummy data if the table is empty.
    Called once on app startup.
    """
    print("[DB] Initialising database...")
    with db_connection() as (conn, cur):
        cur.execute(CREATE_AGENTS_TABLE)
        cur.execute(CREATE_INDEX)
        cur.execute(CREATE_TRIGGER)
        print("[DB] Table and indexes ready")

    _seed_dummy_data()
    print("[DB] Database ready")


# ── Dummy data ─────────────────────────────────────────────────────────────

DUMMY_AGENTS = [
    {
        "agent_id":         "agent_doc_extractor",
        "name":             "Document Extractor",
        "description":      "Extracts and parses text from documents including PDFs, Word files, spreadsheets, and images. Handles multi-page documents and preserves structure.",
        "capabilities":     ["document_extraction", "file_parsing", "pdf_extraction", "text_extraction", "ocr"],
        "speed_score":      5,
        "cost_score":       1,
        "accuracy_score":   0.95,
        "performance_score": 0.95,
        "trust_level":      "verified",
        "version":          "1.2.0",
    },
    {
        "agent_id":         "agent_summariser",
        "name":             "Summariser",
        "description":      "Summarises long documents, contracts, reports, and articles into concise, structured outputs. Supports multiple summary formats: bullet points, executive summary, detailed breakdown.",
        "capabilities":     ["summarisation", "text_summarisation", "legal_summarisation", "document_summarisation", "content_summarisation"],
        "speed_score":      4,
        "cost_score":       2,
        "accuracy_score":   0.91,
        "performance_score": 0.91,
        "trust_level":      "verified",
        "version":          "2.0.1",
    },
    {
        "agent_id":         "agent_risk_analyst",
        "name":             "Risk Analyst",
        "description":      "Identifies risk clauses, compliance issues, legal flags, and liability concerns in documents. Specialised in contract review, regulatory compliance, and financial risk identification.",
        "capabilities":     ["risk_analysis", "risk_identification", "legal_risk", "compliance_check", "clause_flagging", "liability_analysis"],
        "speed_score":      3,
        "cost_score":       3,
        "accuracy_score":   0.93,
        "performance_score": 0.88,
        "trust_level":      "verified",
        "version":          "1.5.0",
    },
    {
        "agent_id":         "agent_data_analyst",
        "name":             "Data Analyst",
        "description":      "Analyses structured data, spreadsheets, and databases. Performs calculations, trend analysis, aggregations, and generates reports with insights. Handles CSV, Excel, and JSON data.",
        "capabilities":     ["data_analysis", "data_extraction", "spreadsheet_analysis", "number_crunching", "reporting", "trend_analysis", "aggregation"],
        "speed_score":      4,
        "cost_score":       2,
        "accuracy_score":   0.94,
        "performance_score": 0.92,
        "trust_level":      "verified",
        "version":          "1.3.2",
    },
    {
        "agent_id":         "agent_researcher",
        "name":             "Researcher",
        "description":      "Gathers information from knowledge bases, performs deep research tasks, retrieves and cross-references relevant information. Good for background research and fact-finding tasks.",
        "capabilities":     ["research", "information_gathering", "fact_finding", "knowledge_retrieval", "background_research"],
        "speed_score":      2,
        "cost_score":       3,
        "accuracy_score":   0.89,
        "performance_score": 0.85,
        "trust_level":      "standard",
        "version":          "1.1.0",
    },
    {
        "agent_id":         "agent_writer",
        "name":             "Writer",
        "description":      "Drafts documents, emails, reports, proposals, and other written content. Adapts tone and style. Handles formal business writing, client communications, and internal memos.",
        "capabilities":     ["writing", "content_generation", "drafting", "email_drafting", "report_writing", "copywriting", "proposal_writing"],
        "speed_score":      3,
        "cost_score":       2,
        "accuracy_score":   0.90,
        "performance_score": 0.90,
        "trust_level":      "verified",
        "version":          "1.4.0",
    },
    {
        "agent_id":         "agent_classifier",
        "name":             "Classifier",
        "description":      "Classifies and categorises documents, items, or data into predefined or dynamic categories. Fast and efficient for bulk classification tasks and tagging pipelines.",
        "capabilities":     ["classification", "categorisation", "tagging", "labelling", "sorting", "routing"],
        "speed_score":      5,
        "cost_score":       1,
        "accuracy_score":   0.96,
        "performance_score": 0.96,
        "trust_level":      "verified",
        "version":          "2.1.0",
    },
    {
        "agent_id":         "agent_qa",
        "name":             "QA Agent",
        "description":      "Answers specific questions from a provided document set or knowledge base. Performs precise information retrieval and question answering with source references.",
        "capabilities":     ["question_answering", "qa", "knowledge_base_query", "faq", "information_lookup"],
        "speed_score":      4,
        "cost_score":       2,
        "accuracy_score":   0.92,
        "performance_score": 0.91,
        "trust_level":      "standard",
        "version":          "1.2.0",
    },
    {
        "agent_id":         "agent_finance",
        "name":             "Finance Agent",
        "description":      "Handles financial document processing, invoice extraction, expense categorisation, budget analysis, and financial reporting. Understands accounting terminology and financial statements.",
        "capabilities":     ["invoice_processing", "financial_analysis", "expense_categorisation", "budget_analysis", "financial_reporting", "accounting"],
        "speed_score":      3,
        "cost_score":       3,
        "accuracy_score":   0.94,
        "performance_score": 0.93,
        "trust_level":      "verified",
        "version":          "1.0.0",
    },
    {
        "agent_id":         "agent_general",
        "name":             "General Agent",
        "description":      "Handles general tasks that do not match a specific capability. Used as fallback when no specialised agent is available. Lower performance on specialised tasks.",
        "capabilities":     ["general", "fallback", "misc", "unknown"],
        "speed_score":      3,
        "cost_score":       2,
        "accuracy_score":   0.80,
        "performance_score": 0.78,
        "trust_level":      "standard",
        "version":          "1.0.0",
    },
]


def _seed_dummy_data():
    """Seeds dummy agents if the table is empty. Safe to call on every startup."""
    with db_connection() as (conn, cur):
        cur.execute("SELECT COUNT(*) as count FROM agents")
        result = cur.fetchone()
        count = result["count"] if result else 0

        if count > 0:
            print(f"[DB] Registry already has {count} agents — skipping seed")
            return

        print(f"[DB] Seeding {len(DUMMY_AGENTS)} dummy agents...")
        for agent in DUMMY_AGENTS:
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
            """, agent)
        print(f"[DB] Seeded {len(DUMMY_AGENTS)} agents successfully")

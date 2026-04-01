# Smart Research Agent

OpenAI Agents SDK — full example with MCP server, orchestration, local memory, guardrails, and tracing.

## Quick start

```bash
# 1. Copy and fill in your API key
cp .env.example .env

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests (no API key needed)
pytest tests/ -v

# 5. Run the agent
python -m agent.main

# 6. Run with Docker
docker compose up --build
```

## Structure

| Path | Purpose |
|------|---------|
| `agent/main.py` | Orchestrator + specialist agents + SQLite session + tracing |
| `agent/guardrails.py` | Input & output safety guardrails |
| `mcp_server/server.py` | MCP stdio server — web_search, calculate, get_top_news |
| `tests/test_agent.py` | 21 unit + protocol tests (no API key needed) |
| `Dockerfile` | Multi-stage production image |
| `docker-compose.yml` | One-command local run with persistent volume |
| `.github/workflows/ci.yml` | CI: test on Python 3.11+3.12 → Docker build |

See the full guide in the previous conversation for step-by-step instructions.

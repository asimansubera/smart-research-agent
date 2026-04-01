#!/usr/bin/env bash
# =============================================================================
# setup_project.sh
# Creates the full smart-research-agent project structure on your local machine.
# Usage:
#   chmod +x setup_project.sh
#   ./setup_project.sh
# =============================================================================

set -e  # exit on any error

ROOT="smart-research-agent"

echo ""
echo "========================================"
echo "  Creating smart-research-agent project"
echo "========================================"
echo ""

# ── 1. Create all directories ─────────────────────────────────────────────────
echo "📁 Creating directories..."
mkdir -p "$ROOT/agent"
mkdir -p "$ROOT/mcp_server"
mkdir -p "$ROOT/tests"
mkdir -p "$ROOT/.github/workflows"
mkdir -p "$ROOT/.vscode"
mkdir -p "$ROOT/data"

# ── 2. agent/__init__.py ──────────────────────────────────────────────────────
cat > "$ROOT/agent/__init__.py" << 'EOF'
# Agent package
EOF

# ── 3. agent/guardrails.py ───────────────────────────────────────────────────
cat > "$ROOT/agent/guardrails.py" << 'EOF'
"""
Guardrails — two layers of safety with no human-in-the-loop:

  1. input_guardrail  : blocks off-topic or harmful queries before the agent acts
  2. output_guardrail : ensures the agent's response is safe before it reaches the user

Both run in parallel with the main agent loop (fail-fast).
"""

from pydantic import BaseModel
from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    input_guardrail,
    output_guardrail,
    GuardrailFunctionOutput,
)


# ── Pydantic schemas for structured guardrail outputs ─────────────────────────

class InputSafetyResult(BaseModel):
    is_safe: bool
    reason: str


class OutputSafetyResult(BaseModel):
    is_safe: bool
    reason: str


# ── Lightweight classifier agents ─────────────────────────────────────────────

_input_classifier = Agent(
    name="Input Safety Classifier",
    instructions="""
You are a content safety classifier for a research assistant.
Analyse the user's message and respond ONLY with a JSON object:
  { "is_safe": true/false, "reason": "brief explanation" }

Mark as UNSAFE (is_safe: false) if the message:
  - Asks for illegal information or instructions for harm
  - Contains abusive or hateful language
  - Is completely unrelated to research, facts, news, or calculations

Mark as SAFE (is_safe: true) for all legitimate research, learning,
news, maths, or factual questions.
""",
    output_type=InputSafetyResult,
)

_output_classifier = Agent(
    name="Output Safety Classifier",
    instructions="""
You are an output safety reviewer for a research assistant.
Review the agent's response and reply ONLY with a JSON object:
  { "is_safe": true/false, "reason": "brief explanation" }

Mark as UNSAFE (is_safe: false) if the response:
  - Contains step-by-step instructions for illegal activities
  - Includes hate speech or targeted harassment
  - Fabricates dangerous misinformation presented as fact

Mark as SAFE (is_safe: true) in all other cases.
""",
    output_type=OutputSafetyResult,
)


# ── Guardrail functions ───────────────────────────────────────────────────────

@input_guardrail
async def research_input_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    user_input: str,
) -> GuardrailFunctionOutput:
    """Runs BEFORE the main agent sees the user's message."""
    result = await Runner.run(
        _input_classifier,
        user_input,
        context=ctx.context,
    )
    check: InputSafetyResult = result.final_output
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=not check.is_safe,
    )


@output_guardrail
async def research_output_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: str,
) -> GuardrailFunctionOutput:
    """Runs AFTER the main agent produces a response, BEFORE it reaches the user."""
    result = await Runner.run(
        _output_classifier,
        output,
        context=ctx.context,
    )
    check: OutputSafetyResult = result.final_output
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=not check.is_safe,
    )
EOF

# ── 4. agent/main.py ─────────────────────────────────────────────────────────
cat > "$ROOT/agent/main.py" << 'EOF'
"""
Smart Research Agent — main entry point.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Orchestrator Agent (triage + combines outputs)     │
  │  ├── Input guardrail  (safety, topic filter)        │
  │  ├── Output guardrail (safe response check)         │
  │  ├── SQLite session   (persistent memory)           │
  │  └── Handoffs ──► Specialist agents                 │
  │        ├── Research Specialist (MCP: web_search)    │
  │        ├── News Specialist     (MCP: get_top_news)  │
  │        └── Math Specialist     (MCP: calculate)     │
  └─────────────────────────────────────────────────────┘
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from agents.tracing import trace
from agents.extensions.memory import SQLiteSession

from agent.guardrails import research_input_guardrail, research_output_guardrail

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smart-research-agent")

# ── Session (local SQLite memory) ─────────────────────────────────────────────
SESSION_DB = Path(__file__).parent.parent / "data" / "sessions.db"
SESSION_DB.parent.mkdir(parents=True, exist_ok=True)


def get_session(user_id: str) -> SQLiteSession:
    return SQLiteSession(
        session_id=user_id,
        database_path=str(SESSION_DB),
    )


# ── MCP server path ───────────────────────────────────────────────────────────
MCP_SERVER_PATH = Path(__file__).parent.parent / "mcp_server" / "server.py"


# ── Agent factory ─────────────────────────────────────────────────────────────
def build_agents(mcp_server: MCPServerStdio) -> Agent:

    research_specialist = Agent(
        name="Research Specialist",
        handoff_description="Handles in-depth web research questions.",
        instructions="""
You are a meticulous research specialist.
Use the web_search tool to find relevant information.
Search at least twice with refined queries if the first result is vague.
Present findings clearly with bullet points and a short summary paragraph.
""",
        mcp_servers=[mcp_server],
    )

    news_specialist = Agent(
        name="News Specialist",
        handoff_description="Handles current-events and news headline requests.",
        instructions="""
You are a news analyst.
Use the get_top_news tool to fetch headlines for the relevant topic.
Summarise the headlines in 2-3 sentences and highlight the most important story.
""",
        mcp_servers=[mcp_server],
    )

    math_specialist = Agent(
        name="Math Specialist",
        handoff_description="Handles all mathematical calculations and expressions.",
        instructions="""
You are a precise mathematics specialist.
Use the calculate tool to evaluate expressions.
Show your working: restate the expression, then show the result clearly.
""",
        mcp_servers=[mcp_server],
    )

    orchestrator = Agent(
        name="Research Orchestrator",
        instructions="""
You are an intelligent research orchestrator. Your job is to:
1. Understand what the user is asking.
2. Route the question to the right specialist using handoffs:
   - Factual / research questions  →  Research Specialist
   - Current events / news         →  News Specialist
   - Calculations / math           →  Math Specialist
3. Always greet the user warmly on their first message.
4. Maintain continuity — refer back to earlier parts of the conversation.
Never answer directly on topics where a specialist exists — always delegate.
""",
        handoffs=[research_specialist, news_specialist, math_specialist],
        input_guardrails=[research_input_guardrail],
        output_guardrails=[research_output_guardrail],
    )

    return orchestrator


# ── Main conversation loop ────────────────────────────────────────────────────
async def run_session(user_id: str = "default-user") -> None:
    session = get_session(user_id)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Smart Research Agent — powered by OpenAI Agents SDK")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60 + "\n")

    mcp_params = {
        "command": sys.executable,
        "args": [str(MCP_SERVER_PATH)],
    }

    async with MCPServerStdio(params=mcp_params, cache_tools_list=True) as mcp_server:
        orchestrator = build_agents(mcp_server)

        with trace(f"research-session-{user_id}"):
            turn = 0
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue
                if user_input.lower() in {"quit", "exit", "bye"}:
                    print("Agent: Goodbye! Your session has been saved.")
                    break

                turn += 1
                log.info("Turn %d — routing query: %r", turn, user_input[:80])

                try:
                    result = await Runner.run(
                        orchestrator,
                        user_input,
                        session=session,
                        max_turns=10,
                    )
                    print(f"\nAgent [{result.last_agent.name}]:\n{result.final_output}\n")

                except Exception as exc:
                    exc_name = type(exc).__name__
                    if "Guardrail" in exc_name or "Tripwire" in exc_name:
                        print(
                            "\nAgent: I'm sorry, I can't help with that request. "
                            "Please ask a research, news, or maths question.\n"
                        )
                        log.warning("Guardrail tripped on turn %d: %s", turn, exc)
                    else:
                        log.error("Unexpected error on turn %d: %s", turn, exc, exc_info=True)
                        print(f"\nAgent: Something went wrong ({exc_name}). Please try again.\n")


def main() -> None:
    user_id = os.getenv("AGENT_USER_ID", "default-user")
    asyncio.run(run_session(user_id))


if __name__ == "__main__":
    main()
EOF

# ── 5. mcp_server/__init__.py ─────────────────────────────────────────────────
cat > "$ROOT/mcp_server/__init__.py" << 'EOF'
# MCP Server package
EOF

# ── 6. mcp_server/server.py ──────────────────────────────────────────────────
cat > "$ROOT/mcp_server/server.py" << 'EOF'
"""
MCP Server — Smart Research Toolkit
Exposes three tools over stdio JSON-RPC:
  • web_search    : simulates a web search
  • calculate     : evaluates safe math expressions
  • get_top_news  : returns trending news headlines
"""

import json
import math
import sys
from datetime import datetime


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line.strip())


# ── Tool implementations ──────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 3) -> str:
    """Simulate a web search. Replace with real Bing/Brave/SerpAPI call."""
    results = [
        {
            "title": f"Result {i+1} for '{query}'",
            "url": f"https://example.com/result-{i+1}",
            "snippet": (
                f"This article covers key aspects of '{query}'. "
                f"Published {datetime.now().strftime('%Y-%m-%d')}."
            ),
        }
        for i in range(min(max_results, 5))
    ]
    return json.dumps(results, indent=2)


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Security model: eval runs with __builtins__ disabled and only the
    math module namespace exposed, allowing sqrt, sin, cos, log, pi, e, etc.
    The ^ -> ** substitution lets users write natural power notation.
    """
    blocked = ["__", "import", "open", "exec", "eval", "compile", "globals", "locals"]
    for token in blocked:
        if token in expression:
            return json.dumps({"error": f"Expression contains forbidden token: '{token}'."})
    safe_expr = expression.replace("^", "**")
    safe_namespace = {k: v for k, v in vars(math).items() if not k.startswith("_")}
    try:
        result = eval(safe_expr, {"__builtins__": {}}, safe_namespace)  # type: ignore
        return json.dumps({"expression": expression, "result": result})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def get_top_news(topic: str = "technology", count: int = 3) -> str:
    """Return mock top-news headlines for a given topic."""
    headlines = {
        "technology": [
            "AI agents are reshaping enterprise workflows in 2025",
            "OpenAI releases new Agents SDK with MCP support",
            "Python 3.14 ships with faster async I/O primitives",
        ],
        "science": [
            "JWST captures deepest infrared image of early universe",
            "Researchers achieve room-temperature superconductivity milestone",
            "CRISPR therapy shows 94% efficacy in latest clinical trial",
        ],
        "finance": [
            "Fed signals two rate cuts in the second half of 2025",
            "Global EV adoption surpasses 30% of new car sales",
            "S&P 500 hits record high on strong earnings season",
        ],
    }
    items = headlines.get(topic.lower(), headlines["technology"])
    return json.dumps(
        {"topic": topic, "headlines": items[:count], "fetched_at": datetime.now().isoformat()},
        indent=2,
    )


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = {
    "web_search": {
        "fn": web_search,
        "description": "Search the web for a query and return top results.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results": {"type": "integer", "description": "Max results (1-5)", "default": 3},
            },
            "required": ["query"],
        },
    },
    "calculate": {
        "fn": calculate,
        "description": "Evaluate a safe math expression (supports +,-,*,/,**,sqrt,etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '2 ** 10 + sqrt(144)'"}
            },
            "required": ["expression"],
        },
    },
    "get_top_news": {
        "fn": get_top_news,
        "description": "Retrieve top news headlines for a topic.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "'technology', 'science', or 'finance'", "default": "technology"},
                "count": {"type": "integer", "description": "Number of headlines (1-3)", "default": 3},
            },
            "required": [],
        },
    },
}


# ── JSON-RPC dispatcher ───────────────────────────────────────────────────────

def handle(msg: dict) -> dict | None:
    method = msg.get("method", "")
    msg_id = msg.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "smart-research-mcp", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "tools": [
                    {"name": n, "description": m["description"], "inputSchema": m["inputSchema"]}
                    for n, m in TOOLS.items()
                ]
            },
        }

    if method == "tools/call":
        params = msg.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        if tool_name not in TOOLS:
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}
        try:
            result_text = TOOLS[tool_name]["fn"](**arguments)
            return {"jsonrpc": "2.0", "id": msg_id,
                    "result": {"content": [{"type": "text", "text": result_text}]}}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32000, "message": str(exc)}}

    if method == "notifications/initialized":
        return None

    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main() -> None:
    while True:
        msg = read()
        if msg is None:
            break
        response = handle(msg)
        if response is not None:
            send(response)


if __name__ == "__main__":
    main()
EOF

# ── 7. tests/__init__.py ─────────────────────────────────────────────────────
cat > "$ROOT/tests/__init__.py" << 'EOF'
# Tests package
EOF

# ── 8. tests/test_agent.py ───────────────────────────────────────────────────
cat > "$ROOT/tests/test_agent.py" << 'EOF'
"""
Tests for Smart Research Agent.
Run with:  pytest tests/ -v
"""

import json
import importlib.util
import pathlib
import pytest

_server_path = pathlib.Path(__file__).parent.parent / "mcp_server" / "server.py"
_spec = importlib.util.spec_from_file_location("mcp_server.server", _server_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_mod)                  # type: ignore

web_search   = _mod.web_search
calculate    = _mod.calculate
get_top_news = _mod.get_top_news
handle       = _mod.handle


class TestWebSearch:
    def test_returns_list(self):
        assert isinstance(json.loads(web_search("Python async")), list)

    def test_respects_max_results(self):
        assert len(json.loads(web_search("AI agents", max_results=2))) == 2

    def test_result_has_required_keys(self):
        for r in json.loads(web_search("OpenAI")):
            assert {"title", "url", "snippet"} <= r.keys()

    def test_query_appears_in_title(self):
        results = json.loads(web_search("quantum computing"))
        assert any("quantum computing" in r["title"] for r in results)


class TestCalculate:
    def test_simple_addition(self):
        assert json.loads(calculate("2 + 2"))["result"] == 4

    def test_power_operator(self):
        assert json.loads(calculate("2 ^ 10"))["result"] == 1024

    def test_sqrt(self):
        assert json.loads(calculate("sqrt(144)"))["result"] == pytest.approx(12.0)

    def test_complex_expression(self):
        assert json.loads(calculate("(10 + 5) * 2 - 3"))["result"] == 27

    def test_unsafe_expression(self):
        assert "error" in json.loads(calculate("__import__('os').system('ls')"))

    def test_division(self):
        assert json.loads(calculate("100 / 4"))["result"] == 25.0


class TestGetTopNews:
    def test_technology_topic(self):
        out = json.loads(get_top_news("technology"))
        assert out["topic"] == "technology" and len(out["headlines"]) == 3

    def test_science_topic(self):
        assert len(json.loads(get_top_news("science"))["headlines"]) > 0

    def test_count_respected(self):
        assert len(json.loads(get_top_news("finance", count=2))["headlines"]) == 2

    def test_unknown_topic_falls_back(self):
        assert "headlines" in json.loads(get_top_news("sports"))

    def test_has_fetched_at(self):
        assert "fetched_at" in json.loads(get_top_news())


class TestMCPProtocol:
    def test_initialize(self):
        resp = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert resp["result"]["serverInfo"]["name"] == "smart-research-mcp"

    def test_tools_list(self):
        resp = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = [t["name"] for t in resp["result"]["tools"]]
        assert {"web_search", "calculate", "get_top_news"} <= set(names)

    def test_tools_call_web_search(self):
        resp = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "web_search", "arguments": {"query": "test"}}})
        assert isinstance(json.loads(resp["result"]["content"][0]["text"]), list)

    def test_tools_call_unknown(self):
        resp = handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                       "params": {"name": "nonexistent", "arguments": {}}})
        assert "error" in resp

    def test_unknown_method(self):
        resp = handle({"jsonrpc": "2.0", "id": 5, "method": "unknown/method", "params": {}})
        assert resp["error"]["code"] == -32601

    def test_notifications_initialized_returns_none(self):
        assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
EOF

# ── 9. .github/workflows/ci.yml ──────────────────────────────────────────────
cat > "$ROOT/.github/workflows/ci.yml" << 'EOF'
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install --upgrade pip && pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short

  docker:
    name: Docker build check
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: smart-research-agent:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
EOF

# ── 10. .vscode/launch.json ───────────────────────────────────────────────────
cat > "$ROOT/.vscode/launch.json" << 'EOF'
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run Agent",
      "type": "debugpy",
      "request": "launch",
      "module": "agent.main",
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/.env",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Run MCP Server (standalone test)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/mcp_server/server.py",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    },
    {
      "name": "Run Tests",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v", "--tb=short"],
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/.env",
      "console": "integratedTerminal"
    }
  ]
}
EOF

# ── 11. .vscode/settings.json ─────────────────────────────────────────────────
cat > "$ROOT/.vscode/settings.json" << 'EOF'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/", "-v"],
  "python.testing.unittestEnabled": false,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.rulers": [88]
  },
  "python.analysis.typeCheckingMode": "basic",
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".venv": true
  }
}
EOF

# ── 12. .env.example ──────────────────────────────────────────────────────────
cat > "$ROOT/.env.example" << 'EOF'
# Copy this file to .env and fill in your values.
# .env is listed in .gitignore — never commit real secrets.

# Required
OPENAI_API_KEY=sk-...your-key-here...

# Optional
AGENT_USER_ID=my-local-user
LOG_LEVEL=INFO
EOF

# ── 13. .gitignore ────────────────────────────────────────────────────────────
cat > "$ROOT/.gitignore" << 'EOF'
# Secrets
.env
*.pem
*.key

# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/
env/
*.egg-info/
dist/
build/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Runtime data
data/
*.db
*.sqlite3

# IDEs
.idea/
*.swp
*~
.DS_Store
Thumbs.db
EOF

# ── 14. Dockerfile ────────────────────────────────────────────────────────────
cat > "$ROOT/Dockerfile" << 'EOF'
# Stage 1: dependency builder
FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libffi-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: runtime image
FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="Smart Research Agent"
LABEL org.opencontainers.image.version="1.0.0"
WORKDIR /app
COPY --from=builder /install /usr/local
COPY mcp_server/ ./mcp_server/
COPY agent/      ./agent/
COPY tests/      ./tests/
RUN mkdir -p /app/data
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENT_USER_ID=docker-user
CMD ["python", "-m", "agent.main"]
EOF

# ── 15. docker-compose.yml ────────────────────────────────────────────────────
cat > "$ROOT/docker-compose.yml" << 'EOF'
version: "3.9"

services:
  agent:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
    image: smart-research-agent:latest
    container_name: smart-research-agent
    stdin_open: true
    tty: true
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - AGENT_USER_ID=${AGENT_USER_ID:-docker-user}
    volumes:
      - agent_data:/app/data
    restart: "no"

volumes:
  agent_data:
    driver: local
EOF

# ── 16. pytest.ini ────────────────────────────────────────────────────────────
cat > "$ROOT/pytest.ini" << 'EOF'
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
EOF

# ── 17. requirements.txt ──────────────────────────────────────────────────────
cat > "$ROOT/requirements.txt" << 'EOF'
# Core
openai-agents>=0.0.19
openai>=1.75.0
mcp>=1.0.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.24.0

# Utilities
python-dotenv>=1.0.0
EOF

# ── 18. README.md (brief) ─────────────────────────────────────────────────────
cat > "$ROOT/README.md" << 'EOF'
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
EOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "✅  Project created at: $(pwd)/$ROOT"
echo ""
echo "Next steps:"
echo "  cd $ROOT"
echo "  cp .env.example .env          # add your OPENAI_API_KEY"
echo "  python -m venv .venv"
echo "  source .venv/bin/activate     # Windows: .venv\\Scripts\\activate"
echo "  pip install -r requirements.txt"
echo "  pytest tests/ -v              # run all tests"
echo "  python -m agent.main          # run the agent"
echo "  docker compose up --build     # run in Docker"
echo ""
echo "Happy building! 🚀"
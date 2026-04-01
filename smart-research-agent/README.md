# Smart Research Agent

A production-ready agentic AI application built with the **OpenAI Agents SDK**.
It demonstrates every major concept from the SDK in one working project:
orchestration, MCP tools, persistent memory, guardrails, and tracing.

---

## What this project does

You type a question. The agent figures out what kind of question it is, routes it
to the right specialist, uses a tool to get the answer, checks the response is
safe, and replies — all automatically. Your conversation is remembered across
sessions using a local SQLite database.

---

## Architecture
```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Input Guardrail                                            │
│  Runs BEFORE the agent sees your message.                   │
│  Blocks harmful or off-topic requests immediately.          │
└─────────────────────────────────────────────────────────────┘
    │ (safe input passes through)
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Research Orchestrator (main agent)                         │
│  Reads conversation history from SQLite session.            │
│  Decides which specialist to hand off to.                   │
│                                                             │
│  ├── Research Specialist                                    │
│  │     Tool: web_search(query, max_results)                 │
│  │     Used for: facts, explanations, topic overviews       │
│  │                                                          │
│  ├── News Specialist                                        │
│  │     Tool: get_top_news(topic, count)                     │
│  │     Used for: current events, headlines, trends          │
│  │                                                          │
│  └── Math Specialist                                        │
│        Tool: calculate(expression)                          │
│        Used for: arithmetic, algebra, sqrt, trig            │
└─────────────────────────────────────────────────────────────┘
    │ (specialist produces response)
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Output Guardrail                                           │
│  Runs AFTER the specialist replies, BEFORE you see it.      │
│  Blocks unsafe or harmful responses.                        │
└─────────────────────────────────────────────────────────────┘
    │ (safe response passes through)
    ▼
Final Response → User
(Every step is traced → platform.openai.com/traces)
```

---

## Project structure
```
smart-research-agent/
├── agent/
│   ├── __init__.py
│   ├── main.py            # Orchestrator + specialists + session + tracing
│   └── guardrails.py      # Input & output guardrail definitions
├── mcp_server/
│   ├── __init__.py
│   └── server.py          # MCP stdio server — 3 tools over JSON-RPC
├── tests/
│   ├── __init__.py
│   └── test_agent.py      # 21 unit + protocol tests (no API key needed)
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions: test → Docker build
├── .vscode/
│   ├── launch.json        # Run/debug configs for VSCode
│   └── settings.json      # Python interpreter, formatter, test runner
├── data/                  # Created at runtime — holds sessions.db
├── .env.example           # Template — copy to .env and fill in your key
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## How each part works

### 1. The Agent (`agent/main.py`)

An Agent in the OpenAI Agents SDK is three things combined:
```
Agent = LLM + Instructions + Tools
```

- **LLM** — the language model (GPT-4o by default) that reads the situation
  and decides what to do next.
- **Instructions** — the system prompt. Defines the agent's role, personality,
  and rules. Written in plain English.
- **Tools** — Python functions or MCP server tools the agent can call to take
  real-world actions.

This project has four agents:

| Agent | Role | Tools | Handoffs |
|-------|------|-------|----------|
| Research Orchestrator | Triage — reads intent and delegates | None | Research, News, Math specialists |
| Research Specialist | Deep web research | `web_search` | None |
| News Specialist | Current events | `get_top_news` | None |
| Math Specialist | Calculations | `calculate` | None |

---

### 2. The Runner (`Runner.run()`)

The Runner is the execution engine. You call it once and it handles everything:
```python
result = await Runner.run(
    orchestrator,      # which agent to start with
    user_input,        # the user's message
    session=session,   # persistent memory
    max_turns=10,      # safety limit on back-and-forth
)
print(result.final_output)      # the agent's final reply
print(result.last_agent.name)   # which specialist answered
```

Internally the Runner loop works like this:
```
1. Call the LLM with the agent's instructions + conversation history
2. LLM decides: reply directly, call a tool, or hand off to another agent
3. If tool call → execute the tool → send result back to LLM → go to step 1
4. If handoff  → switch to the specialist agent → go to step 1
5. If reply    → run output guardrail → return result to you
```

This loop repeats until the agent produces a final reply or `max_turns` is hit.

---

### 3. Orchestration — Handoffs

Handoffs let one agent pass the entire conversation to a specialist. The
orchestrator never answers directly — it only routes.
```python
# Specialist agents — each knows one domain
research_specialist = Agent(
    name="Research Specialist",
    handoff_description="Handles in-depth web research questions.",
    instructions="Use web_search to find information...",
    mcp_servers=[mcp_server],
)

news_specialist = Agent(
    name="News Specialist",
    handoff_description="Handles current-events and news headline requests.",
    instructions="Use get_top_news to fetch headlines...",
    mcp_servers=[mcp_server],
)

math_specialist = Agent(
    name="Math Specialist",
    handoff_description="Handles all mathematical calculations.",
    instructions="Use calculate to evaluate expressions...",
    mcp_servers=[mcp_server],
)

# Orchestrator — reads handoff_description to decide who to call
orchestrator = Agent(
    name="Research Orchestrator",
    instructions="Route questions to the right specialist...",
    handoffs=[research_specialist, news_specialist, math_specialist],
)
```

The `handoff_description` is what the orchestrator reads to decide who to
delegate to. Make it precise — it is the routing signal.

When a handoff happens:
- The specialist takes over the conversation completely
- The specialist uses its own tools to answer
- `result.last_agent.name` tells you which specialist answered

---

### 4. MCP Server (`mcp_server/server.py`)

MCP (Model Context Protocol) is a standard way to give agents tools that run
as a separate process. The agent starts the MCP server as a subprocess and
communicates with it over stdio using JSON-RPC.

This project's MCP server exposes three tools:

#### `web_search(query, max_results=3)`
Searches the web for a query and returns top results. Replace the mock
implementation with a real Brave/Bing/SerpAPI call for production.
```python
def web_search(query: str, max_results: int = 3) -> str:
    # Returns a JSON list of {title, url, snippet}
    ...
```

#### `calculate(expression)`
Safely evaluates a mathematical expression. Uses Python's `eval` with
`__builtins__` disabled and only the `math` module exposed — so `sqrt`,
`sin`, `cos`, `log`, `pi`, `e` all work but nothing dangerous can run.
```python
def calculate(expression: str) -> str:
    # Blocks: __, import, open, exec, eval, compile
    # Allows: +, -, *, /, **, sqrt, sin, cos, log, pi, e, ...
    # Returns: {"expression": "...", "result": 42.0}
    ...
```

#### `get_top_news(topic="technology", count=3)`
Returns top news headlines for a topic. Supports `technology`, `science`,
and `finance`. Replace with a real news API for production.
```python
def get_top_news(topic: str, count: int) -> str:
    # Returns: {"topic": "...", "headlines": [...], "fetched_at": "..."}
    ...
```

The MCP server speaks JSON-RPC 2.0 over stdio. The agent connects to it like this:
```python
async with MCPServerStdio(
    params={"command": sys.executable, "args": ["mcp_server/server.py"]},
    cache_tools_list=True,   # cache the tool list — avoids re-fetching every turn
) as mcp_server:
    agent = Agent(
        name="Specialist",
        mcp_servers=[mcp_server],   # agent can now call all 3 tools
    )
```

---

### 5. Guardrails (`agent/guardrails.py`)

Guardrails are safety checks that run in parallel with the agent. They are
fast (use a cheap model) and fail-fast (stop the run immediately if triggered).

This project has two guardrails:

#### Input guardrail — runs before the agent sees your message
```python
@input_guardrail
async def research_input_guardrail(ctx, agent, user_input) -> GuardrailFunctionOutput:
    # Asks a classifier: "Is this message safe and on-topic?"
    # If not safe → tripwire_triggered=True → run stops immediately
    # If safe     → orchestrator proceeds normally
```

Blocks: illegal requests, hate speech, completely off-topic messages.
Allows: all legitimate research, news, maths, and factual questions.

#### Output guardrail — runs after the specialist replies
```python
@output_guardrail
async def research_output_guardrail(ctx, agent, output) -> GuardrailFunctionOutput:
    # Asks a classifier: "Is this response safe to show the user?"
    # If not safe → tripwire_triggered=True → response is blocked
    # If safe     → response reaches the user
```

Blocks: instructions for illegal activities, hate speech, dangerous misinformation.

Both guardrails use a lightweight classifier agent with `output_type=BaseModel`
to get structured `{"is_safe": true/false, "reason": "..."}` responses.
The main agent never runs if the input guardrail fires, which keeps latency low.

When a guardrail fires the main loop catches it gracefully:
```python
except Exception as exc:
    if "Guardrail" in type(exc).__name__ or "Tripwire" in type(exc).__name__:
        print("I'm sorry, I can't help with that request.")
```

---

### 6. SQLite Memory (`AsyncSQLiteSession`)

By default each `Runner.run()` call is stateless — the agent remembers nothing
from previous turns. Sessions give the agent persistent memory.
```python
from agents.extensions.memory.async_sqlite_session import AsyncSQLiteSession as SQLiteSession

# One session per user — stored in a local .db file
session = SQLiteSession(
    session_id="user-123",          # unique key per user
    db_path="data/sessions.db",     # created automatically
)

# Pass the session to every Runner.run() call
result = await Runner.run(agent, user_input, session=session)
```

The SDK automatically:
1. Loads conversation history from the database before calling the LLM
2. Appends the new turn to the database after the response

The session data is stored in two SQLite tables:
- `agent_sessions` — session metadata
- `agent_messages` — full conversation history per session

The database file lives in `data/sessions.db` (gitignored). In Docker it is
stored in a named volume (`agent_data`) so it persists across container restarts.

To use Redis instead (for multi-server deployments):
```python
from agents.extensions.memory.redis_session import RedisSession

session = RedisSession(session_id="user-123", redis_url="redis://localhost:6379")
```

---

### 7. Tracing

Every `Runner.run()` call is traced automatically — no setup needed.
```python
from agents.tracing import trace

# Group all turns in one session under a single named trace
with trace("research-session-user-123"):
    result = await Runner.run(agent, user_input, session=session)
```

Each trace records:
- Every LLM call — model, input, output, token usage, latency
- Every tool call — name, arguments, result
- Every handoff — which agent handed off to which specialist
- Every guardrail check — input, output, tripwire result

View traces at: **https://platform.openai.com/traces**

---

## Quick start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11 or 3.12 |
| Docker Desktop | 4.x+ |
| Git | any |
| OpenAI API key | from platform.openai.com |

### Local setup
```bash
# Clone
git clone https://github.com/asimansubera/smart-research-agent.git
cd smart-research-agent

# Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Dependencies
pip install -r requirements.txt

# API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run tests (no API key needed)
pytest tests/ -v

# Run the agent
python -m agent.main
```

### Docker
```bash
# Start in background
docker compose up --build -d

# Interact in a second terminal
docker attach smart-research-agent

# Stop
docker compose down
```

---

## Sample conversation
```
============================================================
  Smart Research Agent — powered by OpenAI Agents SDK
  Type 'quit' or 'exit' to end the session.
============================================================

You: What is the latest news in AI?

Agent [News Specialist]:
Top headlines in technology today:
- AI agents are reshaping enterprise workflows in 2025
- OpenAI releases new Agents SDK with MCP support
- Python 3.14 ships with faster async I/O primitives

You: Calculate 2^16 + sqrt(256)

Agent [Math Specialist]:
Expression: 2^16 + sqrt(256)
  2^16   = 65,536
  sqrt(256) = 16
  Result = 65,552

You: Research the history of Python programming language

Agent [Research Specialist]:
Based on web search results:
- Python was created by Guido van Rossum, first released in 1991...
- Key milestones: Python 2.0 (2000), Python 3.0 (2008)...

You: exit
Agent: Goodbye! Your session has been saved.
```

---

## Extending the project

| What to add | Where to change |
|-------------|-----------------|
| Real web search | `mcp_server/server.py` → `web_search()` — add Brave/SerpAPI call |
| Real news API | `mcp_server/server.py` → `get_top_news()` — add NewsAPI call |
| New specialist agent | `agent/main.py` → add Agent + add to `handoffs=[]` |
| Redis session | `agent/main.py` → swap `AsyncSQLiteSession` for `RedisSession` |
| REST API interface | Add FastAPI endpoint that calls `Runner.run()` |
| Slack bot | Add Slack Bolt listener that calls `Runner.run()` |
| Stricter guardrails | `agent/guardrails.py` → update classifier instructions |

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | Your OpenAI API key |
| `AGENT_USER_ID` | No | `default-user` | Unique session ID per user |
| `LOG_LEVEL` | No | `INFO` | Set to `DEBUG` for verbose output |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `OPENAI_API_KEY not set` | Copy `.env.example` to `.env` and add your key |
| `ModuleNotFoundError: agents` | Run `pip install -r requirements.txt` in `.venv` |
| `ModuleNotFoundError: aiosqlite` | Run `pip install aiosqlite` |
| `MCP server not found` | Check `mcp_server/server.py` path is correct |
| Tests failing | Run `pytest tests/ -v --tb=long` for full tracebacks |
| Guardrail fires on valid input | Loosen classifier instructions in `guardrails.py` |
| Docker: missing module | Run `docker compose up --build` to rebuild the image |

---

## CI/CD

GitHub Actions runs automatically on every push to `main`:

1. **Test** — runs all 21 tests on Python 3.11 and 3.12
2. **Docker build** — builds the Docker image to verify the Dockerfile is valid

See `.github/workflows/ci.yml`.

---

## Licence

MIT — free to use, modify, and distribute.

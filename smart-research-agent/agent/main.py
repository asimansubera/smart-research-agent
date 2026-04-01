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
from agents.extensions.memory.async_sqlite_session import AsyncSQLiteSession as SQLiteSession

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
        db_path=str(SESSION_DB),
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

    try:
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
    finally:
        # Close the async SQLite connection cleanly before the event loop shuts
        # down — prevents the noisy RuntimeError: Event loop is closed on exit.
        if hasattr(session, "aclose"):
            await session.aclose()
        elif hasattr(session, "close"):
            await session.close()


def main() -> None:
    user_id = os.getenv("AGENT_USER_ID", "default-user")
    asyncio.run(run_session(user_id))


if __name__ == "__main__":
    main()

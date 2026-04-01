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
    """Safely evaluate a math expression.

    Security: __builtins__ disabled; only math module exposed.
    Allows sqrt, sin, cos, log, pi, e, etc.
    """
    blocked = ["__", "import", "open", "exec", "eval", "compile", "globals", "locals"]
    for token in blocked:
        if token in expression:
            return __import__("json").dumps({"error": f"Forbidden token: '{token}'."})

    safe_expr = expression.replace("^", "**")
    safe_ns = {k: v for k, v in __import__("math").__dict__.items() if not k.startswith("_")}

    try:
        result = eval(safe_expr, {"__builtins__": {}}, safe_ns)
        return __import__("json").dumps({"expression": expression, "result": result})
    except Exception as exc:
        return __import__("json").dumps({"error": str(exc)})
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

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

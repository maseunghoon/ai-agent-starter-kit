"""Offline tests — they run without an OpenAI API key or network access.

We test the parts we own (tools, memory, HTTP wiring) and stub out the LLM so
the suite is fast and deterministic. Run with:  pytest -q
"""

from fastapi.testclient import TestClient

from app.agents.tools import calculator, current_time
from app.memory.store import InMemoryMemoryStore


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def test_calculator_basic():
    # `.invoke` is how you call a LangChain tool with its arguments.
    assert "345" in calculator.invoke({"expression": "15 * 23"})


def test_calculator_rejects_unsafe_input():
    # Anything that isn't whitelisted arithmetic returns a friendly error,
    # never raises, and never executes arbitrary code.
    result = calculator.invoke({"expression": "__import__('os').system('ls')"})
    assert "Could not evaluate" in result


def test_current_time_returns_iso_string():
    result = current_time.invoke({})
    # ISO 8601 timestamps contain a 'T' between date and time.
    assert "T" in result


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def test_memory_store_roundtrip():
    store = InMemoryMemoryStore()
    store.add_message("s1", "user", "hello")
    store.add_message("s1", "assistant", "hi there")

    messages = store.get_messages("s1")
    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    # Sessions are isolated from each other.
    assert store.get_messages("other") == []

    store.clear("s1")
    assert store.get_messages("s1") == []


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
def test_health_endpoint():
    import app.main as main_module

    # TestClient triggers the app's lifespan (startup) just like a real server.
    with TestClient(main_module.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_chat_endpoint_with_stubbed_agent(monkeypatch):
    """The /chat endpoint should persist history and report tool calls.

    We replace `run_agent` with a fake so no real LLM call happens.
    """
    import app.main as main_module

    def fake_run_agent(graph, history, user_message):
        # Echo back how many prior turns existed, proving memory is wired up.
        return {"response": f"seen {len(history)} prior msgs", "tool_calls": ["calculator"]}

    monkeypatch.setattr(main_module, "run_agent", fake_run_agent)

    with TestClient(main_module.app) as client:
        first = client.post("/chat", json={"session_id": "t1", "message": "hi"})
        assert first.status_code == 200
        assert first.json()["tool_calls"] == ["calculator"]

        # Second call should see the 2 messages (user + assistant) from the first.
        second = client.post("/chat", json={"session_id": "t1", "message": "again"})
        assert "seen 2 prior msgs" in second.json()["response"]

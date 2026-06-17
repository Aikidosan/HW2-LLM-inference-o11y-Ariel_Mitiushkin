"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.

When LANGFUSE_* keys are set, every run is wrapped in one Langfuse trace
("answer:<db>") so the LangGraph node spans (generate_sql, execute, verify,
revise) nest into a single waterfall, and the trace is tagged with post-run
facts (db_id, iteration_count, verify_ok, revised) for Phase 6 filtering.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from agent.graph import AgentState, graph  # noqa: E402

# Langfuse callback handler + client. If keys are set we initialize them; init
# failures are NOT swallowed - a misconfigured Langfuse should not silently
# produce zero traces.
_lf_handler: Any = None
_lf_client: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    _lf_handler = CallbackHandler()
    _lf_client = get_client()


app = FastAPI()


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _build_response(final: dict[str, Any]) -> AnswerResponse:
    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")
    if execution is None:
        return AnswerResponse(sql=sql, rows=None, iterations=iteration, ok=False,
                              error="agent produced no execution result", history=history)
    if not execution.ok:
        return AnswerResponse(sql=sql, rows=None, iterations=iteration, ok=False,
                              error=execution.error, history=history)
    return AnswerResponse(sql=sql, rows=[list(r) for r in (execution.rows or [])],
                          iterations=iteration, ok=True, history=history)


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    state = AgentState(question=req.question, db_id=req.db)
    # `langfuse_tags` is read by the callback handler and applied to the trace;
    # other metadata keys propagate as trace metadata. db_id is known up front.
    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler is not None else [],
        "metadata": {"langfuse_tags": [f"db:{req.db}"], "db_id": req.db, **req.tags},
    }
    # Wrap the run in one root span so the LangGraph node spans nest into a single
    # waterfall and we can enrich it with post-run facts for Phase 6 filtering.
    span_ctx = (
        _lf_client.start_as_current_observation(
            as_type="span", name=f"answer:{req.db}",
            input={"question": req.question, "db": req.db})
        if _lf_client is not None else nullcontext()
    )
    try:
        with span_ctx as root:
            final = graph.invoke(state, config=config)
            resp = _build_response(final)
            if _lf_client is not None and root is not None:
                _enrich_trace(root, req, final, resp)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return resp


def _enrich_trace(root: Any, req: AnswerRequest, final: dict[str, Any],
                  resp: AnswerResponse) -> None:
    """Best-effort: attach post-run facts (iteration_count, verify_ok, revised) to
    the trace via its root span, plus a trace-level summary I/O. Tracing problems
    must not fail an /answer call - the node spans are captured regardless.
    """
    try:
        verify_ok = bool(final.get("verify_ok"))
        revised = any(h.get("node") == "revise" for h in final.get("history", []))
        out = {"sql": resp.sql, "ok": resp.ok, "rows": len(resp.rows or [])}
        root.update(metadata={
            "db_id": req.db,
            "iteration_count": resp.iterations,
            "verify_ok": verify_ok,
            "revised": revised,
            **req.tags,
        }, output=out)
        root.set_trace_io(input={"question": req.question, "db": req.db}, output=out)
    except Exception as e:  # noqa: BLE001
        print(f"[langfuse] trace enrichment failed (non-fatal): {type(e).__name__}: {e}")

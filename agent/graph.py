"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import asyncio
import json as _json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# Tuned to 2 in Phase 6: the baseline eval showed iter-3 pass rate == iter-2
# (the 3rd attempt added zero accuracy), so capping at 2 removes up to 2 serial
# LLM calls from the worst case (revise+verify) with no measured quality loss.
MAX_ITERATIONS = 2

# Verifier mode (Phase 6). The Phase-5 eval showed the LLM plausibility check
# converted ZERO net questions (per-iteration pass rate iter1 == iter2), because
# the failures that survive execution are plausible-but-wrong rows the verifier
# can't catch without the gold answer. Meanwhile that check costs one extra
# serial LLM call on EVERY request — the dominant agent-side latency under load.
# So the default verifier is PROGRAMMATIC: it revises only on the failure modes a
# rule can detect for free (execution error / zero rows), which is where the
# loop's value actually came from. Set VERIFY_LLM=1 to restore the LLM verifier
# (used for the Phase 3 loop demo and the Phase 4 Langfuse waterfall).
VERIFY_LLM = os.environ.get("VERIFY_LLM", "0") == "1"

# Self-consistency (accuracy lever, Phase 5). With K>1 the generate step samples K
# candidate queries in parallel at SELF_CONSISTENCY_TEMP, executes each, and keeps
# the SQL whose *executed result* is the most common — a majority vote over answers,
# which filters one-off reasoning slips that a single greedy decode commits to. K
# parallel calls cost K x the generate latency, so this is a QUALITY-mode lever:
# default K=1 (the low-latency SLO config); the accuracy benchmark runs K=5.
SELF_CONSISTENCY_K = int(os.environ.get("SELF_CONSISTENCY_K", "1"))
SELF_CONSISTENCY_TEMP = float(os.environ.get("SELF_CONSISTENCY_TEMP", "0.7"))

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm(temperature: float = 0.0) -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default).

    Phase 6 tuning: cap output length only. SQL answers and the verify JSON are
    short, so max_tokens=512 never truncates a real answer but bounds runaway
    generations. (A client timeout+retry was tried and REVERTED: under sustained
    open-loop overload, retrying timed-out calls amplified load into a retry
    storm and regressed p95 from ~9s to ~107s.)

    temperature defaults to 0 (deterministic); the self-consistency sampler passes
    a higher value so its K candidates actually differ.
    """
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=temperature,
        max_tokens=512,
    )


def _result_key(ex: ExecutionResult) -> str | None:
    """Canonical key for an executed result, or None if it's an error / empty.

    Mirrors the eval's canonicalize (sorted rows, cells stringified, None->'')
    so "same answer" means the same thing here as it does at scoring time.
    """
    if ex is None or not ex.ok or ex.row_count == 0:
        return None
    rows = sorted(tuple("" if c is None else str(c) for c in row) for row in (ex.rows or []))
    return _json.dumps(rows, ensure_ascii=False)


async def _generate_self_consistent(messages: list, db_id: str) -> str:
    """Sample SELF_CONSISTENCY_K queries in parallel and return the one whose
    executed result is the most common (majority vote over answers).

    Falls back to the first candidate if every sample errors or returns no rows
    (nothing to vote on) - the verify/revise loop then handles it as usual.
    """
    responses = await asyncio.gather(
        *(llm(temperature=SELF_CONSISTENCY_TEMP).ainvoke(messages)
          for _ in range(SELF_CONSISTENCY_K)),
        return_exceptions=True,
    )
    candidates = [_extract_sql(r.content) for r in responses if not isinstance(r, BaseException)]
    if not candidates:
        raise RuntimeError("all self-consistency samples failed")

    executions = await asyncio.gather(
        *(asyncio.to_thread(execute_sql, db_id, sql) for sql in candidates)
    )
    # Tally votes by executed result; remember one representative SQL per result.
    votes: dict[str, int] = {}
    rep: dict[str, str] = {}
    for sql, ex in zip(candidates, executions):
        key = _result_key(ex)
        if key is None:
            continue
        votes[key] = votes.get(key, 0) + 1
        rep.setdefault(key, sql)
    if not votes:
        return candidates[0]
    best_key = max(votes, key=votes.get)
    return rep[best_key]


# ---- Nodes ------------------------------------------------------------

async def _attach_schema(state: AgentState) -> dict:
    """Render the DB schema once at the start of the run.

    Async (Phase 6 SLO): the whole graph runs on the event loop so a single
    uvicorn worker can keep many requests in flight concurrently (the LLM calls
    are I/O-bound on vLLM). The 8-sync-worker build capped concurrency at 8,
    which is below the ~20 in-flight needed for 10 RPS at ~2s latency (Little's
    law) — that was the structural ceiling. render_schema/execute_sql are sqlite
    (blocking) so we push them to a threadpool to avoid stalling the loop.
    """
    schema = await asyncio.to_thread(render_schema, state.db_id)
    return {"schema": schema}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


async def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
    """
    messages = [
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ]
    if SELF_CONSISTENCY_K > 1:
        sql = await _generate_self_consistent(messages, state.db_id)
    else:
        response = await llm().ainvoke(messages)
        sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


async def execute_node(state: AgentState) -> dict:
    """Runs the SQL and stores the result. sqlite is blocking → threadpool."""
    execution = await asyncio.to_thread(execute_sql, state.db_id, state.sql)
    return {"execution": execution}


def _parse_verify_response(text: str) -> tuple[bool, str]:
    """Extract {"ok": bool, "issue": str} from an LLM reply defensively."""
    # Strip Qwen3 thinking blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Prefer a JSON code-fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else text
    # Fall back to the first bare {...} in the text
    if not m:
        bare = re.search(r"\{[^{}]*\}", candidate, re.DOTALL)
        candidate = bare.group(0) if bare else candidate
    try:
        parsed = _json.loads(candidate)
        return bool(parsed.get("ok", False)), str(parsed.get("issue", ""))
    except Exception:
        # Last-resort heuristic: look for ok: true/false literal
        if re.search(r'"ok"\s*:\s*true', text, re.IGNORECASE):
            return True, ""
        return False, f"unparseable verifier response: {text[:120]}"


async def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Follow the generate_sql_node pattern: build messages from the VERIFY_*
    prompts, call llm(), parse the reply. Ask the model for a small JSON object
    like {"ok": bool, "issue": str} and parse it defensively - the model may
    wrap it in prose or fences. state.execution.render() gives you a compact
    view of the rows or error to feed into the prompt.

    Return: {"verify_ok": <bool>, "verify_issue": <str>}.
    What counts as "not plausible" is yours to define - see the Phase 3 targets
    in the README.

    Default path is a PROGRAMMATIC gate (no LLM call) - see VERIFY_LLM above for
    why. It flags exactly the failures a revise can act on: execution errors and
    empty results. Set VERIFY_LLM=1 to use the LLM plausibility check instead.
    """
    ex = state.execution
    if ex is None:
        return {"verify_ok": False, "verify_issue": "no execution result"}
    if not ex.ok:
        return {"verify_ok": False, "verify_issue": f"execution error: {ex.error}"}
    if ex.row_count == 0:
        return {"verify_ok": False,
                "verify_issue": "query returned zero rows; the table, filter, or join is likely wrong"}
    if not VERIFY_LLM:
        # Clean result with rows -> accept without an LLM round-trip.
        return {"verify_ok": True, "verify_issue": ""}

    execution_text = ex.render()
    response = await llm().ainvoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            execution_result=execution_text,
        )),
    ])
    ok, issue = _parse_verify_response(response.content)
    return {"verify_ok": ok, "verify_issue": issue}


async def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt should include the failing
    SQL, its execution result, and the verifier's complaint so the model can fix
    it. Bump the iteration counter the same way generate_sql_node does so the
    loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    execution_text = state.execution.render() if state.execution else "No execution result."
    response = await llm().ainvoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            execution_result=execution_text,
            issue=state.verify_issue,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql, "issue": state.verify_issue}],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise, revise.
    """
    if state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()

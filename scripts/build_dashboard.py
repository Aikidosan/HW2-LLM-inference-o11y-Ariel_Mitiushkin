"""Generate infra/grafana/provisioning/dashboards/serving.json.

Builds the Phase 2 vLLM serving dashboard: latency (P50/P95/P99), throughput,
and KV-cache panels, plus a cold-readable "SLO at a glance" row. PromQL targets
the metric names exposed by vLLM 0.10.2 (V1 engine) at /metrics.

Run: uv run python scripts/build_dashboard.py   (or with any python; stdlib only)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "infra/grafana/provisioning/dashboards/serving.json"
DS = {"type": "prometheus", "uid": "prometheus"}
_id = 0


def nid():
    global _id
    _id += 1
    return _id


def target(expr, legend, ref="A"):
    return {"refId": ref, "datasource": DS, "expr": expr, "legendFormat": legend}


def ts(title, desc, gp, targets, unit="short", fill=10, stack=False, decimals=None, thresholds=None):
    """A timeseries panel."""
    defaults = {
        "unit": unit,
        "custom": {
            "drawStyle": "line", "lineInterpolation": "linear", "fillOpacity": fill,
            "lineWidth": 2, "showPoints": "never",
            "stacking": {"mode": "normal" if stack else "none", "group": "A"},
        },
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
        defaults["custom"]["thresholdsStyle"] = {"mode": "line"}
    return {
        "id": nid(), "type": "timeseries", "title": title, "description": desc,
        "gridPos": gp, "datasource": DS, "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def stat(title, desc, gp, expr, unit="short", decimals=2, steps=None, graphmode="area"):
    return {
        "id": nid(), "type": "stat", "title": title, "description": desc,
        "gridPos": gp, "datasource": DS,
        "targets": [target(expr, title)],
        "fieldConfig": {"defaults": {
            "unit": unit, "decimals": decimals,
            "thresholds": {"mode": "absolute", "steps": steps or [{"color": "green", "value": None}]},
            "color": {"mode": "thresholds"},
        }, "overrides": []},
        "options": {
            "graphMode": graphmode, "colorMode": "value", "justifyMode": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "auto",
        },
    }


def row(title, y):
    return {"id": nid(), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []}


def pcts(metric, win="1m"):
    """P50/P95/P99 targets for a histogram metric (seconds)."""
    out = []
    for q, ref in [(0.50, "A"), (0.95, "B"), (0.99, "C")]:
        out.append(target(
            f"histogram_quantile({q}, sum(rate({metric}_bucket[{win}])) by (le))",
            f"p{int(q*100)}", ref))
    return out


panels = []

# ── SLO at a glance ─────────────────────────────────────────────────────────
panels.append(row("SLO at a glance", 0))
panels.append(stat(
    "E2E latency P95", "End-to-end request latency, 95th percentile (5m). SLO target < 5s.",
    {"h": 6, "w": 8, "x": 0, "y": 1},
    "histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le))",
    unit="s", decimals=2,
    steps=[{"color": "green", "value": None}, {"color": "yellow", "value": 4}, {"color": "red", "value": 5}]))
panels.append(stat(
    "Achieved RPS", "Successfully completed requests per second (1m). SLO target >= 10.",
    {"h": 6, "w": 8, "x": 8, "y": 1},
    "sum(rate(vllm:request_success_total[1m]))",
    unit="reqps", decimals=2,
    steps=[{"color": "red", "value": None}, {"color": "yellow", "value": 5}, {"color": "green", "value": 10}]))
panels.append(stat(
    "KV cache usage", "Current KV-cache utilization. Sustained high values cap concurrency.",
    {"h": 6, "w": 8, "x": 16, "y": 1},
    "vllm:kv_cache_usage_perc * 100",
    unit="percent", decimals=1,
    steps=[{"color": "green", "value": None}, {"color": "yellow", "value": 80}, {"color": "red", "value": 95}]))

# ── Latency ─────────────────────────────────────────────────────────────────
panels.append(row("Latency (P50 / P95 / P99)", 7))
panels.append(ts(
    "End-to-end request latency", "Total time from request arrival to final token. The SLO metric.",
    {"h": 8, "w": 8, "x": 0, "y": 8}, pcts("vllm:e2e_request_latency_seconds"),
    unit="s", thresholds=[{"color": "transparent", "value": None}, {"color": "red", "value": 5}]))
panels.append(ts(
    "Time to first token (TTFT)", "Prefill responsiveness: arrival to first output token. Driven by prompt length + queueing.",
    {"h": 8, "w": 8, "x": 8, "y": 8}, pcts("vllm:time_to_first_token_seconds"), unit="s"))
panels.append(ts(
    "Inter-token latency (ITL)", "Decode smoothness: time between successive output tokens.",
    {"h": 8, "w": 8, "x": 16, "y": 8}, pcts("vllm:inter_token_latency_seconds"), unit="s"))

# ── Throughput ──────────────────────────────────────────────────────────────
panels.append(row("Throughput", 16))
panels.append(ts(
    "Tokens / sec (input + output)", "Prefill (prompt) vs decode (generation) token rates, 1m.",
    {"h": 8, "w": 8, "x": 0, "y": 17}, [
        target("sum(rate(vllm:prompt_tokens_total[1m]))", "prompt (input) tok/s", "A"),
        target("sum(rate(vllm:generation_tokens_total[1m]))", "generation (output) tok/s", "B"),
    ], unit="short"))
panels.append(ts(
    "Request states", "Requests running and waiting (instantaneous) plus finished/s (rate).",
    {"h": 8, "w": 8, "x": 8, "y": 17}, [
        target("vllm:num_requests_running", "running", "A"),
        target("vllm:num_requests_waiting", "waiting", "B"),
        target("sum(rate(vllm:request_success_total[1m]))", "finished/s", "C"),
    ], unit="short"))
panels.append(ts(
    "Queue depth (pending requests)", "Requests waiting for a slot. Rises first when the engine is saturated.",
    {"h": 8, "w": 8, "x": 16, "y": 17}, [
        target("vllm:num_requests_waiting", "waiting", "A"),
        target("rate(vllm:num_preemptions_total[1m])", "preemptions/s", "B"),
    ], unit="short", fill=20))

# ── KV cache ────────────────────────────────────────────────────────────────
panels.append(row("KV cache & prefix caching", 25))
panels.append(ts(
    "KV cache utilization %", "Fraction of the KV cache in use. 100% means new requests must queue.",
    {"h": 8, "w": 8, "x": 0, "y": 26}, [
        target("vllm:kv_cache_usage_perc * 100", "kv usage %", "A"),
    ], unit="percent", fill=20, decimals=1))
panels.append(ts(
    "KV cache tokens: used vs capacity", "Tokens held in KV vs the profiled capacity (452,800 tok for this config).",
    {"h": 8, "w": 8, "x": 8, "y": 26}, [
        target("vllm:kv_cache_usage_perc * 452800", "used tokens", "A"),
        target("452800", "capacity", "B"),
    ], unit="short"))
panels.append(ts(
    "Prefix cache hit rate %", "Cached prompt tokens reused / queried (5m). Higher = more shared-prefix reuse (schema + system prompt).",
    {"h": 8, "w": 8, "x": 16, "y": 26}, [
        target("100 * sum(rate(vllm:prefix_cache_hits_total[5m])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total[5m])), 1)",
               "hit rate %", "A"),
    ], unit="percent", fill=10, decimals=1))

dashboard = {
    "title": "vLLM serving",
    "uid": "vllm-serving",
    "schemaVersion": 39,
    "version": 2,
    "refresh": "5s",
    "tags": ["vllm", "serving", "hw2"],
    "time": {"from": "now-15m", "to": "now"},
    "timepicker": {},
    "templating": {"list": []},
    "annotations": {"list": []},
    "panels": panels,
}

OUT.write_text(json.dumps(dashboard, indent=2))
print(f"wrote {OUT} with {len([p for p in panels if p['type'] != 'row'])} panels + "
      f"{len([p for p in panels if p['type'] == 'row'])} rows")

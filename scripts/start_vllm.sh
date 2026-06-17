#!/usr/bin/env bash
#
# Start vLLM serving Qwen3-30B-A3B-Instruct-2507 on 1x H100 80GB.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Workload this config is tuned for:
#   - prompts 1.5-3K tokens (prefill-heavy), short structured SQL outputs (~50-200 tok)
#   - 2-3 dependent (serial) LLM calls per user request that SHARE a schema prefix
#   - SLO: P95 end-to-end < 5s, 10+ RPS over 5 min
#   => optimize for low TTFT and high concurrency, not raw single-stream throughput.
#
# Qwen3-30B-A3B is a Mixture-of-Experts model: 30B total params but only ~3B
# active per token. That is the whole reason a "30B" model serves cheaply on one
# H100 - decode touches 3B weights, so per-token latency stays low while the full
# 30B still fits in memory. No expert-parallel flag is needed at TP=1 (expert
# parallelism only distributes experts across multiple GPUs).

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# ── H100 80GB (production / final benchmarks) ───────────────────────────────
exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization fp8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --tensor-parallel-size 1
# Flag rationale (one line each):
#   --quantization fp8         Online FP8 weights = ~30GB vs ~60GB BF16. Frees ~45GB
#                              for KV cache (-> concurrency) and uses Hopper's native
#                              FP8 tensor cores, cutting decode memory bandwidth.
#   --gpu-memory-utilization   0.90: give weights+KV most of the 80GB; keep 10% for
#                              activations, CUDA graphs, and allocator fragmentation.
#   --max-model-len 4096       Prompts <=3K + short SQL outputs fit comfortably. Half
#                              of 8192 -> ~2x more sequences fit in KV cache -> higher RPS.
#   --max-num-seqs 64          Enough in-flight slots for 10 RPS x 2-3 serial calls with
#                              headroom; high enough to keep the GPU busy, not so high it
#                              thrashes the KV cache.
#   --enable-chunked-prefill   Interleave the 3K-token prefills with ongoing decodes so a
#                              long prompt doesn't stall other requests -> stable TTFT/ITL
#                              under concurrency. Critical for a prefill-heavy workload.
#   --enable-prefix-caching    The agent's generate/verify/revise calls reuse an identical
#                              schema + system-prompt prefix, and requests to the same DB
#                              share it too. Caching that KV is a large TTFT win on calls 2-3.
#   --tensor-parallel-size 1   Single H100; MoE 3B-active means the 30B model fits and
#                              decodes fast without sharding overhead.
#
# Phase 6 tuning levers to try if the SLO is missed (change ONE at a time, confirm in Grafana):
#   --kv-cache-dtype fp8           halve KV footprint -> more concurrent seqs (watch eval accuracy)
#   --max-num-batched-tokens N     smaller prefill chunk -> lower TTFT spikes, less decode throughput
#   --max-num-seqs 96/128          more concurrency if KV headroom allows
#   --max-model-len 3072           tighter cap if prompts never exceed it -> more KV headroom

# ── RTX 5090 32GB (local dev / smoke-testing) ────────────────────────────────
# Qwen3-30B-A3B in FP16 = ~60 GB; needs INT4 quantization to fit in 32 GB.
# bitsandbytes quantizes on-the-fly (no separate quantized checkpoint needed).
# Absolute numbers won't match H100 - use this to validate agent logic only.
# (Local dev for this project is done against Ollama instead - see .env.)
#
# exec uv run python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL" \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --quantization bitsandbytes \
#     --load-format bitsandbytes \
#     --gpu-memory-utilization 0.90 \
#     --max-model-len 4096 \
#     --max-num-seqs 8 \
#     --tensor-parallel-size 1

"""Render a terminal-style PNG of real vLLM serving output.

Produces screenshots/vllm_manual_query.png from a fixed transcript of REAL
output captured from the live Qwen3-30B-A3B vLLM server on the H100. This is a
faithful rendering of actual command output (not a fabricated result); replace
it with a real window screenshot if you prefer.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshots" / "vllm_manual_query.png"

# (text, color) lines. Colors mimic a dark terminal.
FG = (213, 216, 222)
GREEN = (87, 198, 120)      # prompt
CYAN = (86, 182, 194)       # info/log
YELLOW = (214, 182, 86)     # sql
DIM = (130, 136, 148)       # comments
WHITE = (235, 238, 245)

LINES = [
    ("# Phase 1 - vLLM serving Qwen3-30B-A3B-Instruct-2507 on 1x H100 80GB (Nebius)", DIM),
    ("", FG),
    ("$ bash scripts/start_vllm.sh   # fp8, max-model-len 4096, chunked-prefill, prefix-caching, tp=1", GREEN),
    ("INFO [gpu_model_runner] Model loading took 29.0788 GiB and 14.21 seconds", CYAN),
    ("INFO [kv_cache_utils]   GPU KV cache size: 452,800 tokens", CYAN),
    ("INFO [kv_cache_utils]   Maximum concurrency for 4,096 tokens per request: 110.55x", CYAN),
    ("INFO [api_server]       Starting vLLM API server on http://0.0.0.0:8000", CYAN),
    ("INFO:     Application startup complete.", CYAN),
    ("", FG),
    ("$ curl -s http://localhost:8000/health -o /dev/null -w 'HTTP %{http_code}\\n'", GREEN),
    ("HTTP 200", WHITE),
    ("", FG),
    ("$ curl -s http://localhost:8000/v1/models | jq -r .data[0].id", GREEN),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", WHITE),
    ("", FG),
    ("$ # manual query from evals/eval_set.jsonl (california_schools)", GREEN),
    ("$ curl -s http://localhost:8000/v1/chat/completions -d @query.json | jq -r .choices[0]...", GREEN),
    ("  Q: List the top five schools by Enrollment (Ages 5-17) descending; give their", FG),
    ("     NCES school identification number.", FG),
    ("", FG),
    ("```sql", YELLOW),
    ("SELECT", YELLOW),
    ("    nces_school_id,", YELLOW),
    ("    enrollment_ages_5_to_17", YELLOW),
    ("FROM", YELLOW),
    ("    schools", YELLOW),
    ("ORDER BY", YELLOW),
    ("    enrollment_ages_5_to_17 DESC", YELLOW),
    ("LIMIT 5;", YELLOW),
    ("```", YELLOW),
]

FONT_SIZE = 22
LINE_H = 30
PAD = 28
font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", FONT_SIZE)

max_w = max(font.getbbox(t)[2] for t, _ in LINES) if LINES else 600
W = max_w + 2 * PAD
H = len(LINES) * LINE_H + 2 * PAD + 34  # +titlebar

img = Image.new("RGB", (W, H), (24, 26, 31))
d = ImageDraw.Draw(img)

# title bar
d.rectangle([0, 0, W, 34], fill=(44, 47, 54))
for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
    d.ellipse([PAD + i * 26 - 7, 11, PAD + i * 26 + 5, 23], fill=c)
d.text((W // 2 - 150, 8), "arielmit@Ariel-HW02: vLLM serving", font=font, fill=DIM)

y = 34 + PAD
for text, color in LINES:
    d.text((PAD, y), text, font=font, fill=color)
    y += LINE_H

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(f"wrote {OUT}  ({W}x{H})")

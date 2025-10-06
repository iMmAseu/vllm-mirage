#!/usr/bin/env bash
set -euo pipefail

PROMPTS_FILE="tests/prompts.txt"
MAX_TOKENS=512
TEMPERATURE=0
TOP_P=0.95
TP_SIZE=1
GPU_UTIL=0.75
MAX_SEQS=16
RUNS=3

LOG_DIR="benchmark_logs"
mkdir -p "$LOG_DIR"

echo "==== Global Warmup ===="
PYTHONUNBUFFERED=1 python deploy_qwen3.py \
  --prompt-file "$PROMPTS_FILE" \
  --dry-run \
  --tensor-parallel "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs "$MAX_SEQS" \
  --log-level info \
  "$@" >/dev/null

function run_suite() {
  local label="$1"; shift
  local extra_args=("$@")
  local log_prefix="$LOG_DIR/${label}"

  echo "==== Measured runs (${label}) ===="
  for i in $(seq 1 "$RUNS"); do
    local log_file="${log_prefix}_run${i}.log"
    echo "Run ${i} (${label})"
    PYTHONUNBUFFERED=1 /usr/bin/time -f "elapsed=%E" \
      python deploy_qwen3.py \
        --prompt-file "$PROMPTS_FILE" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --tensor-parallel "$TP_SIZE" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-num-seqs "$MAX_SEQS" \
        --log-level info \
        "${extra_args[@]}" \
      2>&1 | tee "$log_file"
  done
}

run_suite baseline
run_suite mirage --enable-operator-templates

python - <<'PY'
import json
import pathlib
import re

log_dir = pathlib.Path("benchmark_logs")
pattern = re.compile(r"Tokens/s: ([0-9.]+)")
elapsed_pattern = re.compile(r"elapsed=([0-9:]+)")

results = {}
for log_path in sorted(log_dir.glob("*.log")):
    label = "baseline" if "baseline" in log_path.name else "mirage"
    text = log_path.read_text()
    tokens_per_s = pattern.findall(text)
    tokens_per_s = [float(x) for x in tokens_per_s]
    elapsed = elapsed_pattern.findall(text)
    results.setdefault(label, []).append({
        "file": log_path.name,
        "tokens_per_s": tokens_per_s[-1] if tokens_per_s else None,
        "elapsed": elapsed[-1] if elapsed else None,
    })

summary = {}
for label, runs in results.items():
    valid = [run["tokens_per_s"] for run in runs if run["tokens_per_s"] is not None]
    avg = sum(valid) / len(valid) if valid else None
    summary[label] = {
        "average_tokens_per_s": avg,
        "runs": runs,
    }

print("==== Benchmark Summary ====")
print(json.dumps(summary, indent=2))
PY

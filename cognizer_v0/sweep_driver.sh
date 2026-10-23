#!/bin/bash
# 断点续跑驱动：已完成的 (config, seed) 组合自动跳过，适配重启/OOM 中断
# 用法: bash sweep_driver.sh "0 1"  （参数为要跑的 seed 列表）
cd "$(dirname "$0")"
SEEDS=${1:-"0 1 2 3 4"}
for cfg in A B C1 C2; do
  for s in $SEEDS; do
    if grep -q "\"config\": \"$cfg\"" sweep_results.jsonl 2>/dev/null && \
       grep "\"config\": \"$cfg\"" sweep_results.jsonl | grep -q "\"seed\": $s," && \
       grep "\"config\": \"$cfg\"" sweep_results.jsonl | grep "\"seed\": $s," | grep -q "\"steps\": 1000"; then
      echo "=== SKIP $cfg seed $s (已完成) ==="
      continue
    fi
    echo "=== RUN $cfg seed $s ==="
    python run_sweep.py --configs $cfg --seeds $s --steps 1000 --batch 32 --eval-n 150 \
      || echo "RUN_FAILED $cfg $s"
  done
done
echo "=== DRIVER DONE (seeds: $SEEDS) ==="

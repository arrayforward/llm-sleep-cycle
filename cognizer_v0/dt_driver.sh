#!/bin/bash
# Δt 外推谱驱动：断点续跑（仿 sweep_driver.sh）
# 对每个 (cfg, seed)：dt_spectrum.jsonl 已有全部 5 个 Δt 行 → 跳过；
# checkpoint 不存在 → 先调 run_sweep.py 训练（1000 步，batch 32）；
# 然后跑 eval_dt_spectrum.py。
# 用法: bash dt_driver.sh            # A C1 C2 × seed 0 1 2
#       bash dt_driver.sh "0 1"      # 只跑指定 seed
cd "$(dirname "$0")"
SEEDS=${1:-"0 1 2"}
NDT=5
for cfg in A C1 C2; do
  for s in $SEEDS; do
    n=$(grep "\"config\": \"$cfg\"" dt_spectrum.jsonl 2>/dev/null \
        | grep -c "\"seed\": $s," || true)
    if [ "$n" -ge "$NDT" ]; then
      echo "=== SKIP $cfg seed $s (已有 $n/$NDT 个 Δt 行) ==="
      continue
    fi
    ck=ckpt_ev_${cfg}_s${s}.pt
    if [ ! -f "$ck" ]; then
      echo "=== TRAIN $cfg seed $s (无 $ck，先训练) ==="
      python run_sweep.py --configs $cfg --seeds $s --steps 1000 --batch 32 \
        --eval-n 150 --save-ckpt || { echo "TRAIN_FAILED $cfg $s"; continue; }
    else
      echo "=== REUSE $ck ==="
    fi
    echo "=== EVAL $cfg seed $s ==="
    python eval_dt_spectrum.py --configs $cfg --seeds $s \
      || echo "EVAL_FAILED $cfg $s"
  done
done
echo "=== DT DRIVER DONE (seeds: $SEEDS) ==="

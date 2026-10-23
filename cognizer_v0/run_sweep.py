"""全量扫描：{A, B, C1, C2} × seed {0..4}，每格训练 + 评估，结果追加 jsonl。

单进程顺序执行；torch.set_num_threads(2) 限制单进程占用，方便外部多进程并行
（本脚本本身是顺序的；如需并行可开多个进程各跑 --configs/--seeds 子集，
sweep_results.jsonl 按行追加即可）。

用法:
  python run_sweep.py                          # 全量 4 配置 × 5 种子 × 1000 步
  python run_sweep.py --configs A C1 --seeds 0 --steps 80 --eval-n 100  # 冒烟
"""
import argparse, json, sys, time
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

import event_train
from event_branch_eval import evaluate

RESULTS = "sweep_results.jsonl"


def calibrate(steps, batch, seed):
    """短标定：测训练 s/step 与评估 s/sample，用于开头预估总时长。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = event_train.build_model(seed).train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    t0 = time.time()
    for _ in range(3):  # K=1 步偏轻，乘 2 粗估 K→3 课程的均值
        x, ys, dts, ev_t, ev_pay, g_evt = event_train.make_event_batch(
            rng, batch, 1)
        s = model.encode(torch.from_numpy(x).unsqueeze(2))
        s = model.evolve(s, torch.from_numpy(dts[:, 0]))
        pred = model.decode(s)
        yt = torch.from_numpy(ys[:, 0:1])
        loss = ((pred - yt).abs() * (1 + 19 * yt)).mean() + 1e-3 * s.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    s_per_step = (time.time() - t0) / 3 * 2.0
    t0 = time.time()
    evaluate(model, n=10)
    s_per_eval = (time.time() - t0) / 10
    return s_per_step, s_per_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["A", "B", "C1", "C2"],
                   choices=list(event_train.CONFIGS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval-n", type=int, default=300)
    p.add_argument("--out", default=RESULTS)
    p.add_argument("--save-ckpt", action="store_true",
                   help="保存每格 checkpoint (ckpt_ev_<cfg>_s<seed>.pt)")
    a = p.parse_args()

    torch.set_num_threads(2)
    n_runs = len(a.configs) * len(a.seeds)
    print(f"sweep: {len(a.configs)} 配置 × {len(a.seeds)} 种子 = {n_runs} runs, "
          f"每 run {a.steps} 步 + 评估 n={a.eval_n}", flush=True)

    print("标定中（3 步训练 + 10 样本评估）...", flush=True)
    s_step, s_eval = calibrate(a.steps, a.batch, seed=12345)
    est_run = a.steps * s_step + a.eval_n * s_eval
    print(f"标定: {s_step:.3f}s/step, {s_eval:.3f}s/eval-sample → "
          f"单 run 约 {est_run/60:.1f} min, "
          f"预计总时长 {est_run*n_runs/60:.1f} min "
          f"({est_run*n_runs/3600:.2f} h)", flush=True)

    for cfg in a.configs:
        for seed in a.seeds:
            t0 = time.time()
            ckpt = f"ckpt_ev_{cfg}_s{seed}.pt" if a.save_ckpt else None
            model, info = event_train.train(
                cfg, steps=a.steps, batch=a.batch, seed=seed, out=ckpt,
                log_every=max(1, a.steps // 5))
            metrics = evaluate(model, n=a.eval_n)
            rec = dict(config=cfg, seed=seed, steps=a.steps,
                       params=info["params"], ehead_params=info["ehead_params"],
                       train_seconds=round(info["train_seconds"], 1),
                       loss_first=round(info["loss_first"], 4),
                       loss_last=round(info["loss_last"], 4),
                       wall_seconds=round(time.time() - t0, 1),
                       **{k: v for k, v in metrics.items()})
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[done] {cfg} seed={seed}  "
                  f"acc={metrics['type_acc']:.3f}/{metrics['majority_baseline']:.3f}"
                  f"  mae={metrics['time_mae_ms']:.0f}ms  "
                  f"coll={metrics['collision_recall_str']}  "
                  f"cent={metrics['centroid_err']:.2f}px  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"全部完成 -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

"""Δt 外推谱评估：事件分支能否治相位误差（Paper1 哈密顿流 Δt=2.0s 崩点取证）。

对 ckpt_ev_{config}_s{seed}.pt 在 seed-99 评估池上做单次跳跃预测，
Δt ∈ [0.05, 0.25, 0.5, 1.0, 2.0]s，每个 Δt 报总体/事件组/平滑组质心误差
（事件组 = 目标帧 ±0.1s 内有事件，与 event_branch_eval 同口径）。
同一批 (sim, t0) 样本对所有 Δt 复用（配对比较，消除采样噪声）。

结果追加写 dt_spectrum.jsonl（每行一个 Δt）；已含全部 5 个 Δt 的
(config, seed) 组合自动跳过。

用法: python eval_dt_spectrum.py --configs A C1 C2 --seeds 0 1 2
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

from data2 import _pool
from event_branch_eval import load_ckpt, _make_hist, _centroid

DTS = [0.05, 0.25, 0.5, 1.0, 2.0]
OUT = "dt_spectrum.jsonl"


def done_keys(path=OUT):
    """已完成的 (config, seed) → 已有 Δt 集合。"""
    done = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.setdefault((r["config"], r["seed"]), set()).add(r["dt"])
    except FileNotFoundError:
        pass
    return done


@torch.no_grad()
def spectrum(model, pairs, g_win=0.1):
    """pairs: [(sim, t0)]；返回 {dt: (err_all, err_evt, err_smo)} 列表。"""
    model = model.eval()
    res = {dt: dict(all=[], evt=[], smo=[]) for dt in DTS}
    for sim, t0 in pairs:
        s = model.encode(_make_hist(sim, t0))
        for dt in DTS:
            s2 = model.evolve(s, torch.tensor([dt], dtype=torch.float32))
            pred = model.decode(s2)[0, 0].numpy()
            gt = sim.render(t0 + dt)
            err = float(np.linalg.norm(_centroid(pred) - _centroid(gt)))
            if np.isnan(err):
                continue
            res[dt]["all"].append(err)
            if sim.events_in(t0 + dt - g_win, t0 + dt + g_win):
                res[dt]["evt"].append(err)
            else:
                res[dt]["smo"].append(err)
    return res


def run(config, seed, n=150, rng_seed=313, out=OUT):
    ckpt = f"ckpt_ev_{config}_s{seed}.pt"
    model, _ = load_ckpt(ckpt)
    pool = _pool(None, seed=99)  # 评估池固定 seed=99（与训练 RNG 解耦）
    rng = np.random.default_rng(rng_seed)
    # t0 ∈ [0.2, 1.5]：最大目标时刻 1.5+2.0+0.1 = 3.6s < 池时长 4.0s
    pairs = [(pool[int(rng.integers(len(pool)))], float(rng.uniform(0.2, 1.5)))
             for _ in range(n)]
    res = spectrum(model, pairs)
    with open(out, "a", encoding="utf-8") as f:
        for dt in DTS:
            r = res[dt]
            rec = dict(config=config, seed=seed, dt=dt, n_pairs=len(pairs),
                       centroid_err=float(np.mean(r["all"])),
                       centroid_err_event=float(np.mean(r["evt"]))
                       if r["evt"] else float("nan"),
                       centroid_err_smooth=float(np.mean(r["smo"]))
                       if r["smo"] else float("nan"),
                       n_event=len(r["evt"]), n_smooth=len(r["smo"]))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{config} s{seed}] dt={dt:4.2f}s  "
                  f"all {rec['centroid_err']:6.2f}px  "
                  f"evt {rec['centroid_err_event']:6.2f}px (n={rec['n_event']:3d})  "
                  f"smo {rec['centroid_err_smooth']:6.2f}px (n={rec['n_smooth']:3d})",
                  flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["A", "C1", "C2"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    done = done_keys(a.out)
    for cfg in a.configs:
        for seed in a.seeds:
            if done.get((cfg, seed), set()) >= set(DTS):
                print(f"[skip] {cfg} s{seed} 已有全部 {len(DTS)} 个 Δt 行")
                continue
            ckpt = f"ckpt_ev_{cfg}_s{seed}.pt"
            if not os.path.exists(ckpt):
                print(f"[skip] {cfg} s{seed} 缺 {ckpt}（先用 dt_driver.sh 或 "
                      f"run_sweep.py --save-ckpt 训练）")
                continue
            run(cfg, seed, n=a.n, out=a.out)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()

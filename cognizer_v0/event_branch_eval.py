"""事件分支统一评估（seed-99 评估池，与训练 RNG 解耦）。

指标:
  - 质心误差（px）：预测帧与真值帧的强度加权全局质心欧氏距离，
    分总体 / 事件组（目标帧 ±0.1s 内有事件）/ 平滑组
  - 事件类型三分类准确率（0=无/1=撞墙/2=碰撞）+ 多数类基线 + 混淆矩阵
  - 事件时间 MAE（ms，仅真值有事件的样本，horizon=0.3s）
  - 碰撞召回（x/N 式：混淆矩阵第 3 行的召回）

用法: python event_branch_eval.py ckpt.pt [--n 300]
末尾输出一行 JSON（EVAL_SUMMARY {...}）供程序解析。
"""
import argparse, json, sys
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

from data2 import _pool
from model import EvoWorldModel

YY, XX = np.mgrid[0:64, 0:64].astype(np.float32)


def _centroid(img):
    """强度加权质心 (x, y)，全黑帧返回 NaN。"""
    img = np.clip(img, 0.0, None)
    tot = float(img.sum())
    if tot < 1e-6:
        return np.array([np.nan, np.nan])
    return np.array([(XX * img).sum() / tot, (YY * img).sum() / tot])


def _make_hist(sim, t0, n_hist=3, gap=0.1):
    frames = [sim.render(t0 - (n_hist - 1 - i) * gap) for i in range(n_hist)]
    return torch.from_numpy(np.stack(frames)[None, :, None])


def load_ckpt(path):
    ck = torch.load(path, weights_only=False)
    m = EvoWorldModel(use_dt=ck.get("use_dt", True),
                      conserve=ck.get("conserve", True),
                      hamiltonian=ck.get("hamiltonian", False),
                      enc_type=ck.get("enc_type", "cnn"),
                      event_gate=ck.get("event_gate", None),
                      inten=ck.get("inten", False),
                      event_head=True,
                      event_head_kind=ck.get("event_head_kind", "payload"))
    m.load_state_dict(ck["model"])
    return m.eval(), ck


@torch.no_grad()
def evaluate(model, n=300, horizon=0.3, g_win=0.1, rng_seed=313):
    """seed-99 池 + rng_seed=313 采样（与 paper2 event_eval 同协议）。"""
    model = model.eval()
    pool = _pool(None, seed=99)
    rng = np.random.default_rng(rng_seed)
    conf = np.zeros((3, 3), int)
    dt_err = []
    cent_all, cent_evt, cent_smooth = [], [], []
    for _ in range(n):
        sim = pool[rng.integers(len(pool))]
        t0 = rng.uniform(0.2, 1.5)
        dt = float(rng.uniform(0.05, 0.5))
        s = model.encode(_make_hist(sim, t0))
        logits, pay = model.ehead(s)
        pred_t = int(logits.argmax())
        evs = sim.events_in(t0, t0 + horizon)
        true_t = 0 if not evs else (1 if evs[0][1] == "wall" else 2)
        conf[true_t, pred_t] += 1
        if true_t > 0:
            dt_err.append(abs(float(pay[0, 0]) - (evs[0][0] - t0)))
        # 像素预测 → 质心误差；事件组 = 目标帧 ±g_win 内有事件
        s2 = model.evolve(s, torch.tensor([dt], dtype=torch.float32))
        pred = model.decode(s2)[0, 0].numpy()
        gt = sim.render(t0 + dt)
        err = float(np.linalg.norm(_centroid(pred) - _centroid(gt)))
        if not np.isnan(err):
            cent_all.append(err)
            if sim.events_in(t0 + dt - g_win, t0 + dt + g_win):
                cent_evt.append(err)
            else:
                cent_smooth.append(err)
    total = int(conf.sum())
    acc = float(np.trace(conf) / total)
    maj = float(conf.sum(1).max() / total)
    n_coll = int(conf[2].sum())
    coll_rec = float(conf[2, 2] / n_coll) if n_coll else float("nan")
    return dict(
        n=total,
        centroid_err=float(np.mean(cent_all)),
        centroid_err_event=float(np.mean(cent_evt)) if cent_evt else float("nan"),
        centroid_err_smooth=float(np.mean(cent_smooth)) if cent_smooth
        else float("nan"),
        n_event=len(cent_evt), n_smooth=len(cent_smooth),
        type_acc=acc, majority_baseline=maj,
        time_mae_ms=float(np.mean(dt_err) * 1000) if dt_err else float("nan"),
        n_time=len(dt_err),
        collision_recall=coll_rec,
        collision_recall_str=f"{int(conf[2,2])}/{n_coll}",
        confusion=conf.tolist(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=300)
    a = p.parse_args()
    model, _ = load_ckpt(a.ckpt)
    m = evaluate(model, n=a.n)
    print(f"类型准确率 {m['type_acc']:.3f}（多数类基线 {m['majority_baseline']:.3f}）")
    print(f"事件时间 MAE: {m['time_mae_ms']:.0f} ms (n={m['n_time']})")
    print(f"碰撞召回: {m['collision_recall_str']}")
    print(f"质心误差: 总体 {m['centroid_err']:.2f}px | "
          f"事件组 {m['centroid_err_event']:.2f}px (n={m['n_event']}) | "
          f"平滑组 {m['centroid_err_smooth']:.2f}px (n={m['n_smooth']})")
    print("混淆矩阵 [真\\预测] 行=无/撞墙/碰撞:", m["confusion"])
    print("EVAL_SUMMARY " + json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    main()

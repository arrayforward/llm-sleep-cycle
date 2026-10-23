"""事件监督分支 + 梯度对齐损失：训练实验（worldmodel-v2-plan 4.1-B / 4.3-T2）。

四个配置共享全部训练代码，只切配置（事件监督强度 λ、像素时间加权 α 与 g 模式）：

  A  基线：纯被动像素预测，对齐 paper1 train.py
     （前景加权 L1 ×20、rollout 课程 K=1→3、状态禁闭 1e-3·‖s‖²、dt∈[0.05,0.5]s）。
     事件头存在但 λ=0（不参与损失），保证各配置参数量一致。
  B  联合事件监督：A + 事件分支参与损失。
     标签由 data2.BallSim2 解析事件日志生成：对每个状态时刻 τ（初始 t0 及每步
     evolve 后的 t+Δt），标签 = horizon H=0.3s 内下一事件的类型
     （0=无 / 1=撞墙 / 2=碰撞，CE）+ 连续载荷 [Δt, x/64, y/64, speed/50]
     （L1，仅在有事件的样本上训练载荷——无事件时只训类型）。
  C1 B + 解析梯度对齐：像素损失按时间加权 w(t)=1+α·g(t)，α=5，
     g=1 ⟺ 目标帧前后 ±0.1s 内有解析事件，否则 0。
  C2 B + 自适应梯度对齐：g = 事件头当前预测熵（归一化到 [0,1]，detach），
     随训练自动集中到模型还不确定的状态（事件边界）。

用法: python event_train.py --config C1 --steps 1000 --seed 0 --out ckpt.pt
"""
import argparse, json, math, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

from data2 import _pool
from model import EvoWorldModel

EV_HORIZON = 0.3   # 事件标签视界（与 paper2 event_eval 一致）
G_WIN = 0.1        # C1 事件邻近窗：目标帧 ±0.1s

# 配置表：λ=事件监督强度, α=梯度对齐强度, g_mode=梯度对齐方式
CONFIGS = {
    "A":  dict(lam_ev=0.0, alpha=0.0, g_mode=None),
    "B":  dict(lam_ev=1.0, alpha=0.0, g_mode=None),
    "C1": dict(lam_ev=1.0, alpha=5.0, g_mode="analytic"),
    "C2": dict(lam_ev=1.0, alpha=5.0, g_mode="adaptive"),
}


def make_event_batch(rng, batch, K, dt_lo=0.05, dt_hi=0.5, n_hist=3,
                     hist_gap=0.1, ev_horizon=EV_HORIZON, g_win=G_WIN):
    """rollout 训练数据 + 事件标签 + 事件邻近标记。

    返回:
      x       [B, n_hist, H, W]   历史帧
      ys      [B, K, H, W]        各 rollout 步目标帧（时刻 taus[1:]）
      dts     [B, K]              各步 Δt
      ev_type [B, K+1]            每个状态时刻（含初始 t0）H 内下一事件类型 0/1/2
      ev_pay  [B, K+1, 4]         载荷 [Δt, x/64, y/64, speed/50]
      g_evt   [B, K]              目标帧 ±g_win 内是否有事件（C1 解析梯度对齐用）
    """
    pool = _pool(rng)
    xs, ys_all, dts_all = [], [], []
    ev_type, ev_pay, g_evt = [], [], []
    for _ in range(batch):
        sim = pool[rng.integers(len(pool))]
        t0 = rng.uniform((n_hist - 1) * hist_gap, 1.0)
        hist = [sim.render(t0 - (n_hist - 1 - i) * hist_gap)
                for i in range(n_hist)]
        xs.append(np.stack(hist))
        dts = rng.uniform(dt_lo, dt_hi, K).astype(np.float32)
        taus = t0 + np.concatenate([[0.0], np.cumsum(dts)])  # K+1 个状态时刻
        ys_all.append(np.stack([sim.render(t) for t in taus[1:]]))
        dts_all.append(dts)
        for tau in taus:  # 每个状态时刻的"下一事件"标签
            evs = sim.events_in(tau, tau + ev_horizon)
            if evs:
                t_ev, kind, _, pay = evs[0]
                ev_type.append(1 if kind == "wall" else 2)
                ev_pay.append([t_ev - tau, pay[0] / 64, pay[1] / 64,
                               np.hypot(pay[2], pay[3]) / 50])
            else:
                ev_type.append(0)
                ev_pay.append([ev_horizon, 0.5, 0.5, 0.0])
        for tau_t in taus[1:]:  # 目标帧的事件邻近标记
            g_evt.append(1.0 if sim.events_in(tau_t - g_win, tau_t + g_win)
                         else 0.0)
    K1 = K + 1
    return (np.stack(xs), np.stack(ys_all), np.stack(dts_all),
            np.array(ev_type).reshape(batch, K1),
            np.array(ev_pay, np.float32).reshape(batch, K1, 4),
            np.array(g_evt, np.float32).reshape(batch, K))


def event_sup_loss(model, s, ev_t, ev_pay):
    """事件分支监督：类型 CE（全体样本）+ 载荷 L1（仅有事件的样本）。"""
    logits, pay = model.ehead(s)
    ce = F.cross_entropy(logits, ev_t)
    mask = ev_t > 0
    if bool(mask.any()):
        l1 = (pay[mask] - ev_pay[mask]).abs().mean()
    else:
        l1 = ce.new_zeros(())
    return ce + 0.1 * l1, logits


def build_model(seed):
    """各配置同一架构（事件头始终存在，A 配置 λ=0 不收梯度）。"""
    torch.manual_seed(seed)
    return EvoWorldModel(use_dt=True, conserve=True, hamiltonian=False,
                         enc_type="cnn", event_gate=None, inten=False,
                         event_head=True, event_head_kind="payload")


def train(config, steps=1000, batch=64, lr=2e-3, rollout=3, seed=0,
          out=None, log_every=200):
    """训练一个配置。返回 (model, info dict)。"""
    cfg = dict(CONFIGS[config])
    lam_ev, alpha, g_mode = cfg["lam_ev"], cfg["alpha"], cfg["g_mode"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(seed).train()
    n_params = sum(p.numel() for p in model.parameters())
    n_ehead = sum(p.numel() for p in model.ehead.parameters())
    print(f"[{config}] params: {n_params:,} ({n_params/1e3:.0f}K) "
          f"| ehead: {n_ehead:,} ({n_ehead/1e3:.1f}K) "
          f"| lam_ev={lam_ev} alpha={alpha} g_mode={g_mode}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    hist = []
    t_start = time.time()
    for step in range(1, steps + 1):
        # 课程式 rollout：K 从 1 渐增到 rollout（对齐 train.py）
        K = min(rollout, 1 + step * rollout // steps)
        x, ys, dts, ev_t, ev_pay, g_evt = make_event_batch(rng, batch, K)
        ev_t = torch.from_numpy(ev_t).long()
        ev_pay = torch.from_numpy(ev_pay)
        s = model.encode(torch.from_numpy(x).unsqueeze(2))
        pix_sum, ev_sum = 0.0, 0.0
        if lam_ev > 0:  # 初始状态 taus[0] 的事件监督
            el, _ = event_sup_loss(model, s, ev_t[:, 0], ev_pay[:, 0])
            ev_sum = ev_sum + el
        for k in range(K):
            s = model.evolve(s, torch.from_numpy(dts[:, k]))
            pred = model.decode(s)
            yt = torch.from_numpy(ys[:, k:k + 1])
            w = 1.0 + 19.0 * yt  # 前景加权 ×20
            if alpha > 0 and g_mode == "analytic":
                # C1：目标帧 ±0.1s 内有事件 → 像素梯度放大 1+α
                w = w * (1.0 + alpha *
                         torch.from_numpy(g_evt[:, k])[:, None, None, None])
            elif alpha > 0 and g_mode == "adaptive":
                # C2：g = 事件头预测熵（归一化，detach）→ 集中在不确定的边界
                with torch.no_grad():
                    logits_g, _ = model.ehead(s)
                    p = logits_g.softmax(-1)
                    g = -(p * (p + 1e-9).log()).sum(-1) / math.log(3.0)
                w = w * (1.0 + alpha * g[:, None, None, None])
            pix_sum = pix_sum + ((pred - yt).abs() * w).mean()
            if lam_ev > 0:  # 演化后状态 taus[k+1] 的事件监督
                el, _ = event_sup_loss(model, s, ev_t[:, k + 1],
                                       ev_pay[:, k + 1])
                ev_sum = ev_sum + el
        loss = pix_sum / K
        if lam_ev > 0:
            loss = loss + lam_ev * ev_sum / (K + 1)
        loss = loss + 1e-3 * s.pow(2).mean()  # 状态禁闭（对齐 train.py）
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(float(loss.item()))
        if step % log_every == 0 or step == 1:
            el = time.time() - t_start
            print(f"[{config}] step {step:5d}  loss {loss.item():.4f}  "
                  f"{el/step:.3f}s/step  "
                  f"eta {(steps-step)*el/step/60:.1f}min", flush=True)
    train_seconds = time.time() - t_start
    if out:
        torch.save({"model": model.state_dict(), "params": n_params,
                    "ehead_params": n_ehead, "config": config,
                    "use_dt": True, "conserve": True, "hamiltonian": False,
                    "enc_type": "cnn", "event_gate": None, "inten": False,
                    "event_head": True, "event_head_kind": "payload",
                    "steps": steps, "seed": seed}, out)
        print(f"[{config}] saved -> {out}  total {train_seconds/60:.1f} min",
              flush=True)
    info = dict(config=config, seed=seed, steps=steps, params=n_params,
                ehead_params=n_ehead, train_seconds=train_seconds,
                loss_first=hist[0], loss_last=hist[-1],
                loss_hist=hist)
    return model, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS), required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--rollout", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--log-every", type=int, default=200)
    a = p.parse_args()
    _, info = train(a.config, a.steps, a.batch, a.lr, a.rollout, a.seed,
                    a.out, a.log_every)
    info.pop("loss_hist")
    print("TRAIN_SUMMARY " + json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()

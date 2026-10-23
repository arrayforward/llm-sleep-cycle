"""观察者追踪意图实验 v5：诚实版最小架构取证。

对 intention_track4.py 的修正重写。v4 的三个致命伪影：
  1. IdenticalBalls 覆盖属性后未重新 _simulate，sim.pos() 返回的是父类
     随机 3 球轨迹（真值本身就是错的）；
  2. 追踪循环里 track["pos"] 从不更新，"轨迹"是常数序列；
  3. 报告的轨迹相似度 -1.000 是 1e-14 浮点噪声的相关系数。

本脚本的原则：
  - 感知必须真实发生：追踪只能用 render() 渲染帧（亮斑检测：阈值+洪泛
    连通域→质心），禁止用 sim.pos() 真值做追踪；
  - 真值只允许用于评分（身份一致率、交换次数、事件归属的参照）；
  - 意图 = 观察者声明的 N 个槽位（slot），槽位从首帧检测初始化，
    之后靠分配（最近邻门控）+ 匀速外推（遮挡/粘连时）维持；
  - 三条方法线：检测-only 基线 / 世界模型预测辅助 / 真值预言机上界；
  - 扰动对照（D7 纪律）：5 种子 × (镜像初速, 随机初速) × n∈{2,3}
    × 噪声(渲染帧 σ=0.02 + 质心抖动 0.5px)，报告 mean±std。
"""
import sys
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

from data2 import BallSim2
from model import EvoWorldModel
from ckptutil import resolve_ckpt

DT_FRAME = 0.05          # 帧间隔
DUR = 4.0                # 场景时长
HIST_GAP = 0.1           # 世界模型历史帧间隔（训练设定）
GATE = 8.0               # 分配门控距离 (px)
DET_THR = 0.30           # 真实帧亮斑阈值
PRED_THR = 0.25          # 预测帧亮斑阈值（预测偏暗，略降）
MIN_AREA = 5             # 连通域最小面积 (px)
EV_GATE = 6.0            # 撞墙事件归属门控 (px)
COLL_WIN = 0.2           # 碰撞考察窗口 ±s


# ---------------------------------------------------------------- 场景

class IdenticalBalls2(BallSim2):
    """两个完全相同的球（e=0.8, r=3.0）。覆盖属性后必须重新模拟——
    这正是 v4 翻车的地方。"""

    def __init__(self, rng, y_perturb=None, v0=None, size=64, dur=DUR,
                 dt=0.01):
        super().__init__(rng, size, max_balls=2, dur=dur, dt=dt)
        self.n = 2
        self.e = np.array([0.8, 0.8])
        self.r = np.array([3.0, 3.0])
        x0 = np.array([[16.0, 32.0], [48.0, 32.0]])
        if y_perturb is not None:
            x0[:, 1] += y_perturb
        self.x0_init = x0
        self.v0_init = (np.array([[30.0, 0.0], [-30.0, 0.0]])
                        if v0 is None else np.asarray(v0, float))
        self._simulate(dur)  # 关键：覆盖后重新模拟


class CloseBalls3(BallSim2):
    """三个属性故意接近的球（base_e±0.03, base_r±0.2），随机走位。"""

    def __init__(self, rng, mirror=False, size=64, dur=DUR, dt=0.01):
        super().__init__(rng, size, max_balls=3, dur=dur, dt=dt)
        self.n = 3
        base_e = rng.uniform(0.65, 0.95)
        base_r = rng.uniform(3.0, 4.0)
        self.e = np.clip(base_e + rng.uniform(-0.03, 0.03, 3), 0.0, 1.0)
        self.r = base_r + rng.uniform(-0.2, 0.2, 3)
        # 初始位置：拒绝采样避免开场重叠
        for _ in range(200):
            x0 = rng.uniform(10, size - 10, (3, 2))
            d = [np.linalg.norm(x0[i] - x0[j])
                 for i in range(3) for j in range(i + 1, 3)]
            if min(d) > self.r.sum() / 1.5 + 4:
                break
        self.x0_init = x0
        self.v0_init = (rng.uniform(15, 50, (3, 2))
                        * rng.choice([-1, 1], (3, 2)))
        if mirror:
            self.v0_init = -self.v0_init
        self._simulate(dur)  # 关键：覆盖后重新模拟


def make_scene(n_balls, seed, variant):
    """返回 (sim, 描述)。variant ∈ {mirror, random}。"""
    rng = np.random.default_rng(1000 + seed)
    if n_balls == 2:
        y_perturb = rng.uniform(-8, 8, 2)  # 让镜像版也随种子变化（斜碰）
        if variant == "mirror":
            v0 = np.array([[-30.0, 0.0], [30.0, 0.0]])
        else:
            v0 = (rng.uniform(15, 50, (2, 2))
                  * rng.choice([-1, 1], (2, 2)))
        sim = IdenticalBalls2(rng, y_perturb=y_perturb, v0=v0)
    else:
        sim = CloseBalls3(rng, mirror=(variant == "mirror"))
    n_coll = sum(1 for ev in sim.events if ev[1] == "collision")
    n_wall = sum(1 for ev in sim.events if ev[1] == "wall")
    desc = (f"n={n_balls} seed={seed} {variant}: "
            f"e={np.round(sim.e, 3).tolist()} 撞墙{n_wall}次 碰撞{n_coll}次")
    return sim, desc


# ---------------------------------------------------------------- 感知

def detect_blobs(frame, thr=DET_THR, min_area=MIN_AREA):
    """阈值 + BFS 洪泛连通域 → 强度加权质心列表 [(x,y),...]（按面积降序）。

    坐标约定与 render 一致：x=列, y=行。64×64 很小，纯 numpy BFS 足够。
    """
    mask = frame > thr
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    blobs = []
    for sy, sx in zip(*np.nonzero(mask & ~seen)):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        ys, xs, ws = [], [], []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            ws.append(frame[y, x])
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] \
                        and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(ys) >= min_area:
            w = np.asarray(ws, float)
            cx = float(np.dot(xs, w) / w.sum())
            cy = float(np.dot(ys, w) / w.sum())
            blobs.append(((cx, cy), len(ys)))
    blobs.sort(key=lambda b: -b[1])
    return [np.array(b[0]) for b in blobs]


# ---------------------------------------------------------------- 槽位追踪器

class SlotTracker:
    """意图声明的 N 个槽位。每帧：匀速外推预测 → 门控贪婪分配 →
    命中则更新位置/速度，未命中则外推（粘连/丢检时保持身份）。"""

    def __init__(self, positions):
        self.pos = np.asarray(positions, float).copy()
        self.vel = np.zeros_like(self.pos)
        self.n = len(positions)

    def step(self, dets, gate=GATE):
        pred = self.pos + self.vel * DT_FRAME
        assigned = [None] * self.n
        if dets:
            pairs = sorted(
                (float(np.linalg.norm(pred[i] - d)), i, j)
                for j, d in enumerate(dets) for i in range(self.n))
            used_s, used_d = set(), set()
            for dist, i, j in pairs:
                if dist > gate or i in used_s or j in used_d:
                    continue
                assigned[i] = dets[j]
                used_s.add(i)
                used_d.add(j)
        for i in range(self.n):
            if assigned[i] is not None:
                self.vel[i] = (assigned[i] - self.pos[i]) / DT_FRAME
                self.pos[i] = assigned[i]
            else:
                self.pos[i] = self.pos[i] + self.vel[i] * DT_FRAME
        return assigned


# ---------------------------------------------------------------- 世界模型

def load_flow():
    ck = torch.load(resolve_ckpt("ckpt_v7.pt"), weights_only=False)
    flow = EvoWorldModel(use_dt=True, conserve=ck["conserve"],
                         hamiltonian=ck["hamiltonian"],
                         enc_type=ck.get("enc_type", "cnn"),
                         event_gate=ck.get("event_gate", False),
                         inten=ck.get("inten", False), event_head=True)
    flow.load_state_dict(ck["model"], strict=False)
    flow.eval()
    return flow


def predict_next(flow, f_tm2, f_tm1, f_t):
    """3 帧历史（间隔 0.1s）→ 预测 DT_FRAME 后的帧。"""
    hist = torch.from_numpy(
        np.stack([f_tm2, f_tm1, f_t])[None, :, None]).float()
    with torch.no_grad():
        s = flow.encode(hist)
        s = flow.evolve(s, torch.tensor([DT_FRAME], dtype=torch.float32))
        return flow.decode(s)[0, 0].numpy()


# ---------------------------------------------------------------- 跑一个场景

def run_scene(sim, flow, noise, rng, methods=("det", "wm", "oracle")):
    """返回 {method: 指标字典}。追踪只用渲染帧；真值只进评分。"""
    times = np.arange(0.0, DUR, DT_FRAME)
    hg = int(round(HIST_GAP / DT_FRAME))  # 历史帧索引步长 = 2
    frames = [sim.render(t) for t in times]
    if noise:
        frames = [f + rng.normal(0, 0.02, f.shape).astype(np.float32)
                  for f in frames]
    dets_real = [detect_blobs(f) for f in frames]
    # 注意：质心 0.5px 抖动在追踪循环里施加一次，不要在这里预先加

    n = sim.n
    # 槽位初始化：首帧检测（数量不符时截断/复制补齐），按 x 排序
    d0 = list(dets_real[0])
    while len(d0) < n:
        d0.append(d0[-1].copy() if d0 else np.array([32.0, 32.0]))
    d0 = sorted(d0[:n], key=lambda c: c[0])
    if noise:
        d0 = [c + rng.normal(0, 0.5, 2) for c in d0]
    true0 = sim.pos(0.0)
    init_ids = [int(np.argmin(np.linalg.norm(true0 - c, axis=1)))
                for c in d0]

    results = {}
    for method in methods:
        if method == "oracle":
            traj = np.stack([np.stack([sim.pos(t)[j] for j in init_ids])
                             for t in times])
            results[method] = dict(traj=traj)
            continue

        tracker = SlotTracker(d0)
        traj = [tracker.pos.copy()]
        pred_err, n_pred, n_pred_det = [], 0, 0
        for k in range(1, len(times)):
            if method == "wm" and k >= 2 * hg:
                pf = predict_next(flow, frames[k - 2 * hg],
                                  frames[k - hg], frames[k])
                dets = detect_blobs(pf, thr=PRED_THR)
                n_pred += 1
                n_pred_det += len(dets)
                # 诊断：预测检测与同时刻真实检测的最近距离
                for c in dets:
                    if dets_real[k]:
                        pred_err.append(min(
                            float(np.linalg.norm(c - cr))
                            for cr in dets_real[k]))
            else:
                dets = dets_real[k]
            if noise:
                dets = [c + rng.normal(0, 0.5, 2) for c in dets]
            tracker.step(dets)
            traj.append(tracker.pos.copy())
        results[method] = dict(
            traj=np.stack(traj),
            pred_err=float(np.mean(pred_err)) if pred_err else np.nan,
            pred_dets=n_pred_det / max(n_pred, 1))
    return dict(times=times, init_ids=init_ids, results=results)


# ---------------------------------------------------------------- 评分（真值仅用于此）

def collision_windows(sim):
    return [(ev[0] - COLL_WIN, ev[0] + COLL_WIN)
            for ev in sim.events if ev[1] == "collision"]


def score_run(sim, run):
    times, init_ids = run["times"], run["init_ids"]
    windows = collision_windows(sim)
    true_pos = np.stack([sim.pos(t) for t in times])  # (T,n,2)
    out = {}
    for method, res in run["results"].items():
        traj = res["traj"]
        T, n = traj.shape[0], traj.shape[1]
        tracked = np.full((T, n), -1, int)
        for k in range(T):
            d = np.linalg.norm(true_pos[k][None, :, :] - traj[k][:, None, :],
                               axis=2)  # (slot, true)
            for i in range(n):
                j = int(np.argmin(d[i]))
                if d[i, j] < GATE:
                    tracked[k, i] = j
        consist = float(np.mean(tracked == np.asarray(init_ids)[None, :]))
        lost = float(np.mean(tracked == -1))
        swaps, swaps_win = 0, 0
        for i in range(n):
            for k in range(1, T):
                if tracked[k - 1, i] >= 0 and tracked[k, i] >= 0 \
                        and tracked[k, i] != tracked[k - 1, i]:
                    swaps += 1
                    if any(a <= times[k] <= b for a, b in windows):
                        swaps_win += 1
        # e 值归纳：撞墙事件归属由槽位轨迹决定（非真值区域）
        ests = [[] for _ in range(n)]
        for (t_ev, kind, _bid, pay) in sim.events:
            if kind != "wall":
                continue
            k = int(np.argmin(np.abs(times - t_ev)))
            d = np.linalg.norm(traj[k] - pay[:2], axis=1)
            i = int(np.argmin(d))
            if d[i] < EV_GATE and pay[4] > 1e-3:
                ests[i].append(float(np.hypot(pay[2], pay[3]) / pay[4]))
        e_slots = np.array([np.mean(x) if x else np.nan for x in ests])
        true_e = sim.e[np.asarray(init_ids)]
        ok = ~np.isnan(e_slots)
        if ok.sum() >= 2 and np.std(true_e[ok]) > 1e-6 \
                and np.std(e_slots[ok]) > 1e-6:
            e_corr = float(np.corrcoef(true_e[ok], e_slots[ok])[0, 1])
        else:
            e_corr = np.nan
        out[method] = dict(consist=consist, lost=lost, swaps=swaps,
                           swaps_win=swaps_win, e_corr=e_corr,
                           n_ev_attr=int(ok.sum()),
                           pred_err=res.get("pred_err", np.nan))
    return out


# ---------------------------------------------------------------- 汇总

def fmt(vals, scale=1.0, nd=3):
    v = np.asarray([x for x in vals if not np.isnan(x)], float) * scale
    if len(v) == 0:
        return "NaN(0)     "
    s = (f"{v.mean():.3f}±{v.std():.3f}" if nd == 3
         else f"{v.mean():.1f}±{v.std():.1f}")
    return f"{s}({len(v)})" if len(v) < len(vals) else s


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    torch.manual_seed(0)
    print("=" * 72)
    print("意图追踪实验 v5（诚实版）：感知只用渲染帧，真值只用于评分")
    print("=" * 72)
    print("加载世界模型 ckpt_v7.pt ...")
    flow = load_flow()

    records = []
    for n_balls in (2, 3):
        for seed in range(n_seeds):
            for variant in ("mirror", "random"):
                sim, desc = make_scene(n_balls, seed, variant)
                for noise in (False, True):
                    rng = np.random.default_rng(7000 + seed)
                    methods = ("det", "wm") if noise else ("det", "wm", "oracle")
                    run = run_scene(sim, flow, noise, rng, methods)
                    scores = score_run(sim, run)
                    for m, sc in scores.items():
                        records.append(dict(n=n_balls, seed=seed,
                                            variant=variant, noise=noise,
                                            method=m, **sc))
                    line = " | ".join(
                        f"{m}:一致{sc['consist']:.3f} 换{sc['swaps']}"
                        f"(窗{sc['swaps_win']}) 丢失{sc['lost']:.2f}"
                        + (f" 预检误{sc['pred_err']:.1f}px"
                           if m == "wm" else "")
                        for m, sc in scores.items())
                    print(f"  {desc} 噪声={'Y' if noise else 'N'} | {line}")

    # 汇总表
    print("\n" + "=" * 72)
    print("汇总表（mean±std，5种子×2变体=10 个场景/格）")
    print("=" * 72)
    header = (f"{'配置':<14}{'方法':<6}{'身份一致率':<16}{'交换次数':<12}"
              f"{'窗内交换':<12}{'丢失率':<14}{'e相关':<14}{'预检误差px':<12}")
    print(header)
    name = {"det": "检测", "wm": "世界模型", "oracle": "预言机"}
    for n_balls in (2, 3):
        for noise in (False, True):
            for method in ("det", "wm", "oracle"):
                rs = [r for r in records if r["n"] == n_balls
                      and r["noise"] == noise and r["method"] == method]
                if not rs:
                    continue
                cfg = f"n={n_balls} 噪声={'Y' if noise else 'N'}"
                print(f"{cfg:<14}{name[method]:<6}"
                      f"{fmt([r['consist'] for r in rs]):<16}"
                      f"{fmt([r['swaps'] for r in rs], nd=1):<12}"
                      f"{fmt([r['swaps_win'] for r in rs], nd=1):<12}"
                      f"{fmt([r['lost'] for r in rs]):<14}"
                      f"{fmt([r['e_corr'] for r in rs]):<14}"
                      f"{fmt([r['pred_err'] for r in rs], nd=1):<12}")

    # 碰撞窗口聚焦：n=2 干净条件下 det vs wm 的窗内交换差
    print("\n判决参考（窗内交换：世界模型 vs 检测，负值=世界模型更少交换）")
    for n_balls in (2, 3):
        for noise in (False, True):
            d = [r["swaps_win"] for r in records if r["n"] == n_balls
                 and r["noise"] == noise and r["method"] == "det"]
            w = [r["swaps_win"] for r in records if r["n"] == n_balls
                 and r["noise"] == noise and r["method"] == "wm"]
            print(f"  n={n_balls} 噪声={'Y' if noise else 'N'}: "
                  f"检测 {np.mean(d):.2f} vs 世界模型 {np.mean(w):.2f} "
                  f"(差 {np.mean(w) - np.mean(d):+.2f})")


if __name__ == "__main__":
    main()

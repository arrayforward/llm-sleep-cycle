"""扩展弹球台（v6 场景）：弹性系数属性 + 球-球碰撞 + 事件日志。

- 每球有弹性系数 e∈[0.6,1.0]，渲染亮度=0.4+0.6e（语义属性挂在外观上，
  撞墙时 v'=-e·v：规则响应依赖属性 → 检验"语义作为规则挂载点"）
- 球-球碰撞：等质量非完全弹性碰撞（恢复系数取两球 e 的较小值），
  第二事件类型 → 检验原型门的事件分化
- 事件日志 events: [(t, 'wall'|'collision', ball_ids)]，监督/分析用
- 轨迹预计算（dt=0.01），render(t) 插值；SimPool 复用轨迹摊薄生成成本
"""
import numpy as np


class BallSim2:
    def __init__(self, rng, size=64, max_balls=3, dur=4.0, dt=0.01,
                 impulse=None):
        self.size, self.dt = size, dt
        self.n = int(rng.integers(2, max_balls + 1))  # ≥2 球才有碰撞
        self.r = rng.uniform(2.5, 4.5, self.n)
        self.e = rng.uniform(0.6, 1.0, self.n)
        self.x0_init = rng.uniform(10, size - 10, (self.n, 2))
        self.v0_init = rng.uniform(15, 50, (self.n, 2)) \
            * rng.choice([-1, 1], (self.n, 2))
        # v11 动作：t_imp 时刻施加全局阵风冲量 dv（作用于所有球）
        self.impulse = impulse  # (t_imp, np.array([dvx, dvy])) 或 None
        self._simulate(dur)

    def resim(self, e_new=None, impulse=None):
        """用相同初始条件重放（反事实动力学）：可换弹性或换动作。"""
        import copy
        sim = copy.copy(self)
        if e_new is not None:
            sim.e = np.array(e_new, float)
        if impulse is not None:
            sim.impulse = impulse
        sim._simulate(self.xs.shape[0] * self.dt)
        return sim

    def _simulate(self, dur):
        size, dt = self.size, self.dt
        xx = self.x0_init.copy()
        vv = self.v0_init.copy().astype(float)
        T = int(dur / dt)
        self.xs = np.empty((T, self.n, 2))
        self.events = []
        applied = False
        for k in range(T):
            t = k * dt
            # v11 动作：冲量在 t_imp 时刻施加（全局阵风）
            if self.impulse is not None and not applied and t >= self.impulse[0]:
                vv = vv + self.impulse[1]
                self.events.append((t, "impulse", -1,
                                    np.array([0., 0., *self.impulse[1]])))
                applied = True
            xx = xx + vv * dt
            # 撞墙：越界且朝外 → 反弹 v'=-e·v
            for axis in (0, 1):
                lo, hi = self.r, size - self.r
                for i in range(self.n):
                    if xx[i, axis] < lo[i] and vv[i, axis] < 0:
                        xx[i, axis] = lo[i]
                        v_pre = float(np.hypot(*vv[i]))
                        vv[i, axis] = -vv[i, axis] * self.e[i]
                        # 载荷: [x, y, vx后, vy后, v前]（L3 定律发现用）
                        self.events.append((t, "wall", i,
                                            np.array([*xx[i], *vv[i], v_pre])))
                    elif xx[i, axis] > hi[i] and vv[i, axis] > 0:
                        xx[i, axis] = hi[i]
                        v_pre = float(np.hypot(*vv[i]))
                        vv[i, axis] = -vv[i, axis] * self.e[i]
                        self.events.append((t, "wall", i,
                                            np.array([*xx[i], *vv[i], v_pre])))
            # 球-球碰撞：重叠且互相接近 → 法向恢复系数碰撞
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    d = xx[j] - xx[i]
                    dist = np.linalg.norm(d)
                    if dist < self.r[i] + self.r[j] and dist > 1e-6:
                        n = d / dist
                        vn = (vv[i] - vv[j]) @ n
                        if vn > 0:  # 接近中
                            e_eff = min(self.e[i], self.e[j])
                            imp = (1 + e_eff) / 2 * vn * n
                            vv[i] -= imp; vv[j] += imp
                            # 位置分离防粘连
                            gap = (self.r[i] + self.r[j] - dist) / 2
                            xx[i] -= gap * n; xx[j] += gap * n
                            self.events.append((t, "collision", (i, j),
                                                np.array([*np.mean([xx[i], xx[j]], 0),
                                                          *np.mean([vv[i], vv[j]], 0)])))
            self.xs[k] = xx

    def pos(self, t):
        """线性插值到任意连续时刻。"""
        k = np.clip(t / self.dt, 0, len(self.xs) - 2)
        k0 = int(k)
        w = k - k0
        return self.xs[k0] * (1 - w) + self.xs[k0 + 1] * w

    def render(self, t, e_show=None):
        """e_show: 显示用弹性（亮度），与动力学 e 解耦——反事实实验用。"""
        yy, xx = np.mgrid[0:self.size, 0:self.size].astype(np.float32)
        img = np.zeros((self.size, self.size), np.float32)
        es = self.e if e_show is None else e_show
        for (px, py), r, e in zip(self.pos(t), self.r, es):
            d2 = (xx - px) ** 2 + (yy - py) ** 2
            body = np.clip(1.0 - (np.sqrt(d2) - r), 0, 1)
            img = np.maximum(img, body * (0.4 + 0.6 * e))  # 亮度=弹性
        return img.astype(np.float32)

    def events_in(self, t0, t1):
        return [ev for ev in self.events if t0 < ev[0] <= t1]


_POOL = {}


def _pool(rng, n=256, seed=None):
    """轨迹池。

    注意（2026-09 修正）：这里曾是进程级单例缓存，导致"训练脚本内调用
    event_eval"与"独立运行 event_eval.py"使用**不同的池**——前者继承训练
    时传入的 rng（seed=0），后者用 seed=99。v8 与 v9a/v12 的数字因此是在
    不同评估样本集上算出的，不能相互比较。

    现在显式按 seed 分键：评估方一律用 seed=99，训练方仍用传入的 rng，
    两者互不污染，脚本内评估与独立评估结果一致。
    """
    if seed is not None:
        key = ("pool", seed)
        if key not in _POOL:
            rng_ = np.random.default_rng(seed)
            _POOL[key] = [BallSim2(rng_) for _ in range(n)]
        return _POOL[key]
    if "pool" not in _POOL:
        _POOL["pool"] = [BallSim2(rng) for _ in range(n)]
    return _POOL["pool"]


def make_batch2(rng, batch, dt_lo=0.05, dt_hi=0.5, n_hist=3, hist_gap=0.1):
    """与 data.make_batch 同接口，从轨迹池采样。"""
    pool = _pool(rng)
    xs, ys, dts = [], [], []
    for _ in range(batch):
        sim = pool[rng.integers(len(pool))]
        t0 = rng.uniform((n_hist - 1) * hist_gap, 1.0)
        dt = rng.uniform(dt_lo, dt_hi)
        hist = [sim.render(t0 - (n_hist - 1 - i) * hist_gap)
                for i in range(n_hist)]
        xs.append(np.stack(hist))
        ys.append(sim.render(t0 + dt))
        dts.append(dt)
    return (np.stack(xs), np.stack(ys)[:, None], np.array(dts, np.float32))


def make_batch_rollout2(rng, batch, K, dt_lo=0.05, dt_hi=0.5,
                        n_hist=3, hist_gap=0.1, ev_horizon=0.3):
    pool = _pool(rng)
    xs, ys_all, dts_all = [], [], []
    ev_type, ev_dt, ev_pay = [], [], []
    # 事件类型编码: 0=无, 1=撞墙, 2=碰撞；载荷=[Δt, x/64, y/64, speed/50]
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
        # 每个状态时刻（含初始 t0）的"下一个事件"标签
        for tau in taus:
            evs = sim.events_in(tau, tau + ev_horizon)
            if evs:
                t_ev, kind, _, pay = evs[0]
                ev_type.append(1 if kind == "wall" else 2)
                ev_dt.append(t_ev - tau)
                ev_pay.append([t_ev - tau, pay[0] / 64, pay[1] / 64,
                               np.hypot(pay[2], pay[3]) / 50])
            else:
                ev_type.append(0)
                ev_dt.append(ev_horizon)
                ev_pay.append([ev_horizon, 0.5, 0.5, 0.0])
    K1 = K + 1
    return (np.stack(xs), np.stack(ys_all), np.stack(dts_all),
            np.array(ev_type).reshape(batch, K1),
            np.array(ev_dt, np.float32).reshape(batch, K1),
            np.array(ev_pay, np.float32).reshape(batch, K1, 4))


def make_batch_rollout_action2(rng, batch, K, dt_lo=0.05, dt_hi=0.5,
                               n_hist=3, hist_gap=0.1, dur=3.0):
    """v11 动作条件数据：随机时刻全局阵风，返回动作向量 [t_rel, dvx, dvy]。
    冲量落在预测窗口内，模型不使用动作信息必错——无平均化退路。"""
    xs, ys_all, dts_all, acts = [], [], [], []
    for _ in range(batch):
        dts = rng.uniform(dt_lo, dt_hi, K).astype(np.float32)
        t0 = rng.uniform((n_hist - 1) * hist_gap, 0.8)
        t_imp = t0 + rng.uniform(0.05, max(0.15, float(dts.sum()) * 0.8))
        dv = (rng.uniform(10, 30, 2) * rng.choice([-1, 1], 2)).astype(np.float32)
        sim = BallSim2(rng, dur=dur, impulse=(t_imp, dv))
        hist = [sim.render(t0 - (n_hist - 1 - i) * hist_gap)
                for i in range(n_hist)]
        xs.append(np.stack(hist))
        ys_all.append(np.stack([sim.render(t) for t in t0 + np.cumsum(dts)]))
        dts_all.append(dts)
        acts.append(np.array([t_imp - t0, dv[0] / 50, dv[1] / 50],
                             np.float32))
    return (np.stack(xs), np.stack(ys_all), np.stack(dts_all),
            np.stack(acts))

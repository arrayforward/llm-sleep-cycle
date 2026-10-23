"""解析弹球模拟器：匀速运动 + 镜面反弹，任意连续时刻 t 的位置有解析解。"""
import numpy as np


def _reflect(p, lo, hi):
    """三角波反射：p 在 [lo, hi] 内匀速往返的解析位置。"""
    w = hi - lo
    t = (p - lo) % (2.0 * w)
    return np.where(t < w, lo + t, 2.0 * hi - t - lo)


class BallSim:
    def __init__(self, rng, size=64, max_balls=3):
        self.size = size
        self.n = rng.integers(1, max_balls + 1)
        self.r = rng.uniform(2.5, 4.5, self.n)
        self.x0 = rng.uniform(8, size - 8, (self.n, 2))
        v = rng.uniform(15, 55, (self.n, 2)) * rng.choice([-1, 1], (self.n, 2))
        self.v = v

    def pos(self, t):
        """t 可以是标量，返回 (n, 2) 的 (x, y) 位置。"""
        lo = self.r[:, None]
        hi = self.size - self.r[:, None]
        return _reflect(self.x0 + self.v * t, lo, hi)

    def render(self, t):
        yy, xx = np.mgrid[0:self.size, 0:self.size].astype(np.float32)
        img = np.zeros((self.size, self.size), np.float32)
        for (px, py), r in zip(self.pos(t), self.r):
            d2 = (xx - px) ** 2 + (yy - py) ** 2
            img = np.maximum(img, np.clip(1.0 - (np.sqrt(d2) - r), 0, 1))
        return img.astype(np.float32)


def make_batch_rollout(rng, batch, K, size=64, dt_lo=0.05, dt_hi=0.5,
                       dt_fixed=None, n_hist=3, hist_gap=0.1):
    """隐空间 rollout 训练数据：hist[B,n_hist,H,W], dts[B,K], ys[B,K,H,W]。
    第 k 步目标 = t0 + cumsum(dts)[k] 时刻的帧。"""
    xs, ys_all, dts_all = [], [], []
    for _ in range(batch):
        sim = BallSim(rng, size)
        t0 = rng.uniform((n_hist - 1) * hist_gap, 1.0)
        hist = [sim.render(t0 - (n_hist - 1 - i) * hist_gap)
                for i in range(n_hist)]
        xs.append(np.stack(hist))
        if dt_fixed is not None:
            dts = np.full(K, dt_fixed, np.float32)
        else:
            dts = rng.uniform(dt_lo, dt_hi, K).astype(np.float32)
        ts = t0 + np.cumsum(dts)
        ys_all.append(np.stack([sim.render(t) for t in ts]))
        dts_all.append(dts)
    return (np.stack(xs), np.stack(ys_all), np.stack(dts_all))


def make_batch(rng, batch, size=64, dt_lo=0.05, dt_hi=0.5, dt_fixed=None,
               n_hist=3, hist_gap=0.1):
    """返回 (frames[B,n_hist,H,W], frame_t+dt, dt)。多帧历史使速度可观测。
    dt_fixed 用于定步长基线。"""
    xs, ys, dts = [], [], []
    for _ in range(batch):
        sim = BallSim(rng, size)
        t0 = rng.uniform((n_hist - 1) * hist_gap, 1.0)
        dt = dt_fixed if dt_fixed is not None else rng.uniform(dt_lo, dt_hi)
        hist = [sim.render(t0 - (n_hist - 1 - i) * hist_gap)
                for i in range(n_hist)]
        xs.append(np.stack(hist))
        ys.append(sim.render(t0 + dt))
        dts.append(dt)
    return (np.stack(xs), np.stack(ys)[:, None],
            np.array(dts, np.float32))

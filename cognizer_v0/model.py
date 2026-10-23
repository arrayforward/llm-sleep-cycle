"""连续时间世界模型 v2：显式状态 + QKV 空间交互 + exp(A·Δt) 演化。

对应 REPORT.md §4 的核因子分解：
  语义因子: QKV 注意力（状态网格 64 token × 64 通道）
  空间因子: G(Δx) = ALiBi 式欧氏距离偏置（每头斜率可学）→ score = q·k/√d − λ_h·|x_i−x_j|
  时间因子: 对角复演化 (α+iβ)⊙s = exp(A·Δt) 的可学习对角化
  循环结构: 固定步长(dt0=0.1s) RK4 积分，步数 n=ceil(Δt/dt0) 自适应
  保能量(v3): conserve=True 时 A 纯旋转(α≡0) + RK4，长时间积分无衰减漂移
  状态充分性: 输入最近 3 帧（间隔 0.1s），速度可观测（v1 问题1修复）
"""
import torch
import torch.nn as nn
import math


class AttnMixer(nn.Module):
    """64 token 注意力：语义 q·k + 几何偏置 −λ·|Δx|，输出有界(tanh)防积分发散。"""

    def __init__(self, ch=64, heads=4, grid=8):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(ch, 3 * ch)
        self.proj = nn.Linear(ch, ch)
        ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid),
                                indexing="ij")
        pos = torch.stack([xs.flatten(), ys.flatten()], 1).float()  # [N,2]
        dist = (pos[:, None] - pos[None]).norm(dim=-1)  # [N,N] 欧氏距离
        self.register_buffer("dist", dist)
        self.slope = nn.Parameter(torch.tensor([2.0 ** -(i + 1)
                                                for i in range(heads)]))

    def forward(self, s):
        B, C, g, _ = s.shape
        x = s.flatten(2).transpose(1, 2)  # [B, N, C]
        q, k, v = self.qkv(x).chunk(3, -1)
        d = C // self.heads
        q = q.view(B, -1, self.heads, d).transpose(1, 2)
        k = k.view(B, -1, self.heads, d).transpose(1, 2)
        v = v.view(B, -1, self.heads, d).transpose(1, 2)
        score = q @ k.transpose(-1, -2) / math.sqrt(d)
        score = score - self.slope[None, :, None, None] * self.dist  # 几何核
        out = (score.softmax(-1) @ v).transpose(1, 2).flatten(2)
        out = self.proj(out.view(B, -1, C)).transpose(1, 2).view(B, C, g, g)
        return torch.tanh(out)


class AttnEnergy(nn.Module):
    """标量相互作用能 H_int(s)：与 AttnMixer 同一核（语义q·k + 几何−λ|Δx|），
    但输出每样本一个标量能量。力 = -∂H/∂s 由 autograd 给出 → 向量场自动哈密顿。"""

    def __init__(self, ch=64, heads=4, grid=8, causal_range=None):
        super().__init__()
        self.heads = heads
        self.qk = nn.Linear(ch, 2 * ch)
        self.v = nn.Linear(ch, ch)
        self.w = nn.Linear(ch, ch)
        ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid),
                                indexing="ij")
        pos = torch.stack([xs.flatten(), ys.flatten()], 1).float()
        dist = (pos[:, None] - pos[None]).norm(dim=-1)
        self.register_buffer("dist", dist)
        self.slope = nn.Parameter(torch.tensor([2.0 ** -(i + 1)
                                                for i in range(heads)]))
        # 小初始化：初始能量力接近零，训练从纯旋转起步（课程友好）
        nn.init.normal_(self.v.weight, std=0.02)
        nn.init.normal_(self.w.weight, std=0.02)

        # 因果掩码（可选）：把注意力限制在因果锥内（|Δx| ≤ causal_range 网格）。
        # 默认 None = 不掩码，与论文 §3.2 描述的注意力公式一致，也与 2026-09
        # 之前训练并保存的 checkpoint 一致（那些 checkpoint 里没有该 buffer）。
        # 非持久化 buffer：它是 dist/causal_range 的确定性函数，不需要存盘，
        # 这样旧 checkpoint 才能 strict=True 加载。
        self.causal_range = causal_range
        if causal_range is not None:
            self.register_buffer("causal_mask",
                                 (dist <= causal_range).float(),
                                 persistent=False)
        else:
            self.causal_mask = None

    def forward(self, s):
        """s: [B, C, g, g] -> 每样本标量能量 [B]"""
        B, C, g, _ = s.shape
        x = s.flatten(2).transpose(1, 2)  # [B, N, C]
        N = x.shape[1]
        q, k = self.qk(x).chunk(2, -1)
        v, w = self.v(x), self.w(x)
        d = C // self.heads
        q = q.view(B, N, self.heads, d).transpose(1, 2)
        k = k.view(B, N, self.heads, d).transpose(1, 2)
        v = v.view(B, N, self.heads, d).transpose(1, 2)
        w = w.view(B, N, self.heads, d).transpose(1, 2)
        score = q @ k.transpose(-1, -2) / math.sqrt(d)
        score = score - self.slope[None, :, None, None] * self.dist
        if self.causal_mask is not None:  # 可选因果锥掩码（默认关闭）
            score = score * self.causal_mask - 1e9 * (1 - self.causal_mask)
        a = score.softmax(-1)                      # [B, H, N, N]
        e = (w * (a @ v)).sum(dim=(1, 2, 3))       # 能量汇聚为标量
        return e


class ViTEncoder(nn.Module):
    """ViT 式感知编码器：8×8 像素 patch -> 64 token，2 层 transformer encoder。
    与 CNN 编码器同输出形状 [B,C,g,g]，用于感知接口消融。"""

    def __init__(self, ch=64, heads=4, grid=8, depth=3):
        super().__init__()
        self.patch = nn.Conv2d(1, ch, 8, 8)  # 64×64 -> 8×8 token 网格
        self.pos = nn.Parameter(torch.randn(1, grid * grid, ch) * 0.02)
        layer = nn.TransformerEncoderLayer(ch, heads, ch * 2, dropout=0.0,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tr = nn.TransformerEncoder(layer, depth)
        self.norm = nn.GroupNorm(8, ch)

    def forward(self, frame):
        x = self.patch(frame)               # [B, C, g, g]
        B, C, g, _ = x.shape
        t = x.flatten(2).transpose(1, 2) + self.pos
        t = self.tr(t)
        return self.norm(t.transpose(1, 2).view(B, C, g, g))


class EventAttnGate(nn.Module):
    """v5b：原型注意力事件门。事件检测 = 在可学习原型库中的联想检索：
    query=状态token(含绝对位置), keys=K 个事件原型, values=跳变响应。
    gate=max_k softmax(q·k)，jump=Σa·v —— 检测与响应是同一次检索。"""

    def __init__(self, ch=64, grid=8, n_proto=8, hard=False):
        super().__init__()
        self.n_proto = n_proto
        self.hard = hard  # v14: 硬读出（argmax+直通），排他性
        self.proto_k = nn.Parameter(torch.randn(n_proto, ch) * 0.1)
        self.proto_v = nn.Parameter(torch.randn(n_proto, ch) * 0.1)
        self.pos = nn.Parameter(torch.randn(1, grid * grid, ch) * 0.02)
        self.q = nn.Linear(ch, ch)
        self.grid = grid

    def forward(self, s):
        B, C, g, _ = s.shape
        x = s.flatten(2).transpose(1, 2) + self.pos  # [B,N,C] 绝对位置接地
        q = self.q(x)                                # [B,N,C]
        a = (q @ self.proto_k.T / (C ** 0.5)).softmax(-1)  # [B,N,K]
        if self.hard:  # 硬读出：前向独热，梯度走软——读出端不许平均
            a = torch.zeros_like(a).scatter(-1, a.argmax(-1, True), 1.0)                 + (a - a.detach())
        self.last_a = a  # 供负载均衡损失使用
        gate = a.max(dim=-1, keepdim=True).values    # [B,N,1] 最强匹配=触发强度
        jump = (a @ self.proto_v).transpose(1, 2).view(B, C, g, g)
        return gate.transpose(1, 2).view(B, 1, g, g), jump


class EventHead(nn.Module):
    """v8/v9a/v12：从状态 S 预测下一个事件。
    检验"事件信息是否在 S 里可读"+ 给语义层直接的监督信号。

    kind="dtime"   -> (cls, dtime)     dtime 是标量"距今时间"（v8/v9a，纯类型+时间）
    kind="payload" -> (cls, payload)   payload 是 [Δt, x, y, speed]（v12，混合符号）

    两个变体的 forward 接口一致（都返回 (logits[B,3], scalar_or_vector[B,k])），
    因此 event_eval.py 里的 pay[0, 0] 对两者都成立。kind 在加载旧 checkpoint 时
    由 eval.load() 依据 state_dict 的键自动推断。
    """

    def __init__(self, ch=64, grid=8, kind="payload"):
        super().__init__()
        self.kind = kind
        self.body = nn.Sequential(
            nn.Conv2d(ch, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.GELU(),
            nn.Flatten(), nn.Linear(32 * grid * grid, 128), nn.GELU())
        self.cls = nn.Linear(128, 3)   # 0=无, 1=撞墙, 2=碰撞
        if kind == "dtime":
            self.dtime = nn.Linear(128, 1)    # 标量时间
        else:
            self.payload = nn.Linear(128, 4)  # 载荷: [Δt, x, y, speed]

    def forward(self, s):
        h = self.body(s)
        cls = self.cls(h)
        if self.kind == "dtime":
            return cls, self.dtime(h)
        return cls, self.payload(h)


class EvoWorldModel(nn.Module):
    def __init__(self, grid=8, ch=32, use_dt=True, dt0=0.1, n_hist=3,
                 chain_sub=8, conserve=True, hamiltonian=False,
                 enc_type="cnn", event_gate=False, inten=False,
                 event_head=False, use_act=False, event_head_kind="payload"):
        super().__init__()
        self.inten = inten  # v7: 强度旁路，属性保真（GN 会抹掉绝对亮度）
        self.use_dt = use_dt      # False = 定步长接龙基线：evolve 忽略 Δt
        self.dt0 = dt0            # 连续模式的固定积分步长
        self.chain_sub = chain_sub  # 基线模式的每步内部细分
        self.conserve = conserve  # True = 保能量：A 纯旋转(α≡0)，RK4 积分
        C = 2 * ch  # 实通道数
        # v11: 动作嵌入（时刻+方向+强度 → 状态通道偏置）
        self.act_embed = nn.Linear(3, C) if use_act else None
        self.enc = ViTEncoder(C, grid=grid) if enc_type == "vit" else nn.Sequential(
            nn.Conv2d(1, 48, 3, 2, 1), nn.GELU(),
            nn.Conv2d(48, 96, 3, 2, 1), nn.GELU(),
            nn.Conv2d(96, C, 3, 2, 1),
            nn.GroupNorm(8, C),
        )
        # 多帧融合：3 帧各编码后 1×1 卷积融合 → 速度可观测
        # inten 模式额外拼入每帧 8×8 局部亮度图（原始物理量，不归一化）
        n_in = C * n_hist + (n_hist if inten else 0)
        self.fuse = nn.Sequential(nn.Conv2d(n_in, C, 1), nn.GELU(),
                                  nn.GroupNorm(8, C))
        # 对角复演化 A = α + iβ；conserve 模式下 α≡0 纯旋转（保能量）
        self.alpha = None if conserve else nn.Parameter(torch.full((ch,), -0.1))
        self.beta = nn.Parameter(torch.randn(ch) * 0.5)
        self.hamiltonian = hamiltonian
        if hamiltonian:
            self.energy = AttnEnergy(C, heads=4, grid=grid)
        else:
            self.mix = AttnMixer(C, heads=4, grid=grid)
        # v5: 事件门。conv=卷积软门(局部模式)；attn=原型注意力门(联想检索，
        # K 个事件原型=语义字母表胚胎，检测与响应同一次检索，query 含绝对位置)
        # ds/dt = 哈密顿流 + gate(s)⊙tanh(J(s))：质变点熵增由门吸收，不污染相位
        self.event_gate = event_gate
        if event_gate == "attn":
            self.egate = EventAttnGate(C, grid=grid)
        elif event_gate == "attn_hard":
            self.egate = EventAttnGate(C, grid=grid, hard=True)
        elif event_gate == "rel":
            # v13: 相对性事件门。gate = σ(raw − EMA(raw))：
            # 事件性 = 超出近期基线的量；训练时动量更新基线
            self.gate = nn.Sequential(
                nn.Conv2d(C, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 1, 3, 1, 1))
            self.jump = nn.Sequential(
                nn.Conv2d(C, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, C, 3, 1, 1))
            self.register_buffer("g_ema", torch.zeros(1))
        elif event_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(C, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 1, 3, 1, 1))
            self.jump = nn.Sequential(
                nn.Conv2d(C, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, C, 3, 1, 1))
        self.ehead = (EventHead(C, grid, kind=event_head_kind)
                      if event_head else None)
        self.dec = nn.Sequential(
            nn.GroupNorm(8, C),
            nn.ConvTranspose2d(C, 96, 3, 2, 1, 1), nn.GELU(),
            nn.ConvTranspose2d(96, 48, 3, 2, 1, 1), nn.GELU(),
            nn.ConvTranspose2d(48, 1, 3, 2, 1, 1),
        )

    def encode(self, frames, act=None):
        """frames: [B, n_hist, 1, H, W], act: [B,3] 动作 -> 状态 s"""
        feats = [self.enc(frames[:, i]) for i in range(frames.shape[1])]
        if self.inten:  # 强度旁路：每帧 8×8 局部亮度图直接进融合
            feats += [torch.nn.functional.avg_pool2d(frames[:, i], 8)
                      for i in range(frames.shape[1])]
        s = self.fuse(torch.cat(feats, dim=1))
        if self.act_embed is not None and act is not None:
            s = s + self.act_embed(act)[:, :, None, None]
        return s

    def _deriv(self, s):
        """哈密顿模式：ds/dt = J·∇H(s)，J 为辛形式（re=q, im=p），
        整个向量场由标量能量 H 的梯度生成，结构自动保辛。
        非哈密顿模式：ds/dt = (α+iβ)·s + AttnMixer(s)。"""
        if self.hamiltonian:
            with torch.enable_grad():
                sg = s.requires_grad_(True)
                re, im = sg[:, 0::2], sg[:, 1::2]
                # 旋转谐振能 H_rot = -β/2·(q²+p²)（对角旋转本来就是哈密顿的）
                H = (-0.5 * self.beta[None, :, None, None]
                     * (re ** 2 + im ** 2)).sum(dim=(1, 2, 3))
                # 相互作用能作用在压扁状态上：H_int = E(τ·tanh(s/τ))
                # 大振幅处力趋于零（自稳定），仍是标量函数的精确梯度，保辛
                sq = 3.0 * torch.tanh(sg / 3.0)
                H = H + self.energy(sq)  # 注意力相互作用能
                g = torch.autograd.grad(H.sum(), sg, create_graph=True)[0]
            dq = g[:, 1::2]    # q̇ =  ∂H/∂p
            dp = -g[:, 0::2]   # ṗ = -∂H/∂q
            base = torch.stack([dq, dp], dim=2).flatten(1, 2)
        else:
            re, im = s[:, 0::2], s[:, 1::2]
            b = self.beta
            if self.conserve:
                a = torch.zeros_like(b)
            else:
                a = self.alpha.clamp(max=0.0)
            dre = a[None, :, None, None] * re - b[None, :, None, None] * im
            dim = b[None, :, None, None] * re + a[None, :, None, None] * im
            m = self.mix(s)
            base = torch.stack([dre + m[:, 0::2], dim + m[:, 1::2]],
                               dim=2).flatten(1, 2)
        if self.event_gate in ("attn", "attn_hard"):  # 原型注意力门
            gate, jump = self.egate(s)
            base = base + gate * torch.tanh(jump)
        elif self.event_gate == "rel":  # v13 相对性门：超出近期基线才算事件
            raw = self.gate(s)
            if self.training:
                with torch.no_grad():
                    self.g_ema.mul_(0.99).add_(0.01 * raw.mean())
            g = torch.sigmoid(raw - self.g_ema)
            base = base + g * torch.tanh(self.jump(s))
        elif self.event_gate:  # 卷积软门：质变点的熵增由门吸收
            base = base + torch.sigmoid(self.gate(s)) * torch.tanh(self.jump(s))
        return base

    def _rk4(self, s, h):
        """一步 RK4：h: [B] 每样本步长。保能量旋转下比欧拉幅度误差低两阶。"""
        B = s.shape[0]
        hv = h.view(B, 1, 1, 1)
        k1 = self._deriv(s)
        k2 = self._deriv(s + 0.5 * hv * k1)
        k3 = self._deriv(s + 0.5 * hv * k2)
        k4 = self._deriv(s + hv * k3)
        return s + hv * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def _euler(self, s, h):
        return s + h.view(s.shape[0], 1, 1, 1) * self._deriv(s)

    def _integrate(self, s, h):
        s = self._rk4(s, h) if (self.conserve or self.hamiltonian) \
            else self._euler(s, h)
        # 安全网：径向限幅，只截断失控尾部，正常 regime (|s|<5) 不受影响
        if self.hamiltonian:
            s = 10.0 * torch.tanh(s / 10.0)
        return s

    def evolve(self, s, dt_total):
        """连续模式：固定步长 dt0、步数自适应；基线模式：固定单位步。"""
        B = s.shape[0]
        if not self.use_dt:
            h = torch.full((B,), 1.0 / self.chain_sub, device=s.device)
            for _ in range(self.chain_sub):
                s = self._integrate(s, h)
            return s
        rem = dt_total.to(s.device).clone().float()
        while bool((rem > 1e-9).any()):
            h = rem.clamp(max=self.dt0)  # 完成的样本 h=0，自动不再更新
            s = self._integrate(s, h)
            rem = rem - h
        return s

    def decode(self, s):
        return torch.sigmoid(self.dec(s))

    def forward(self, frames, dt):
        return self.decode(self.evolve(self.encode(frames), dt))

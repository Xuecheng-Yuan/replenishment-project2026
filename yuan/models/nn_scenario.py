"""models.nn_scenario: actual model/processing implementation, not a legacy runner alias.

Refactored mechanically from the existing project. See matching docs for
information timing, historical results, variants and limitations.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[1]
if str(HERE) in sys.path:sys.path.remove(str(HERE))
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
PROJECT_ROOT=ROOT
DATA_DIR=ROOT/'data'
OUTPUT_DIR=ROOT/'results/generated'
OUT=ROOT/'results/generated/ready_immediate'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
import argparse
import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def choose_device(requested: str='auto') -> torch.device:
    if requested != 'auto':
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def gaussian_nll_log1p(mu: torch.Tensor, sigma: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL on y = log(1 + demand)."""
    y = torch.log1p(torch.clamp(demand, min=0.0))
    z = (y - mu) / sigma
    return (0.5 * z.square() + torch.log(sigma)).mean()

def normalized_softmin(x: torch.Tensor, tau: float, dim: int) -> torch.Tensor:
    """Smooth minimum, normalized so equal inputs return that same input."""
    n = x.shape[dim]
    return -tau * (torch.logsumexp(-x / tau, dim=dim) - math.log(n))

@dataclass
class Config:
    train_periods: int = 60
    test_periods: int = 28
    n_features: int = 20
    hidden_dim: int = 96
    sku_emb_dim: int = 8
    min_sigma: float = 0.05
    pretrain_epochs: int = 10
    pretrain_lr: float = 0.002
    pretrain_batch_size: int = 4096
    num_scenarios: int = 8
    decision_steps: int = 10
    decision_lr: float = 1.0
    capacity_multiplier: float = 1.05
    initial_inventory_cover: float = 1.25
    inventory_epsilon: float = 0.0001
    equal_sku_weighting: bool = True
    service_target: float = 0.95
    service_rho_inner: float = 8.0
    service_rho_task: float = 15.0
    service_kappa: float = 10.0
    positive_inventory_delta: float = 0.02
    factor_rank: int = 8
    dro_radius: float = 0.4
    dro_mix: float = 0.65
    robust_tau: float = 0.08
    task_horizon: int = 7
    dfl_epochs: int = 3
    task_windows_per_epoch: int = 10
    dfl_lr: float = 0.0002
    pred_loss_weight: float = 0.05
    seed: int = 42

@dataclass
class PanelData:
    X: torch.Tensor
    D: torch.Tensor
    periods: list
    sku_labels: list
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    @property
    def T(self) -> int:
        return self.X.shape[0]

    @property
    def N(self) -> int:
        return self.X.shape[1]

    @property
    def F(self) -> int:
        return self.X.shape[2]

def make_synthetic_data(T: int, N: int, Fdim: int, seed: int) -> PanelData:
    """Synthetic panel with known-current features and a common demand factor."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, N, Fdim)).astype(np.float32)
    t = np.arange(T, dtype=np.float32)
    X[:, :, 0] = np.sin(2 * np.pi * t[:, None] / 7.0)
    X[:, :, 1] = np.cos(2 * np.pi * t[:, None] / 7.0)
    X[:, :, 2] = (rng.random((T, N)) < 0.14).astype(np.float32)
    X[:, :, 3] = rng.normal(0.0, 0.6, size=(T, N)).astype(np.float32)
    base = rng.normal(np.log(8.0), 0.65, size=N).astype(np.float32)
    beta = np.zeros(Fdim, dtype=np.float32)
    beta[:8] = np.array([0.1, 0.08, 0.35, -0.15, 0.08, -0.05, 0.06, 0.04], dtype=np.float32)
    common = np.zeros(T, dtype=np.float32)
    for k in range(1, T):
        common[k] = 0.75 * common[k - 1] + rng.normal(0, 0.12)
    n_cat = max(4, min(20, N // 20 + 1))
    cat = rng.integers(0, n_cat, size=N)
    cat_shock = rng.normal(0, 0.12, size=(T, n_cat)).astype(np.float32)
    log_mean = base[None, :] + np.einsum('tnf,f->tn', X, beta)
    log_mean += common[:, None]
    log_mean += cat_shock[:, cat]
    mean = np.exp(log_mean).astype(np.float32)
    mean = np.clip(mean, 0.2, 80.0)
    shape = 8.0
    gamma_scale = mean / shape
    lam = rng.gamma(shape, gamma_scale).astype(np.float32)
    D = rng.poisson(lam).astype(np.float32)
    fmean = X[:min(60, T)].mean(axis=(0, 1))
    fstd = X[:min(60, T)].std(axis=(0, 1)) + 1e-06
    X = (X - fmean[None, None, :]) / fstd[None, None, :]
    return PanelData(X=torch.from_numpy(X), D=torch.from_numpy(D), periods=list(range(T)), sku_labels=[f'SKU_{i:05d}' for i in range(N)], feature_mean=torch.from_numpy(fmean), feature_std=torch.from_numpy(fstd))

def load_long_csv(path: str, cfg: Config) -> PanelData:
    df = pd.read_csv(path)
    required = {'period', 'sku_id', 'demand'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'CSV缺少必要列: {sorted(missing)}')
    feature_cols = [f'x{i}' for i in range(1, cfg.n_features + 1)]
    missing_x = [c for c in feature_cols if c not in df.columns]
    if missing_x:
        raise ValueError(f'CSV缺少特征列: {missing_x}')
    periods = sorted(df['period'].unique().tolist())
    skus = sorted(df['sku_id'].unique().tolist())
    need_T = cfg.train_periods + cfg.test_periods
    if len(periods) < need_T:
        raise ValueError(f'至少需要 {need_T} 个周期，目前只有 {len(periods)} 个')
    periods = periods[:need_T]
    df = df[df['period'].isin(periods)].copy()
    pmap = {p: i for i, p in enumerate(periods)}
    smap = {s: i for i, s in enumerate(skus)}
    T, N, Fdim = (len(periods), len(skus), len(feature_cols))
    X = np.full((T, N, Fdim), np.nan, dtype=np.float32)
    D = np.full((T, N), np.nan, dtype=np.float32)
    for row in df.itertuples(index=False):
        ti = pmap[getattr(row, 'period')]
        si = smap[getattr(row, 'sku_id')]
        D[ti, si] = float(getattr(row, 'demand'))
        for j, c in enumerate(feature_cols):
            X[ti, si, j] = float(getattr(row, c))
    if np.isnan(X).any() or np.isnan(D).any():
        raise ValueError('CSV必须是完整的 period × sku 网格，当前存在缺失的SKU-周期记录。')
    train_X = X[:cfg.train_periods]
    fmean = train_X.mean(axis=(0, 1))
    fstd = train_X.std(axis=(0, 1)) + 1e-06
    X = (X - fmean[None, None, :]) / fstd[None, None, :]
    return PanelData(X=torch.from_numpy(X), D=torch.from_numpy(D), periods=periods, sku_labels=skus, feature_mean=torch.from_numpy(fmean.astype(np.float32)), feature_std=torch.from_numpy(fstd.astype(np.float32)))

class GlobalDemandNet(nn.Module):

    def __init__(self, n_sku: int, n_features: int, hidden: int, emb_dim: int, min_sigma: float):
        super().__init__()
        self.emb = nn.Embedding(n_sku, emb_dim)
        self.net = nn.Sequential(nn.Linear(n_features + emb_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu_head = nn.Linear(hidden, 1)
        self.sigma_head = nn.Linear(hidden, 1)
        self.min_sigma = min_sigma

    def forward(self, x: torch.Tensor, sku_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.emb(sku_idx)
        h = self.net(torch.cat([x, e], dim=-1))
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.sigma_head(h).squeeze(-1)) + self.min_sigma
        return (mu, sigma)

def pretrain_forecaster(model: GlobalDemandNet, data: PanelData, cfg: Config, device: torch.device) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.pretrain_lr)
    Ttr, N = (cfg.train_periods, data.N)
    X = data.X[:Ttr].reshape(-1, data.F)
    D = data.D[:Ttr].reshape(-1)
    sku = torch.arange(N).repeat(Ttr)
    nobs = X.shape[0]
    history = []
    for epoch in range(cfg.pretrain_epochs):
        perm = torch.randperm(nobs)
        total = 0.0
        count = 0
        for start in range(0, nobs, cfg.pretrain_batch_size):
            idx = perm[start:start + cfg.pretrain_batch_size]
            xb = X[idx].to(device)
            db = D[idx].to(device)
            sb = sku[idx].to(device)
            mu, sigma = model(xb, sb)
            loss = gaussian_nll_log1p(mu, sigma, db)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
            count += len(idx)
        print(f'[Pretrain] epoch {epoch + 1:02d}/{cfg.pretrain_epochs}  NLL={total / count:.5f}')
        history.append({'epoch': epoch + 1, 'nll': total / count})
    return history

@torch.no_grad()
def compute_residual_factor_basis(model: GlobalDemandNet, data: PanelData, cfg: Config, device: torch.device) -> torch.Tensor:
    """
    PCA/SVD-style low-rank basis of cross-SKU residuals in log-demand space.
    Returns loadings [N, r].
    """
    model.eval()
    N = data.N
    sku = torch.arange(N, device=device)
    rows = []
    for t in range(cfg.train_periods):
        x = data.X[t].to(device)
        d = data.D[t].to(device)
        mu, _ = model(x, sku)
        resid = torch.log1p(d) - mu
        rows.append(resid.detach().cpu())
    R = torch.stack(rows, dim=0)
    R = R - R.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    r = min(cfg.factor_rank, Vh.shape[0])
    scale = S[:r] / math.sqrt(max(1, cfg.train_periods - 1))
    basis = Vh[:r].T * scale[None, :]
    return basis.to(device)

def make_scenarios(mu: torch.Tensor, sigma: torch.Tensor, factor_basis: Optional[torch.Tensor], cfg: Config, dro_radius: float, seed: int) -> torch.Tensor:
    """
    Create K differentiable demand scenarios in original demand units.

    Nominal uncertainty: Gaussian in log1p demand space.
    Robust stress: low-rank residual-factor shifts with bounded magnitude.
    """
    device = mu.device
    K, N = (cfg.num_scenarios, mu.numel())
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    eps = torch.randn((K, N), generator=gen, dtype=mu.dtype).to(device)
    shift = torch.zeros((K, N), device=device, dtype=mu.dtype)
    if factor_basis is not None and dro_radius > 0:
        r = factor_basis.shape[1]
        z = torch.randn((K, r), generator=gen, dtype=mu.dtype).to(device)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-08)
        shift = dro_radius * (z @ factor_basis.T)
    logd = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps + shift
    logd = torch.clamp(logd, min=-3.0, max=20.0)
    return torch.expm1(logd).clamp_min(0.0)

def order_from_logits(logits: torch.Tensor, capacity: torch.Tensor) -> torch.Tensor:
    """
    N SKU logits + 1 'unused capacity' logit.
    This guarantees q_i >= 0 and sum_i q_i <= capacity.
    """
    alloc = torch.softmax(logits, dim=0)
    return capacity * alloc[:-1]

def decision_score(q: torch.Tensor, inventory: torch.Tensor, demand_scen: torch.Tensor, sku_weights: torch.Tensor, cum_sales: torch.Tensor, cum_inventory: torch.Tensor, cfg: Config, robust: bool) -> torch.Tensor:
    """Differentiable one-period rolling decision score."""
    available = inventory.unsqueeze(0) + q.unsqueeze(0)
    sales = torch.minimum(demand_scen, available)
    end_inv = F.relu(available - demand_scen)
    w = sku_weights.unsqueeze(0)
    s = (sales * w).sum(dim=1)
    h = (end_inv * w).sum(dim=1)
    ratio = (cum_sales + s) / (cum_inventory + h + cfg.inventory_epsilon)
    if robust and cfg.dro_mix > 0:
        worst_ratio = normalized_softmin(ratio, cfg.robust_tau, dim=0)
        turnover_term = (1.0 - cfg.dro_mix) * ratio.mean() + cfg.dro_mix * worst_ratio
    else:
        turnover_term = ratio.mean()
    z = torch.sigmoid(cfg.service_kappa * (end_inv - cfg.positive_inventory_delta))
    if robust and cfg.dro_mix > 0:
        z_robust = normalized_softmin(z, cfg.robust_tau, dim=0)
        service_est = (1.0 - cfg.dro_mix) * z.mean(dim=0) + cfg.dro_mix * z_robust
    else:
        service_est = z.mean(dim=0)
    service_pen = F.relu(cfg.service_target - service_est).square().mean()
    return turnover_term - cfg.service_rho_inner * service_pen

def optimize_order_unrolled(mu: torch.Tensor, sigma: torch.Tensor, inventory: torch.Tensor, sku_weights: torch.Tensor, capacity: torch.Tensor, cum_sales: torch.Tensor, cum_inventory: torch.Tensor, factor_basis: Optional[torch.Tensor], cfg: Config, create_graph: bool, robust: bool, seed: int, deterministic_forecast: bool=False) -> torch.Tensor:
    """
    Differentiable order optimizer by unrolled gradient ascent.

    The final q satisfies q>=0 and sum(q)<=capacity exactly by construction.
    """
    N = mu.numel()
    if deterministic_forecast:
        sigma_use = torch.full_like(sigma, 0.0001)
        radius = 0.0
    else:
        sigma_use = sigma
        radius = cfg.dro_radius if robust else 0.0
    demand_scen = make_scenarios(mu, sigma_use, factor_basis, cfg, radius, seed)
    init = torch.zeros(N + 1, device=mu.device, dtype=mu.dtype)
    with torch.no_grad():
        mean_d = torch.expm1(mu).clamp_min(0.001)
        init[:-1] = torch.log(mean_d + 0.001)
        init[-1] = torch.log(torch.tensor(0.5, device=mu.device, dtype=mu.dtype))
    logits = init.detach().requires_grad_(True)
    for _ in range(cfg.decision_steps):
        q = order_from_logits(logits, capacity)
        score = decision_score(q=q, inventory=inventory, demand_scen=demand_scen, sku_weights=sku_weights, cum_sales=cum_sales, cum_inventory=cum_inventory, cfg=cfg, robust=robust)
        grad = torch.autograd.grad(score, logits, create_graph=create_graph)[0]
        logits = logits + cfg.decision_lr * grad
        if not create_graph:
            logits = logits.detach().requires_grad_(True)
    return order_from_logits(logits, capacity)

def build_sku_weights(data: PanelData, cfg: Config, device: torch.device) -> torch.Tensor:
    mean_d = data.D[:cfg.train_periods].mean(dim=0).to(device)
    if cfg.equal_sku_weighting:
        w = 1.0 / torch.clamp(mean_d, min=0.5)
        w = w / w.mean()
    else:
        w = torch.ones_like(mean_d)
    return w

def build_initial_inventory(data: PanelData, cfg: Config, device: torch.device) -> torch.Tensor:
    mean_d = data.D[:cfg.train_periods].mean(dim=0).to(device)
    return cfg.initial_inventory_cover * mean_d

def build_capacity(data: PanelData, cfg: Config, device: torch.device) -> torch.Tensor:
    avg_total = data.D[:cfg.train_periods].sum(dim=1).float().mean().to(device)
    return cfg.capacity_multiplier * avg_total

def fine_tune_decision_focused(model: GlobalDemandNet, data: PanelData, factor_basis: torch.Tensor, cfg: Config, device: torch.device) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.dfl_lr)
    sku_idx = torch.arange(data.N, device=device)
    weights = build_sku_weights(data, cfg, device)
    capacity = build_capacity(data, cfg, device)
    base_inventory = build_initial_inventory(data, cfg, device)
    max_start = cfg.train_periods - cfg.task_horizon
    if max_start < 0:
        raise ValueError('task_horizon 大于训练期长度')
    starts_all = list(range(max_start + 1))
    history = []
    for epoch in range(cfg.dfl_epochs):
        random.shuffle(starts_all)
        starts = starts_all[:min(cfg.task_windows_per_epoch, len(starts_all))]
        running = []
        for start in starts:
            inv = base_inventory.clone()
            cum_sales = torch.zeros((), device=device)
            cum_inv = torch.zeros((), device=device)
            ready_soft = []
            pred_losses = []
            for j in range(cfg.task_horizon):
                t = start + j
                x = data.X[t].to(device)
                d_true = data.D[t].to(device)
                mu, sigma = model(x, sku_idx)
                pred_losses.append(gaussian_nll_log1p(mu, sigma, d_true))
                q = optimize_order_unrolled(mu=mu, sigma=sigma, inventory=inv, sku_weights=weights, capacity=capacity, cum_sales=cum_sales, cum_inventory=cum_inv, factor_basis=factor_basis, cfg=cfg, create_graph=True, robust=True, seed=cfg.seed + 1000 * epoch + 97 * start + t)
                available = inv + q
                sales = torch.minimum(d_true, available)
                end_inv = F.relu(available - d_true)
                cum_sales = cum_sales + (weights * sales).sum()
                cum_inv = cum_inv + (weights * end_inv).sum()
                ready_soft.append(torch.sigmoid(cfg.service_kappa * (end_inv - cfg.positive_inventory_delta)))
                inv = end_inv
            turnover = cum_sales / (cum_inv + cfg.inventory_epsilon)
            service = torch.stack(ready_soft, dim=0).mean(dim=0)
            service_pen = F.relu(cfg.service_target - service).square().mean()
            pred_loss = torch.stack(pred_losses).mean()
            task_loss = -torch.log(turnover + 1e-06) + cfg.service_rho_task * service_pen
            loss = task_loss + cfg.pred_loss_weight * pred_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            running.append((float(loss.detach().cpu()), float(turnover.detach().cpu()), float(service.mean().detach().cpu()), float(pred_loss.detach().cpu())))
        arr = np.array(running)
        history.append(dict(loss=float(arr[:, 0].mean()), turnover=float(arr[:, 1].mean()), soft_service=float(arr[:, 2].mean()), pred_nll=float(arr[:, 3].mean())))
        print(f'[DFL] epoch {epoch + 1:02d}/{cfg.dfl_epochs}  loss={arr[:, 0].mean():.4f}  turnover={arr[:, 1].mean():.4f}  soft_service={arr[:, 2].mean():.4f}  predNLL={arr[:, 3].mean():.4f}')
    return history

def evaluate_rolling(model: GlobalDemandNet, data: PanelData, factor_basis: torch.Tensor, cfg: Config, device: torch.device, *, robust: bool, deterministic_forecast: bool, name: str) -> Dict[str, float]:
    """
    Strict out-of-sample rolling evaluation:
    at period t use X_t and current inventory; only after q_t is fixed reveal D_t.
    """
    model.eval()
    sku_idx = torch.arange(data.N, device=device)
    weights = build_sku_weights(data, cfg, device)
    capacity = build_capacity(data, cfg, device)
    inv = build_initial_inventory(data, cfg, device)
    weighted_sales = torch.zeros((), device=device)
    weighted_inv = torch.zeros((), device=device)
    raw_sales = torch.zeros((), device=device)
    raw_demand = torch.zeros((), device=device)
    ready_hist = []
    order_hist = []
    start_t = cfg.train_periods
    end_t = cfg.train_periods + cfg.test_periods
    for t in range(start_t, end_t):
        x = data.X[t].to(device)
        d_true = data.D[t].to(device)
        with torch.no_grad():
            mu, sigma = model(x, sku_idx)
        mu = mu.detach()
        sigma = sigma.detach()
        with torch.enable_grad():
            q = optimize_order_unrolled(mu=mu, sigma=sigma, inventory=inv.detach(), sku_weights=weights, capacity=capacity, cum_sales=weighted_sales.detach(), cum_inventory=weighted_inv.detach(), factor_basis=factor_basis.detach() if factor_basis is not None else None, cfg=cfg, create_graph=False, robust=robust, seed=cfg.seed + 10000 + t, deterministic_forecast=deterministic_forecast).detach()
        available = inv + q
        sales = torch.minimum(d_true, available)
        end_inv = F.relu(available - d_true)
        weighted_sales += (weights * sales).sum()
        weighted_inv += (weights * end_inv).sum()
        raw_sales += sales.sum()
        raw_demand += d_true.sum()
        ready_hist.append((end_inv > 1e-08).float())
        order_hist.append(q.sum())
        inv = end_inv
    ready = torch.stack(ready_hist, dim=0)
    service_per_sku = ready.mean(dim=0)
    turnover = weighted_sales / (weighted_inv + cfg.inventory_epsilon)
    fill_rate = raw_sales / (raw_demand + 1e-08)
    mean_service = service_per_sku.mean()
    service_pass = (service_per_sku >= cfg.service_target - 1e-12).float().mean()
    avg_order = torch.stack(order_hist).mean()
    out = {'weighted_turnover': float(turnover.cpu()), 'fill_rate': float(fill_rate.cpu()), 'mean_ready_rate': float(mean_service.cpu()), 'sku_service_pass_rate': float(service_pass.cpu()), 'avg_total_order': float(avg_order.cpu())}
    print(f"[{name}] turnover={out['weighted_turnover']:.4f}  fill_rate={out['fill_rate']:.4f}  mean_ready={out['mean_ready_rate']:.4f}  SKU>=95%={out['sku_service_pass_rate']:.4f}  avg_order={out['avg_total_order']:.2f}")
    return out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Decision-focused robust rolling replenishment prototype')
    p.add_argument('--csv', type=str, default=None, help='Long-format CSV. If omitted, use synthetic data.')
    p.add_argument('--n-sku', type=int, default=200, help='Synthetic demo SKU count.')
    p.add_argument('--train-periods', type=int, default=60)
    p.add_argument('--test-periods', type=int, default=28)
    p.add_argument('--pretrain-epochs', type=int, default=10)
    p.add_argument('--dfl-epochs', type=int, default=3)
    p.add_argument('--task-horizon', type=int, default=7)
    p.add_argument('--decision-steps', type=int, default=10)
    p.add_argument('--num-scenarios', type=int, default=8)
    p.add_argument('--factor-rank', type=int, default=8)
    p.add_argument('--capacity-multiplier', type=float, default=1.05)
    p.add_argument('--dro-radius', type=float, default=0.4)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true', help='Very small/fast smoke test settings.')
    return p.parse_args()
p = sys.modules[__name__]
import sys, json, copy
from common.ready_evaluation import ReadyRateEvaluator, selection_key

class ScaledDemandNet(p.GlobalDemandNet):
    """Predict log-demand deviations around a training-only SKU offset."""

    def __init__(self, n, f, offset):
        super().__init__(n, f, 96, 8, 0.05)
        self.register_buffer('offset', torch.tensor(offset, dtype=torch.float32))

    def forward(self, x, sku_idx):
        mu, sigma = super().forward(x, sku_idx)
        return (mu + self.offset[sku_idx], sigma)

def setup(raw, split):
    d = raw['demand']
    n = d.shape[0]
    e = ReadyRateEvaluator(d[:, :split], raw['item_ids'])
    f = raw['features'].copy()
    scale = np.maximum(e.mean, 0.5)
    f[:, :, :13] /= scale[None, :, None]
    f[:, :, 14] /= scale[None, :]
    fm = f[:split - 12].mean(axis=(0, 1))
    fs = np.maximum(f[:split - 12].std(axis=(0, 1)), 1e-06)
    f = (f - fm) / fs
    data = p.PanelData(X=torch.tensor(f, dtype=torch.float32), D=torch.tensor(d[:, 12:].T, dtype=torch.float32), periods=list(range(13, 89)), sku_labels=raw['item_ids'].tolist(), feature_mean=torch.tensor(fm), feature_std=torch.tensor(fs))
    data.raw_demand = d[:, 12:]
    cfg = p.Config(train_periods=split - 12, test_periods=18 if split == 42 else 28, n_features=f.shape[-1])
    p.build_sku_weights = lambda data, cfg, device: torch.tensor(e.weights, dtype=torch.float32, device=device)
    p.build_initial_inventory = lambda data, cfg, device: torch.tensor(e.initial, dtype=torch.float32, device=device)
    p.build_capacity = lambda data, cfg, device: torch.tensor(e.capacity, dtype=torch.float32, device=device)
    model = ScaledDemandNet(n, f.shape[-1], np.log1p(scale))
    return (data, cfg, e, model, fm, fs)

def evaluate(model, data, cfg, e, basis, robust, deterministic, output=None):
    model.eval()
    idx = torch.arange(data.N)
    cs = 0.0
    ci = 0.0
    predictions = []

    def policy(t, inv):
        nonlocal cs, ci
        with torch.no_grad():
            mu, sigma = model(data.X[cfg.train_periods + t], idx)
            predictions.append(torch.expm1(mu.clamp(0, 20)).numpy())
        q = p.optimize_order_unrolled(mu, sigma, torch.tensor(inv, dtype=torch.float32), torch.tensor(e.weights, dtype=torch.float32), torch.tensor(e.capacity, dtype=torch.float32), torch.tensor(cs), torch.tensor(ci), basis, cfg, False, robust, cfg.seed + 10000 + t, deterministic)
        order = q.detach().numpy()
        dt = data.raw_demand[:, cfg.train_periods + t]
        sales = np.minimum(inv + order, dt)
        endings = inv + order - sales
        cs += float(e.weights @ sales)
        ci += float(e.weights @ endings)
        return order
    demand = data.raw_demand[:, cfg.train_periods:cfg.train_periods + cfg.test_periods]
    m = e.evaluate(demand, policy, output)
    m['test_wape'] = float(np.abs(np.stack(predictions, axis=1) - demand).sum() / demand.sum())
    return m

def train_legacy():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    raw = np.load(OUT / 'data.npz', allow_pickle=False)
    device = torch.device('cpu')
    data, cfg, e, model, fm, fs = setup(raw, 42)
    p.set_seed(42)
    model = ScaledDemandNet(data.N, data.F, np.log1p(np.maximum(e.mean, 0.5)))
    trace = dict(validation_pretrain=p.pretrain_forecaster(model, data, cfg, device), validation_dfl=[])
    basis = p.compute_residual_factor_basis(model, data, cfg, device)
    valrows = []
    radius_options = []
    for radius in (0.0, 0.2, 0.4, 0.8):
        cfg.dro_radius = radius
        m = evaluate(model, data, cfg, e, basis, True, False)
        valrows.append(dict(stage='pretrain', radius=radius, epoch=0, **m))
        radius_options.append((selection_key(m), radius))
    _, radius = max(radius_options, key=lambda a: a[0])
    cfg.dro_radius = radius
    initial = copy.deepcopy(model.state_dict())
    best = initial
    best_epoch = 0
    best_key = selection_key(evaluate(model, data, cfg, e, basis, True, False))
    cfg.dfl_epochs = 1
    for epoch in range(1, 4):
        trace['validation_dfl'] += p.fine_tune_decision_focused(model, data, basis, cfg, device)
        m = evaluate(model, data, cfg, e, basis, True, False)
        valrows.append(dict(stage='dfl', radius=radius, epoch=epoch, **m))
        print('Validation DFL', epoch, m, flush=True)
        if selection_key(m) > best_key:
            best_key = selection_key(m)
            best = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    pd.DataFrame(valrows).to_csv(OUT / 'dfl_validation.csv', index=False)
    p.set_seed(42)
    data, cfg, e, model, fm, fs = setup(raw, 60)
    cfg.dro_radius = radius
    trace['final_pretrain'] = p.pretrain_forecaster(model, data, cfg, device)
    trace['final_dfl'] = []
    basis = p.compute_residual_factor_basis(model, data, cfg, device)
    rows = []
    torch.save(dict(state_dict=model.state_dict(), feature_mean=fm, feature_std=fs, factor_basis=basis, config=vars(cfg)), OUT / 'nn_pretrained.pt')
    for name, robust, deterministic in [('nn_deterministic', False, True), ('nn_stochastic', False, False), ('nn_robust', True, False)]:
        m = evaluate(model, data, cfg, e, basis, robust, deterministic, OUT / name)
        rows.append(dict(model=name, radius=radius, dfl_epochs=0, **m))
        print(name, m, flush=True)
    cfg.dfl_epochs = 1
    for _ in range(best_epoch):
        trace['final_dfl'] += p.fine_tune_decision_focused(model, data, basis, cfg, device)
    m = evaluate(model, data, cfg, e, basis, True, False, OUT / 'dfl_robust_selected')
    rows.append(dict(model='dfl_robust_selected', radius=radius, dfl_epochs=best_epoch, **m))
    torch.save(dict(state_dict=model.state_dict(), feature_mean=fm, feature_std=fs, factor_basis=basis, config={**vars(cfg), 'dfl_epochs': best_epoch}), OUT / 'dfl_robust_selected/model.pt')
    for _ in range(3 - best_epoch):
        trace['final_dfl'] += p.fine_tune_decision_focused(model, data, basis, cfg, device)
    m = evaluate(model, data, cfg, e, basis, True, False, OUT / 'dfl_robust_3epochs')
    rows.append(dict(model='dfl_robust_3epochs', radius=radius, dfl_epochs=3, **m))
    torch.save(dict(state_dict=model.state_dict(), feature_mean=fm, feature_std=fs, factor_basis=basis, config={**vars(cfg), 'dfl_epochs': 3}), OUT / 'dfl_robust_3epochs/model.pt')
    (OUT / 'dfl_training_trace.json').write_text(json.dumps(trace, indent=2))
    pd.DataFrame(rows).to_csv(OUT / 'dfl_comparison.csv', index=False)
    (OUT / 'dfl_selection.json').write_text(json.dumps(dict(radius=radius, selected_epochs=best_epoch, seed=42, validation_split=42, scenario_log_cap=20), indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

if __name__=='__main__':
    from common.cli import start
    start('nn_scenario', globals(), default_protocol='ready_immediate')

"""概率网络、有限步DFL、直接策略和在线学习；训练、决策损失不混淆。"""
import json
import time
import numpy as np
import pandas as pd
from ..runtime import ROOT, save, sha, freeze_selection
from ..evaluation import key, simulate
from ..features import common_features
import copy
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(2)
FAMILIES = ['nn_scenario', 'dfl_improved', 'service_policy', 'online_learning', 'turnover_dfl']

class DemandNet(nn.Module):
    """共享概率需求网络：38个历史特征加8维商品嵌入。"""

    def __init__(self, n, offset):
        super().__init__()
        self.emb = nn.Embedding(n, 8)
        self.body = nn.Sequential(nn.Linear(46, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU())
        self.mu = nn.Linear(96, 1)
        self.sig = nn.Linear(96, 1)
        self.register_buffer('offset', torch.tensor(offset, dtype=torch.float32))

    def forward(self, x, idx):
        h = self.body(torch.cat([x, self.emb(idx)], -1))
        return (self.mu(h).squeeze(-1) + self.offset[idx], F.softplus(self.sig(h).squeeze(-1)) + 0.05)

class DirectNet(nn.Module):
    """直接目标库存网络，输入历史信息及当前真实库存。"""

    def __init__(self, n):
        super().__init__()
        self.emb = nn.Embedding(n, 4)
        self.net = nn.Sequential(nn.Linear(44, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.2)

    def forward(self, x, idx, inv, ref, scale):
        a = torch.cat([x, self.emb(idx), (inv / scale).unsqueeze(-1), (ref / scale).unsqueeze(-1)], -1)
        return ref * torch.exp(self.net(a).squeeze(-1).clamp(-3, 2))

def nll(mu, sig, d):
    """log(1+需求)上的高斯负对数似然，省略与参数无关常数。"""
    return (0.5 * ((torch.log1p(d) - mu) / sig).square() + torch.log(sig)).mean()

def prepare(raw, split):
    """尺度与标准化只在拟合前缀计算；历史特征严格滞后。"""
    d, X, _ = common_features(raw)
    scale = np.maximum(d[:, :split].mean(1), 0.5)
    a = np.concatenate([X[:, :, :17], X[:, :, 57:58], X[:, :, 17:37]], 2)
    a[:, :, :13] /= scale[None, :, None]
    a[:, :, 14] /= scale[None, :]
    mean = a[:split - 12].mean((0, 1))
    std = np.maximum(a[:split - 12].std((0, 1)), 1e-06)
    a = np.clip((a - mean) / std, -12, 12)
    ref = np.column_stack([d[:, max(0, t - 3):t].mean(1) for t in range(12, 89)])
    return dict(
        X=torch.tensor(a, dtype=torch.float32),
        D=torch.tensor(d[:, 12:].T, dtype=torch.float32),
        d=d,
        scale=torch.tensor(scale, dtype=torch.float32),
        feature_mean=mean,
        feature_std=std,
        offset=np.log1p(scale),
        reference=torch.tensor(ref.T, dtype=torch.float32),
        split=split,
        initial=torch.tensor(raw['initial'], dtype=torch.float32)
    )

def prediction(model, pack):
    """逐期概率预测，不更新模型参数。"""
    model.eval()
    XX = pack['X']
    out = []
    sig = []
    idx = torch.arange(XX.shape[1])
    with torch.no_grad():
        for k in range(len(XX)):
            m, s = model(XX[k], idx)
            out.append(m.numpy())
            sig.append(s.numpy())
    return (np.stack(out), np.stack(sig))

def pretrained(raw, split):
    """训练10轮概率网络；只复用当前运行目录内的基础网络缓存。"""
    out = ROOT / 'models' / f'_fresh_nn_base_{split}'
    out.mkdir(parents=True, exist_ok=True)
    pack = prepare(raw, split)
    torch.manual_seed(42)
    model = DemandNet(len(raw['initial']), pack['offset'])
    if (out / 'model.pt').exists():
        model.load_state_dict(torch.load(out / 'model.pt', weights_only=False)['state_dict'])
        return (model, pack)
    n = len(raw['initial'])
    count = split - 12
    xx = pack['X'][:count].reshape(-1, 38)
    dd = pack['D'][:count].reshape(-1)
    ids = torch.arange(n).repeat(count)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for epoch in range(10):
        order = torch.randperm(len(dd))
        total = 0.0
        model.train()
        for batch in order.split(4096):
            opt.zero_grad()
            mu, sig = model(xx[batch], ids[batch])
            loss = nll(mu, sig, dd[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total += float(loss.detach()) * len(batch)
        history.append(dict(epoch=epoch + 1, nll=total / len(dd)))
        print('NN PRETRAIN', split, epoch + 1, history[-1]['nll'], flush=True)
    mu, sig = prediction(model, pack)
    res = (np.log1p(pack['d'][:, 12:split]).T - mu[:count]) / np.maximum(sig[:count], 0.05)
    _, _, vt = np.linalg.svd(res, full_matrices=False)
    pack['basis'] = vt[:8].T.astype(np.float32) * np.sqrt(n / 8)
    torch.save(
        dict(
            state_dict=model.state_dict(),
            basis=pack['basis'],
            feature_mean=pack['feature_mean'],
            feature_std=pack['feature_std'],
            offset=pack['offset'],
            split=split,
            history=history
        ),
        out / 'model.pt'
    )
    save(out / 'history.json', history)
    return (model, pack)

def with_basis(pack, split):
    """载入当前运行内的训练残差相关性基。"""
    pack['basis'] = torch.load(
        ROOT / 'models' / f'_fresh_nn_base_{split}' / 'model.pt',
        weights_only=False
    )['basis']
    return pack

def scenarios(mu, sig, pack, idx, seed, radius):
    """固定种子生成8个非负需求情景，含低秩相关扰动。"""
    g = torch.Generator().manual_seed(int(seed))
    eps = torch.randn((8, len(idx)), generator=g)
    if radius:
        basis = torch.tensor(pack['basis'][idx.numpy()], dtype=torch.float32)
        common = torch.randn((8, basis.shape[1]), generator=g) @ basis.T
        eps = (eps + radius * common) / np.sqrt(1 + radius * radius)
    return torch.expm1((mu[None] + sig[None] * eps).clamp(0, 18))

def objective(q, b, scen, scale, rho):
    """内层代理目标：库存需求比，加服务水平95%—96%的双侧罚项。"""
    available = b + q
    ending = F.relu(available[None] - scen)
    service = torch.sigmoid((available[None] - scen - 1) / (0.05 * scale[None])).mean()
    gap = F.relu(0.95 - service).square() + F.relu(service - 0.96).square()
    return ending.sum(1).mean() / scen.sum(1).mean().clamp_min(1) + rho * gap

def inner(mu, sig, b, pack, idx, z, rho, radius, seed, steps, logits=False, create_graph=False):
    """有限步内层优化；梯度方向detach，不是精确隐式求导。"""
    scale = pack['scale'][idx]
    scen = scenarios(mu, sig, pack, idx, seed, radius)
    gap = F.relu(torch.expm1((mu + z * sig).clamp(0, 18)) - b)
    if logits:
        cap = torch.maximum(gap.sum() * 1.3, scen.max(0).values.sum()).clamp_min(1)
        v = torch.log(torch.cat([gap.clamp_min(0.001 * scale), (cap - gap.sum()).clamp_min(1).reshape(1)]))
    else:
        v = gap / scale
    if not create_graph:
        v = v.detach().requires_grad_(True)
    for _ in range(steps):
        q = cap * torch.softmax(v, 0)[:-1] if logits else scale * v.clamp_min(0)
        loss = objective(q, b, scen, scale, rho)
        grad = torch.autograd.grad(loss, v, retain_graph=create_graph)[0].detach()
        v = v - 0.03 * (len(idx) * grad).clamp(-4, 4)
        if not logits:
            v = v.clamp_min(0)
        if not create_graph:
            v = v.detach().requires_grad_(True)
    return cap * torch.softmax(v, 0)[:-1] if logits else scale * v.clamp_min(0)

def fine_tune(model, pack, name, rho, epochs):
    """从真实回放状态开始做4期或7期DFL微调，不读取验证需求训练。"""
    n = pack['X'].shape[1]
    count = pack['split'] - 12
    opt = torch.optim.Adam(model.parameters(), lr=0.0001 if name != 'service_policy' else 0.0004)
    history = []
    for epoch in range(epochs):
        torch.manual_seed(42 + epoch)
        ids = torch.randperm(n)
        values = []
        model.train()
        _, states = simulate(
            pack['d'][:, 12:pack['split']],
            pack['initial'].numpy(),
            runtime_policy(name, model, pack, 0, 1.65, rho, 0.2)
        )
        horizon = 7 if name == 'turnover_dfl' else 4
        for idx in ids.split(1000):
            for origin in range(4, count - horizon, 4):
                b = torch.tensor(states[1][idx.numpy(), origin - 1], dtype=torch.float32)
                pending = torch.tensor(states[3][idx.numpy(), origin - 1], dtype=torch.float32)
                sum_b = 0.0
                sum_d = 0.0
                ready = []
                nlls = []
                for h in range(horizon):
                    t = origin + h
                    actual = pack['D'][t, idx]
                    a = b + pending
                    b = F.relu(a - actual)
                    sum_b = sum_b + b.sum()
                    sum_d = sum_d + actual.sum()
                    ready.append(torch.sigmoid((a - actual - 1) / (0.05 * pack['scale'][idx])).mean())
                    if name == 'service_policy':
                        target = model(
                            pack['X'][t + 1, idx],
                            idx,
                            b,
                            pack['reference'][t + 1, idx],
                            pack['scale'][idx]
                        )
                        pending = F.relu(target - b)
                    else:
                        mu, sig = model(pack['X'][t + 1, idx], idx)
                        nlls.append(nll(mu, sig, pack['D'][t + 1, idx]))
                        pending = inner(
                            mu,
                            sig,
                            b,
                            pack,
                            idx,
                            1.65,
                            rho,
                            0.2,
                            42 + 100 * epoch + t,
                            4,
                            False,
                            True
                        )
                r = torch.stack(ready).mean()
                loss = sum_b / sum_d.clamp_min(1) + rho * (F.relu(0.95 - r).square() + F.relu(r - 0.96).square())
                if nlls:
                    loss += (0.2 if name == 'turnover_dfl' else 0.1) * torch.stack(nlls).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                values.append(float(loss.detach()))
        history.append(dict(epoch=epoch + 1, teacher_task_loss=float(np.mean(values))))
        print('NN DFL', name, pack['split'], epoch + 1, history[-1]['teacher_task_loss'], flush=True)
    return history

def runtime_policy(name, model, pack, start, z, rho, radius, seed=42, lr=0, alpha=0):
    """每次回放创建独立模型副本；在线模型仅在需求发生后更新。"""
    model = copy.deepcopy(model)
    model.eval()
    n = pack['X'].shape[1]
    idx = torch.arange(n)
    bias = torch.zeros(n)
    opt = torch.optim.Adam(model.parameters(), lr=lr) if lr else None
    mu, sig = prediction(model, pack) if name != 'service_policy' and (not lr) else (None, None)

    def policy(t, b):
        """当天需求已发生，使用当前库存决定下一期到货的订单。"""
        k = start + t
        inventory = torch.tensor(b, dtype=torch.float32)
        if name == 'online_learning':
            with torch.no_grad():
                m, s = model(pack['X'][k], idx)
                bias.mul_(1 - alpha).add_(alpha * (torch.log1p(pack['D'][k]) - m))
            if opt:
                opt.zero_grad()
                m, s = model(pack['X'][k], idx)
                loss = nll(m, s, pack['D'][k])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                opt.step()
        if name == 'service_policy':
            with torch.no_grad():
                target = model(
                    pack['X'][k + 1],
                    idx,
                    inventory,
                    pack['reference'][k + 1],
                    pack['scale']
                ) + z * 0.2 * pack['scale']
                q = F.relu(target - inventory)
        else:
            if mu is None:
                with torch.no_grad():
                    m, s = model(pack['X'][k + 1], idx)
                    m = m + bias
            else:
                m = torch.tensor(mu[k + 1])
                s = torch.tensor(sig[k + 1])
            q = inner(
                m,
                s,
                inventory,
                pack,
                idx,
                z,
                rho,
                radius,
                seed + 10000 + k,
                8,
                name == 'nn_scenario',
                False
            )
        return q.detach().numpy()
    return policy

def calibration(name, model, pack, raw, rho, radius, lr=0, alpha=0):
    """只用43—60期验证选择z，不能保证测试期一定满足区间。"""
    records = []

    def run(z):
        m, _ = simulate(
            raw['train_demand'][:, 12:60],
            raw['initial'],
            runtime_policy(name, model, pack, 0, z, rho, radius, lr=lr, alpha=alpha),
            30
        )
        records.append(dict(z=float(z), **m))
        print('NN CAL', name, round(z, 3), round(m['ready_rate'], 4), flush=True)
    for z in [-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        run(z)
    for _ in range(2):
        best = min(records, key=key)
        zs = sorted(set((r['z'] for r in records)))
        j = zs.index(best['z'])
        left = zs[max(0, j - 1)]
        right = zs[min(len(zs) - 1, j + 1)]
        for z in np.linspace(left, right, 5):
            if all((abs(z - r['z']) > 1e-08 for r in records)):
                run(z)
    best = min(records, key=key)
    return (best['z'], best, records)

def train_one(name, raw, base42, pack42, base60, pack60):
    start = time.time()
    out = ROOT / 'models' / name
    out.mkdir(parents=True, exist_ok=True)
    options = []
    search = []
    for ci, rho in enumerate([80.0, 240.0]):
        torch.manual_seed(42)
        model = DirectNet(len(raw['initial'])) if name == 'service_policy' else copy.deepcopy(base42)
        epochs = 4 if name == 'service_policy' else 2 if name in ['dfl_improved', 'turnover_dfl'] else 0
        history = fine_tune(model, pack42, name, rho, epochs) if epochs else []
        lr = 1e-05 if name == 'online_learning' and ci == 0 else 5e-05 if name == 'online_learning' else 0.0
        alpha = 0.1 if name == 'online_learning' else 0.0
        z, val, records = calibration(name, model, pack42, raw, rho, 0.2, lr, alpha)
        options.append((key(val), rho, z, val, epochs, lr, alpha))
        search.extend((dict(candidate=ci, rho=rho, **r) for r in records))
        torch.save(dict(state_dict=model.state_dict(), history=history), out / f'prefix_candidate_{ci}.pt')
    _, rho, z, val, epochs, lr, alpha = min(options, key=lambda r: r[0])
    freeze_selection(
        name,
        dict(
            validation=val,
            z=z,
            rho=rho,
            radius=0.2,
            epochs=epochs,
            online_lr=lr,
            online_alpha=alpha,
            fit_cutoff=42,
            refit_cutoff=60,
            training_status='refitting',
            loss='sum ending / sum demand + two-sided pooled ready interval penalty; NLL regularizer',
            inner='capacity/slack logits' if name == 'nn_scenario' else 'inventory-gap projected steps',
            teacher_no_external_capacity=True
        )
    )
    torch.manual_seed(42)
    model = DirectNet(len(raw['initial'])) if name == 'service_policy' else copy.deepcopy(base60)
    history = fine_tune(model, pack60, name, rho, epochs) if epochs else []
    torch.save(
        dict(
            state_dict=model.state_dict(),
            feature_mean=pack60['feature_mean'],
            feature_std=pack60['feature_std'],
            offset=pack60['offset'],
            basis=pack60['basis'],
            history=history
        ),
        out / 'model.pt'
    )
    selection = json.loads((out / 'selection.json').read_text())
    selection.update(
        training_status='trained',
        training_seconds=time.time() - start,
        final_model_sha256=sha(out / 'model.pt')
    )
    save(out / 'selection.json', selection)
    pd.DataFrame(search).to_csv(out / 'validation_search.csv', index=False)
    print('TRAINED', name, 'validation', val['ready_rate'], flush=True)

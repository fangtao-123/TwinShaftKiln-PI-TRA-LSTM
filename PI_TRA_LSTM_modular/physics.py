# -*- coding: utf-8 -*-
from .common import *
from .config import CONFIG
from .data_process import load_csv_file
from .models.model_zoo import PI_LSTM, PI_TRA_LSTM

PHYS_CLASSES = (PI_LSTM, PI_TRA_LSTM)

# -------------------------
# 4. 物理损失与工具
# -------------------------
def inverse_transform_y(y_norm: torch.Tensor, scaler_y: MinMaxScaler) -> torch.Tensor:
    y_np = y_norm.detach().cpu().numpy()
    y_inv = scaler_y.inverse_transform(y_np)
    return torch.tensor(y_inv, dtype=torch.float32, device=y_norm.device)


class PhysicsLoss(nn.Module):
    def __init__(self, cfg, scaler_X: MinMaxScaler, a_tg=None, b_tg=None, learn_tg=False):
        super().__init__()
        self.cfg = cfg
        self.scaler_X = scaler_X
        if learn_tg:
            self.a_tg = nn.Parameter(torch.tensor([a_tg], dtype=torch.float32))
            self.b_tg = nn.Parameter(torch.tensor([b_tg], dtype=torch.float32))
        else:
            self.register_buffer("a_tg", torch.tensor([a_tg], dtype=torch.float32))
            self.register_buffer("b_tg", torch.tensor([b_tg], dtype=torch.float32))

    def forward(self, y_pred, y_prev, X_seq_norm, Savg_seq, mode_seq):
        a = self.a_tg.to(y_pred.device)
        b = self.b_tg.to(y_pred.device)

        Tavg_t = y_pred.squeeze(-1)
        Tavg_tm1 = y_prev.squeeze(-1)

        Tg_t = a * Tavg_t + b
        Tg_tm1 = a * Tavg_tm1 + b
        dTg_dt_pred = (Tg_t - Tg_tm1)

        ctrl_norm_last = X_seq_norm[:, -1, :4].detach().cpu().numpy()
        ctrl_last_real = self.scaler_X.inverse_transform(ctrl_norm_last)
        Q_fuel = torch.tensor(ctrl_last_real[:, 0], dtype=torch.float32, device=y_pred.device)

        mode = mode_seq[:, -1]
        Savg_t = Savg_seq[:, -1]
        Ts_t = Tavg_t

        k1 = self.cfg["k1"]
        k3 = self.cfg["k3"]
        HGS = self.cfg["HGS"]

        dTg_dt_phys = (k1 * Q_fuel
                       - HGS * (1.0 - 0.1 * Savg_t) * (Tg_t - Ts_t)
                       - k3 * mode)

        L_phys_temp = torch.mean((dTg_dt_pred - dTg_dt_phys) ** 2)

        gamma = self.cfg["gamma_rb"]
        burn_raw = torch.relu(Savg_t - 5.0 + gamma * (Tavg_t - 927.0))
        L_phys_burn = torch.mean(burn_raw ** 2)

        return L_phys_temp, L_phys_burn


def gated_reattn_consistency(alpha, alpha_re, tau=0.3):
    """
    alpha, alpha_re: (B,L)
    returns scalar loss = mean( G(alpha_re) * |alpha_re - alpha| )
    """
    if (alpha is None) or (alpha_re is None):
        dev = alpha.device if alpha is not None else "cpu"
        return torch.tensor(0.0, device=dev)

    diff = torch.abs(alpha_re - alpha)  # (B,L)

    eps = 1e-8
    # entropy of re-attention
    H_re = -torch.sum(alpha_re * torch.log(alpha_re + eps), dim=1)  # (B,)
    L = alpha_re.shape[1]
    H_uniform = math.log(L + eps)  # scalar

    # normalized concentration indicator
    indicator = (H_uniform - H_re) / (H_uniform + eps)  # (B,)
    gate_mask = (indicator > tau).float()  # (B,)

    gate = torch.sigmoid(H_uniform / (H_re + eps)) * gate_mask  # (B,)
    return torch.mean(gate.unsqueeze(1) * diff)


def metrics_mae_rmse_mape_r2(y_pred, y_true):
    y_pred = y_pred.reshape(-1);
    y_true = y_true.reshape(-1)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mape = float(np.mean(np.abs((y_pred - y_true) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0)
    sse = float(np.sum((y_pred - y_true) ** 2))
    y_mean = float(np.mean(y_true))
    sst = float(np.sum((y_true - y_mean) ** 2))
    r2 = (np.nan if sst <= 1e-12 else (1.0 - sse / sst))
    mse = rmse ** 2
    return mae, rmse, mape, r2, mse


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# 4.5 PSO：离线机理参数校准
# -------------------------
def _finite_diff(vec):
    return vec[1:] - vec[:-1]


def _prepare_physics_arrays(csv_path):
    df = load_csv_file(csv_path)
    df = df.dropna().reset_index(drop=True)
    Q = df["Q_fuel"].values.astype(np.float64)
    S = df["S_avg"].values.astype(np.float64)
    mode = df["mode"].values.astype(np.float64)
    Tavg = df["T_avg"].values.astype(np.float64)
    if len(Tavg) < 3:
        raise RuntimeError("历史数据太短，无法做离散导数。")
    return Q, S, mode, Tavg


def _physics_residuals(a_tg, b_tg, k1, k3, HGS, gamma_rb, Q, S, mode, Tavg):
    Tg = a_tg * Tavg + b_tg
    dTg_obs = _finite_diff(Tg)

    Q1 = Q[1:];
    S1 = S[1:];
    mode1 = mode[1:];
    Tg1 = Tg[1:];
    Ts1 = Tavg[1:]
    dTg_hat = (k1 * Q1) - HGS * (1.0 - 0.1 * S1) * (Tg1 - Ts1) - k3 * mode1

    resid = dTg_obs - dTg_hat
    burn = np.maximum(0.0, S1 - 5.0 + gamma_rb * (Ts1 - 927.0))
    J = np.mean(resid ** 2) + 0.05 * np.mean(burn ** 2)
    return J


def _rand_init(bounds, size, rng):
    X = np.zeros((size, len(bounds)), dtype=np.float64)
    for j, (lo, hi) in enumerate(bounds):
        X[:, j] = rng.uniform(lo, hi, size=size)
    return X


def run_pso_calibration(csv_path, cfg):
    rng = np.random.default_rng(cfg["pso_seed"])
    Q, S, mode, Tavg = _prepare_physics_arrays(csv_path)

    keys = ["a_tg", "b_tg", "k1", "k3", "HGS", "gamma_rb"]
    bnds = [cfg["pso_bounds"][k] for k in keys]

    nP = cfg["pso_particles"];
    nI = cfg["pso_iters"]
    w, c1, c2 = cfg["pso_inertia"], cfg["pso_c1"], cfg["pso_c2"]

    X = _rand_init(bnds, nP, rng)
    V = np.zeros_like(X)

    def clip_to_bounds(x):
        for j, (lo, hi) in enumerate(bnds):
            x[j] = np.clip(x[j], lo, hi)
        return x

    def obj(x):
        a_tg, b_tg, k1, k3, HGS, gamma_rb = x
        return _physics_residuals(a_tg, b_tg, k1, k3, HGS, gamma_rb, Q, S, mode, Tavg)

    pbest = X.copy()
    pbest_val = np.array([obj(x) for x in X])
    g_idx = np.argmin(pbest_val)
    gbest = pbest[g_idx].copy()
    gbest_val = pbest_val[g_idx]

    for it in range(nI):
        r1 = rng.random((nP, X.shape[1]))
        r2 = rng.random((nP, X.shape[1]))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        for i in range(nP):
            X[i] = clip_to_bounds(X[i])

        vals = np.array([obj(x) for x in X])
        better = vals < pbest_val
        pbest[better] = X[better]
        pbest_val[better] = vals[better]
        if pbest_val.min() < gbest_val:
            gi = np.argmin(pbest_val)
            gbest = pbest[gi].copy()
            gbest_val = pbest_val[gi]

        if (it + 1) % max(1, nI // 10) == 0:
            print(f"[PSO] iter {it + 1}/{nI} | best J={gbest_val:.6f}")

    os.makedirs(os.path.dirname(cfg["pso_outfile"]), exist_ok=True)
    out = {k: float(v) for k, v in zip(keys, gbest)}
    out["J"] = float(gbest_val)
    with open(cfg["pso_outfile"], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[PSO] done. best = {out}")

    if cfg.get("use_pso_init", True):
        CONFIG["a_tg"] = out["a_tg"]
        CONFIG["b_tg"] = out["b_tg"]
        CONFIG["k1"] = out["k1"]
        CONFIG["k3"] = out["k3"]
        CONFIG["HGS"] = out["HGS"]
        CONFIG["gamma_rb"] = out["gamma_rb"]
    return out

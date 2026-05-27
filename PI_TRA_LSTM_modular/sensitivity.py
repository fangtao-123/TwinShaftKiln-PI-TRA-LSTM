# -*- coding: utf-8 -*-
from .common import *
from .config import CONFIG
from .data_process import LimeKilnSeqDataset, make_loaders_from_datasets
from .models.model_zoo import TRA_LSTM
from .physics import PhysicsLoss
from .train_torch import train_one_epoch, evaluate

# =========================
# Sensitivity for TRA-LSTM (Fig.6)
# =========================

def _rankdata(x: np.ndarray) -> np.ndarray:
    """不用 scipy 的 rankdata：处理 ties 用平均秩"""
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)

    # tie handling: average ranks for equal values
    # find groups in sorted order
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            avg = (i + (j - 1)) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman_abs(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman |rho|，不依赖 scipy"""
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if len(x) < 3:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum())) + 1e-12
    return float(abs((rx * ry).sum() / denom))


def _anova_importance_for_categorical(cat: np.ndarray, y: np.ndarray) -> float:
    """
    分类变量重要性：组间均值方差解释率（类似 one-way ANOVA 的 R^2）
    importance = SS_between / SS_total
    """
    cat = np.asarray(cat)
    y = np.asarray(y).astype(float)
    y_mean = y.mean()
    ss_total = ((y - y_mean) ** 2).sum() + 1e-12

    ss_between = 0.0
    for v in np.unique(cat):
        idx = (cat == v)
        if idx.sum() == 0:
            continue
        m = y[idx].mean()
        ss_between += idx.sum() * ((m - y_mean) ** 2)

    return float(ss_between / ss_total)


def compute_importance_from_trials(trials: pd.DataFrame, metric_col: str = "val_loss") -> pd.DataFrame:
    """
    输入：trials，包含每次 trial 的超参取值 + 性能指标（val_loss 或 val_rmse 等）
    输出：hyperparam, importance（归一化到和为1）
    """
    y = trials[metric_col].values.astype(float)

    # 数值超参用 Spearman |rho|
    num_cols = ["lr", "dropout", "batch_size", "hidden", "layers", "beta_att", "attn_drop"]
    # 分类超参：optimizer
    cat_cols = ["optimizer"]

    imps = []
    for c in num_cols:
        if c not in trials.columns:
            continue
        x = trials[c].values.astype(float)
        # 如果全常数，重要性为0
        if np.allclose(x, x[0]):
            imp = 0.0
        else:
            imp = _spearman_abs(x, y)
        imps.append((c, imp))

    for c in cat_cols:
        if c not in trials.columns:
            continue
        x = trials[c].values.astype(str)
        imp = _anova_importance_for_categorical(x, y)
        imps.append((c, imp))

    df_imp = pd.DataFrame(imps, columns=["hyperparam", "raw_importance"])

    # 归一化
    s = df_imp["raw_importance"].sum()
    if s <= 1e-12:
        df_imp["importance"] = 0.0
    else:
        df_imp["importance"] = df_imp["raw_importance"] / s

    df_imp = df_imp[["hyperparam", "importance"]].sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp


def _build_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr)
    if name == "adadelta":
        return torch.optim.Adadelta(params, lr=lr)
    # fallback
    return torch.optim.Adam(params, lr=lr)


def train_eval_one_trial_TRA(
        ds: Dataset,
        scalers: dict,
        L: int,
        H: int,
        device: str,
        trial_cfg: dict,
        sens_epochs: int = 50,
):
    """
    单次 trial：用 trial_cfg 的超参训练 TRA_LSTM 若干 epoch，返回 val_loss（越小越好）
    - 为了“敏感性分析”，建议禁用早停，固定训练步数 sens_epochs
    - scheduler 这里不做（否则把 lr 和 step 混在一起影响重要性解释）
    """
    batch_size = int(trial_cfg["batch_size"])
    train_L, valid_L, test_L, *_ = make_loaders(ds, CONFIG["split_ratios"], batch_size=batch_size)

    # 用 trial 超参建模
    model = TRA_LSTM(
        input_dim=7,
        hidden=int(trial_cfg["hidden"]),
        layers=int(trial_cfg["layers"]),
        dropout=float(trial_cfg["dropout"]),
        beta_att=float(trial_cfg["beta_att"]),
        stronger=CONFIG["stronger"],
        attn_drop=float(trial_cfg["attn_drop"]),
    ).to(device)

    physics = PhysicsLoss(
        CONFIG,
        scaler_X=scalers["scaler_X"],
        a_tg=CONFIG["a_tg"], b_tg=CONFIG["b_tg"],
        learn_tg=CONFIG["learn_tg_map"]
    ).to(device)

    params = list(model.parameters())
    if CONFIG["learn_tg_map"]:
        params += list(physics.parameters())

    opt = _build_optimizer(trial_cfg["optimizer"], params, lr=float(trial_cfg["lr"]))

    # 固定训练 sens_epochs，不早停，不 scheduler
    curves = {"train": [], "val": []}
    for ep in range(sens_epochs):
        _ = train_one_epoch(model, train_L, opt, scheduler=None, physics_loss=physics,
                            device=device, scaler_y=scalers["scaler_y"],
                            record_curve=False, curve_list=None, epoch_idx=ep)
        va, _, *_ = evaluate(model, valid_L, physics, device, scalers["scaler_y"],
                             record_curve=False, curve_list=None)

    return float(va)  # val loss


def run_sensitivity_TRA_LSTM_and_save_csv(
        ds: Dataset,
        scalers: dict,
        L: int,
        H_list: list,
        out_dir: str,
        n_trials: int = 40,
        sens_epochs: int = 50,
        seed: int = 2025
):
    """
    对每个 H（任务）：
      1) 随机采样 n_trials 组超参
      2) 每组训练 TRA-LSTM sens_epochs
      3) 根据 trials -> 计算 importance -> 保存 sens_{task}_L{L}.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 超参采样空间（你可以微调范围）
    space = dict(
        optimizer=["adam", "sgd", "rmsprop", "adagrad", "adadelta"],
        lr=(1e-4, 2e-3),  # log-uniform
        dropout=(0.0, 0.30),
        batch_size=[64, 128, 256],
        hidden=[32, 64, 128],
        layers=[1, 2],
        beta_att=(0.0, 1.2),  # risk-att strength
        attn_drop=(0.0, 0.30),
    )

    def _sample_one():
        # log-uniform for lr
        lr_lo, lr_hi = space["lr"]
        lr = float(10 ** rng.uniform(np.log10(lr_lo), np.log10(lr_hi)))

        d_lo, d_hi = space["dropout"]
        dropout = float(rng.uniform(d_lo, d_hi))

        b = int(rng.choice(space["batch_size"]))
        h = int(rng.choice(space["hidden"]))
        lay = int(rng.choice(space["layers"]))

        ba_lo, ba_hi = space["beta_att"]
        beta_att = float(rng.uniform(ba_lo, ba_hi))

        ad_lo, ad_hi = space["attn_drop"]
        attn_drop = float(rng.uniform(ad_lo, ad_hi))

        opt = str(rng.choice(space["optimizer"]))

        return dict(
            optimizer=opt,
            lr=lr,
            dropout=dropout,
            batch_size=b,
            hidden=h,
            layers=lay,
            beta_att=beta_att,
            attn_drop=attn_drop,
        )

    device = CONFIG["device"]

    # 对每个任务 H 输出一个 trials + 一个 importance csv
    for H in H_list:
        print(f"\n[Sensitivity] TRA-LSTM | L={L}, H={H} | trials={n_trials}, epochs={sens_epochs}")

        # 重新构造 ds（因为 H 变化会影响样本构造）
        ds_H = LimeKilnSeqDataset(
            [CONFIG["original_csv"]],
            time_step=L,
            pred_horizon=H,
            fit_scaler=False,
            global_scaler=scalers
        )

        trial_rows = []
        for t in range(n_trials):
            cfg_t = _sample_one()
            val_loss = train_eval_one_trial_TRA(
                ds=ds_H, scalers=scalers, L=L, H=H, device=device,
                trial_cfg=cfg_t, sens_epochs=sens_epochs
            )
            row = dict(**cfg_t, val_loss=val_loss, H=H, L=L, trial=t + 1)
            trial_rows.append(row)
            print(f"[Sensitivity] H={H} trial {t + 1:02d}/{n_trials} | val_loss={val_loss:.6f} | {cfg_t}")

        df_trials = pd.DataFrame(trial_rows)

        # 计算重要性
        df_imp = compute_importance_from_trials(df_trials, metric_col="val_loss")

        # 任务名映射：按你论文术语
        if H == 1:
            task_tag = "Single-step"
        elif H in (2, 3):
            task_tag = "Short-term"
        elif H in (4, 5):
            task_tag = "Middle-term"
        else:
            task_tag = "Long-term"

        # 保存
        trials_path = os.path.join(out_dir, f"sens_trials_{task_tag}_L{L}_H{H}.csv")
        imp_path = os.path.join(out_dir, f"sens_{task_tag}_L{L}.csv")

        df_trials.to_csv(trials_path, index=False)
        df_imp.to_csv(imp_path, index=False)

        print(f"[Sensitivity] trials saved → {trials_path}")
        print(f"[Sensitivity] importance saved → {imp_path}")


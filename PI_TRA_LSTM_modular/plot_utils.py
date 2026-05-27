# -*- coding: utf-8 -*-
from .common import *
from .config import CONFIG
from .models.model_zoo import *
from .physics import metrics_mae_rmse_mape_r2
from .train_torch import build_model, make_loaders

# -------------------------
# 7. 绘图
# -------------------------
def plot_error_series_and_boxplot(model_prefix_map: dict, L: int, H: int, outname=None,
                                  y_limit=(-25, 25), max_points=None):
    """
    Fig.12风格：
      左：误差序列 e(t)=T_pred-T_true
      右：误差箱线图（带均值红点）

    model_prefix_map: {"显示名": save_prefix, ...}
      读取文件：pred_{save_prefix}.csv（列：T_true,T_pred）
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    errors = {}
    N_ref = None

    # 1) 读入误差
    for disp, prefix in model_prefix_map.items():
        f = os.path.join(CONFIG["result_dir"], f"pred_{prefix}.csv")
        if not os.path.exists(f):
            print(f"[ErrPlot] missing: {f}")
            continue

        df = pd.read_csv(f)
        if ("T_true" not in df.columns) or ("T_pred" not in df.columns):
            print(f"[ErrPlot] invalid columns in: {f}")
            continue

        e = (df["T_pred"].values.astype(float) - df["T_true"].values.astype(float))
        if max_points is not None:
            e = e[:max_points]

        errors[disp] = e
        N_ref = len(e) if N_ref is None else min(N_ref, len(e))

    if len(errors) == 0:
        print("[ErrPlot] no pred_*.csv found. Skip.")
        return

    # 对齐长度，确保同一 x 轴
    for k in list(errors.keys()):
        errors[k] = errors[k][:N_ref]

    # 2) 画图：1×2
    fig = plt.figure(figsize=(12.5, 4.6))

    # ---------- 左：误差序列 ----------
    ax1 = plt.subplot(1, 2, 1)
    x = np.arange(N_ref)
    for disp, e in errors.items():
        ax1.plot(x, e, label=disp, linewidth=1.0, alpha=0.95)

    ax1.axhline(0.0, linewidth=1.0, linestyle="--", alpha=0.8)
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Temperature prediction error (°C)")
    ax1.set_ylim(y_limit[0], y_limit[1])
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)

    # ---------- 右：箱线图 ----------
    ax2 = plt.subplot(1, 2, 2)
    labels = list(errors.keys())
    data = [errors[k] for k in labels]

    bp = ax2.boxplot(
        data,
        labels=labels,
        showfliers=True,
        patch_artist=True,
        widths=0.5
    )

    # 箱体淡色（不指定具体颜色也能用默认，但 patch_artist=True 会变成白色；
    # 这里给一个很淡的灰蓝，不会像原图那样“抄袭感”强）
    for box in bp["boxes"]:
        box.set_alpha(0.35)

    # 均值红点
    means = [np.mean(d) for d in data]
    ax2.scatter(np.arange(1, len(labels) + 1), means, marker="o", s=30, zorder=3)

    ax2.axhline(0.0, linewidth=1.0, linestyle="--", alpha=0.8)
    ax2.set_xlabel("Model name")
    ax2.set_ylabel("Temperature prediction error (°C)")
    ax2.set_ylim(y_limit[0], y_limit[1])
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    if outname is None:
        outname = f"Fig12_error_compare_L{L}_H{H}.png"
    outpath = os.path.join(CONFIG["result_dir"], outname)
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[ErrPlot] saved → {outpath}")


def plot_convergence_curves_from_curve_csv(
        model_prefix_map: dict,
        L: int,
        H: int,
        use="val",  # "val" or "train"
        max_epochs=500,
        outname=None,
):
    """
    model_prefix_map: {"显示名": save_prefix, ...}
      save_prefix 对应 curve_{save_prefix}.csv
    use: 画验证损失还是训练损失
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.0, 4.2))
    has_any = False

    for disp, prefix in model_prefix_map.items():
        f = os.path.join(CONFIG["result_dir"], f"curve_{prefix}.csv")
        if not os.path.exists(f):
            # 没曲线就跳过（例如EN/SVM）
            print(f"[Convergence] missing curve: {f}")
            continue

        dfc = pd.read_csv(f)
        if use not in dfc.columns:
            print(f"[Convergence] curve file has no column '{use}': {f}")
            continue

        y = dfc[use].values.astype(float)
        if max_epochs is not None:
            y = y[:max_epochs]

        x = dfc["epoch"].values[:len(y)] if "epoch" in dfc.columns else (np.arange(len(y)) + 1)
        plt.plot(x, y, label=disp, linewidth=1.6)
        has_any = True

    if not has_any:
        print("[Convergence] no curve files found. Skip.")
        plt.close()
        return

    plt.xlabel("Training epochs")
    plt.ylabel("Loss" if use in ("train", "val") else use)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

    title_use = "Validation" if use == "val" else "Training"
    plt.title(f"Convergence curves ({title_use} loss), L={L}, H={H}")
    plt.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    if outname is None:
        outname = f"Fig7_convergence_{title_use.lower()}_L{L}_H{H}.png"
    outpath = os.path.join(CONFIG["result_dir"], outname)
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[Convergence] saved → {outpath}")


def plot_sensitivity_panels(
        task_csv_list,
        panel_labels=("a", "b", "c"),
        fig_title=None,
        outname="Fig6_sensitivity.png",
        topk=10
):
    """
    task_csv_list: list of tuples (task_name, csv_path)
      csv columns required: ["hyperparam","importance"]
      (task_name 用作子图标题)
    输出：一个拼图（最多3个子图），并且每个 task 也可单独输出（见下面单图函数）
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    n = len(task_csv_list)
    n = min(n, 3)  # 复刻 Fig.6 三联图
    fig, axes = plt.subplots(1, n, figsize=(12.6, 4.0))
    if n == 1:
        axes = [axes]

    for i in range(n):
        task_name, csv_path = task_csv_list[i]
        ax = axes[i]

        df = pd.read_csv(csv_path)
        df = df[["hyperparam", "importance"]].copy()
        df["importance"] = df["importance"].astype(float)

        df = df.sort_values("importance", ascending=False).head(topk)
        df = df.iloc[::-1]  # 反过来：最大在最上面（横向条形更像论文图）
        y = df["hyperparam"].values
        x = df["importance"].values

        ax.barh(y, x)
        ax.set_xlabel("Hyperparameter importance")
        ax.set_title(f"({panel_labels[i]})")

        # 网格风格（类似你图中的虚线网格）
        ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.5)

        # 数值标注（条形末端）
        for yy, xx in zip(range(len(y)), x):
            ax.text(xx + 0.01 * (x.max() + 1e-9), yy, f"{xx:.2f}", va="center", fontsize=9)

        # 用 task_name 作为更外层标题（放在图内靠上）
        ax.text(0.5, 1.05, task_name, transform=ax.transAxes, ha="center", fontsize=11)

    if fig_title is not None:
        fig.suptitle(fig_title, y=1.02, fontsize=12)

    plt.tight_layout()
    outpath = os.path.join(CONFIG["result_dir"], outname)
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[Sensitivity] panel figure saved → {outpath}")


def plot_sensitivity_single(
        task_name,
        csv_path,
        outname,
        topk=10
):
    """你说“我是想输出每一张图”：这个就是单独输出每个 task 一张图"""
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    df = pd.read_csv(csv_path)
    df = df[["hyperparam", "importance"]].copy()
    df["importance"] = df["importance"].astype(float)
    df = df.sort_values("importance", ascending=False).head(topk)
    df = df.iloc[::-1]

    plt.figure(figsize=(4.2, 3.8))
    plt.barh(df["hyperparam"].values, df["importance"].values)
    plt.xlabel("Hyperparameter importance")
    plt.title(task_name)
    plt.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.8)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    for i, (hp, val) in enumerate(zip(df["hyperparam"].values, df["importance"].values)):
        plt.text(val + 0.01 * (df["importance"].max() + 1e-9), i, f"{val:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(CONFIG["result_dir"], outname)
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[Sensitivity] single figure saved → {outpath}")


def plot_metric_bars(df_metric: pd.DataFrame, tag_prefix, metric_col, title):
    models = df_metric["model"].tolist()
    values = df_metric[metric_col].values
    plt.figure(figsize=(10, 4))
    plt.bar(models, values)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    fname = os.path.join(CONFIG["result_dir"], f"{tag_prefix}_{metric_col}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[图] {title} → {fname}")


def plot_attention_heatmaps(trainer_outputs, ds, scalers, L, H):
    """
    Attn(PI-TRA-LSTM) vs Attn(TRA-LSTM)
    """
    device = CONFIG["device"]
    _, _, test_L, *_ = make_loaders(ds, CONFIG["split_ratios"], batch_size=CONFIG["base_batch"])

    tra_info = next((x for x in trainer_outputs if x["model"] in ("TRA_LSTM", "TRA-LSTM", "TRA")), None)
    ptra_info = next((x for x in trainer_outputs if x["model"] in ("PI_TRA_LSTM", "PI-TRA-LSTM")), None)
    if (tra_info is None) or (ptra_info is None):
        print("[注意力图] 缺少 TRA-LSTM 或 PI-TRA-LSTM 结果，跳过绘图。")
        return

    tra = build_model("TRA_LSTM", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"],
                                       dropout=CONFIG["base_dropout"], stronger=CONFIG["stronger"])).to(device)
    tra_ckpt = os.path.join(CONFIG["ckpt_dir"],
                            f"TRA_LSTM_L{L}_H{H}_E{CONFIG['base_epochs']}_B{CONFIG['base_batch']}_best.pt")
    if not os.path.exists(tra_ckpt):
        cands = [f for f in os.listdir(CONFIG["ckpt_dir"]) if f.startswith("TRA_LSTM_") and f.endswith("_best.pt")]
        if cands:
            tra_ckpt = os.path.join(CONFIG["ckpt_dir"], cands[0])
    tra.load_state_dict(torch.load(tra_ckpt, map_location=device))

    ptra = build_model("PI_TRA_LSTM", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"],
                                           dropout=CONFIG["base_dropout"], stronger=CONFIG["stronger"])).to(device)
    ptra_ckpt = os.path.join(CONFIG["ckpt_dir"],
                             f"PI_TRA_LSTM_L{L}_H{H}_E{CONFIG['base_epochs']}_B{CONFIG['base_batch']}_best.pt")
    if not os.path.exists(ptra_ckpt):
        cands = [f for f in os.listdir(CONFIG["ckpt_dir"]) if f.startswith("PI_TRA_LSTM_") and f.endswith("_best.pt")]
        if cands:
            ptra_ckpt = os.path.join(CONFIG["ckpt_dir"], cands[0])
    ptra.load_state_dict(torch.load(ptra_ckpt, map_location=device))

    X_seq, _, S_seq, _, _, _ = next(iter(test_L))
    X_seq = X_seq.to(device);
    S_seq = S_seq.to(device)

    with torch.no_grad():
        _ = tra(X_seq, savg=S_seq)
        alpha_tra = tra.last_alpha[0].detach().cpu().numpy()

    with torch.no_grad():
        _ = ptra(X_seq, savg=S_seq)
        alpha_ptra = ptra.last_alpha[0].detach().cpu().numpy()

    def _heat(vec, title, fname):
        plt.figure(figsize=(8, 1.8))
        plt.imshow(vec.reshape(1, -1), aspect="auto", cmap="viridis")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.yticks([])
        plt.xlabel("Time steps (last L)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["result_dir"], fname), dpi=150)
        plt.close()

    _heat(alpha_ptra, f"(a) Attention by PI-TRA-LSTM (H={H})", f"att_PI_TRA_LSTM_L{L}_H{H}.png")
    _heat(alpha_tra, f"(b) Attention by TRA-LSTM (H={H})", f"att_TRA_LSTM_L{L}_H{H}.png")
    print("[图] PI-TRA-LSTM vs TRA-LSTM attention 热力图已完成。")


def save_curve_csv(curves: dict, save_prefix: str):
    os.makedirs(CONFIG["result_dir"], exist_ok=True)

    # 真实长度（早停后 < base_epochs）
    n_tr = len(curves.get("train", []))
    n_va = len(curves.get("val", []))

    if n_tr == 0 and n_va == 0:
        print(f"[曲线] curves为空，跳过保存：{save_prefix}")
        return None

    # 对齐长度：用最短长度，确保 dataframe 不报错
    n = min(n_tr if n_tr > 0 else n_va, n_va if n_va > 0 else n_tr)

    dfc = pd.DataFrame({
        "epoch": np.arange(1, n + 1),
        "train": curves.get("train", [])[:n] if n_tr > 0 else [np.nan] * n,
        "val": curves.get("val", [])[:n] if n_va > 0 else [np.nan] * n,
    })

    f = os.path.join(CONFIG["result_dir"], f"curve_{save_prefix}.csv")
    dfc.to_csv(f, index=False)
    print(f"[曲线] curve csv saved → {f}")
    return f


def _segments_from_binary(arr01):
    arr01 = np.asarray(arr01).astype(int)
    if len(arr01) == 0:
        return []
    segs = []
    s = 0
    cur = arr01[0]
    for i in range(1, len(arr01)):
        if arr01[i] != cur:
            segs.append((s, i, cur))
            s = i
            cur = arr01[i]
    segs.append((s, len(arr01), cur))
    return segs


def plot_6model_val_loss(curve_prefix_map: dict, L: int, H: int, outname=None):
    plt.figure(figsize=(7.2, 3.8))
    has_any = False
    for disp, prefix in curve_prefix_map.items():
        f = os.path.join(CONFIG["result_dir"], f"curve_{prefix}.csv")
        if not os.path.exists(f):
            continue
        dfc = pd.read_csv(f)
        if "val" not in dfc.columns:
            continue
        plt.plot(dfc["val"].values, label=disp)
        has_any = True

    if not has_any:
        print("[Val-loss 6模型] 缺少 curve_*.csv，跳过。")
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss (SmoothL1)")
    plt.title(f"Val-loss curve of compared algorithms (L={L}, H={H})")
    plt.text(0.01, 0.02, "Note: PI models may converge slightly slower but more stably.",
             transform=plt.gca().transAxes, fontsize=9)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    if outname is None:
        outname = f"VALLOSS_models_L{L}_H{H}.png"
    out = os.path.join(CONFIG["result_dir"], outname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[图] Val-loss 模型对比 → {out}")


def plot_6model_prediction(predmeta_prefix_map: dict, L: int, H: int,
                           zoom1=(100, 200), zoom2=(300, 400),
                           sample_interval_s=60,
                           x_unit="min",
                           outname=None):
    first_key = next(iter(predmeta_prefix_map.keys()))
    f0 = os.path.join(CONFIG["result_dir"], f"predmeta_{predmeta_prefix_map[first_key]}.csv")
    if not os.path.exists(f0):
        print("[Pred] 缺少 predmeta_*.csv，跳过。")
        return

    df0 = pd.read_csv(f0)
    y_true = df0["T_true"].values.astype(float)
    mode = df0["mode_last"].values.astype(int)
    N = len(y_true)

    preds = []
    for disp, prefix in predmeta_prefix_map.items():
        f = os.path.join(CONFIG["result_dir"], f"predmeta_{prefix}.csv")
        if not os.path.exists(f):
            continue
        dfi = pd.read_csv(f)
        n = min(N, len(dfi))
        preds.append((disp, dfi["T_pred"].values.astype(float)[:n]))

    if len(preds) == 0:
        print("[Pred] 没有可用预测文件，跳过。")
        return

    # --------- 真实时间轴 ---------
    t_s = np.arange(N) * float(sample_interval_s)
    if x_unit == "min":
        t = t_s / 60.0
        xlabel = "Time (min)"
    elif x_unit == "h":
        t = t_s / 3600.0
        xlabel = "Time (h)"
    else:
        t = t_s
        xlabel = "Time (s)"

    fig = plt.figure(figsize=(11.2, 6.8))

    # (a) 全段
    ax1 = plt.subplot(3, 1, 1)
    for s, e, v in _segments_from_binary(mode[:N]):
        if v == 1:
            ax1.axvspan(t[s], t[e - 1] if e - 1 < N else t[-1], alpha=0.08)
    ax1.plot(t, y_true, label="Ground Truth", linewidth=1.6)
    for disp, yp in preds:
        ax1.plot(t[:len(yp)], yp, label=disp, linewidth=1.0, alpha=0.90)
    ax1.set_title("(a) Test prediction (shaded regions indicate mode=1)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(ncol=3, fontsize=8)

    # (b) zoom1
    b0, b1 = zoom1
    b0 = max(0, b0);
    b1 = min(N, b1)
    ax2 = plt.subplot(3, 1, 2)
    for s, e, v in _segments_from_binary(mode[b0:b1]):
        if v == 1:
            ax2.axvspan(t[b0 + s], t[b0 + e - 1] if (b0 + e - 1) < N else t[-1], alpha=0.08)
    ax2.plot(t[b0:b1], y_true[b0:b1], label="Ground Truth", linewidth=1.6)
    for disp, yp in preds:
        ax2.plot(t[b0:b1], yp[b0:b1], label=disp, linewidth=1.0, alpha=0.90)
    ax2.set_title(f"(b) Zoom-in: idx={b0}–{b1}  (≈ {t[b0]:.1f}–{t[b1 - 1]:.1f} {xlabel.split()[-1].strip('()')})")
    ax2.set_ylabel("Temperature (°C)")

    # (c) zoom2
    c0, c1 = zoom2
    c0 = max(0, c0);
    c1 = min(N, c1)
    ax3 = plt.subplot(3, 1, 3)
    for s, e, v in _segments_from_binary(mode[c0:c1]):
        if v == 1:
            ax3.axvspan(t[c0 + s], t[c0 + e - 1] if (c0 + e - 1) < N else t[-1], alpha=0.08)
    ax3.plot(t[c0:c1], y_true[c0:c1], label="Ground Truth", linewidth=1.6)
    for disp, yp in preds:
        ax3.plot(t[c0:c1], yp[c0:c1], label=disp, linewidth=1.0, alpha=0.90)
    ax3.set_title(f"(c) Zoom-in: idx={c0}–{c1}  (≈ {t[c0]:.1f}–{t[c1 - 1]:.1f} {xlabel.split()[-1].strip('()')})")
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel("Temperature (°C)")

    plt.tight_layout()
    if outname is None:
        outname = f"PRED_model_L{L}_H{H}.png"
    out = os.path.join(CONFIG["result_dir"], outname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[图] Prediction（三段）→ {out}")


def plot_prediction_for_model_list(prefix_map: dict, L: int, H: int,
                                   zoom1=(100, 200), zoom2=(300, 400),
                                   sample_interval_s=60,
                                   x_unit="min",
                                   outname=None):
    first_key = next(iter(prefix_map.keys()))
    f0 = os.path.join(CONFIG["result_dir"], f"predmeta_{prefix_map[first_key]}.csv")
    if not os.path.exists(f0):
        print("[Pred] 缺少 predmeta 文件，跳过。")
        return

    df0 = pd.read_csv(f0)
    y_true = df0["T_true"].values.astype(float)
    mode = df0["mode_last"].values.astype(int)
    N = len(y_true)

    preds = []
    for disp, prefix in prefix_map.items():
        f = os.path.join(CONFIG["result_dir"], f"predmeta_{prefix}.csv")
        if not os.path.exists(f):
            print(f"[Pred] missing: {f}")
            continue
        dfi = pd.read_csv(f)
        n = min(N, len(dfi))
        preds.append((disp, dfi["T_pred"].values.astype(float)[:n]))

    if len(preds) == 0:
        print("[Pred] 没有可用预测文件，跳过。")
        return

    # --------- 真实时间轴 ---------
    t_s = np.arange(N) * float(sample_interval_s)
    if x_unit == "min":
        t = t_s / 60.0
        xlabel = "Time (min)"
    elif x_unit == "h":
        t = t_s / 3600.0
        xlabel = "Time (h)"
    else:
        t = t_s
        xlabel = "Time (s)"

    plt.figure(figsize=(11.2, 6.8))

    ax1 = plt.subplot(3, 1, 1)
    for s, e, v in _segments_from_binary(mode[:N]):
        if v == 1:
            ax1.axvspan(t[s], t[e - 1] if e - 1 < N else t[-1], alpha=0.08)
    ax1.plot(t, y_true, label="Ground Truth", linewidth=1.6)
    for disp, yp in preds:
        ax1.plot(t[:len(yp)], yp, label=disp, linewidth=1.0, alpha=0.90)
    ax1.set_title("(a) Test prediction (shaded: mode=1)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(ncol=3, fontsize=8)

    b0, b1 = zoom1
    b0 = max(0, b0);
    b1 = min(N, b1)
    ax2 = plt.subplot(3, 1, 2)
    for s, e, v in _segments_from_binary(mode[b0:b1]):
        if v == 1:
            ax2.axvspan(t[b0 + s], t[b0 + e - 1] if (b0 + e - 1) < N else t[-1], alpha=0.08)
    ax2.plot(t[b0:b1], y_true[b0:b1], label="Ground Truth", linewidth=1.6)
    for disp, yp in preds:
        ax2.plot(t[b0:b1], yp[b0:b1], label=disp, linewidth=1.0, alpha=0.90)
    ax2.set_title(f"(b) Zoom-in: idx={b0}–{b1}")
    ax2.set_ylabel("Temperature (°C)")

    c0, c1 = zoom2
    c0 = max(0, c0);
    c1 = min(N, c1)
    ax3 = plt.subplot(3, 1, 3)
    for s, e, v in _segments_from_binary(mode[c0:c1]):
        if v == 1:
            ax3.axvspan(t[c0 + s], t[c0 + e - 1] if (c0 + e - 1) < N else t[-1], alpha=0.08)
    ax3.plot(t[c0:c1], y_true[c0:c1], label="Ground Truth", linewidth=1.6)
    for disp, yp in preds:
        ax3.plot(t[c0:c1], yp[c0:c1], label=disp, linewidth=1.0, alpha=0.90)
    ax3.set_title(f"(c) Zoom-in: idx={c0}–{c1}")
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel("Temperature (°C)")

    plt.tight_layout()
    if outname is None:
        outname = f"PRED_custom_L{L}_H{H}.png"
    out = os.path.join(CONFIG["result_dir"], outname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[图] Prediction（三段）→ {out}")


@torch.no_grad()
def collect_pred_and_attention_matrix(model, loader, device, scaler_y, which="att"):
    """
    which:
      - "att": 取 model.last_alpha
      - "reatt": 取 model.last_alpha_re
    return:
      y_true_c, y_pred_c: (T,)
      A: (T, L) 其中 L=history window length
    """
    model.eval()
    ys, yhs = [], []
    As = []

    # 修复：解包 6 个返回值（包括 raw_idx）
    for X_seq, y_next, S_seq, mode_seq, y_prev_true, raw_idx in loader:
        X_seq = X_seq.to(device)
        S_seq = S_seq.to(device)

        # 与你训练/评估一致的 forward 入口
        if isinstance(model, (TRA_LSTM, PI_TRA_LSTM)):
            y_hat = model(X_seq, savg=S_seq)
        else:
            y_hat = model(X_seq)

        ys.append(y_next.detach().cpu().numpy())
        yhs.append(y_hat.detach().cpu().numpy())

        if which == "att":
            a = getattr(model, "last_alpha", None)
        else:
            a = getattr(model, "last_alpha_re", None)

        if a is None:
            raise RuntimeError(f"[collect] model has no weights for which={which}")

        As.append(a.detach().cpu().numpy())  # (B, L)

    y_true = np.concatenate(ys, axis=0)  # (T,1)
    y_pred = np.concatenate(yhs, axis=0)  # (T,1)
    A = np.concatenate(As, axis=0)  # (T,L)

    y_true_c = scaler_y.inverse_transform(y_true).reshape(-1)
    y_pred_c = scaler_y.inverse_transform(y_pred).reshape(-1)

    return y_true_c, y_pred_c, A


def plot_pair_curve_heatmap(
        left_pack, right_pack,
        sample_interval_s=60,
        L=30,
        time_unit="min",  # "s" / "min" / "hour"
        fig_title="Comparison of attention weights",
        outpath="Fig_pair.png",
        cmap="coolwarm",
        mode="deviation",  # "raw" or "deviation"
        use_quantile=True,  # raw模式下是否用分位数压缩色条
        q_low=0.05,
        q_high=0.95,
        focus_range=None,  # 例如 (2500, 3500)；None表示全局
        pred_ylim=None,  # 例如 (875, 970)
        linewidth_true=1.2,
        linewidth_pred=1.0,
        bottom_title=None,  # 新增：在图下方显示的标题，如 "(a)"
):
    """
    left_pack / right_pack:
        {
            "y_true": array(T,),
            "y_pred": array(T,),
            "A": array(T, L),
            "title": "(a) PI-TRA-LSTM"
        }

    参数说明
    ----------
    mode:
        - "raw": 直接画原始注意力权重
        - "deviation": 画相对均匀分布(1/L)的偏离，更适合突出热点，推荐

    focus_range:
        - None: 全时间范围
        - (start_idx, end_idx): 只截取指定预测时间段，例如 (2500, 3500)

    pred_ylim:
        - 上方预测曲线y轴范围，None表示自动

    bottom_title:
        - 在图下方显示的标题，如 "(a)" 或 "(b)"
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # ========= 1) 时间单位换算 =========
    unit_scale = {"s": 1.0, "min": 60.0, "hour": 3600.0}
    if time_unit not in unit_scale:
        time_unit = "min"
    scale = unit_scale[time_unit]

    # ========= 2) 取数据 =========
    y_true_L = np.asarray(left_pack["y_true"]).reshape(-1)
    y_pred_L = np.asarray(left_pack["y_pred"]).reshape(-1)
    A_L = np.asarray(left_pack["A"])

    y_true_R = np.asarray(right_pack["y_true"]).reshape(-1)
    y_pred_R = np.asarray(right_pack["y_pred"]).reshape(-1)
    A_R = np.asarray(right_pack["A"])

    # 对齐长度
    T1 = min(len(y_true_L), len(y_pred_L), A_L.shape[0])
    T2 = min(len(y_true_R), len(y_pred_R), A_R.shape[0])
    T = min(T1, T2)

    y_true_L = y_true_L[:T]
    y_pred_L = y_pred_L[:T]
    A_L = A_L[:T, :L]

    y_true_R = y_true_R[:T]
    y_pred_R = y_pred_R[:T]
    A_R = A_R[:T, :L]

    # ========= 3) 可选：只截取局部区间 =========
    if focus_range is not None:
        s_idx, e_idx = focus_range
        s_idx = max(0, int(s_idx))
        e_idx = min(T, int(e_idx))
        if e_idx <= s_idx:
            raise ValueError(f"focus_range={focus_range} 非法。")

        y_true_L = y_true_L[s_idx:e_idx]
        y_pred_L = y_pred_L[s_idx:e_idx]
        A_L = A_L[s_idx:e_idx, :]

        y_true_R = y_true_R[s_idx:e_idx]
        y_pred_R = y_pred_R[s_idx:e_idx]
        A_R = A_R[s_idx:e_idx, :]

        x_idx = np.arange(s_idx, e_idx)
    else:
        x_idx = np.arange(T)

    T_plot = len(x_idx)

    # ========= 4) 横轴 / 纵轴 =========
    x = x_idx * (sample_interval_s / scale)
    y_hist = np.arange(L) * (sample_interval_s / scale)

    # ========= 5) 热力图数据处理 =========
    if mode == "deviation":
        # 相对均匀注意力偏离
        uniform_val = 1.0 / L
        A_L_plot = A_L - uniform_val
        A_R_plot = A_R - uniform_val

        vmax = max(np.max(np.abs(A_L_plot)), np.max(np.abs(A_R_plot)))
        vmax = max(vmax, 1e-6)
        vmin = -vmax
        cbar_label = "Attention deviation from uniform"
    elif mode == "raw":
        A_L_plot = A_L.copy()
        A_R_plot = A_R.copy()

        all_vals = np.concatenate([A_L_plot.reshape(-1), A_R_plot.reshape(-1)])
        if use_quantile:
            vmin = np.quantile(all_vals, q_low)
            vmax = np.quantile(all_vals, q_high)
            if vmax <= vmin:
                vmin = float(np.min(all_vals))
                vmax = float(np.max(all_vals))
        else:
            vmin = float(np.min(all_vals))
            vmax = float(np.max(all_vals))

        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-6

        vmin = float(np.min(vmin, -0.1))
        vmax = float(np.max(vmax, 0.1))

        cbar_label = "Attention weight"
    else:
        raise ValueError("mode 必须是 'raw' 或 'deviation'")

    # ========= 6) 画图 =========
    fig = plt.figure(figsize=(13.5, 6.4))
    fig.suptitle(fig_title, y=0.98, fontsize=12)

    # -------- 左上：预测曲线 --------
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x, y_true_L, label="Ground Truth", linewidth=linewidth_true)
    ax1.plot(x, y_pred_L, label="Predicted", linewidth=linewidth_pred, linestyle="--")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(left_pack.get("title", "(a)"))
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    if pred_ylim is not None:
        ax1.set_ylim(pred_ylim)

    # -------- 左下：热力图 --------
    ax2 = plt.subplot(2, 2, 3)
    im1 = ax2.imshow(
        A_L_plot.T,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], y_hist[0], y_hist[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    ax2.set_xlabel(f"Prediction time ({time_unit})")
    ax2.set_ylabel(f"History time ({time_unit})")

    cb1 = plt.colorbar(im1, ax=ax2, fraction=0.03, pad=0.02)
    cb1.set_label(cbar_label, fontsize=8)
    cb1.ax.tick_params(labelsize=8)

    # -------- 右上：预测曲线 --------
    ax3 = plt.subplot(2, 2, 2)
    ax3.plot(x, y_true_R, label="Ground Truth", linewidth=linewidth_true)
    ax3.plot(x, y_pred_R, label="Predicted", linewidth=linewidth_pred, linestyle="--")
    ax3.set_ylabel("Temperature (°C)")
    ax3.set_title(right_pack.get("title", "(b)"))
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    if pred_ylim is not None:
        ax3.set_ylim(pred_ylim)

    # -------- 右下：热力图 --------
    ax4 = plt.subplot(2, 2, 4)
    im2 = ax4.imshow(
        A_R_plot.T,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], y_hist[0], y_hist[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    ax4.set_xlabel(f"Prediction time ({time_unit})")
    ax4.set_ylabel(f"History time ({time_unit})")

    cb2 = plt.colorbar(im2, ax=ax4, fraction=0.03, pad=0.02)
    cb2.set_label(cbar_label, fontsize=8)
    cb2.ax.tick_params(labelsize=8)

    # ========= 7) 添加底部标题 =========
    if bottom_title is not None:
        fig.text(0.5, 0.01, bottom_title, ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[Fig] saved → {outpath}")


def plot_pair_curve_heatmap1(
        left_pack, right_pack,
        sample_interval_s=60,
        L=30,
        time_unit="min",  # "s" / "min" / "hour"
        fig_title="Comparison of attention weights",
        outpath="Fig_pair.png",
        cmap="coolwarm",
        mode="deviation",  # "raw" or "deviation"
        use_quantile=True,  # raw模式下是否用分位数压缩色条
        q_low=0.05,
        q_high=0.95,
        focus_range=None,  # 例如 (2500, 3500)；None表示全局
        pred_ylim=None,  # 例如 (875, 970)
        linewidth_true=1.2,
        linewidth_pred=1.0,
        sym_colorbar=True,  # 是否使用对称色条（deviation模式下自动对称）
        colorbar_limit=None,  # 手动指定色条范围，例如 0.08
):
    """
    left_pack / right_pack:
        {
            "y_true": array(T,),
            "y_pred": array(T,),
            "A": array(T, L),
            "title": "(a) PI-TRA-LSTM"
        }

    参数说明
    ----------
    mode:
        - "raw": 直接画原始注意力权重
        - "deviation": 画相对均匀分布(1/L)的偏离，更适合突出热点，推荐

    focus_range:
        - None: 全时间范围
        - (start_idx, end_idx): 只截取指定预测时间段，例如 (2500, 3500)

    pred_ylim:
        - 上方预测曲线y轴范围，None表示自动

    sym_colorbar:
        - 是否使用对称色条（deviation模式下默认为True）

    colorbar_limit:
        - 手动指定色条范围，例如 0.08 表示 [-0.08, 0.08]
        - 如果为 None，则自动计算
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # ========= 1) 时间单位换算 =========
    unit_scale = {"s": 1.0, "min": 60.0, "hour": 3600.0}
    if time_unit not in unit_scale:
        time_unit = "min"
    scale = unit_scale[time_unit]

    # ========= 2) 取数据 =========
    y_true_L = np.asarray(left_pack["y_true"]).reshape(-1)
    y_pred_L = np.asarray(left_pack["y_pred"]).reshape(-1)
    A_L = np.asarray(left_pack["A"])

    y_true_R = np.asarray(right_pack["y_true"]).reshape(-1)
    y_pred_R = np.asarray(right_pack["y_pred"]).reshape(-1)
    A_R = np.asarray(right_pack["A"])

    # 对齐长度
    T1 = min(len(y_true_L), len(y_pred_L), A_L.shape[0])
    T2 = min(len(y_true_R), len(y_pred_R), A_R.shape[0])
    T = min(T1, T2)

    y_true_L = y_true_L[:T]
    y_pred_L = y_pred_L[:T]
    A_L = A_L[:T, :L]

    y_true_R = y_true_R[:T]
    y_pred_R = y_pred_R[:T]
    A_R = A_R[:T, :L]

    # ========= 3) 可选：只截取局部区间 =========
    if focus_range is not None:
        s_idx, e_idx = focus_range
        s_idx = max(0, int(s_idx))
        e_idx = min(T, int(e_idx))
        if e_idx <= s_idx:
            raise ValueError(f"focus_range={focus_range} 非法。")

        y_true_L = y_true_L[s_idx:e_idx]
        y_pred_L = y_pred_L[s_idx:e_idx]
        A_L = A_L[s_idx:e_idx, :]

        y_true_R = y_true_R[s_idx:e_idx]
        y_pred_R = y_pred_R[s_idx:e_idx]
        A_R = A_R[s_idx:e_idx, :]

        x_idx = np.arange(s_idx, e_idx)
    else:
        x_idx = np.arange(T)

    T_plot = len(x_idx)

    # ========= 4) 横轴 / 纵轴 =========
    x = x_idx * (sample_interval_s / scale)
    y_hist = np.arange(L) * (sample_interval_s / scale)

    # ========= 5) 热力图数据处理 =========
    if mode == "deviation":
        # 相对均匀注意力偏离
        uniform_val = 1.0 / L
        A_L_plot = A_L - uniform_val
        A_R_plot = A_R - uniform_val

        cbar_label = "Attention deviation from uniform"

        # 处理色条范围
        if colorbar_limit is not None:
            # 手动指定对称范围
            limit = float(colorbar_limit)
            vmin = -limit
            vmax = limit
        else:
            # 自动计算对称范围
            max_abs = max(np.max(np.abs(A_L_plot)), np.max(np.abs(A_R_plot)))
            max_abs = max(max_abs, 1e-6)
            if sym_colorbar:
                vmin = -max_abs
                vmax = max_abs
            else:
                vmin = float(np.min(A_L_plot))
                vmax = float(np.max(A_R_plot))

    elif mode == "raw":
        A_L_plot = A_L.copy()
        A_R_plot = A_R.copy()

        all_vals = np.concatenate([A_L_plot.reshape(-1), A_R_plot.reshape(-1)])

        if colorbar_limit is not None:
            # 手动指定对称范围
            limit = float(colorbar_limit)
            vmin = -limit
            vmax = limit
        elif use_quantile:
            vmin = np.quantile(all_vals, q_low)
            vmax = np.quantile(all_vals, q_high)
            if vmax <= vmin:
                vmin = float(np.min(all_vals))
                vmax = float(np.max(all_vals))
        else:
            vmin = float(np.min(all_vals))
            vmax = float(np.max(all_vals))

        # 确保范围有效
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-6

        cbar_label = "Attention weight"
    else:
        raise ValueError("mode 必须是 'raw' 或 'deviation'")

    # 将 vmin 和 vmax 转换为 float
    vmin = float(vmin)
    vmax = float(vmax)

    # ========= 6) 画图 =========
    fig = plt.figure(figsize=(13.5, 6.4))
    fig.suptitle(fig_title, y=0.98, fontsize=12)

    # -------- 左上：预测曲线 --------
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x, y_true_L, label="Ground Truth", linewidth=linewidth_true, color='black')
    ax1.plot(x, y_pred_L, label="Predicted", linewidth=linewidth_pred, linestyle="--", color='orange')
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(left_pack.get("title", "(a)"))
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    if pred_ylim is not None:
        ax1.set_ylim(pred_ylim)

    # -------- 左下：热力图 --------
    ax2 = plt.subplot(2, 2, 3)
    im1 = ax2.imshow(
        A_L_plot.T,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], y_hist[0], y_hist[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    ax2.set_xlabel(f"Prediction time ({time_unit})")
    ax2.set_ylabel(f"History time ({time_unit})")

    cb1 = plt.colorbar(im1, ax=ax2, fraction=0.03, pad=0.02)
    cb1.set_label(cbar_label, fontsize=8)
    cb1.ax.tick_params(labelsize=8)

    # -------- 右上：预测曲线 --------
    ax3 = plt.subplot(2, 2, 2)
    ax3.plot(x, y_true_R, label="Ground Truth", linewidth=linewidth_true, color='black')
    ax3.plot(x, y_pred_R, label="Predicted", linewidth=linewidth_pred, linestyle="--", color='orange')
    ax3.set_ylabel("Temperature (°C)")
    ax3.set_title(right_pack.get("title", "(b)"))
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    if pred_ylim is not None:
        ax3.set_ylim(pred_ylim)

    # -------- 右下：热力图 --------
    ax4 = plt.subplot(2, 2, 4)
    im2 = ax4.imshow(
        A_R_plot.T,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], y_hist[0], y_hist[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    ax4.set_xlabel(f"Prediction time ({time_unit})")
    ax4.set_ylabel(f"History time ({time_unit})")

    cb2 = plt.colorbar(im2, ax=ax4, fraction=0.03, pad=0.02)
    cb2.set_label(cbar_label, fontsize=8)
    cb2.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[Fig] saved → {outpath}")


def _load_best_ckpt_for(prefix: str):
    ckpt = os.path.join(CONFIG["ckpt_dir"], f"{prefix}_best.pt")
    if os.path.exists(ckpt):
        return ckpt
    # 兜底：找同模型前缀的 best
    head = prefix.split("_L")[0] + "_"
    cands = [f for f in os.listdir(CONFIG["ckpt_dir"]) if f.startswith(head) and f.endswith("_best.pt")]
    if cands:
        return os.path.join(CONFIG["ckpt_dir"], cands[0])
    return None


def draw_attention_pairs_for_L30H1(ds, scalers, L, H, sample_interval_s=60):
    """
    TRA-LSTM vs PI-TRA-LSTM attention comparison.
    """
    device = CONFIG["device"]
    _, _, test_loader, *_ = make_loaders(ds, CONFIG["split_ratios"], batch_size=CONFIG["base_batch"])

    # ---------- 1) TRA-LSTM ----------
    tra = build_model("TRA_LSTM", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"],
                                       dropout=CONFIG["base_dropout"], stronger=CONFIG["stronger"])).to(device)
    tra_prefix = f"TRA_LSTM_L{L}_H{H}_E{CONFIG['base_epochs']}_B{CONFIG['base_batch']}"
    tra_ckpt = _load_best_ckpt_for(tra_prefix)
    if tra_ckpt is None:
        print("[AttPair] TRA-LSTM ckpt not found, skip.")
        return
    tra.load_state_dict(torch.load(tra_ckpt, map_location=device))

    y_true, y_pred_tra, A_tra = collect_pred_and_attention_matrix(
        tra, test_loader, device, scalers["scaler_y"], which="att"
    )

    # ---------- 2) PI-TRA-LSTM ----------
    pi_tra = build_model("PI_TRA_LSTM", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"],
                                             dropout=CONFIG["base_dropout"], stronger=CONFIG["stronger"])).to(device)
    pitra_prefix = f"PI_TRA_LSTM_L{L}_H{H}_E{CONFIG['base_epochs']}_B{CONFIG['base_batch']}"
    pitra_ckpt = _load_best_ckpt_for(pitra_prefix)
    if pitra_ckpt is None:
        print("[AttPair] PI-TRA-LSTM ckpt not found, skip set-1.")
    else:
        pi_tra.load_state_dict(torch.load(pitra_ckpt, map_location=device))
        _, y_pred_pitra, A_pitra = collect_pred_and_attention_matrix(
            pi_tra, test_loader, device, scalers["scaler_y"], which="att"
        )

        # 全量数据图
        plot_pair_curve_heatmap(
            left_pack=dict(
                y_true=y_true,
                y_pred=y_pred_pitra,
                A=A_pitra,
                title="(a) PI-TRA-LSTM"
            ),
            right_pack=dict(
                y_true=y_true,
                y_pred=y_pred_tra,
                A=A_tra,
                title="(b) TRA-LSTM"
            ),
            sample_interval_s=60,
            L=L,
            time_unit="min",
            mode="deviation",
            colorbar_limit=0.08,
            pred_ylim=(875, 970),
            outpath=os.path.join(CONFIG["result_dir"], f"Fig_AttPair_all_data_L{L}_H{H}.png"),
        )

        # 异常段2500至3500 - 添加底部标题
        plot_pair_curve_heatmap(
            left_pack=dict(
                y_true=y_true,
                y_pred=y_pred_pitra,
                A=A_pitra,
                title="PI-TRA-LSTM"
            ),
            right_pack=dict(
                y_true=y_true,
                y_pred=y_pred_tra,
                A=A_tra,
                title="TRA-LSTM"
            ),
            sample_interval_s=60,
            L=L,
            time_unit="min",
            mode="deviation",
            focus_range=(2500, 3500),
            pred_ylim=(875, 970),
            bottom_title="(a)",  # 底部标题
            outpath=os.path.join(CONFIG["result_dir"], f"Fig_AttPair_abnormal_L{L}_H{H}.png"),
        )

        # 正常段3500至15000 - 添加底部标题
        plot_pair_curve_heatmap(
            left_pack=dict(
                y_true=y_true,
                y_pred=y_pred_pitra,
                A=A_pitra,
                title="(a) PI-TRA-LSTM"
            ),
            right_pack=dict(
                y_true=y_true,
                y_pred=y_pred_tra,
                A=A_tra,
                title="(b) TRA-LSTM"
            ),
            sample_interval_s=60,
            L=L,
            time_unit="min",
            mode="deviation",
            focus_range=(3500, 15000),
            pred_ylim=(875, 970),
            bottom_title="Normal operation (3500-15000)",  # 底部标题
            outpath=os.path.join(CONFIG["result_dir"], f"Fig_AttPair_normal_L{L}_H{H}.png"),
        )

        # 尾部段15000-17500 - 添加底部标题
        plot_pair_curve_heatmap(
            left_pack=dict(
                y_true=y_true,
                y_pred=y_pred_pitra,
                A=A_pitra,
                title="PI-TRA-LSTM"
            ),
            right_pack=dict(
                y_true=y_true,
                y_pred=y_pred_tra,
                A=A_tra,
                title="TRA-LSTM"
            ),
            sample_interval_s=60,
            L=L,
            time_unit="min",
            mode="deviation",
            focus_range=(15000, 17000),
            pred_ylim=(875, 970),
            bottom_title="(b)",  # 底部标题
            outpath=os.path.join(CONFIG["result_dir"], f"Fig_AttPair_tail_L{L}_H{H}.png"),
        )

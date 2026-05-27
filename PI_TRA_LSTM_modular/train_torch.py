# -*- coding: utf-8 -*-
from .common import *
from .config import CONFIG
from .data_process import time_series_split, make_loaders_from_datasets
from .models.model_zoo import *
from .physics import PhysicsLoss, inverse_transform_y, gated_reattn_consistency, metrics_mae_rmse_mape_r2, count_params, PHYS_CLASSES

# save_curve_csv / export_pred_with_meta are defined below in this module.

# -------------------------
# 5. 训练 / 评估（Torch）
# -------------------------
def build_model(tag: str, cfg_model) -> nn.Module:
    hid = cfg_model.get("hidden", CONFIG["base_hidden"])
    lay = cfg_model.get("layers", CONFIG["base_layers"])
    drp = cfg_model.get("dropout", CONFIG["base_dropout"])
    stronger = cfg_model.get("stronger", CONFIG["stronger"])

    tag = tag.strip()

    if tag == "LSTM":
        return LSTM_Base(hidden=hid, layers=lay, dropout=drp, stronger=stronger)

    if tag == "BiLSTM":
        return BiLSTM(hidden=hid, layers=lay, dropout=drp, stronger=stronger)

    if tag in ("TRA", "TRA_LSTM", "TRA-LSTM"):
        return TRA_LSTM(hidden=hid, layers=lay, dropout=drp, beta_att=CONFIG["beta_att"], stronger=stronger)

    if tag == "PI_LSTM":
        return PI_LSTM(hidden=hid, layers=lay, dropout=drp, stronger=stronger)

    # 新增：PI_TRA_LSTM
    if tag in ("PI_TRA_LSTM", "PI-TRA-LSTM"):
        return PI_TRA_LSTM(hidden=hid, layers=lay, dropout=drp, beta_att=CONFIG["beta_att"], stronger=stronger)

    if tag == "GRU":
        return GRU_Base(hidden=hid, layers=lay, dropout=drp)

    if tag in ("CNN_LSTM", "CNN-LSTM"):
        return CNNLSTM(hidden=hid, drop=drp)

    if tag in ("TPA_LSTM", "TPA-LSTM"):
        return TPALSTM(hidden=hid, drop=drp)

    return None


def make_loaders(ds, ratios, batch_size):
    tr, va, te = time_series_split(ds, ratios)
    L = DataLoader(tr, batch_size=batch_size, shuffle=False, drop_last=False)
    V = DataLoader(va, batch_size=batch_size, shuffle=False, drop_last=False)
    T = DataLoader(te, batch_size=batch_size, shuffle=False, drop_last=False)
    return L, V, T, tr, va, te


def weight_schedule(epoch):
    """
    returns: w_p1, w_p2, w_re
    linear warmup for first phys_warmup_epochs epochs (or 0 if disabled)
    """
    if not CONFIG.get("use_phys_warmup", False):
        return CONFIG["theta_p1"], CONFIG["theta_p2"], CONFIG["lambda_re"]

    T = int(CONFIG.get("phys_warmup_epochs", 50))
    if epoch >= T:
        return CONFIG["theta_p1"], CONFIG["theta_p2"], CONFIG["lambda_re"]

    ratio = (epoch + 1) / max(1, T)
    return CONFIG["theta_p1"] * ratio, CONFIG["theta_p2"] * ratio, CONFIG["lambda_re"] * ratio


def train_one_epoch(model, loader, optimizer, scheduler, physics_loss, device, scaler_y,
                    record_curve=False, curve_list=None, epoch_idx=0):
    model.train()
    w_p1, w_p2, w_re = weight_schedule(epoch_idx)

    total = 0.0;
    n = 0
    for X_seq, y_next, S_seq, mode_seq, y_prev_true, raw_idx in loader:
        X_seq = X_seq.to(device);
        y_next = y_next.to(device)
        S_seq = S_seq.to(device);
        mode_seq = mode_seq.to(device);
        y_prev_true = y_prev_true.to(device)

        optimizer.zero_grad()

        # forward: get attention weights for re-attn gate
        y_hat = model(X_seq)
        alpha = None
        alpha_re = None

        Ld = nn.functional.smooth_l1_loss(y_hat, y_next)

        # physics loss (only PI classes)
        Lp1 = torch.tensor(0.0, device=device)
        Lp2 = torch.tensor(0.0, device=device)
        if isinstance(model, PHYS_CLASSES):
            y_hat_inv = inverse_transform_y(y_hat, scaler_y)
            Lp1, Lp2 = physics_loss(
                y_pred=y_hat_inv, y_prev=y_prev_true,
                X_seq_norm=X_seq, Savg_seq=S_seq, mode_seq=mode_seq
            )

        # gated re-attention consistency
        Lre = gated_reattn_consistency(alpha, alpha_re, tau=CONFIG["re_gate_tau"])

        loss = CONFIG["theta_d"] * Ld + w_p1 * Lp1 + w_p2 * Lp2 + w_re * Lre
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item() * X_seq.size(0)
        n += X_seq.size(0)

    if scheduler is not None:
        scheduler.step()

    avg = total / max(n, 1)
    if record_curve and curve_list is not None:
        curve_list["train"].append(avg)
    return avg


@torch.no_grad()
def evaluate(model, loader, physics_loss, device, scaler_y, record_curve=False, curve_list=None):
    model.eval()
    total = 0.0;
    n = 0
    ys = [];
    yhs = []

    for X_seq, y_next, S_seq, mode_seq, y_prev_true, raw_idx in loader:
        X_seq = X_seq.to(device)
        y_next = y_next.to(device)

        y_hat = model(X_seq)  # eval阶段不必取attn
        Ld = nn.functional.smooth_l1_loss(y_hat, y_next)

        total += Ld.item() * X_seq.size(0)
        n += X_seq.size(0)

        ys.append(y_next.detach().cpu())
        yhs.append(y_hat.detach().cpu())

    if n == 0:
        return 0.0, (np.nan,) * 5, None, None

    avg = total / n
    if record_curve and curve_list is not None:
        curve_list["val"].append(avg)

    y_true = torch.cat(ys, 0).numpy()
    y_pred = torch.cat(yhs, 0).numpy()
    y_true_c = scaler_y.inverse_transform(y_true)
    y_pred_c = scaler_y.inverse_transform(y_pred)
    mae, rmse, mape, r2, mse = metrics_mae_rmse_mape_r2(y_pred_c, y_true_c)

    return avg, (r2, mse, mae, mape, rmse), y_true_c, y_pred_c


def parse_LH_from_save_prefix(save_prefix: str):
    """
    稳健解析 L/H：
    目标格式：..._L{int}_H{int}_...
    例如：TRA_LSTM_L30_H1_E500_B128
    """
    if not isinstance(save_prefix, str):
        return None, None

    m = re.search(r"_L(\d+)_H(\d+)(?:_|$)", save_prefix)
    if m:
        return int(m.group(1)), int(m.group(2))

    # 回退：保留你原来的 split 思路（但包 try 防止炸）
    L = None
    H = None
    try:
        if "_L" in save_prefix:
            L = int(save_prefix.split("_L")[-1].split("_")[0])  # 用 [-1] 避免命中模型名内部 _L
        if "_H" in save_prefix:
            H = int(save_prefix.split("_H")[-1].split("_")[0])
    except Exception:
        pass
    return L, H


def save_curve_csv(curves: dict, save_prefix: str):
    os.makedirs(CONFIG["result_dir"], exist_ok=True)

    n_tr = len(curves.get("train", []))
    n_va = len(curves.get("val", []))

    if n_tr == 0 and n_va == 0:
        print(f"[曲线] curves为空，跳过保存：{save_prefix}")
        return None

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


def train_and_eval_torch(tag, model, loaders, scaler_pack, device, total_epochs, patience, save_prefix, lr):
    train_L, valid_L, test_L = loaders

    physics = PhysicsLoss(
        CONFIG,
        scaler_X=scaler_pack["scaler_X"],
        a_tg=CONFIG["a_tg"], b_tg=CONFIG["b_tg"],
        learn_tg=CONFIG["learn_tg_map"]
    ).to(device)

    params = list(model.parameters())
    if CONFIG["learn_tg_map"]:
        params += list(physics.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0)

    # 我是去掉了止损点，希望能直接运行epochs=500次
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG["lr_decay_step"],
        gamma=CONFIG["lr_decay_gamma"]
    )

    n_params = count_params(model)
    curves = {"train": [], "val": []}

    best_val = float("inf")
    best_state = None
    best_metrics = None
    no_imp = 0

    t0 = time.time()
    for ep in range(total_epochs):
        tr = train_one_epoch(model, train_L, optimizer, scheduler, physics, device,
                             scaler_pack["scaler_y"], record_curve=True, curve_list=curves, epoch_idx=ep)

        va, va_m, *_ = evaluate(model, valid_L, physics, device, scaler_pack["scaler_y"],
                                record_curve=True, curve_list=curves)

        print(f"[{tag}] Epoch {ep + 1:03d}/{total_epochs} | Train {tr:.4f} | Val {va:.4f} | "
              f"R2 {va_m[0]:.4f} MSE {va_m[1]:.4f} MAE {va_m[2]:.3f} MAPE {va_m[3]:.2f}% RMSE {va_m[4]:.3f}")

        if va < best_val - 1e-6:
            best_val = va
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = va_m
            no_imp = 0
            os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)
            torch.save(best_state, os.path.join(CONFIG["ckpt_dir"], f"{save_prefix}_best.pt"))
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"[{tag}] 早停")
                break

    train_sec = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_m, yt, yp = evaluate(model, test_L, physics, device, scaler_pack["scaler_y"])

    os.makedirs(CONFIG["result_dir"], exist_ok=True)

    # 保存学习曲线图 + 曲线csv（无论是否有yt/yp都保存）
    plt.figure(figsize=(6, 3))
    plt.plot(curves["train"], label="train")
    plt.plot(curves["val"], label="val")
    plt.title(f"Learning Curve - {tag}")
    plt.xlabel("epoch");
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["result_dir"], f"lc_{save_prefix}.png"), dpi=150)
    plt.close()

    save_curve_csv(curves, save_prefix)

    if (yt is not None) and (yp is not None):
        err = (yp - yt).reshape(-1)
        plt.figure(figsize=(4.8, 3.2))
        plt.hist(err, bins=60)
        plt.title(f"Error Hist - {tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["result_dir"], f"err_{save_prefix}.png"), dpi=150)
        plt.close()

        Np = min(1000, len(yt))
        plt.figure(figsize=(10, 3.2))
        plt.plot(yt[:Np], label="True")
        plt.plot(yp[:Np], label="Pred")
        plt.legend()
        plt.title(f"{tag} Prediction (Test, first {Np})")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["result_dir"], f"pred_{save_prefix}.png"), dpi=150)
        plt.close()

        pd.DataFrame(dict(T_true=yt.reshape(-1), T_pred=yp.reshape(-1))).to_csv(
            os.path.join(CONFIG["result_dir"], f"pred_{save_prefix}.csv"), index=False
        )

        export_pred_with_meta(model, test_L, device, scaler_pack["scaler_y"], save_prefix)

        # ====== 解析 L/H（修复：避免模型名包含 _L 导致 split 误命中）======
        L_parsed, H_parsed = parse_LH_from_save_prefix(save_prefix)

        # 元信息
        meta = dict(
            model=str(tag),
            save_prefix=str(save_prefix),
            # L=int(save_prefix.split("_L")[1].split("_")[0]) if "_L" in save_prefix else None,
            # H=int(save_prefix.split("_H")[1].split("_")[0]) if "_H" in save_prefix else None,
            # 使用稳健解析结果
            L=L_parsed,
            H=H_parsed,
            n_params=int(n_params),
            train_sec=float(train_sec),
            best_val=float(best_val),
            val_best_R2=float(best_metrics[0]) if best_metrics is not None else None,
            val_best_MSE=float(best_metrics[1]) if best_metrics is not None else None,
            val_best_MAE=float(best_metrics[2]) if best_metrics is not None else None,
            val_best_MAPE=float(best_metrics[3]) if best_metrics is not None else None,
            val_best_RMSE=float(best_metrics[4]) if best_metrics is not None else None,
            test_R2=float(test_m[0]),
            test_MSE=float(test_m[1]),
            test_MAE=float(test_m[2]),
            test_MAPE=float(test_m[3]),
            test_RMSE=float(test_m[4]),
            cfg_snapshot=dict(
                base_epochs=CONFIG["base_epochs"],
                base_batch=CONFIG["base_batch"],
                base_lr=CONFIG["base_lr"],
                patience=CONFIG["base_patience"],
                theta_d=CONFIG["theta_d"],
                theta_p1=CONFIG["theta_p1"],
                theta_p2=CONFIG["theta_p2"],
                beta_att=CONFIG["beta_att"],
                risk_threshold=CONFIG["risk_threshold"],
                use_reattn_loss=CONFIG["use_reattn_loss"],
                lambda_re=CONFIG["lambda_re"],
                re_gate_tau=CONFIG["re_gate_tau"],
                use_phys_warmup=CONFIG["use_phys_warmup"],
                phys_warmup_epochs=CONFIG["phys_warmup_epochs"],
            )
        )
        with open(os.path.join(CONFIG["result_dir"], f"run_{save_prefix}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[{tag}] Params={n_params:,} | TrainSec={train_sec:.1f}s | "
          f"Test ⇒ R2={test_m[0]:.4f}, MSE={test_m[1]:.4f}, MAE={test_m[2]:.3f}, "
          f"MAPE={test_m[3]:.2f}%, RMSE={test_m[4]:.3f}")

    # best_metrics 可能为空（极端情况下），这里做兜底
    if best_metrics is None:
        best_metrics = (np.nan, np.nan, np.nan, np.nan, np.nan)

    return dict(
        params=n_params, train_sec=round(train_sec, 2),
        val_loss=best_val, val_R2=best_metrics[0], val_MSE=best_metrics[1], val_MAE=best_metrics[2],
        val_MAPE=best_metrics[3], val_RMSE=best_metrics[4],
        test_loss=float(test_loss), test_R2=test_m[0], test_MSE=test_m[1], test_MAE=test_m[2],
        test_MAPE=test_m[3], test_RMSE=test_m[4]
    )

    # ===== 分段评估 =====
    # 在 train_and_eval_torch 函数末尾
    predmeta_file = os.path.join(CONFIG["result_dir"], f"predmeta_{save_prefix}.csv")
    if os.path.exists(predmeta_file):
        seg_csv = os.path.join(CONFIG["result_dir"], f"segment_metrics_{save_prefix}.csv")
        df_seg = evaluate_segments_from_predmeta(predmeta_file, out_csv=seg_csv)


# -------------------------
# 6. 传统 ML
# -------------------------
def make_flat_features(loader):
    Xs, ys = [], []
    for X_seq, y_next, *_ in loader:
        B, L, D = X_seq.shape
        Xs.append(X_seq.reshape(B, L * D).numpy())
        ys.append(y_next.reshape(B).numpy())
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


@torch.no_grad()
def export_pred_with_meta(model, loader, device, scaler_y, save_prefix: str):
    """
    导出测试集预测 + 元信息（包含完整的索引，用于分段评估）
    产物：predmeta_{save_prefix}.csv
    包含：idx, T_true, T_pred, mode_last, Savg_last
    """
    model.eval()
    ys = []
    yhs = []
    mode_last = []
    savg_last = []
    # 需要获取测试集的起始索引
    start_idx = loader.dataset.indices[0] if hasattr(loader.dataset, 'indices') else 0

    for X_seq, y_next, S_seq, mode_seq, y_prev_true, raw_idx in loader:
        X_seq = X_seq.to(device)
        S_seq = S_seq.to(device)

        if isinstance(model, (TRA_LSTM, PI_TRA_LSTM)):
            y_hat = model(X_seq, savg=S_seq)
        else:
            y_hat = model(X_seq)

        ys.append(y_next.detach().cpu())
        yhs.append(y_hat.detach().cpu())

        mode_last.append(mode_seq[:, -1].detach().cpu())
        savg_last.append(S_seq[:, -1].detach().cpu())

    y_true = torch.cat(ys, 0).numpy()
    y_pred = torch.cat(yhs, 0).numpy()

    y_true_c = scaler_y.inverse_transform(y_true)
    y_pred_c = scaler_y.inverse_transform(y_pred)

    mode_last = torch.cat(mode_last, 0).numpy().astype(int).reshape(-1)
    savg_last = torch.cat(savg_last, 0).numpy().astype(float).reshape(-1)

    # 生成连续的索引
    indices = np.arange(start_idx, start_idx + len(y_true_c))

    df = pd.DataFrame({
        "idx": indices,
        "T_true": y_true_c.reshape(-1),
        "T_pred": y_pred_c.reshape(-1),
        "mode_last": mode_last,
        "Savg_last": savg_last
    })

    f = os.path.join(CONFIG["result_dir"], f"predmeta_{save_prefix}.csv")
    df.to_csv(f, index=False)
    print(f"[预测] predmeta saved → {f}")
    return f


def run_sklearn(tag, train_L, valid_L, test_L, scaler_y, save_prefix):
    Xtr, ytr = make_flat_features(train_L)
    Xva, yva = make_flat_features(valid_L)
    Xte, yte = make_flat_features(test_L)

    t0 = time.time()
    if tag == "EN":
        model = ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=3000, random_state=CONFIG["seed"])
    elif tag == "MLP":
        model = MLPRegressor(
            hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-4,
            batch_size=128, learning_rate_init=1e-3, max_iter=400,
            early_stopping=True, random_state=CONFIG["seed"]
        )
    elif tag == "SVM":
        model = SVR(C=10.0, epsilon=0.05, kernel="rbf", gamma="scale")
    else:
        raise ValueError(tag)

    model.fit(Xtr, ytr)
    train_sec = time.time() - t0

    inv = lambda z: scaler_y.inverse_transform(z.reshape(-1, 1)).reshape(-1)
    yva_p = inv(model.predict(Xva));
    yva = inv(yva)
    yte_p = inv(model.predict(Xte));
    yte = inv(yte)

    def _eval(yhat, y):
        mae = np.mean(np.abs(yhat - y))
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        mape = np.mean(np.abs((yhat - y) / np.clip(np.abs(y), 1e-6, None))) * 100.0
        sse = np.sum((yhat - y) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        r2 = (np.nan if sst <= 1e-12 else (1.0 - sse / sst))
        mse = rmse ** 2
        return r2, mse, mae, mape, rmse

    val_m = _eval(yva_p, yva)
    test_m = _eval(yte_p, yte)

    os.makedirs(CONFIG["result_dir"], exist_ok=True)
    err = (yte_p - yte).reshape(-1)
    plt.figure(figsize=(4.8, 3.2))
    plt.hist(err, bins=60)
    plt.title(f"Error Hist - {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["result_dir"], f"err_{save_prefix}.png"), dpi=150)
    plt.close()

    pd.DataFrame(dict(T_true=yte, T_pred=yte_p)).to_csv(
        os.path.join(CONFIG["result_dir"], f"pred_{save_prefix}.csv"), index=False
    )

    print(f"[{tag}] Params=0 | TrainSec={train_sec:.1f}s | "
          f"Test ⇒ R2={test_m[0]:.4f}, MSE={test_m[1]:.4f}, MAE={test_m[2]:.3f}, "
          f"MAPE={test_m[3]:.2f}%, RMSE={test_m[4]:.3f}")

    return dict(
        params=0, train_sec=round(train_sec, 2),
        val_loss=val_m[1], val_R2=val_m[0], val_MSE=val_m[1], val_MAE=val_m[2],
        val_MAPE=val_m[3], val_RMSE=val_m[4],
        test_loss=np.nan, test_R2=test_m[0], test_MSE=test_m[1], test_MAE=test_m[2],
        test_MAPE=test_m[3], test_RMSE=test_m[4]
    )

# -*- coding: utf-8 -*-
from .common import *
from .physics import metrics_mae_rmse_mape_r2

# -------------------------
# 7.5 分段评估函数
# -------------------------
def evaluate_segments_from_predmeta(predmeta_csv: str, out_csv: str = None):
    """
    分段评估：全局(0-17500) + 异常段(2500-3500) + 尾部段(15000-17500)
    输出指标：MSE, MAE, RMSE, MAPE, R²
    """
    df = pd.read_csv(predmeta_csv)

    # 分段定义（只评估三个关键段）
    segments_to_eval = [
        {"name": "global", "start": 0, "end": 17500, "desc": "Full dataset (0-17500)"},
        {"name": "abnormal", "start": 2500, "end": 3500, "desc": "Abnormal disturbance (2500-3500)"},
        {"name": "tail", "start": 15000, "end": 17500, "desc": "Tail drift (15000-17500)"},
    ]

    rows = []
    for seg in segments_to_eval:
        # 根据raw_idx筛选
        dfi = df[(df["idx"] >= seg["start"]) & (df["idx"] < seg["end"])].copy()

        if len(dfi) == 0:
            print(f"[警告] 分段 {seg['name']} 无数据 (区间: {seg['start']}-{seg['end']})")
            continue

        # 计算有效窗口数
        n_windows = len(dfi)

        y_true = dfi["T_true"].values.astype(float)
        y_pred = dfi["T_pred"].values.astype(float)

        mae, rmse, mape, r2, mse = metrics_mae_rmse_mape_r2(y_pred, y_true)

        rows.append({
            "segment": seg["name"],
            "description": seg["desc"],
            "n_samples": n_windows,
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2
        })

    df_seg = pd.DataFrame(rows)

    # 打印分段评估结果
    print("\n" + "=" * 60)
    print("分段评估结果 (测试集)")
    print("=" * 60)
    for _, row in df_seg.iterrows():
        print(f"{row['description']}:")
        print(f"  样本数: {row['n_samples']}")
        print(f"  MSE: {row['MSE']:.4f}, MAE: {row['MAE']:.3f}, RMSE: {row['RMSE']:.3f}")
        print(f"  MAPE: {row['MAPE']:.2f}%, R²: {row['R2']:.4f}")
        print("-" * 40)

    if out_csv is not None:
        df_seg.to_csv(out_csv, index=False)
        print(f"[分段评估] saved → {out_csv}")

    return df_seg



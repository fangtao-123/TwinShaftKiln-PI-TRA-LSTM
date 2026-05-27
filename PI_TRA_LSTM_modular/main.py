# -*- coding: utf-8 -*-
from .common import *
from .config import CONFIG, set_seed
from .data_process import *
from .models.model_zoo import *
from .physics import run_pso_calibration
from .train_torch import build_model, train_and_eval_torch, run_sklearn
from .plot_utils import *
from .sensitivity import run_sensitivity_TRA_LSTM_and_save_csv
from .segment_eval import evaluate_segments_from_predmeta

# -------------------------
# 8. 主流程（PSO → 训练对比 → 指标对比 → 附图）
# -------------------------
def run_one_combo(L, H, epochs, batch_size, seed=2025):
    set_seed(seed)

    # 1) 读原始全量数据
    df_all = load_csv_file(CONFIG["original_csv"])

    # 2) 先按时间切 raw dataframe
    df_train, df_valid, df_test = split_df_by_time(df_all, CONFIG["split_ratios"])

    # 3) 只用训练集拟合 scaler
    scalers = fit_scalers_on_train_df(df_train)

    # 4) 各自独立构造 dataset
    n_train = len(df_train)
    n_valid = len(df_valid)

    train_ds = LimeKilnSeqDatasetFromDF(
        df=df_train,
        time_step=L,
        pred_horizon=H,
        scaler_pack=scalers,
        start_raw_idx=0
    )

    valid_ds = LimeKilnSeqDatasetFromDF(
        df=df_valid,
        time_step=L,
        pred_horizon=H,
        scaler_pack=scalers,
        start_raw_idx=n_train
    )

    test_ds = LimeKilnSeqDatasetFromDF(
        df=df_test,
        time_step=L,
        pred_horizon=H,
        scaler_pack=scalers,
        start_raw_idx=n_train + n_valid
    )

    train_L, valid_L, test_L = make_loaders_from_datasets(
        train_ds, valid_ds, test_ds, batch_size
    )

    ds = LimeKilnSeqDataset(
        [CONFIG["original_csv"]],
        time_step=L,
        pred_horizon=H,
        fit_scaler=False,
        global_scaler=scalers
    )

    device = CONFIG["device"]
    results = []

    # 传统 ML
    for tag in ["EN", "MLP", "SVM"]:
        rec = run_sklearn(
            tag, train_L, valid_L, test_L,
            scalers["scaler_y"],
            save_prefix=f"{tag}_L{L}_H{H}_E{epochs}_B{batch_size}"
        )
        rec.update(dict(model=tag, L=L, H=H, epochs=epochs, batch_size=batch_size, lr=None))
        results.append(rec)

    DL_LIST = [
        ("LSTM", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"], dropout=CONFIG["base_dropout"])),
        ("BiLSTM", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"], dropout=CONFIG["base_dropout"])),
        ("GRU", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"], dropout=CONFIG["base_dropout"])),
        ("CNN_LSTM", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"], dropout=CONFIG["base_dropout"])),
        ("TPA_LSTM", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"], dropout=CONFIG["base_dropout"])),
        ("TRA_LSTM", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"], dropout=CONFIG["base_dropout"])),
        ("PI_LSTM", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"], dropout=CONFIG["base_dropout"])),
        ("PI_TRA_LSTM", dict(hidden=CONFIG["base_hidden"], layers=CONFIG["base_layers"], dropout=CONFIG["base_dropout"])),
    ]

    for tag, mc in DL_LIST:
        model = build_model(tag, {**mc, "stronger": CONFIG["stronger"]})
        if model is None:
            continue
        model = model.to(device)

        save_prefix = f"{tag}_L{L}_H{H}_E{epochs}_B{batch_size}"
        rec = train_and_eval_torch(
            tag, model, (train_L, valid_L, test_L),
            scalers, device,
            total_epochs=epochs,
            patience=CONFIG["base_patience"],
            save_prefix=save_prefix,
            lr=CONFIG["base_lr"]
        )
        rec.update(dict(
            model=tag, L=L, H=H, epochs=epochs, batch_size=batch_size, lr=CONFIG["base_lr"],
            hidden=mc["hidden"], layers=mc["layers"], dropout=mc["dropout"]
        ))
        results.append(rec)

    draw_attention_pairs_for_L30H1(ds, scalers, L, H, sample_interval_s=60)
    # 注意力图这里如果继续用，也要传 test_ds 体系下的 loader
    return results

"""
一键基准（PSO + 机理蒸馏 + 全指标对比）- 多数据段版本
在4个不同数据段上分别训练和评估：
- Segment 1: 0-5000
- Segment 2: 5000-15000
- Segment 3: 15000-17500
- Segment 4: 0-17500 (全量)
"""


def main():
    os.makedirs(CONFIG["result_dir"], exist_ok=True)
    os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)
    print(f"Running on: {CONFIG['device']} | seed={CONFIG['seed']}")
    print(f"Data: {CONFIG['original_csv']}")

    # # ===== 新增：定义4个数据段 =====
    # DATA_SEGMENTS = [
    #     {"name": "seg1_0-3500", "start": 0, "end": 3500},
    #     {"name": "seg2_3500-17000", "start": 3500, "end": 17000},
    #     {"name": "seg3_15000-17500", "start": 15000, "end": 17500},
    #     {"name": "seg4_full", "start": 0, "end": 17500},  # 全量数据
    # ]
    # 先加载原始数据并获取总长度（仅用于信息展示，不用于分段训练）
    df_full = load_csv_file(CONFIG["original_csv"])
    total_samples = len(df_full)
    print(f"\n原始数据总长度: {total_samples} 条")
    print(f"训练模式: 全量数据 (0-{total_samples}) 训练，然后在测试集上进行分段验证")
    print(f"分段验证区间: 0-17500 (全局), 2500-3500 (异常段), 15000-17500 (尾部段)")

    # PSO校准（在整个数据集上做一次，因为物理参数应该与数据段无关）
    if CONFIG.get("do_pso_calibration", False):
        print("\n[Stage 0] PSO calibration for physics params ...")
        run_pso_calibration(CONFIG["original_csv"], CONFIG)
        CONFIG["learn_tg_map"] = False
        print("[Stage 0] PSO calibration finished.\n")

    all_rec = []
    combo_id = 0

    def _sp(tagname: str, L: int, H: int):
        return f"{tagname}_L{L}_H{H}_E{CONFIG['base_epochs']}_B{CONFIG['base_batch']}"

    for L in CONFIG["time_steps"]:
        for H in CONFIG["horizons"]:
            combo_id += 1
            print(
                f"\n====== Combo #{combo_id}: L={L} | H={H} | epochs={CONFIG['base_epochs']} | batch={CONFIG['base_batch']} ======")
            results = run_one_combo(L, H, epochs=CONFIG["base_epochs"], batch_size=CONFIG["base_batch"],
                                    seed=CONFIG["seed"])
            all_rec.extend(results)
            pd.DataFrame(all_rec).to_csv(os.path.join(CONFIG["result_dir"], "benchmark_report.csv"), index=False)

            # 6模型对比图（Val-loss + Prediction三段图）----PI-TRA-LSTM模型图
            # ========= 画图：按你最新要求输出三套 =========

            # 统一前缀生成
            def _sp(tagname: str, L: int, H: int):
                tagname = tagname.replace("-", "_")  # 关键：统一到落盘命名
                return f"{tagname}_L{L}_H{H}_E{CONFIG['base_epochs']}_B{CONFIG['base_batch']}"

            # (1) 3模型 Val-loss 同图：LSTM、PI-LSTM、PI-TRA-LSTM
            curve_map_3_val = {
                "LSTM": _sp("LSTM", L, H),
                "PI-LSTM": _sp("PI_LSTM", L, H),
                "PI-TRA-LSTM": _sp("PI_TRA_LSTM", L, H),
            }
            plot_6model_val_loss(
                curve_map_3_val,
                L=L, H=H,
                outname=f"VALLOSS_3models_PI_chain_L{L}_H{H}.png"
            )

            # (2) 3模型 Prediction 三段图：LSTM、PI-LSTM、PI-TRA-LSTM   （你说要把“6模型 Prediction”改成只画这3个）
            pred_map_3 = {
                "LSTM": _sp("LSTM", L, H),
                "PI-LSTM": _sp("PI_LSTM", L, H),
                "PI-TRA-LSTM": _sp("PI_TRA_LSTM", L, H),
            }
            plot_prediction_for_model_list(
                pred_map_3,
                L=L, H=H,
                zoom1=(2500, 3500),  # 异常扰动段
                zoom2=(15000, 17500),  # 尾部漂移段
                outname=f"PRED_3models_PI_chain_L{L}_H{H}.png"
            )

            # (3) 6模型 Prediction 三段图：EN、SVM、BiLSTM、GRU、CNN_LSTM、PI-TRA-LSTM   （你说把“9模型”改成只画这6个）
            pred_map_6_cross_1 = {
                "BiLSTM": _sp("BiLSTM", L, H),
                "GRU": _sp("GRU", L, H),
                "CNN-LSTM": _sp("CNN_LSTM", L, H),
                "TPA-LSTM": _sp("TPA-LSTM", L, H),
                "TRA-LSTM": _sp("TRA-LSTM", L, H),
                "PI-TRA-LSTM": _sp("PI_TRA_LSTM", L, H),
            }
            plot_prediction_for_model_list(
                pred_map_6_cross_1,
                L=L, H=H,
                zoom1=(2500, 3500),  # 异常扰动段
                zoom2=(15000, 17500),  # 尾部漂移段
                outname=f"PRED_6models_cross_L{L}_H{H}.png"
            )

            # loss-epochs关系图
            pred_map_6_cross = {
                "EN": _sp("EN", L, H),
                "SVM": _sp("SVM", L, H),
                "BiLSTM": _sp("BiLSTM", L, H),
                "GRU": _sp("GRU", L, H),
                "CNN-LSTM": _sp("CNN_LSTM", L, H),
                "TPA-LSTM": _sp("TPA-LSTM", L, H),
                "TRA-LSTM": _sp("TRA-LSTM", L, H),
                "PI-TRA-LSTM": _sp("PI_TRA_LSTM", L, H),
            }

            plot_convergence_curves_from_curve_csv(pred_map_6_cross, L=L, H=H, use="train",
                                                   max_epochs=CONFIG["base_epochs"])

            # 误差分析图
            err_map_4 = {
                "BiLSTM": _sp("BiLSTM", L, H),
                "GRU": _sp("GRU", L, H),
                "CNN-LSTM": _sp("CNN_LSTM", L, H),
                "TPA-LSTM": _sp("TPA-LSTM", L, H),
                "TRA-LSTM": _sp("TRA-LSTM", L, H),
                "PI-TRA-LSTM": _sp("PI_TRA_LSTM", L, H),
            }

            plot_error_series_and_boxplot(
                err_map_4,
                L=L,
                H=H,
                outname=f"Fig12_error_4models_L{L}_H{H}.png",
                y_limit=(-25, 25),  # 你想跟参考图一样就用这个范围
                max_points=None  # 或者写 12000 之类限制长度
            )

            print(os.path.join(CONFIG["result_dir"], f"predmeta_{_sp('TPA-LSTM', L, H)}.csv"))
            print(os.path.exists(os.path.join(CONFIG["result_dir"], f"predmeta_{_sp('TPA-LSTM', L, H)}.csv")))

    df = pd.DataFrame(all_rec)

    # 展示列表
    keep_models = [
        "EN", "SVM",
        "LSTM", "BiLSTM", "GRU", "CNN_LSTM", "TPA_LSTM",
        "PI_LSTM", "PI_TRA_LSTM"
    ]
    df_vis = df[df["model"].isin(keep_models)].copy()

    # 统一显示名
    df_vis["model"] = df_vis["model"].replace({
        "CNN_LSTM": "CNN-LSTM",
        "TPA_LSTM": "TPA-LSTM",
        "TRA_LSTM": "TRA-LSTM",
        "PI_TRA_LSTM": "PI-TRA-LSTM",
    })
    df_vis = df_vis.sort_values("model")

    plot_metric_bars(df_vis[["model", "test_R2"]], "cmp", "test_R2", "R² Comparison (Test)")
    plot_metric_bars(df_vis[["model", "test_MSE"]], "cmp", "test_MSE", "MSE Comparison (Test)")
    plot_metric_bars(df_vis[["model", "test_MAE"]], "cmp", "test_MAE", "MAE Comparison (Test)")
    plot_metric_bars(df_vis[["model", "test_MAPE"]], "cmp", "test_MAPE", "MAPE (%) Comparison (Test)")
    plot_metric_bars(df_vis[["model", "test_RMSE"]], "cmp", "test_RMSE", "RMSE Comparison (Test)")

    # 在 main 函数末尾，生成分段对比表格
    print("\n" + "=" * 80)
    print("分段评估汇总表 (测试集)")
    print("=" * 80)

    # 选择关键模型
    key_models = ["BiLSTM", "GRU", "CNN_LSTM", "TPA_LSTM", "TRA_LSTM", "PI_LSTM", "PI_TRA_LSTM"]
    seg_summary = []

    for model_name in key_models:
        for L in CONFIG["time_steps"]:
            for H in CONFIG["horizons"][:1]:  # 只取H=1作为代表
                prefix = f"{model_name}_L{L}_H{H}_E{CONFIG['base_epochs']}_B{CONFIG['base_batch']}"
                seg_file = os.path.join(CONFIG["result_dir"], f"segment_metrics_{prefix}.csv")

                if os.path.exists(seg_file):
                    df_seg = pd.read_csv(seg_file)
                    for _, row in df_seg.iterrows():
                        seg_summary.append({
                            "model": model_name.replace("_", "-"),
                            "segment": row["segment"],
                            "RMSE": row["RMSE"],
                            "MAE": row["MAE"],
                            "MAPE": row["MAPE"],
                            "R2": row["R2"]
                        })

    if seg_summary:
        df_summary = pd.DataFrame(seg_summary)

        # 打印表格
        print("\n全局指标:")
        global_data = df_summary[df_summary["segment"] == "global"]
        for _, row in global_data.iterrows():
            print(
                f"  {row['model']}: RMSE={row['RMSE']:.3f}, MAE={row['MAE']:.3f}, MAPE={row['MAPE']:.2f}%, R²={row['R2']:.4f}")

        print("\n异常段 (2500-3500):")
        abnormal_data = df_summary[df_summary["segment"] == "abnormal"]
        for _, row in abnormal_data.iterrows():
            print(
                f"  {row['model']}: RMSE={row['RMSE']:.3f}, MAE={row['MAE']:.3f}, MAPE={row['MAPE']:.2f}%, R²={row['R2']:.4f}")

        print("\n尾部段 (15000-17500):")
        tail_data = df_summary[df_summary["segment"] == "tail"]
        for _, row in tail_data.iterrows():
            print(
                f"  {row['model']}: RMSE={row['RMSE']:.3f}, MAE={row['MAE']:.3f}, MAPE={row['MAPE']:.2f}%, R²={row['R2']:.4f}")

        # 保存汇总表
        df_summary.to_csv(os.path.join(CONFIG["result_dir"], "segment_summary_all.csv"), index=False)
        print(f"\n分段汇总表已保存: {os.path.join(CONFIG['result_dir'], 'segment_summary_all.csv')}")

    print("\n===== 完成 =====")
    print(f"汇总表：{os.path.join(CONFIG['result_dir'], 'benchmark_report.csv')}")
    print(f"图表目录：{CONFIG['result_dir']} ；权重目录：{CONFIG['ckpt_dir']}")
    if CONFIG.get("do_pso_calibration", False):
        print(f"PSO 结果：{CONFIG['pso_outfile']}")

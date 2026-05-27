# -*- coding: utf-8 -*-
from .common import *

# -------------------------
# 0. 全局配置
# -------------------------
CONFIG = dict(
    original_csv="data_src/limekiln.csv",
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=2025,

    # 数据与切分
    time_steps=[30],
    horizons=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # horizons=[1],

    split_ratios=(0.7, 0.1, 0.2),

    # 统一训练协议（主对比）
    base_epochs=500,
    base_batch=128,
    base_lr=5e-4,
    base_patience=20,
    lr_decay_step=200,
    lr_decay_gamma=0.5,

    # 结构预算
    base_hidden=64,
    base_layers=1,
    base_dropout=0.05,
    stronger=True,  # 轻增强：LayerNorm + ResidualFFN
    # PI-RA-TRA-LSTM 稍强预算
    m5_hidden=128,
    m5_layers=2,
    m5_dropout=0.15,
    m5_beta_att=0.8,

    # 生烧率风险把控
    risk_threshold=5.0,  # 或 4.6 / 3.5，看你想怎么开 risk gate
    # 时间间隔
    sample_interval_s=60,

    # ===== 机理参数（将被 PSO 标定覆盖）=====
    learn_tg_map=False,
    a_tg=1.95,
    b_tg=28.0,
    k1=1.0,
    k3=500.0,
    HGS=280.0,
    gamma_rb=2e-4,

    # ===== 物理/一致性损失权重（训练阶段）=====
    theta_d=100.0,
    theta_p1=0.2,
    theta_p2=2.0,

    # 注意力一致性参数（保留为实验开关；默认主模型为 PI_TRA_LSTM）
    beta_att=0.5,
    use_reattn_loss=True,
    lambda_re=0.1,
    re_gate_tau=0.3,

    # ====== 物理权重线性升权（可选）======
    use_phys_warmup=False,
    phys_warmup_epochs=50,

    # 输出
    ckpt_dir="checkpoints_bench_版本modelV21.0",
    result_dir="results_版本modelV21.0",

    # 额外：敏感性分析（TRA-LSTM）
    run_sensitivity=True,
    sensitivity_tasks=[
        ("Single-step", 1),
        ("Short-term", [2, 3, 4, 5]),
        ("Middle-term", [6, 7, 8, 9]),
        ("Long-term", 10),
    ],
    sens_epochs=50,

    # ===== PSO 两阶段标定 =====
    do_pso_calibration=True,
    use_pso_init=True,
    pso_particles=30,
    pso_iters=200,
    pso_seed=2025,
    pso_inertia=0.72,
    pso_c1=1.6,
    pso_c2=1.6,
    pso_bounds=dict(
        a_tg=(0.6, 3.0),
        b_tg=(-50.0, 120.0),
        k1=(1e-3, 5.0),
        k3=(1.0, 800.0),
        HGS=(10.0, 800.0),
        gamma_rb=(1e-6, 2e-3),
    ),
    pso_outfile="results_V21.0/pso_physics_best.json",
)


def set_seed(s):
    random.seed(s);
    np.random.seed(s);
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed(CONFIG["seed"])

# -*- coding: utf-8 -*-
from .common import *
from .config import CONFIG

# -------------------------
# 1. 数据与预处理
# -------------------------
COLMAP_CN2EN = {
    "时间": "time",
    "燃气热能流量实际值": "Q_fuel",
    "冷却空气流量": "F_cool",
    "助燃空气流量": "F_air",
    "总上料量": "M_feed",
    "窑膛模式": "mode",
    "平均生烧率": "S_avg",
    "平均通道温度": "T_avg",
}
INPUT_BASE = ["Q_fuel", "F_cool", "F_air", "M_feed", "mode", "S_avg"]
TARGET_COL = "T_avg"

KJ_PER_MCAL = 4184.0
PER_MIN = 1.0 / 60.0
KG_PER_TON = 1000.0


def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "utf-16", "latin1"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, sep=None, engine="python", encoding="latin1", on_bad_lines="skip")


def _convert_units(df: pd.DataFrame) -> pd.DataFrame:
    if "Q_fuel" in df:
        df["Q_fuel"] = pd.to_numeric(df["Q_fuel"], errors="coerce") * KJ_PER_MCAL * PER_MIN
    for c in ["F_cool", "F_air"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce") * PER_MIN
    if "M_feed" in df:
        df["M_feed"] = pd.to_numeric(df["M_feed"], errors="coerce") * KG_PER_TON * PER_MIN
    return df


def _delay_savg_24h(df: pd.DataFrame, sampling_per_min=1) -> pd.DataFrame:
    if "S_avg" in df:
        shift_steps = int(24 * 60 / sampling_per_min)  # 1440
        df["S_avg"] = df["S_avg"].shift(shift_steps)
    return df


def load_csv_file(path: str) -> pd.DataFrame:
    df = _read_csv_any(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: COLMAP_CN2EN.get(c, c) for c in df.columns})

    miss = [c for c in ["Q_fuel", "F_cool", "F_air", "M_feed", "mode", "S_avg", "T_avg"] if c not in df.columns]
    if miss:
        raise ValueError(f"CSV缺少列：{miss}")

    for c in ["Q_fuel", "F_cool", "F_air", "M_feed", "mode", "S_avg", "T_avg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = _convert_units(df)
    df = _delay_savg_24h(df)

    df["mode"] = (df["mode"] > 0.5).astype(int)
    df["S_avg"] = df["S_avg"].clip(0.0, 20.0)  # 这里进行修改后，之前是生烧率数据自动归纳在0-5之间，没有进行约束

    if "time" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
            except Exception:
                pass

    df = df.dropna(subset=["Q_fuel", "F_cool", "F_air", "M_feed", "mode", "S_avg", "T_avg"]).reset_index(drop=True)
    keep = ["time", "Q_fuel", "F_cool", "F_air", "M_feed", "mode", "S_avg", "T_avg"] if "time" in df.columns else \
        ["Q_fuel", "F_cool", "F_air", "M_feed", "mode", "S_avg", "T_avg"]
    if "time" in keep:
        df = df.sort_values("time").reset_index(drop=True)
    return df[keep]

def split_df_by_time(df: pd.DataFrame, ratios=(0.7, 0.1, 0.2), look_back=30):
    """
    时间顺序切分 + 给 valid/test 留历史窗口，避免边界信息泄漏
    """
    n = len(df)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])

    # ===== train =====
    df_train = df.iloc[:n_train].copy().reset_index(drop=True)

    # ===== valid（向前多取 L 步）=====
    df_valid = df.iloc[n_train - look_back : n_train + n_valid]\
                 .copy().reset_index(drop=True)

    # ===== test（同理）=====
    df_test = df.iloc[n_train + n_valid - look_back :]\
                .copy().reset_index(drop=True)

    return df_train, df_valid, df_test

def fit_scalers_on_train_df(df_train: pd.DataFrame) -> Dict[str, MinMaxScaler]:
    """
    只在训练集上拟合 scaler，避免数据泄漏。
    """
    X_raw = df_train[INPUT_BASE].values.astype(np.float32)
    y_all = df_train[[TARGET_COL]].values.astype(np.float32)

    ctrl = X_raw[:, :4]  # 只归一化控制量
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(ctrl)
    scaler_y.fit(y_all)

    return dict(scaler_X=scaler_X, scaler_y=scaler_y)

class LimeKilnSeqDataset(Dataset):
    """
    X_seq(L,7) = [ctrl_norm(4), mode_raw(1), S_norm(1), T_hist_norm(1)]
    返回:
        X_seq,
        y_next_norm,
        S_seq,
        mode_seq,
        y_prev_true,
        raw_idx   # 目标点在原始序列中的索引
    """

    def __init__(self, files: List[str], time_step: int, pred_horizon: int = 1,
                 fit_scaler=True, global_scaler: Dict = None):
        self.time_step = time_step
        self.pred_horizon = pred_horizon

        frames = [load_csv_file(p) for p in files]
        if len(frames) == 0:
            raise RuntimeError("未找到 CSV")
        df_all = pd.concat(frames, axis=0).reset_index(drop=True)
        if "time" in df_all.columns:
            df_all = df_all.sort_values("time").reset_index(drop=True)

        X_raw = df_all[INPUT_BASE].values.astype(np.float32)
        y_all = df_all[[TARGET_COL]].values.astype(np.float32)

        ctrl = X_raw[:, :4]
        mode_raw = X_raw[:, 4:5]
        S_real = X_raw[:, 5:6]
        S_norm = np.clip(S_real / 5.0, 0.0, 1.0)

        if fit_scaler:
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            scaler_X.fit(ctrl)
            scaler_y.fit(y_all)
            self.scalers = dict(scaler_X=scaler_X, scaler_y=scaler_y)
        else:
            if global_scaler is None:
                raise ValueError("global_scaler 为空，但 fit_scaler=False")
            self.scalers = global_scaler

        ctrl_n = self.scalers["scaler_X"].transform(ctrl)
        y_n = self.scalers["scaler_y"].transform(y_all)
        T_norm = y_n.copy()

        L = self.time_step
        H = self.pred_horizon
        N = len(ctrl_n)

        X_seq, y_next, S_seq, mode_seq, y_prev_true, raw_idx = [], [], [], [], [], []

        # 真正 H 步预测：目标位置 = t + H - 1
        for t in range(L, N - H + 1):
            target_idx = t + H - 1

            x6 = np.concatenate(
                [ctrl_n[t-L:t, :], mode_raw[t-L:t, :], S_norm[t-L:t, :]],
                axis=1
            )
            t_hist = T_norm[t-L:t, :]
            x7 = np.concatenate([x6, t_hist], axis=1)

            X_seq.append(x7.astype(np.float32))
            y_next.append(y_n[target_idx, 0].astype(np.float32))
            S_seq.append(S_real[t-L:t, 0].astype(np.float32))
            mode_seq.append(mode_raw[t-L:t, 0].astype(np.float32))
            y_prev_true.append(y_all[target_idx - 1, 0].astype(np.float32))
            raw_idx.append(int(target_idx))

        self.X_seq = np.array(X_seq, dtype=np.float32)
        self.y_next = np.array(y_next, dtype=np.float32).reshape(-1, 1)
        self.S_seq = np.array(S_seq, dtype=np.float32)
        self.mode_seq = np.array(mode_seq, dtype=np.float32)
        self.y_prev_true = np.array(y_prev_true, dtype=np.float32).reshape(-1, 1)
        self.raw_idx = np.array(raw_idx, dtype=np.int64)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return (
            self.X_seq[idx],
            self.y_next[idx],
            self.S_seq[idx],
            self.mode_seq[idx],
            self.y_prev_true[idx],
            self.raw_idx[idx]
        )

class LimeKilnSeqDatasetFromDF(Dataset):
    """
    用已经切好的 dataframe 构造时序样本，避免先全量窗口再切分。
    X_seq(L,7) = [ctrl_norm(4), mode_raw(1), S_norm(1), T_hist_norm(1)]
    返回:
        X_seq,
        y_next_norm,
        S_seq,
        mode_seq,
        y_prev_true,
        raw_idx
    """

    def __init__(self, df: pd.DataFrame, time_step: int, pred_horizon: int = 1,
                 scaler_pack: Dict = None, start_raw_idx: int = 0):
        self.time_step = time_step
        self.pred_horizon = pred_horizon

        if scaler_pack is None:
            raise ValueError("scaler_pack 不能为空，且必须来自训练集拟合。")

        df = df.copy().reset_index(drop=True)

        X_raw = df[INPUT_BASE].values.astype(np.float32)
        y_all = df[[TARGET_COL]].values.astype(np.float32)

        ctrl = X_raw[:, :4]
        mode_raw = X_raw[:, 4:5]
        S_real = X_raw[:, 5:6]
        S_norm = np.clip(S_real / 5.0, 0.0, 1.0)

        ctrl_n = scaler_pack["scaler_X"].transform(ctrl)
        y_n = scaler_pack["scaler_y"].transform(y_all)
        T_norm = y_n.copy()

        L = self.time_step
        H = self.pred_horizon
        N = len(ctrl_n)

        X_seq, y_next, S_seq, mode_seq, y_prev_true, raw_idx = [], [], [], [], [], []

        for t in range(L, N - H + 1):
            target_idx = t + H - 1

            x6 = np.concatenate(
                [ctrl_n[t-L:t, :], mode_raw[t-L:t, :], S_norm[t-L:t, :]],
                axis=1
            )
            t_hist = T_norm[t-L:t, :]
            x7 = np.concatenate([x6, t_hist], axis=1)

            X_seq.append(x7.astype(np.float32))
            y_next.append(y_n[target_idx, 0].astype(np.float32))
            S_seq.append(S_real[t-L:t, 0].astype(np.float32))
            mode_seq.append(mode_raw[t-L:t, 0].astype(np.float32))
            y_prev_true.append(y_all[target_idx - 1, 0].astype(np.float32))
            raw_idx.append(int(start_raw_idx + target_idx))

        self.X_seq = np.array(X_seq, dtype=np.float32)
        self.y_next = np.array(y_next, dtype=np.float32).reshape(-1, 1)
        self.S_seq = np.array(S_seq, dtype=np.float32)
        self.mode_seq = np.array(mode_seq, dtype=np.float32)
        self.y_prev_true = np.array(y_prev_true, dtype=np.float32).reshape(-1, 1)
        self.raw_idx = np.array(raw_idx, dtype=np.int64)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return (
            self.X_seq[idx],
            self.y_next[idx],
            self.S_seq[idx],
            self.mode_seq[idx],
            self.y_prev_true[idx],
            self.raw_idx[idx]
        )

def make_loaders_from_datasets(train_ds, valid_ds, test_ds, batch_size):
    train_L = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_L = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_L  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_L, valid_L, test_L
def time_series_split(dataset: Dataset, ratios=(0.7, 0.1, 0.2)):
    n = len(dataset)
    n_train = int(n * ratios[0]);
    n_valid = int(n * ratios[1])
    idx_tr = np.arange(0, n_train)
    idx_va = np.arange(n_train, n_train + n_valid)
    idx_te = np.arange(n_train + n_valid, n)
    return Subset(dataset, idx_tr), Subset(dataset, idx_va), Subset(dataset, idx_te)


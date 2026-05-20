"""
feature_engineering.py — CliMedV2
=================================

Chuẩn bị dữ liệu CSV NghiencuuHFLC.csv cho training ML.

Pipeline:
  1. Normalize column names (HctN1..N5 vs hctN6..N10).
  2. Parse target (text → category).
  3. Drop POST-OUTCOME cols (tránh leakage):
     - thoigiannamvien (length of stay): chỉ biết sau xuất viện
     - ketcuc: rỗng trong dataset này
  4. Define series groups (cho missingness pipeline).
  5. Compute trend features (delta day-to-day, peak/nadir).
  6. Encode categorical (gioitinh, BMI, etc.).
  7. Return X, y, feature_names, metadata.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Cấu hình cho dataset NghiencuuHFLC.csv
# ===========================================================================

# Series time-varying columns (mỗi chỉ số có N1..N10)
SERIES_GROUPS_RAW: Dict[str, List[str]] = {
    "bachcau": [f"bachcauN{i}" for i in range(1, 11)],
    "tieucau": [f"tieucauN{i}" for i in range(1, 11)],
    "HFLC":    [f"HFLCN{i}" for i in range(1, 11)],
    "Hct":     ["HctN1", "HctN2", "HctN3", "HctN4", "HctN5",
                "hctN6", "hctN7", "hctN8", "hctN9", "hctN10"],
    "phantramHFLC": [f"PhantramHFLCN1"]  # N1 capital, N2-9 lowercase
                    + [f"phantramHFLCN{i}" for i in range(2, 10)]
                    + [f"PhantramHFLCN10"]   # placeholder, sẽ check sau
}

# Cột phải drop để tránh leakage (chỉ biết SAU outcome)
LEAKY_COLUMNS = [
    "thoigiannamvien",  
    "ketcuc",             
    "Ten",               
    "ghichutinhtrangbenhnang",  
]

# Cột symptom đầu vào — giữ nguyên
SYMPTOM_COLUMNS = [
    "NVnonoi", "NVtieuchay", "NVdaubung", "NVdauco", "NVdaudau",
    "NVdausauhocmat", "NVho", "NVxuathuyet", "NVganto","ghichutinhtrangbenhnang",
]

# Cột demographic
DEMO_COLUMNS = [
    "gioitinh", "Tuoi", "Cannang", "Chieucao", "phanloaiBMI",
    "Doituong", "tiencansxh", "Petechia",
]

# Lab one-shot (không phải time series — chỉ đo 1 lần)
LAB_ONESHOT = ["AST", "ALT", "Creatinin", "Albumin", "Bilirubin",
               "TQphantram", "Fibrinogen", "TCKgiay", "Troponin"]


# ===========================================================================
# Target parsing
# ===========================================================================

@dataclass
class TargetSpec:
    """Spec cho target column."""
    column: str            # cột đầu vào
    mapping: Dict          # mapping value gốc → label
    n_classes: int


def build_target(
    df: pd.DataFrame,
    target_mode: str = "binary_warning",
) -> Tuple[pd.Series, TargetSpec]:
    """
    Build target column theo nhiều cách.

    Modes:
      "binary_warning": 0=thường (lớp 2), 1=cảnh báo, 2 = nặng 
                       Khuyên dùng vì dataset có 96 vs 47+5, đỡ imbalance.
      "multiclass":    giữ nguyên 3 lớp 1, 2, 3
      "from_text":     parse từ ghichutinhtrangbenhnang (free text)
                       sxhcanhbao→1, soc*→2, sxhnang→2, others→0

    Returns: (y_series, spec)
    """
    if target_mode == "binary_warning":
        col = "benhcanhcuoicunglagi"
        mapping = {1: 1, 2: 0, 3: 2}  # 1+3 → cảnh báo, 2 → thường
        y = df[col].map(mapping).astype("Int64")
        spec = TargetSpec(column=col, mapping=mapping, n_classes=2)

    elif target_mode == "multiclass":
        col = "benhcanhcuoicunglagi"
        # 1 → 0, 2 → 1, 3 → 2 (sklearn cần label 0-indexed)
        mapping = {1: 0, 2: 1, 3: 2}
        y = df[col].map(mapping).astype("Int64")
        spec = TargetSpec(column=col, mapping=mapping, n_classes=3)

    elif target_mode == "from_text":
        col = "ghichutinhtrangbenhnang"
        text_map = {}
        for v in df[col].dropna().unique():
            t = str(v).lower().strip()
            if "soc" in t or "sxhnang" in t or "sox" in t or "xuathuyet" in t or "tonthuong" in t or "xhthtren" in t or "shh" in t or "gan" in t or "than" in t:
                text_map[v] = 2  # nặng
            elif "canhbao" in t or "tieuhuyetsacto" in t or "thai" in t:
                text_map[v] = 1  # cảnh báo
            else:
                text_map[v] = 0  # thường
        y = df[col].map(text_map).astype("Int64")
        spec = TargetSpec(column=col, mapping=text_map, n_classes=3)

    else:
        raise ValueError(f"Unknown target_mode: {target_mode!r}")

    n_drop = y.isna().sum()
    if n_drop > 0:
        logger.warning("Target có %d NaN — sẽ drop khi train.", n_drop)

    logger.info("Target distribution (%s): %s",
                target_mode, dict(y.value_counts(dropna=False)))
    return y, spec


# ===========================================================================
# Trend features — delta day-to-day, peak/nadir, rate of change
# ===========================================================================

def compute_trend_features(
    df: pd.DataFrame,
    series_columns: List[str],
    series_name: str,
) -> pd.DataFrame:
    """
    Tính trend features cho 1 chuỗi:
      • {name}_max:   peak observed
      • {name}_min:   nadir observed
      • {name}_first: giá trị observed đầu tiên (smallest day index)
      • {name}_last:  giá trị observed cuối (largest day index)
      • {name}_range: max - min
      • {name}_n_observed: số ngày có observed
      • {name}_max_dropdown: drop lớn nhất giữa 2 ngày liên tiếp (negative)
      • {name}_max_rise:    rise lớn nhất

    Trả về DataFrame với các feature mới (1 hàng / bệnh nhân).
    """
    arr = df[series_columns].values.astype(float)  # (n_patients, n_days)
    n_patients, n_days = arr.shape

    # Stats cơ bản (dùng nanmax/nanmin để bỏ qua NaN; warnings im khi all-NaN)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="All-NaN")
        feat_max = np.nanmax(arr, axis=1)
        feat_min = np.nanmin(arr, axis=1)
        feat_range = feat_max - feat_min
        feat_n_obs = np.sum(~np.isnan(arr), axis=1)

    # First/last observed — tìm vị trí observed sớm nhất / muộn nhất
    feat_first = np.full(n_patients, np.nan)
    feat_last = np.full(n_patients, np.nan)
    for i in range(n_patients):
        obs = np.where(~np.isnan(arr[i]))[0]
        if len(obs) > 0:
            feat_first[i] = arr[i, obs[0]]
            feat_last[i] = arr[i, obs[-1]]

    # Day-to-day delta — cho mỗi cặp (j, j+1) chỉ tính nếu CẢ 2 đều observed
    feat_max_drop = np.full(n_patients, np.nan)
    feat_max_rise = np.full(n_patients, np.nan)
    for i in range(n_patients):
        deltas = []
        for j in range(n_days - 1):
            if not np.isnan(arr[i, j]) and not np.isnan(arr[i, j + 1]):
                deltas.append(arr[i, j + 1] - arr[i, j])
        if deltas:
            feat_max_drop[i] = min(deltas)  # most negative
            feat_max_rise[i] = max(deltas)  # most positive

    out = pd.DataFrame({
        f"{series_name}_max":          feat_max,
        f"{series_name}_min":          feat_min,
        f"{series_name}_range":        feat_range,
        f"{series_name}_first":        feat_first,
        f"{series_name}_last":         feat_last,
        f"{series_name}_n_observed":   feat_n_obs,
        f"{series_name}_max_dropdown": feat_max_drop,
        f"{series_name}_max_rise":     feat_max_rise,
    }, index=df.index)
    return out


# ===========================================================================
# Pipeline chính
# ===========================================================================

def filter_existing_series(df: pd.DataFrame,
                           groups: Dict[str, List[str]]
                           ) -> Dict[str, List[str]]:
    """Giữ lại các cột thực sự tồn tại trong df (handle naming inconsistency)."""
    out = {}
    for name, cols in groups.items():
        existing = [c for c in cols if c in df.columns]
        if existing:
            out[name] = existing
        else:
            logger.warning("Group '%s' không có cột nào tồn tại — bỏ qua", name)
    return out


def prepare_features(
    df: pd.DataFrame,
    target_mode: str = "binary_warning",
    drop_leaky: bool = True,
    add_trend: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    add_split_column: bool = True,
    focus_days: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Pipeline đầy đủ chuẩn bị dataset.

    Args:
        test_size: tỷ lệ holdout test (mặc định 0.2 = 20%).
        random_state: seed cho train/test split (reproducible).
        add_split_column: nếu True, thêm cột '_split' = 'train'/'test'.
        focus_days: list các ngày cần giữ. Mặc định None = giữ tất cả N1-N10.
                    Khuyên dùng [3, 4, 5, 6, 7] (giai đoạn giữa, gồm cả
                    giai đoạn nguy hiểm N4-N6) vì N1-N2 BN ít đến khám,
                    N9-N10 cũng thưa.

    Returns:
        X, y, meta (xem doc cũ + thêm meta['focus_days']).
    """
    from sklearn.model_selection import train_test_split

    df = df.copy()

    # 1. Build target
    target_col_used: Optional[str] = None
    y, spec = build_target(df, target_mode=target_mode)
    target_col_used = spec.column

    # 2. Drop leaky cols + target column
    leaky = list(LEAKY_COLUMNS)
    if target_col_used and target_col_used not in leaky:
        leaky.append(target_col_used)
    if drop_leaky:
        for c in leaky:
            if c in df.columns:
                df = df.drop(columns=c)
                logger.debug("Drop leaky/target col: %s", c)

    # 3. Filter series groups về cột thực có
    series_groups = filter_existing_series(df, SERIES_GROUPS_RAW)

    # 3b. (NEW) Filter theo focus_days nếu có — chỉ giữ N3-N7 (mặc định khuyên)
    if focus_days is not None:
        focus_set = set(int(d) for d in focus_days)
        filtered_groups: Dict[str, List[str]] = {}
        cols_to_drop_outside_focus: List[str] = []
        for name, cols in series_groups.items():
            keep_cols = []
            for c in cols:
                # Trích số ngày từ tên cột (vd 'tieucauN5' → 5, 'HctN10' → 10)
                m = re.search(r"[Nn](\d+)", c)
                if m and int(m.group(1)) in focus_set:
                    keep_cols.append(c)
                else:
                    cols_to_drop_outside_focus.append(c)
            if keep_cols:
                filtered_groups[name] = keep_cols
        # Drop các cột ngày NGOÀI focus khỏi df
        for c in cols_to_drop_outside_focus:
            if c in df.columns:
                df = df.drop(columns=c)
        series_groups = filtered_groups
        logger.info("Focus days: %s → giữ %d cột series, drop %d cột ngoài focus",
                    sorted(focus_set),
                    sum(len(v) for v in series_groups.values()),
                    len(cols_to_drop_outside_focus))

    # 4. Trend features
    if add_trend:
        trend_dfs = []
        for name, cols in series_groups.items():
            trend_dfs.append(compute_trend_features(df, cols, name))
        df = pd.concat([df] + trend_dfs, axis=1)
        logger.info("Đã thêm %d trend features",
                    sum(td.shape[1] for td in trend_dfs))

    # 5. Drop string columns
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in str_cols:
        if c not in ("Ten",):
            logger.info("Drop string column: %s", c)
            df = df.drop(columns=c)

    # 6. Drop hàng có y NaN
    valid_mask = y.notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True).astype(int)

    # 7. Split 80/20 stratified
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    split_arr = np.array(["train"] * len(df), dtype=object)
    split_arr[test_idx] = "test"

    X_test_raw = df.iloc[test_idx].copy().reset_index(drop=True)
    y_test = y.iloc[test_idx].copy().reset_index(drop=True)

    if add_split_column:
        df["_split"] = split_arr

    meta = {
        "series_groups": series_groups,
        "feature_columns": [c for c in df.columns if c != "_split"],
        "target_spec": spec,
        "n_samples": len(df),
        "n_features_initial": df.shape[1],
        "split_indices": {
            "train": train_idx.tolist(),
            "test": test_idx.tolist(),
        },
        "X_test_raw": X_test_raw,
        "y_test": y_test,
        "test_size": test_size,
        "random_state": random_state,
        "focus_days": (sorted(focus_days) if focus_days is not None
                        else list(range(1, 11))),
    }

    logger.info(
        "Feature engineering xong: X=%s, y=%s, classes=%s, train=%d, test=%d, focus=%s",
        df.shape, y.shape, dict(y.value_counts()),
        len(train_idx), len(test_idx),
        meta["focus_days"],
    )

    return df, y, meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    df = pd.read_csv("data/NghiencuuHFLC.csv")
    X, y, meta = prepare_features(df, target_mode="binary_warning")
    print(f"\nX shape: {X.shape}")
    print(f"y distribution: {dict(y.value_counts())}")
    print(f"\nTrend feature samples:")
    trend_cols = [c for c in X.columns if c.endswith("_range")
                  or c.endswith("_max_dropdown")]
    print(X[trend_cols[:5]].head())
"""
missingness.py — CliMedV2
=========================

Pipeline xử lý missing data cho dữ liệu y khoa theo ngày.

Chiến lược (theo guideline xử lý missing data trong y khoa):

Tier A — Giữ NaN + thêm mask + time_delta
    • Dùng cho khoảng trống DÀI (>2 ngày): không nội suy.
    • Tree-based model (LightGBM/XGBoost) tự handle NaN.
    • Mask + time_delta cho phép model học pattern missing-as-signal.

Tier B — Stochastic imputation (với physiologic noise)
    • Dùng cho khoảng trống NGẮN (≤2 ngày).
    • Base = linear interpolation giữa 2 quan sát gần nhất.
    • Thêm noise δ ~ N(0, σ_physio) để chống false confidence.
    • Clip vào reference range để không tạo giá trị vô lý.
    • Đánh `imputed_flag=1` để model phân biệt số thật vs số tạo.

Tier C — GRU-D (file gru_d.py riêng — optional)

LÝ DO STOCHASTIC IMPUTATION:
    Trong y khoa, chỉ số luôn dao động trong "biological variation" (CV%).
    Nội suy deterministic (vd PLT trung bình 200 đúng giữa 2 ngày)
    sẽ khiến model OVER-TRUST số nội suy như fact.
    Thêm noise σ_physio mô phỏng dao động sinh lý → model robust hơn.

References:
  • Westgard biological variation database
  • Little & Rubin (2019), Statistical Analysis with Missing Data
  • Lipton et al. (2016), Modeling Missing Data in Clinical Time Series
"""

from __future__ import annotations
from sklearn.model_selection import StratifiedKFold
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Cấu hình physiologic noise theo từng chỉ số
# ===========================================================================

@dataclass
class PhysioConfig:
    """Cấu hình noise + clip range cho 1 chỉ số y khoa."""
    name: str
    sigma: float                  # SD noise theo đơn vị gốc của chỉ số
    clip_low: Optional[float]     # min sau khi imputed (None = không clip dưới)
    clip_high: Optional[float]    # max sau khi imputed (None = không clip trên)
    description: str = ""

    def add_noise(self, value: float, rng: np.random.Generator) -> float:
        """Thêm noise N(0, sigma) và clip. Trả về NaN nếu input là NaN."""
        if pd.isna(value):
            return np.nan
        noisy = float(value) + float(rng.normal(0.0, self.sigma))
        if self.clip_low is not None:
            noisy = max(noisy, self.clip_low)
        if self.clip_high is not None:
            noisy = min(noisy, self.clip_high)
        return noisy


# Default physio config theo CV% từ Westgard biological variation database.
# Sigma = giá trị tham chiếu trung bình × CV%/100. Có thể override khi dùng.
# Có alias tiếng Anh để hỗ trợ cả 2 cách đặt tên cột.
DEFAULT_PHYSIO: Dict[str, PhysioConfig] = {
    # WBC / Bạch cầu
    "bachcau": PhysioConfig(
        name="WBC", sigma=0.3,
        clip_low=0.5, clip_high=50.0,
        description="WBC ×10⁹/L, CV~5%, ref 4-10",
    ),
    "wbc": PhysioConfig(
        name="WBC", sigma=0.3, clip_low=0.5, clip_high=50.0,
        description="WBC alias",
    ),
    # PLT / Tiểu cầu
    "tieucau": PhysioConfig(
        name="PLT", sigma=15.0,
        clip_low=5.0, clip_high=800.0,
        description="PLT K/µL, CV~7%, ref 150-400. PLT rất thấp vẫn có ý nghĩa.",
    ),
    "plt": PhysioConfig(
        name="PLT", sigma=15.0, clip_low=5.0, clip_high=800.0,
        description="PLT alias",
    ),
    # HCT
    "hct": PhysioConfig(
        name="HCT", sigma=1.5,
        clip_low=15.0, clip_high=70.0,
        description="HCT %, CV~3%, ref 35-50.",
    ),
    # HFLC (high-fluorescent lymphocytes)
    "hflc": PhysioConfig(
        name="HFLC", sigma=1.0,
        clip_low=0.0, clip_high=30.0,
        description="HFLC %, CV~10%.",
    ),
    "phantramhflc": PhysioConfig(
        name="HFLC_pct", sigma=1.0,
        clip_low=0.0, clip_high=30.0,
        description="HFLC % cùng config với hflc.",
    ),
    # Men gan
    "ast": PhysioConfig(name="AST", sigma=5.0, clip_low=0.0, clip_high=5000.0,
                        description="AST U/L, CV~10%"),
    "alt": PhysioConfig(name="ALT", sigma=5.0, clip_low=0.0, clip_high=5000.0,
                        description="ALT U/L, CV~10%"),
    "creatinin": PhysioConfig(name="Cre", sigma=0.05, clip_low=0.1, clip_high=15.0,
                              description="Creatinin mg/dL, CV~5%"),
}

# Catch-all cho chỉ số không có config riêng — default conservative
_GENERIC_CONFIG = PhysioConfig(
    name="generic", sigma=0.0,  # sigma=0 = imputation deterministic
    clip_low=None, clip_high=None,
    description="Generic — no noise, no clip.",
)


def _normalize_var_name(col: str) -> str:
    """
    Trích "stem" của tên cột để map vào physio config.
    Ví dụ: 'bachcauN3' -> 'bachcau', 'HctN5' -> 'hct',
           'phantramHFLCN6' -> 'phantramhflc'
    """
    base = re.sub(r"[Nn]\d+$", "", col).lower()
    base = re.sub(r"[^a-z]", "", base)
    return base


def get_physio_config(col: str,
                      overrides: Optional[Dict[str, PhysioConfig]] = None
                      ) -> PhysioConfig:
    """
    Tìm physio config cho cột. Ưu tiên overrides → DEFAULT_PHYSIO → generic.
    """
    stem = _normalize_var_name(col)
    if overrides and stem in overrides:
        return overrides[stem]
    return DEFAULT_PHYSIO.get(stem, _GENERIC_CONFIG)


# ===========================================================================
# Tier A: Mask + time_delta (always-on layer)
# ===========================================================================

def add_missingness_features(
    df: pd.DataFrame,
    columns: List[str],
    suffix_mask: str = "_mask",
    suffix_delta: str = "_tdelta",
) -> pd.DataFrame:
    """
    Thêm 2 feature cho mỗi cột trong `columns`:
      • {col}_mask:   1 nếu observed, 0 nếu NaN
      • {col}_tdelta: số "tick" (đơn vị tùy domain — thường là ngày)
                      kể từ lần observed gần nhất theo CHIỀU NGANG.
                      Nếu chưa từng observed, đặt = -1 (sentinel).

    Lưu ý: hàm tính theo từng HÀNG (mỗi bệnh nhân), giả sử columns
    đã sắp đúng thứ tự thời gian (vd N1, N2, ..., N10).

    Trả về df mới (không mutate input).
    """
    out = df.copy()
    new_features = {}
    # Mask
    for col in columns:
        new_features[col + suffix_mask] = out[col].notna().astype(int)
    temp_masks = pd.DataFrame({c: new_features[c] for c in new_features if c.endswith(suffix_mask)})
    current_tdeltas = []
    last_delta = None
    # Time delta
    for i, col in enumerate(columns):
        delta_col = col + suffix_delta
        if i == 0:
            # Cột đầu: 0 nếu observed, -1 nếu chưa từng có
            new_val = np.where(out[col].notna(), 0, -1)
        else:
            prev_delta = last_delta
            cur_observed = out[col].notna().values
            new_val = np.where(
                cur_observed, 0,
                np.where(prev_delta >= 0, prev_delta + 1, -1)
            )
        new_features[delta_col] = new_val
        last_delta = new_val
    out = pd.concat([out, pd.DataFrame(new_features, index=out.index)], axis=1)
    
    # Chống phân mảnh thêm một lần nữa cho chắc chắn
    return out.copy()


# ===========================================================================
# Tier B: Stochastic short-gap imputation
# ===========================================================================

@dataclass
class ImputationReport:
    """Báo cáo audit cho 1 lần imputation — quan trọng cho compliance y khoa."""
    n_total_cells: int = 0
    n_observed: int = 0
    n_imputed: int = 0
    n_kept_nan: int = 0          # gap quá dài → giữ NaN
    by_column: Dict[str, Dict[str, int]] = field(default_factory=dict)
    noise_sigma_used: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Imputation Report ===",
            f"  Total cells:  {self.n_total_cells}",
            f"  Observed:     {self.n_observed} ({self.n_observed / max(self.n_total_cells, 1):.1%})",
            f"  Imputed:      {self.n_imputed} ({self.n_imputed / max(self.n_total_cells, 1):.1%})",
            f"  Kept NaN:     {self.n_kept_nan} (gap quá dài hoặc rìa)",
        ]
        if self.by_column:
            lines.append("  Per column (imputed / total):")
            for col, st in self.by_column.items():
                lines.append(f"    {col:30s}  imp={st['imputed']:3d}  "
                             f"obs={st['observed']:3d}  nan={st['kept_nan']:3d}")
        return "\n".join(lines)


def stochastic_short_gap_impute(
    df: pd.DataFrame,
    series_columns: List[str],
    max_gap: int = 2,
    physio_overrides: Optional[Dict[str, PhysioConfig]] = None,
    random_state: int = 42,
    add_imputed_flag: bool = True,
) -> Tuple[pd.DataFrame, ImputationReport]:
    """
    Nội suy NGẪU NHIÊN có kiểm soát cho gap ngắn trong chuỗi thời gian wide.

    Args:
        df: DataFrame, mỗi hàng = 1 bệnh nhân.
        series_columns: list các cột theo thứ tự thời gian (vd ['plt_n1', ..., 'plt_n10']).
                        TẤT CẢ cùng 1 chỉ số (cùng physio config).
        max_gap: chỉ nội suy nếu khoảng trống ≤ max_gap ngày.
        physio_overrides: dict override config (tên_stem → PhysioConfig).
        random_state: seed để reproducible.
        add_imputed_flag: nếu True, thêm cột `{col}_imputed_flag`.

    Pipeline:
      1. Linear interpolation giữa 2 quan sát thật gần nhất.
      2. Nếu gap > max_gap, giữ NaN (Tier A).
      3. Thêm noise δ ~ N(0, σ_physio) cho mỗi giá trị nội suy.
      4. Clip vào (clip_low, clip_high).
      5. Set `{col}_imputed_flag=1` cho ô được nội suy.

    Trả về:
      (df_imputed, report)
    """
    if not series_columns:
        return df.copy(), ImputationReport()

    rng = np.random.default_rng(random_state)
    out = df.copy()
    report = ImputationReport()

    # Lấy physio config (giả định tất cả series_columns cùng chỉ số → cùng config)
    cfg = get_physio_config(series_columns[0], physio_overrides)
    report.noise_sigma_used[series_columns[0]] = cfg.sigma

    # Khởi tạo cột imputed_flag = 0
    flag_cols = []
    if add_imputed_flag:
        for col in series_columns:
            flag_col = col + "_imputed_flag"
            out[flag_col] = 0
            flag_cols.append(flag_col)

    # Stats per column
    for col in series_columns:
        report.by_column[col] = {"observed": 0, "imputed": 0, "kept_nan": 0}

    # Process từng hàng
    arr = out[series_columns].values.astype(float)  # shape (n_patients, n_days)
    n_patients, n_days = arr.shape

    for pi in range(n_patients):
        row = arr[pi]
        # Tìm các vị trí observed
        obs_idx = np.where(~np.isnan(row))[0]

        # Đếm observed
        for k in range(n_days):
            report.n_total_cells += 1
            if not np.isnan(row[k]):
                report.n_observed += 1
                report.by_column[series_columns[k]]["observed"] += 1

        # Nội suy gap giữa 2 obs liên tiếp
        for j in range(len(obs_idx) - 1):
            left = int(obs_idx[j])
            right = int(obs_idx[j + 1])
            gap = right - left - 1   # số ô NaN giữa
            if gap == 0:
                continue
            if gap > max_gap:
                # gap quá dài → bỏ qua, sẽ đếm là kept_nan ở dưới
                continue

            v_left = row[left]
            v_right = row[right]
            for step in range(1, gap + 1):
                # linear interpolate
                base = v_left + (v_right - v_left) * step / (gap + 1)
                noisy = cfg.add_noise(base, rng)
                k = left + step
                arr[pi, k] = noisy
                report.n_imputed += 1
                report.by_column[series_columns[k]]["imputed"] += 1
                if add_imputed_flag:
                    out.iloc[pi, out.columns.get_loc(series_columns[k] + "_imputed_flag")] = 1

        # Đếm còn lại NaN sau khi đã thử impute
        # (rìa: trước obs đầu / sau obs cuối, hoặc gap > max_gap)
        for k in range(n_days):
            if np.isnan(arr[pi, k]):
                report.n_kept_nan += 1
                report.by_column[series_columns[k]]["kept_nan"] += 1

    # Ghi lại vào DataFrame
    for k, col in enumerate(series_columns):
        out[col] = arr[:, k]

    return out, report


def stochastic_impute_all_series(
    df: pd.DataFrame,
    series_groups: Dict[str, List[str]],
    max_gap: int = 2,
    physio_overrides: Optional[Dict[str, PhysioConfig]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, ImputationReport]]:
    """
    Convenience: chạy `stochastic_short_gap_impute` cho NHIỀU chỉ số.

    Args:
        series_groups: {tên_chỉ_số: [cột_n1, cột_n2, ...]}.
                       Mỗi group có cùng physio config.

    Returns:
        (df_imputed, {tên_chỉ_số: report})
    """
    out = df
    reports: Dict[str, ImputationReport] = {}
    for series_name, cols in series_groups.items():
        out, rpt = stochastic_short_gap_impute(
            out, cols,
            max_gap=max_gap,
            physio_overrides=physio_overrides,
            random_state=random_state + hash(series_name) % 1000,
        )
        reports[series_name] = rpt
        logger.info("Series '%s': %d imputed, %d kept NaN",
                    series_name, rpt.n_imputed, rpt.n_kept_nan)
    return out, reports


# ===========================================================================
# Tier B+: Spline imputation cho gap GIỮA (interior gap)
# ===========================================================================
# QUAN TRỌNG: edge gap (rìa) LUÔN giữ NaN — đây là policy y khoa.
# Lý do: rìa = MNAR (Missing Not At Random):
#   • Edge trái (N1, N2 trắng): BN chưa đến khám → có thể bệnh nhẹ chủ động
#     ở nhà, hoặc bệnh nặng cấp cứu thẳng. Cả 2 đều bias.
#   • Edge phải (N9, N10 trắng): BN đã ra viện → đa số là hồi phục.
#     Nội suy ra giá trị "bình thường" sẽ làm model nghĩ BN nặng cũng có
#     PLT bình thường vào N10 → false negative.
# ===========================================================================

def spline_impute_interior_gaps(
    df: pd.DataFrame,
    series_columns: List[str],
    max_gap: int = 3,
    physio_overrides: Optional[Dict[str, PhysioConfig]] = None,
    random_state: int = 42,
    add_imputed_flag: bool = True,
) -> Tuple[pd.DataFrame, ImputationReport]:
    """
    Cubic spline imputation cho interior gaps + Gaussian noise theo σ_physio.

    Khác stochastic_short_gap_impute:
      • Dùng cubic spline (mượt hơn linear) khi BN có ≥3 quan sát thật.
      • Fallback linear khi chỉ có 2 quan sát.
      • Tăng default max_gap lên 3 (spline mượt cho gap 2-3 ngày).

    Edge gap (trước observation đầu, sau observation cuối) → KHÔNG impute.
    """
    if not series_columns:
        return df.copy(), ImputationReport()

    try:
        from scipy.interpolate import CubicSpline
        SCIPY_OK = True
    except ImportError:
        logger.warning("scipy chưa cài → fallback linear. "
                       "pip install scipy để dùng cubic spline.")
        SCIPY_OK = False

    rng = np.random.default_rng(random_state)
    out = df.copy()
    report = ImputationReport()

    cfg = get_physio_config(series_columns[0], physio_overrides)
    report.noise_sigma_used[series_columns[0]] = cfg.sigma

    if add_imputed_flag:
        for col in series_columns:
            out[col + "_imputed_flag"] = 0

    for col in series_columns:
        report.by_column[col] = {"observed": 0, "imputed": 0, "kept_nan": 0}

    arr = out[series_columns].values.astype(float)
    n_patients, n_days = arr.shape

    for pi in range(n_patients):
        row = arr[pi]
        obs_idx = np.where(~np.isnan(row))[0]

        for k in range(n_days):
            report.n_total_cells += 1
            if not np.isnan(row[k]):
                report.n_observed += 1
                report.by_column[series_columns[k]]["observed"] += 1

        # Cần ít nhất 2 anchor để fit
        if len(obs_idx) < 2:
            for k in range(n_days):
                if np.isnan(arr[pi, k]):
                    report.n_kept_nan += 1
                    report.by_column[series_columns[k]]["kept_nan"] += 1
            continue

        # Fit spline (hoặc linear) trên các anchor
        x_obs = obs_idx.astype(float)
        y_obs = row[obs_idx]

        # Cubic spline cần ≥4 anchor; fallback theo số anchor
        if SCIPY_OK and len(obs_idx) >= 4:
            try:
                spline = CubicSpline(x_obs, y_obs, extrapolate=False)
                interp_fn = lambda xs: spline(xs)
            except Exception as e:
                logger.debug("Spline fail BN%d: %s — fallback linear", pi, e)
                interp_fn = lambda xs: np.interp(xs, x_obs, y_obs)
        else:
            interp_fn = lambda xs: np.interp(xs, x_obs, y_obs)

        # Chỉ impute INTERIOR gaps (giữa 2 anchor liền kề)
        for j in range(len(obs_idx) - 1):
            left = int(obs_idx[j])
            right = int(obs_idx[j + 1])
            gap = right - left - 1
            if gap == 0 or gap > max_gap:
                continue

            for step in range(1, gap + 1):
                k = left + step
                base = float(interp_fn(np.array([float(k)]))[0])
                # Spline có thể overshoot → vẫn dùng noise + clip
                noisy = cfg.add_noise(base, rng)
                arr[pi, k] = noisy
                report.n_imputed += 1
                report.by_column[series_columns[k]]["imputed"] += 1
                if add_imputed_flag:
                    flag_col = series_columns[k] + "_imputed_flag"
                    out.iloc[pi, out.columns.get_loc(flag_col)] = 1

        # Đếm các ô NaN còn lại (edge gap + interior gap > max_gap)
        for k in range(n_days):
            if np.isnan(arr[pi, k]):
                report.n_kept_nan += 1
                report.by_column[series_columns[k]]["kept_nan"] += 1

    for k, col in enumerate(series_columns):
        out[col] = arr[:, k]

    return out, report


def spline_impute_all_series(
    df: pd.DataFrame,
    series_groups: Dict[str, List[str]],
    max_gap: int = 3,
    physio_overrides: Optional[Dict[str, PhysioConfig]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, ImputationReport]]:
    """Convenience: chạy spline imputation cho nhiều chỉ số."""
    out = df
    reports: Dict[str, ImputationReport] = {}
    for series_name, cols in series_groups.items():
        out, rpt = spline_impute_interior_gaps(
            out, cols,
            max_gap=max_gap,
            physio_overrides=physio_overrides,
            random_state=random_state + hash(series_name) % 1000,
        )
        reports[series_name] = rpt
        logger.info("Series '%s' (spline): %d imputed, %d kept NaN",
                    series_name, rpt.n_imputed, rpt.n_kept_nan)
    return out, reports


# ===========================================================================
# Tier B++: MICE-like (multivariate imputation) — interior only
# ===========================================================================

def _mark_edge_nans(arr: np.ndarray) -> np.ndarray:
    """
    Trả về mask BOOL cùng shape arr, True ở các ô EDGE NaN
    (NaN trước observation đầu hoặc sau observation cuối CỦA TỪNG HÀNG).
    """
    n_rows, n_cols = arr.shape
    edge_mask = np.zeros_like(arr, dtype=bool)
    for i in range(n_rows):
        obs = np.where(~np.isnan(arr[i]))[0]
        if len(obs) == 0:
            edge_mask[i, :] = True
            continue
        left, right = obs[0], obs[-1]
        # Trước anchor đầu → edge
        edge_mask[i, :left] = True
        # Sau anchor cuối → edge
        edge_mask[i, right + 1:] = True
    return edge_mask


def mice_impute_interior_gaps(
    df: pd.DataFrame,
    series_groups: Dict[str, List[str]],
    physio_overrides: Optional[Dict[str, PhysioConfig]] = None,
    random_state: int = 42,
    max_iter: int = 10,
    add_imputed_flag: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, ImputationReport]]:
    """
    Multivariate imputation (MICE-like) dùng IterativeImputer của sklearn.

    Tận dụng tương quan GIỮA CÁC CHỈ SỐ (vd PLT giảm đi kèm Hct tăng) để
    impute interior gaps tốt hơn linear/spline đơn biến.

    Implementation:
      1. Stack tất cả series_columns thành 1 ma trận lớn.
      2. Save edge_mask (vị trí NaN ở rìa từng hàng).
      3. Chạy IterativeImputer (BayesianRidge per column).
      4. Restore NaN ở các vị trí edge_mask.
      5. Add physiologic noise + clip cho các giá trị đã impute.

    Args:
        max_iter: số vòng lặp iterative imputation.

    Returns: (df_imputed, {series_name: ImputationReport})
    """
    try:
        # IterativeImputer là experimental → cần enable
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
    except ImportError as e:
        raise ImportError(
            "scikit-learn cần version >= 0.21 cho IterativeImputer."
        ) from e

    rng = np.random.default_rng(random_state)
    out = df.copy()
    reports: Dict[str, ImputationReport] = {
        name: ImputationReport() for name in series_groups
    }

    # Khởi tạo flag cols
    if add_imputed_flag:
        for cols in series_groups.values():
            for c in cols:
                out[c + "_imputed_flag"] = 0
            for c in cols:
                reports[next(k for k, v in series_groups.items() if c in v)] \
                    .by_column[c] = {"observed": 0, "imputed": 0, "kept_nan": 0}
    else:
        for name, cols in series_groups.items():
            for c in cols:
                reports[name].by_column[c] = {
                    "observed": 0, "imputed": 0, "kept_nan": 0,
                }

    # Chuẩn bị ma trận stack
    all_cols: List[str] = []
    col_to_series: Dict[str, str] = {}
    for series_name, cols in series_groups.items():
        for c in cols:
            all_cols.append(c)
            col_to_series[c] = series_name

    arr_orig = out[all_cols].values.astype(float)

    # Đếm observed
    for ci, col in enumerate(all_cols):
        series_name = col_to_series[col]
        n_obs = int(np.sum(~np.isnan(arr_orig[:, ci])))
        reports[series_name].by_column[col]["observed"] = n_obs
        reports[series_name].n_observed += n_obs
        reports[series_name].n_total_cells += arr_orig.shape[0]

    # Mark edge NaN ĐÃ CỘT-WISE: edge của 1 series chỉ áp dụng trong series đó
    # (một BN có thể có edge gap ở PLT nhưng không ở Hct)
    edge_mask = np.zeros_like(arr_orig, dtype=bool)
    for series_name, cols in series_groups.items():
        col_idx = [all_cols.index(c) for c in cols]
        sub_arr = arr_orig[:, col_idx]
        sub_edge = _mark_edge_nans(sub_arr)
        for j, ci in enumerate(col_idx):
            edge_mask[:, ci] = sub_edge[:, j]

    # Chạy IterativeImputer
    # LƯU Ý: IterativeImputer drop cols all-NaN → output thiếu cột.
    # Workaround: tự handle các cột all-NaN trước khi pass vào imputer.
    logger.info("MICE: chạy IterativeImputer trên %d cột × %d hàng "
                "(max_iter=%d)...",
                arr_orig.shape[1], arr_orig.shape[0], max_iter)

    # Tìm cột nào có ít nhất 1 obs
    has_obs_per_col = ~np.all(np.isnan(arr_orig), axis=0)
    cols_with_obs = np.where(has_obs_per_col)[0]
    cols_all_nan = np.where(~has_obs_per_col)[0]

    if len(cols_all_nan) > 0:
        logger.warning(
            "MICE: %d cột không có observation nào → giữ nguyên NaN: %s",
            len(cols_all_nan),
            [all_cols[i] for i in cols_all_nan][:5]
        )

    # Chạy imputer chỉ trên các cột có obs
    arr_imputed = arr_orig.copy()
    if len(cols_with_obs) >= 2:   # cần ≥2 cột để imputer chạy
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max_iter,
            random_state=random_state,
            sample_posterior=False,
            initial_strategy="median",
        )
        arr_sub_imputed = imputer.fit_transform(arr_orig[:, cols_with_obs])
        arr_imputed[:, cols_with_obs] = arr_sub_imputed
    elif len(cols_with_obs) == 1:
        # 1 cột → fill bằng median của cột đó
        col_idx = cols_with_obs[0]
        median_val = np.nanmedian(arr_orig[:, col_idx])
        mask = np.isnan(arr_imputed[:, col_idx])
        arr_imputed[mask, col_idx] = median_val

    # Restore edge NaN (giờ shape khớp đảm bảo)
    arr_imputed[edge_mask] = np.nan

    # Detect các vị trí ĐÃ IMPUTE (was NaN, now not NaN, AND not edge)
    was_nan = np.isnan(arr_orig)
    is_filled = (~np.isnan(arr_imputed))
    imputed_positions = was_nan & is_filled  # và bị edge_mask filter rồi

    # Add physio noise + clip cho các giá trị đã impute
    for ci, col in enumerate(all_cols):
        cfg = get_physio_config(col, physio_overrides)
        series_name = col_to_series[col]
        reports[series_name].noise_sigma_used[col] = cfg.sigma

        for ri in range(arr_imputed.shape[0]):
            if imputed_positions[ri, ci]:
                base = arr_imputed[ri, ci]
                noisy = cfg.add_noise(base, rng)
                arr_imputed[ri, ci] = noisy
                reports[series_name].n_imputed += 1
                reports[series_name].by_column[col]["imputed"] += 1
                if add_imputed_flag:
                    flag_col_idx = out.columns.get_loc(col + "_imputed_flag")
                    out.iloc[ri, flag_col_idx] = 1
            elif was_nan[ri, ci]:
                # Vẫn NaN sau impute → kept_nan
                reports[series_name].n_kept_nan += 1
                reports[series_name].by_column[col]["kept_nan"] += 1

    # Ghi lại vào DataFrame
    for ci, col in enumerate(all_cols):
        out[col] = arr_imputed[:, ci]

    for name, rpt in reports.items():
        logger.info("Series '%s' (mice): %d imputed, %d kept NaN",
                    name, rpt.n_imputed, rpt.n_kept_nan)

    return out, reports


# ===========================================================================
# Pipeline chính (public API)
# ===========================================================================

def build_missingness_features(
    df: pd.DataFrame,
    series_groups: Dict[str, List[str]],
    impute_short_gaps: bool = True,
    max_gap: int = 2,
    strategy: str = "linear_noise",
    physio_overrides: Optional[Dict[str, PhysioConfig]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, ImputationReport]]:
    """
    Pipeline missingness ĐẦY ĐỦ — entry point chính cho predictor.

    Args:
        df: DataFrame raw.
        series_groups: dict {tên: [cột_n1, ..., cột_n10]}.
        impute_short_gaps: True = bật imputation, False = chỉ Tier A (mask+delta).
        max_gap: max gap được impute (chỉ áp dụng cho linear_noise / spline).
        strategy: chọn 1 trong:
            • "linear_noise" (default) — linear interpolate + N(0,σ_physio) + clip.
              Tốt cho gap 1-2 ngày. Đơn giản, nhanh.
            • "spline" — cubic spline (≥4 anchor) hoặc linear (2-3 anchor) +
              N(0,σ_physio) + clip. Mượt hơn cho gap 2-3 ngày.
            • "mice" — MICE-like (IterativeImputer/BayesianRidge) tận dụng
              tương quan giữa các chỉ số. Mạnh nhất nhưng chậm nhất.
        physio_overrides: override physio config nếu cần.
        random_state: seed.

    POLICY Y KHOA (mọi strategy):
        Edge gap (NaN ở rìa, trước observation đầu hoặc sau observation cuối)
        LUÔN GIỮ NaN. Lý do: rìa = MNAR (Missing Not At Random) — BN chưa đến
        khám / đã ra viện — không có anchor để nội suy chính xác.

    Returns:
        (df_with_features, {series_name: ImputationReport})

    LƯU Ý: gọi tuần tự (Tier B → Tier A), KHÔNG đảo thứ tự.
    """
    reports: Dict[str, ImputationReport] = {}

    # 1. Tier B: imputation theo strategy đã chọn
    if impute_short_gaps:
        if strategy == "linear_noise":
            df, reports = stochastic_impute_all_series(
                df, series_groups,
                max_gap=max_gap,
                physio_overrides=physio_overrides,
                random_state=random_state,
            )
        elif strategy == "spline":
            df, reports = spline_impute_all_series(
                df, series_groups,
                max_gap=max_gap,
                physio_overrides=physio_overrides,
                random_state=random_state,
            )
        elif strategy == "mice":
            df, reports = mice_impute_interior_gaps(
                df, series_groups,
                physio_overrides=physio_overrides,
                random_state=random_state,
            )
        else:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                f"Chọn 1 trong: linear_noise / spline / mice."
            )

    # 2. Tier A: mask + time_delta (cho TẤT CẢ cột series)
    all_series_cols = [c for cols in series_groups.values() for c in cols]
    df = add_missingness_features(df, all_series_cols)

    return df, reports


# ===========================================================================
# CLI test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Demo: 3 bệnh nhân, 5 ngày, chỉ số PLT
    demo = pd.DataFrame({
        "plt_n1": [200, np.nan, 180],
        "plt_n2": [np.nan, 150, np.nan],   # gap=1 (BN1, BN3 sẽ được nội suy)
        "plt_n3": [150, np.nan, np.nan],   # gap=2 BN2
        "plt_n4": [np.nan, 100, 120],
        "plt_n5": [80, np.nan, np.nan],    # rìa BN3 → giữ NaN
    })
    print("=== INPUT ===")
    print(demo)
    print()

    out, reports = build_missingness_features(
        demo,
        series_groups={"plt": ["plt_n1", "plt_n2", "plt_n3", "plt_n4", "plt_n5"]},
        random_state=42,
    )
    print("=== OUTPUT ===")
    print(out.round(2))
    print()
    for name, rpt in reports.items():
        print(rpt.summary())
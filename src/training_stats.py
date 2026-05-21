"""
training_stats.py — CliMedV2
=============================

Tính thống kê mô tả từ 80% train data, dùng để **định vị BN mới** vs
phân phối từng class (thường/cảnh báo/nặng).

Khác với predictor (ML model), module này là **lookup table thuần** — nhanh,
giải thích được, dùng để bot trả lời kiểu: "PLT N5=41 thấp hơn 82% BN class
`canhbao`, sát median class `nang`".

Pipeline:
    train_df ──► compute_training_stats() ──► dict per-class per-day per-lab
                                                       │
                                                       ▼
    new patient ─► position_patient_against_stats() ─► narrative + percentiles
                ─► find_similar_train_patients() ─► top-k similar BNs

Saved to .cache/train_stats.json — load lại trong app khởi động.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

# Map label int → string
LABEL_INT_TO_STR = {1: "thuong", 2: "canhbao", 3: "nang"}
LABEL_STR_TO_INT = {v: k for k, v in LABEL_INT_TO_STR.items()}
CLASSES_ORDERED = ["thuong", "canhbao", "nang"]
SEVERITY_VI = {
    "thuong":  "SXHD thông thường",
    "canhbao": "SXHD có dấu hiệu cảnh báo",
    "nang":    "SXHD nặng",
}

# Lab series — name → list of column variants per day (handle case-mix in CSV)
LAB_SERIES = {
    "tieucau":      [f"tieucauN{i}" for i in range(1, 11)],
    "bachcau":      [f"bachcauN{i}" for i in range(1, 11)],
    "HFLC":         [f"HFLCN{i}" for i in range(1, 11)],
    "Hct":          [f"HctN{i}" if i <= 5 else f"hctN{i}" for i in range(1, 11)],
    "phantramHFLC": ([f"PhantramHFLCN1"]
                     + [f"phantramHFLCN{i}" for i in range(2, 10)]
                     + ["PhantramHFLCN10"]),
}

# Lab UI labels
LAB_VI = {
    "tieucau": ("Tiểu cầu", "K/uL"),
    "bachcau": ("Bạch cầu", "K/uL"),
    "HFLC":    ("HFLC",     "K/uL"),
    "Hct":     ("Hct",      "%"),
    "phantramHFLC": ("%HFLC", "%"),
}

# Symptoms cols (NV prefix in raw CSV)
SYMPTOM_COLS = ["NVnonoi", "NVtieuchay", "NVdaubung", "NVdauco",
                "NVdaudau", "NVdausauhocmat", "NVho", "NVxuathuyet",
                "NVganto"]
SYMPTOM_VI = {
    "NVnonoi": "nôn ói",
    "NVtieuchay": "tiêu chảy",
    "NVdaubung": "đau bụng",
    "NVdauco": "đau cơ",
    "NVdaudau": "đau đầu",
    "NVdausauhocmat": "đau sau hốc mắt",
    "NVho": "ho",
    "NVxuathuyet": "xuất huyết",
    "NVganto": "gan to",
}

# Day-of-disease cutoffs for "critical phase"
CRITICAL_DAYS = [4, 5, 6]


# ===========================================================================
# Compute stats from train set
# ===========================================================================

@dataclass
class LabDayStat:
    """Stats cho 1 lab × 1 day × 1 class."""
    n: int = 0
    mean: float = 0.0
    median: float = 0.0
    q10: float = 0.0
    q25: float = 0.0
    q75: float = 0.0
    q90: float = 0.0
    min: float = 0.0
    max: float = 0.0
    raw_values: List[float] = field(default_factory=list)   # giữ để percentile


def _safe_numeric(series: pd.Series) -> np.ndarray:
    """Convert series → numpy float, drop NaN."""
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return arr


def _compute_lab_day_stat(values: np.ndarray) -> LabDayStat:
    if len(values) == 0:
        return LabDayStat()
    return LabDayStat(
        n=len(values),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        q10=float(np.percentile(values, 10)),
        q25=float(np.percentile(values, 25)),
        q75=float(np.percentile(values, 75)),
        q90=float(np.percentile(values, 90)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        raw_values=values.tolist(),
    )


def _find_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first column from candidates that exists in df (case-insensitive)."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in df_cols_lower:
            return df_cols_lower[cand.lower()]
    return None


def compute_training_stats(train_df: pd.DataFrame,
                           label_col: str = "benhcanhcuoicunglagi"
                           ) -> Dict[str, Any]:
    """
    Tính thống kê mô tả từ train set, group by class.

    Returns dict structure:
    {
      "n_train": int,
      "class_dist": {"thuong": int, "canhbao": int, "nang": int},
      "labs": {
        "tieucau": {
          "N1": {"thuong": LabDayStat, "canhbao": ..., "nang": ...},
          ...
        }, ...
      },
      "symptoms": {
        "NVnonoi": {"thuong": 0.2, "canhbao": 0.65, "nang": 0.80}, ...
      },
      "demographics": {
        "age": {"thuong": LabDayStat, ...},
        "weight": {...}
      },
      "fever_day_at_admission": {
        "thuong": {1: 10, 2: 20, ...}, ...  # count distribution
      }
    }
    """
    if label_col not in train_df.columns:
        raise ValueError(f"Label col '{label_col}' not in train_df")

    out: Dict[str, Any] = {
        "n_train": len(train_df),
        "class_dist": {},
        "labs": {},
        "symptoms": {},
        "demographics": {},
        "fever_day_at_admission": {},
    }

    # Class distribution
    for int_lbl, str_lbl in LABEL_INT_TO_STR.items():
        out["class_dist"][str_lbl] = int((train_df[label_col] == int_lbl).sum())

    # Lab stats per series × day × class
    for lab_name, day_cols in LAB_SERIES.items():
        out["labs"][lab_name] = {}
        for day_idx, col_name in enumerate(day_cols, start=1):
            actual_col = _find_existing_col(train_df, [col_name])
            if actual_col is None:
                continue
            day_key = f"N{day_idx}"
            out["labs"][lab_name][day_key] = {}
            for int_lbl, str_lbl in LABEL_INT_TO_STR.items():
                mask = train_df[label_col] == int_lbl
                vals = _safe_numeric(train_df.loc[mask, actual_col])
                stat = _compute_lab_day_stat(vals)
                out["labs"][lab_name][day_key][str_lbl] = asdict(stat)

    # Symptom positive rates per class
    for sym_col in SYMPTOM_COLS:
        if sym_col not in train_df.columns:
            continue
        out["symptoms"][sym_col] = {}
        for int_lbl, str_lbl in LABEL_INT_TO_STR.items():
            mask = train_df[label_col] == int_lbl
            n_cls = int(mask.sum())
            if n_cls == 0:
                out["symptoms"][sym_col][str_lbl] = 0.0
                continue
            pos = pd.to_numeric(train_df.loc[mask, sym_col],
                                errors="coerce").fillna(0) > 0
            out["symptoms"][sym_col][str_lbl] = float(pos.sum()) / n_cls

    # Demographics
    for dem_col, key in [("Tuoi", "age"), ("Cannang", "weight"),
                         ("Chieucao", "height")]:
        if dem_col not in train_df.columns:
            continue
        out["demographics"][key] = {}
        for int_lbl, str_lbl in LABEL_INT_TO_STR.items():
            mask = train_df[label_col] == int_lbl
            vals = _safe_numeric(train_df.loc[mask, dem_col])
            out["demographics"][key][str_lbl] = asdict(
                _compute_lab_day_stat(vals))

    # Fever-day-at-admission distribution
    if "ngaybenhlucnhapvien" in train_df.columns:
        for int_lbl, str_lbl in LABEL_INT_TO_STR.items():
            mask = train_df[label_col] == int_lbl
            vc = (pd.to_numeric(train_df.loc[mask, "ngaybenhlucnhapvien"],
                                errors="coerce").dropna().astype(int)
                  .value_counts().sort_index().to_dict())
            out["fever_day_at_admission"][str_lbl] = {int(k): int(v)
                                                       for k, v in vc.items()}

    return out


# ===========================================================================
# Position 1 patient vs training stats
# ===========================================================================

def _percentile_of(value: float, sorted_vals: List[float]) -> float:
    """Percentile của value trong sorted_vals (0-100). 50 = median."""
    if not sorted_vals:
        return 50.0
    n = len(sorted_vals)
    # Số phần tử <= value
    n_below = sum(1 for v in sorted_vals if v <= value)
    return 100.0 * n_below / n


def position_patient_against_stats(
    patient_data: Dict[str, Any],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Đặt BN mới vào phân phối training data, mỗi class.

    patient_data: dict với keys như:
        {
          "age": 24, "sex": "nam", "weight": 65,
          "fever_day_at_admission": 4,
          "symptoms": {"NVdaubung": 1, "NVnonoi": 1, ...},
          "labs": {
            "tieucau": {1: 180, 2: 150, 3: 120, 4: 73, 5: 41, ...},
            "Hct":     {1: 38, 2: 38, 3: 40, 4: 42, 5: 46, ...},
            ...
          }
        }

    Returns dict:
        {
          "lab_positions": [...],   # list các điểm dữ liệu có (giá trị, percentile per class)
          "symptom_alignment": {"thuong": 0.x, "canhbao": 0.x, "nang": 0.x},
          "score_per_class": {"thuong": 0.x, "canhbao": 0.x, "nang": 0.x},
          "best_match_class": "canhbao",
          "narrative": "Bằng văn xuôi tiếng Việt cho LLM dùng làm context"
        }
    """
    out: Dict[str, Any] = {
        "lab_positions": [],
        "symptom_alignment": {},
        "score_per_class": {c: 0.0 for c in CLASSES_ORDERED},
        "best_match_class": None,
        "narrative": "",
    }

    n_lab_signals = 0
    class_lab_dist: Dict[str, List[float]] = {c: [] for c in CLASSES_ORDERED}

    # ----- Lab positioning -----
    patient_labs = patient_data.get("labs", {})
    for lab_name, day_to_value in patient_labs.items():
        if lab_name not in stats.get("labs", {}):
            continue
        for day, value in day_to_value.items():
            if value is None or pd.isna(value):
                continue
            day_key = f"N{day}" if isinstance(day, int) else day
            day_stats = stats["labs"][lab_name].get(day_key)
            if not day_stats:
                continue

            per_class: Dict[str, Dict[str, float]] = {}
            for cls in CLASSES_ORDERED:
                s = day_stats.get(cls, {})
                if not s or s.get("n", 0) == 0:
                    continue
                pct = _percentile_of(float(value), s.get("raw_values", []))
                per_class[cls] = {
                    "percentile": round(pct, 1),
                    "median": s.get("median", 0),
                    "q25": s.get("q25", 0),
                    "q75": s.get("q75", 0),
                    "n_train": s.get("n", 0),
                }
                # Distance to median as "distance"
                if s.get("median", 0):
                    rel_dist = abs(float(value) - s["median"]) / max(
                        s.get("median", 1), 1e-6)
                    class_lab_dist[cls].append(rel_dist)
            if per_class:
                out["lab_positions"].append({
                    "lab": lab_name,
                    "lab_vi": LAB_VI.get(lab_name, (lab_name, ""))[0],
                    "unit": LAB_VI.get(lab_name, ("", ""))[1],
                    "day": day_key,
                    "value": float(value),
                    "by_class": per_class,
                })
                n_lab_signals += 1

    # ----- Symptom alignment -----
    p_symptoms = patient_data.get("symptoms", {}) or {}
    pos_sym_cols = [k for k, v in p_symptoms.items() if v]
    if pos_sym_cols:
        for cls in CLASSES_ORDERED:
            score = 0.0
            for sc in pos_sym_cols:
                rate = stats.get("symptoms", {}).get(sc, {}).get(cls, 0.0)
                score += rate
            out["symptom_alignment"][cls] = round(score / len(pos_sym_cols), 3)

    # ----- Combined score per class (heuristic) -----
    # Lab signal: lower distance = higher score
    for cls in CLASSES_ORDERED:
        dists = class_lab_dist.get(cls, [])
        lab_score = 1.0 / (1.0 + np.mean(dists)) if dists else 0.0
        sym_score = out["symptom_alignment"].get(cls, 0.0)
        # Weight: lab > sym (lab is more objective)
        out["score_per_class"][cls] = round(0.7 * lab_score + 0.3 * sym_score, 3)

    # Best match
    if any(v > 0 for v in out["score_per_class"].values()):
        out["best_match_class"] = max(out["score_per_class"],
                                       key=out["score_per_class"].get)

    # ----- Narrative -----
    out["narrative"] = _build_narrative(out, patient_data, stats)
    return out


def _build_narrative(positioning: Dict[str, Any],
                     patient_data: Dict[str, Any],
                     stats: Dict[str, Any]) -> str:
    """Build văn xuôi tiếng Việt cho bot dùng."""
    lines: List[str] = []

    # 1. Lab vị trí
    lab_lines = []
    for lp in positioning.get("lab_positions", [])[:6]:   # cap 6 dòng
        by_cls = lp["by_class"]
        # Find class closest to median (smallest |percentile - 50|)
        best = min(by_cls.items(),
                   key=lambda kv: abs(kv[1]["percentile"] - 50))
        best_cls, best_info = best
        # Build description
        val = lp["value"]
        unit = lp["unit"]
        lab_lines.append(
            f"  - {lp['lab_vi']} {lp['day']} = {val}{unit}: "
            f"thuộc top {100 - best_info['percentile']:.0f}% (percentile "
            f"{best_info['percentile']:.0f}) của class `{best_cls}` "
            f"(median={best_info['median']:.1f}, "
            f"khoảng q25-q75: {best_info['q25']:.0f}-{best_info['q75']:.0f})"
        )
    if lab_lines:
        lines.append("**Vị trí lab so với 80% train data:**")
        lines.extend(lab_lines)

    # 2. Symptom alignment
    if positioning.get("symptom_alignment"):
        lines.append("\n**Match triệu chứng với từng class:**")
        for cls, score in positioning["symptom_alignment"].items():
            lines.append(f"  - `{cls}`: {score*100:.0f}% (tổng tỉ lệ "
                         f"các triệu chứng BN có, trên class này)")

    # 3. Tổng kết
    if positioning.get("best_match_class"):
        best = positioning["best_match_class"]
        scores = positioning["score_per_class"]
        lines.append(
            f"\n**Phân tích thống kê → BN có hồ sơ giống nhất với class "
            f"`{best}` ({SEVERITY_VI[best]})** (score "
            f"{scores[best]:.2f}, lab + symptom kết hợp). "
            f"Các class khác: " +
            ", ".join(f"{c}={scores[c]:.2f}" for c in CLASSES_ORDERED
                       if c != best)
        )

    return "\n".join(lines).strip()


# ===========================================================================
# Find similar patients in train set
# ===========================================================================

def _patient_feature_vector(patient: Dict[str, Any],
                            lab_keys: List[str],
                            sym_keys: List[str]) -> np.ndarray:
    """Vectorize patient với pattern cố định để Euclidean dist được."""
    vec = []
    # Labs: lấy median value across days
    p_labs = patient.get("labs", {}) or {}
    for lab in lab_keys:
        vals = list(p_labs.get(lab, {}).values())
        vals = [v for v in vals if v is not None and not pd.isna(v)]
        vec.append(float(np.median(vals)) if vals else np.nan)
    # Symptoms: 0/1
    p_syms = patient.get("symptoms", {}) or {}
    for s in sym_keys:
        vec.append(1.0 if p_syms.get(s) else 0.0)
    # Fever day
    vec.append(float(patient.get("fever_day_at_admission") or 0))
    # Age
    vec.append(float(patient.get("age") or 0))
    return np.array(vec, dtype=float)


def find_similar_train_patients(
    patient: Dict[str, Any],
    train_df: pd.DataFrame,
    k: int = 5,
    label_col: str = "benhcanhcuoicunglagi",
) -> List[Dict[str, Any]]:
    """
    Top-k train patients gần nhất với BN mới (Euclidean dist trên feature
    vector chuẩn hóa).

    Returns list dicts:
        [{"patient_idx": int, "distance": float, "label": "canhbao",
          "summary": "..." }, ...]
    """
    if len(train_df) == 0:
        return []

    lab_keys = ["tieucau", "Hct", "bachcau"]
    sym_keys = SYMPTOM_COLS

    # Vector cho BN mới
    p_vec = _patient_feature_vector(patient, lab_keys, sym_keys)

    # Vector cho từng train patient — dùng median across days
    train_vecs = []
    for _, row in train_df.iterrows():
        labs = {}
        for lab in lab_keys:
            day_cols = LAB_SERIES.get(lab, [])
            day_vals = {}
            for i, col in enumerate(day_cols, start=1):
                col = _find_existing_col(train_df, [col])
                if col and pd.notna(row.get(col)):
                    day_vals[i] = float(row[col])
            labs[lab] = day_vals
        syms = {s: int(pd.notna(row.get(s)) and row.get(s) > 0)
                for s in sym_keys}
        train_p = {
            "labs": labs,
            "symptoms": syms,
            "fever_day_at_admission": row.get("ngaybenhlucnhapvien"),
            "age": row.get("Tuoi"),
        }
        train_vecs.append(_patient_feature_vector(
            train_p, lab_keys, sym_keys))
    train_arr = np.vstack(train_vecs)   # (n_train, dim)

    # Normalize columns by std (z-score), ignore NaN
    mu = np.nanmean(train_arr, axis=0)
    sigma = np.nanstd(train_arr, axis=0) + 1e-9
    p_norm = (p_vec - mu) / sigma
    train_norm = (train_arr - mu) / sigma

    # Distance: Euclidean với NaN handling (set NaN diff to 0 = "skip dim")
    diff = train_norm - p_norm[None, :]
    diff = np.where(np.isnan(diff), 0.0, diff)
    distances = np.linalg.norm(diff, axis=1)

    # Top-k
    top_idx = np.argsort(distances)[:k]
    results = []
    for idx in top_idx:
        row = train_df.iloc[int(idx)]
        lbl_int = int(row.get(label_col, 0))
        lbl_str = LABEL_INT_TO_STR.get(lbl_int, "?")
        summary_parts = []
        # PLT min and Hct max for quick summary
        plt_min = float(np.nanmin([row.get(c, np.nan)
                                    for c in LAB_SERIES["tieucau"]]))
        hct_max = float(np.nanmax([row.get(c, np.nan)
                                    for c in LAB_SERIES["Hct"]]))
        if not np.isnan(plt_min):
            summary_parts.append(f"PLT min={plt_min:.0f}")
        if not np.isnan(hct_max):
            summary_parts.append(f"Hct max={hct_max:.0f}")
        if pd.notna(row.get("Tuoi")):
            summary_parts.append(f"{int(row['Tuoi'])}T")
        results.append({
            "patient_idx": int(idx),
            "distance": round(float(distances[idx]), 3),
            "label": lbl_str,
            "label_vi": SEVERITY_VI.get(lbl_str, "?"),
            "summary": ", ".join(summary_parts),
        })
    return results


# ===========================================================================
# Save / Load JSON
# ===========================================================================

def save_stats(stats: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate raw_values to keep file small but enough for percentile
    out = json.loads(json.dumps(stats))   # deep copy
    for lab in out.get("labs", {}).values():
        for day in lab.values():
            for cls in day.values():
                if "raw_values" in cls and len(cls["raw_values"]) > 100:
                    # Sample 100 evenly
                    arr = np.array(cls["raw_values"])
                    idx = np.linspace(0, len(arr) - 1, 100, dtype=int)
                    cls["raw_values"] = arr[idx].tolist()
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    logger.info("Saved training stats to %s", path)


def load_stats(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_positioning_for_chat(positioning: Dict[str, Any],
                                 similar_patients: List[Dict[str, Any]]
                                 ) -> str:
    """Format chuỗi text sẵn để inject vào system prompt của LLM."""
    parts: List[str] = ["## VỊ TRÍ BN TRÊN PHÂN PHỐI 80% TRAIN DATA"]
    parts.append("")
    parts.append(positioning.get("narrative", "(không tính được)"))
    if similar_patients:
        parts.append("\n**5 BN gần nhất trong train set:**")
        label_count = {c: 0 for c in CLASSES_ORDERED}
        for sp in similar_patients:
            parts.append(f"  - BN#{sp['patient_idx']} ({sp['summary']}) "
                         f"→ class `{sp['label']}` (dist={sp['distance']})")
            label_count[sp["label"]] = label_count.get(sp["label"], 0) + 1
        parts.append(f"  → Phân phối nhãn 5 BN gần nhất: " +
                     ", ".join(f"`{c}`: {n}" for c, n
                                in label_count.items() if n > 0))
    return "\n".join(parts)


# ===========================================================================
# CLI smoke test
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) < 2:
        print("Cách dùng: python training_stats.py path/to/train.csv [test_idx]")
        sys.exit(1)

    train_df = pd.read_csv(sys.argv[1])
    print(f"Loaded {len(train_df)} train rows")

    stats = compute_training_stats(train_df)
    print(f"\nClass distribution: {stats['class_dist']}")
    print(f"Lab series: {list(stats['labs'].keys())}")
    print(f"Sample stat for tieucau N5:")
    if "N5" in stats["labs"].get("tieucau", {}):
        for cls, s in stats["labs"]["tieucau"]["N5"].items():
            print(f"  {cls}: n={s['n']}, median={s['median']:.1f}, "
                  f"q25={s['q25']:.1f}, q75={s['q75']:.1f}")

    # Test positioning với BN mẫu
    print("\n=== Test positioning ===")
    sample = {
        "age": 24, "sex": "nam", "weight": 65,
        "fever_day_at_admission": 4,
        "symptoms": {"NVdaubung": 1, "NVnonoi": 1, "NVganto": 1},
        "labs": {
            "tieucau": {3: 120, 4: 73, 5: 41, 6: 22},
            "Hct":     {3: 40, 4: 42, 5: 46, 6: 48},
        }
    }
    pos = position_patient_against_stats(sample, stats)
    print(pos["narrative"])
    print(f"\nBest match: {pos['best_match_class']}")
    print(f"Scores: {pos['score_per_class']}")

    # Similar
    print("\n=== Similar patients ===")
    sim = find_similar_train_patients(sample, train_df, k=5)
    for s in sim:
        print(f"  - {s}")

    # Format for chat
    print("\n=== Formatted block (cho LLM) ===")
    print(format_positioning_for_chat(pos, sim))
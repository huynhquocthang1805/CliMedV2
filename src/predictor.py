"""
predictor.py — CliMedV2
=======================

LightGBM classifier cho phân loại mức độ SXHD.

Đặc điểm:
  • LightGBM hỗ trợ NaN nguyên gốc — KHÔNG cần fill missingness.
  • Patient-level split (mỗi hàng đã = 1 patient → tránh leakage).
  • Stratified K-fold CV để đánh giá robust trên dataset nhỏ (148 BN).
  • Class weight balanced để xử lý imbalance 96 vs 52.
  • Metrics: F1 (macro + per-class), ROC AUC, confusion matrix.
  • Feature importance (gain-based).
  • Save model + feature names + metadata.

Pipeline tích hợp:
  1. feature_engineering.prepare_features → X, y, meta
  2. missingness.build_missingness_features → X_full (mask + delta + imputed)
  3. predictor.train_and_evaluate → model + metrics
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Result dataclasses
# ===========================================================================

@dataclass
class FoldResult:
    fold_id: int
    train_size: int
    test_size: int
    f1_macro: float
    f1_per_class: List[float]
    roc_auc: Optional[float]      # None nếu multiclass < 2 unique trong test
    accuracy: float
    confusion_matrix: List[List[int]]


@dataclass
class TrainResult:
    """Kết quả training đầy đủ."""
    fold_results: List[FoldResult] = field(default_factory=list)
    holdout_metrics: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model: Any = None
    feature_names: List[str] = field(default_factory=list)
    target_spec: Any = None

    @property
    def cv_f1_mean(self) -> float:
        if not self.fold_results:
            return float("nan")
        return float(np.mean([f.f1_macro for f in self.fold_results]))

    @property
    def cv_f1_std(self) -> float:
        if not self.fold_results:
            return float("nan")
        return float(np.std([f.f1_macro for f in self.fold_results]))

    @property
    def cv_auc_mean(self) -> float:
        aucs = [f.roc_auc for f in self.fold_results if f.roc_auc is not None]
        return float(np.mean(aucs)) if aucs else float("nan")

    def summary(self) -> str:
        lines = ["=== Train Result ===",
                 f"  CV F1 (macro):    {self.cv_f1_mean:.3f} ± {self.cv_f1_std:.3f}",
                 f"  CV ROC AUC:       {self.cv_auc_mean:.3f}"]
        if self.holdout_metrics:
            lines += [
                f"  Holdout F1:       {self.holdout_metrics.get('f1_macro'):.3f}",
                f"  Holdout ROC AUC:  {self.holdout_metrics.get('roc_auc'):.3f}"
                f"" if self.holdout_metrics.get("roc_auc") is None
                else f"  Holdout ROC AUC:  {self.holdout_metrics.get('roc_auc'):.3f}",
                f"  Holdout accuracy: {self.holdout_metrics.get('accuracy'):.3f}",
            ]
        lines.append("\n  Top 10 features (by gain):")
        for name, score in list(self.feature_importance.items())[:10]:
            lines.append(f"    {name:40s}  {score:.1f}")
        return "\n".join(lines)


# ===========================================================================
# Helpers
# ===========================================================================

def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray,
                  n_classes: int) -> Optional[float]:
    """Tính ROC AUC, trả None nếu không tính được (vd chỉ 1 class trong y_true)."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return None

    try:
        if n_classes == 2:
            # binary: y_proba có shape (n, 2), lấy cột 1
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            return float(roc_auc_score(y_true, y_proba))
        # multiclass: dùng OVR macro
        return float(roc_auc_score(y_true, y_proba,
                                   multi_class="ovr", average="macro"))
    except ValueError as e:
        logger.warning("ROC AUC failed: %s", e)
        return None


def _make_lgbm(n_classes: int,
               class_weight: str = "balanced",
               random_state: int = 42,
               **overrides) -> Any:
    """Tạo LightGBM classifier với config phù hợp dataset y khoa nhỏ."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "LightGBM chưa cài. Chạy: pip install lightgbm"
        ) from e

    # Hyper-params chọn cho dataset nhỏ (148 BN), nhiều feature, missing nhiều
    params = dict(
        objective="binary" if n_classes == 2 else "multiclass",
        num_class=1 if n_classes == 2 else n_classes,
        learning_rate=0.05,
        num_leaves=15,           # nhỏ vì dataset nhỏ
        min_data_in_leaf=5,      # rất nhỏ vì 148 BN
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=200,
        class_weight=class_weight,
        random_state=random_state,
        verbose=-1,
        force_col_wise=True,
    )
    params.update(overrides)
    return lgb.LGBMClassifier(**params)


# ===========================================================================
# Cross-validation
# ===========================================================================

def stratified_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_classes: int,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[FoldResult]:
    """
    Stratified K-fold CV. Mỗi fold train LightGBM mới.
    Trả về list FoldResult.

    Patient-level split: mỗi hàng đã = 1 BN → KFold thông thường là đủ
    (không có leakage giữa các ngày của cùng BN vì wide format).
    """
    from sklearn.metrics import (
        confusion_matrix, f1_score, accuracy_score
    )
    from sklearn.model_selection import StratifiedKFold

    results: List[FoldResult] = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _make_lgbm(n_classes, random_state=random_state + fold_id)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)

        f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
        f1_per = f1_score(y_te, y_pred, average=None, zero_division=0).tolist()
        cm = confusion_matrix(y_te, y_pred,
                              labels=list(range(n_classes))).tolist()
        auc = _safe_roc_auc(y_te.values, y_proba, n_classes)

        results.append(FoldResult(
            fold_id=fold_id,
            train_size=len(tr_idx),
            test_size=len(te_idx),
            f1_macro=float(f1_macro),
            f1_per_class=f1_per,
            roc_auc=auc,
            accuracy=float(accuracy_score(y_te, y_pred)),
            confusion_matrix=cm,
        ))
        logger.info("Fold %d: F1=%.3f AUC=%s acc=%.3f",
                    fold_id, f1_macro,
                    f"{auc:.3f}" if auc is not None else "n/a",
                    accuracy_score(y_te, y_pred))

    return results


# ===========================================================================
# Train + holdout evaluate
# ===========================================================================

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    target_spec: Any = None,
    n_classes: Optional[int] = None,
    test_size: float = 0.2,
    n_cv_splits: int = 5,
    random_state: int = 42,
    do_cv: bool = True,
    split_indices: Optional[Dict[str, List[int]]] = None,
) -> TrainResult:
    """
    Pipeline đầy đủ:
      1. Holdout split 80/20 (stratified theo y) — HOẶC dùng `split_indices`
         truyền vào (để consistent với feature_engineering output).
      2. (Optional) Stratified K-fold CV trên train set → estimate generalization.
      3. Train final model trên ENTIRE train set.
      4. Evaluate trên holdout test.
      5. Trả về TrainResult với model + metrics.

    Args:
        split_indices: nếu truyền (vd từ meta['split_indices']),
                       sẽ dùng split đã có sẵn thay vì tự split lại
                       → đảm bảo cùng test set với feature_engineering.
    """
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score
    )
    from sklearn.model_selection import train_test_split

    if n_classes is None:
        n_classes = (target_spec.n_classes if target_spec is not None
                     else int(y.nunique()))

    # NẾU X có cột '_split' → loại bỏ trước khi train (LightGBM không xử lý
    # string-typed columns). Dùng nó để ưu tiên split.
    has_split_col = "_split" in X.columns
    if has_split_col and split_indices is None:
        # Dùng cột _split để xác định indices
        train_idx = X.index[X["_split"] == "train"].tolist()
        test_idx = X.index[X["_split"] == "test"].tolist()
        split_indices = {"train": train_idx, "test": test_idx}

    if has_split_col:
        X = X.drop(columns="_split")

    # 1. Holdout split (dùng split_indices nếu có, không thì stratified split)
    if split_indices is not None:
        train_idx = split_indices["train"]
        test_idx = split_indices["test"]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        logger.info("Dùng split_indices có sẵn: train=%d, test=%d",
                    len(train_idx), len(test_idx))
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        logger.info("Holdout split mới: train=%d, test=%d",
                    len(X_train), len(X_test))

    # 2. CV trên TRAIN (không leak test)
    fold_results: List[FoldResult] = []
    if do_cv and n_cv_splits >= 2:
        # Cẩn thận: nếu lớp nhỏ nhất có < n_cv_splits mẫu sau split → giảm folds
        min_class_count = y_train.value_counts().min()
        actual_splits = min(n_cv_splits, int(min_class_count))
        if actual_splits < n_cv_splits:
            logger.warning(
                "Lớp nhỏ nhất chỉ có %d mẫu trong train → giảm CV từ %d xuống %d",
                min_class_count, n_cv_splits, actual_splits)
        if actual_splits >= 2:
            fold_results = stratified_kfold_cv(
                X_train, y_train,
                n_classes=n_classes,
                n_splits=actual_splits,
                random_state=random_state,
            )

    # 3. Train final model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _make_lgbm(n_classes, random_state=random_state)
        model.fit(X_train, y_train)

    # 4. Holdout eval
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    holdout_metrics = {
        "f1_macro": float(f1_score(y_test, y_pred, average="macro",
                                   zero_division=0)),
        "f1_per_class": f1_score(y_test, y_pred, average=None,
                                 zero_division=0).tolist(),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=list(range(n_classes))).tolist(),
        "roc_auc": _safe_roc_auc(y_test.values, y_proba, n_classes),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0),
        "n_test": len(y_test),
    }

    # 5. Feature importance (gain-based, sorted desc)
    importances = model.booster_.feature_importance(importance_type="gain")
    fi = dict(sorted(zip(X.columns, importances),
                     key=lambda x: -x[1]))

    return TrainResult(
        fold_results=fold_results,
        holdout_metrics=holdout_metrics,
        feature_importance=fi,
        model=model,
        feature_names=X.columns.tolist(),
        target_spec=target_spec,
    )


# ===========================================================================
# Persistence
# ===========================================================================

def save_model(result: TrainResult, save_dir: Path) -> Path:
    """Lưu model + metadata vào folder."""
    import joblib

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.model, save_dir / "lgbm_model.joblib")

    # Save metadata (không lưu model trong json)
    target_spec_dict = None
    if result.target_spec is not None:
        target_spec_dict = {
            "column": result.target_spec.column,
            "mapping": {str(k): v for k, v in result.target_spec.mapping.items()},
            "n_classes": result.target_spec.n_classes,
        }

    meta = {
        "feature_names": result.feature_names,
        "target_spec": target_spec_dict,
        "cv_f1_mean": result.cv_f1_mean,
        "cv_f1_std": result.cv_f1_std,
        "cv_auc_mean": result.cv_auc_mean,
        "holdout_metrics": {
            k: v for k, v in result.holdout_metrics.items()
            if k != "classification_report"   # text, không cần json
        },
        "top_features": dict(list(result.feature_importance.items())[:30]),
    }
    with open(save_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Saved model to %s", save_dir)
    return save_dir


def load_model(save_dir: Path) -> Tuple[Any, Dict]:
    """Load model + metadata."""
    import joblib

    save_dir = Path(save_dir)
    model = joblib.load(save_dir / "lgbm_model.joblib")
    with open(save_dir / "model_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def predict_for_patient(model: Any,
                        feature_row: pd.DataFrame,
                        feature_names: List[str]) -> Dict[str, Any]:
    """
    Dự đoán cho 1 BN (DataFrame 1-row).

    Returns:
        {"predicted_class": int, "probabilities": list, "warning_prob": float}
    """
    # Đảm bảo đúng thứ tự cột
    X = feature_row.reindex(columns=feature_names, fill_value=np.nan)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    return {
        "predicted_class": pred,
        "probabilities": proba,
        "warning_prob": float(proba[1]) if len(proba) >= 2 else None,
    }


# ===========================================================================
# End-to-end pipeline
# ===========================================================================

def run_full_pipeline(
    csv_path: str,
    target_mode: str = "binary_warning",
    impute_short_gaps: bool = True,
    max_gap: int = 2,
    do_cv: bool = True,
    save_dir: Optional[str] = None,
    random_state: int = 42,
    test_size: float = 0.2,
    export_test_set: Optional[str] = None,
) -> TrainResult:
    """
    End-to-end:
      1. Load CSV
      2. feature_engineering.prepare_features (split 80/20 stratified)
      3. missingness.build_missingness_features
      4. train_and_evaluate (dùng split từ bước 2)
      5. (Optional) save model
      6. (Optional) export test set ra file để paste vào chatbox

    Args:
        export_test_set: nếu truyền path folder, sẽ xuất:
            • {folder}/test_holdout.csv  — full test data
            • {folder}/test_for_chatbox.txt — text format dễ paste
            • {folder}/test_summary.json — y_test ground truth
    """
    from feature_engineering import prepare_features
    from missingness import build_missingness_features

    logger.info("=== STEP 1: Load CSV ===")
    df = pd.read_csv(csv_path)
    logger.info("Loaded %s: %s", csv_path, df.shape)

    logger.info("=== STEP 2: Feature engineering + split 80/20 ===")
    X, y, meta = prepare_features(
        df, target_mode=target_mode,
        test_size=test_size, random_state=random_state,
        add_split_column=True,
    )

    # Export test set NẾU được yêu cầu (BEFORE missingness pipeline để giữ raw data)
    if export_test_set:
        from eval_chatbot import export_test_set_for_chatbox
        export_test_set_for_chatbox(
            meta["X_test_raw"], meta["y_test"],
            target_spec=meta["target_spec"],
            output_dir=export_test_set,
        )
        logger.info("Đã export test set ra %s", export_test_set)

    logger.info("=== STEP 3: Missingness pipeline ===")
    X_full, miss_reports = build_missingness_features(
        X.drop(columns="_split", errors="ignore"),
        meta["series_groups"],
        impute_short_gaps=impute_short_gaps,
        max_gap=max_gap,
        random_state=random_state,
    )
    # Restore _split sau khi missingness xong
    if "_split" in X.columns:
        X_full["_split"] = X["_split"].values
    logger.info("Final feature count: %d (was %d)",
                X_full.shape[1], X.shape[1])

    logger.info("=== STEP 4: Train + evaluate (dùng split đã có) ===")
    result = train_and_evaluate(
        X_full, y,
        target_spec=meta["target_spec"],
        do_cv=do_cv,
        random_state=random_state,
        split_indices=meta["split_indices"],
    )

    if save_dir:
        save_model(result, Path(save_dir))

    return result


def run_pipeline_from_split_files(
    target_mode: str = "multiclass",
    impute_short_gaps: bool = True,
    max_gap: int = 3,
    strategy: str = "linear_noise",
    focus_days: Optional[List[int]] = None,
    do_cv: bool = True,
    save_dir: Optional[str] = None,
    random_state: int = 42,
) -> TrainResult:
    """
    Pipeline mới — đọc TRỰC TIẾP từ 2 file `data/splits/train_XX.csv` và
    `data/splits/test_YY.csv` đã được tạo bởi `data_split.create_split_files()`.

    Args:
        impute_short_gaps: True = bật imputation cho interior gaps,
                           False = chỉ thêm mask + tdelta.
        max_gap: gap interior tối đa được impute (chỉ cho linear_noise/spline).
        strategy: "linear_noise" / "spline" / "mice".
                  Edge gap LUÔN giữ NaN (policy y khoa).
        focus_days: nếu None (mặc định) = dùng tất cả N1-N10.
                    Chỉ truyền list nếu muốn restrict (vd [3,4,5,6,7]).

    Khác `run_full_pipeline` ở chỗ:
      • Không split lại bên trong (split là việc của data_split.py).
      • Train CHỈ dùng file train, test CHỈ dùng file test → 100% rõ ràng.
      • focus_days mặc định None — fill NaN thay vì restrict ngày.
    """
    from data_split import load_train_df, load_test_df, load_split_meta
    from feature_engineering import prepare_features
    from missingness import build_missingness_features

    meta_split = load_split_meta()
    if meta_split is None:
        raise FileNotFoundError(
            "Chưa có file split. Vào tab 'Data & Train' bấm 'Tạo split file' "
            "hoặc chạy `python src/data_split.py`."
        )

    logger.info("=== STEP 1: Load split files ===")
    train_df = load_train_df()
    test_df = load_test_df()
    logger.info("Train: %s | Test: %s", train_df.shape, test_df.shape)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["__split_flag"] = "train"
    test_df["__split_flag"] = "test"
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    logger.info("=== STEP 2: Feature engineering (focus_days=%s) ===",
                focus_days)
    X, y, fe_meta = prepare_features(
        full_df.drop(columns="__split_flag"),
        target_mode=target_mode,
        focus_days=focus_days,
        add_split_column=False,
        test_size=0.2,
        random_state=random_state,
    )

    flag_arr = full_df["__split_flag"].values
    train_idx = np.where(flag_arr == "train")[0].tolist()
    test_idx = np.where(flag_arr == "test")[0].tolist()
    if len(train_idx) + len(test_idx) != len(X):
        valid = full_df.index[full_df[fe_meta["target_spec"].column].notna()]
        flag_after = full_df.loc[valid, "__split_flag"].reset_index(drop=True)
        train_idx = np.where(flag_after.values == "train")[0].tolist()
        test_idx = np.where(flag_after.values == "test")[0].tolist()

    split_indices = {"train": train_idx, "test": test_idx}

    logger.info("=== STEP 3: Missingness pipeline (strategy=%s) ===", strategy)
    X_full, miss_reports = build_missingness_features(
        X, fe_meta["series_groups"],
        impute_short_gaps=impute_short_gaps,
        max_gap=max_gap,
        strategy=strategy,
        random_state=random_state,
    )
    logger.info("Features: %d → %d", X.shape[1], X_full.shape[1])

    # Log imputation report
    total_imp = sum(r.n_imputed for r in miss_reports.values())
    total_kept = sum(r.n_kept_nan for r in miss_reports.values())
    logger.info("Total imputed: %d | Total kept NaN: %d "
                "(rìa = MNAR, giữ nguyên)", total_imp, total_kept)

    logger.info("=== STEP 4: Train + Evaluate ===")
    result = train_and_evaluate(
        X_full, y,
        target_spec=fe_meta["target_spec"],
        do_cv=do_cv,
        random_state=random_state,
        split_indices=split_indices,
    )

    if save_dir:
        save_model(result, Path(save_dir))

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    result = run_full_pipeline(
        csv_path="data/NghiencuuHFLC.csv",
        target_mode="binary_warning",
        save_dir="models/lgbm_warning",
    )
    print("\n" + result.summary())
    print("\nClassification report (holdout):")
    print(result.holdout_metrics.get("classification_report"))
    print("\nConfusion matrix (holdout):")
    cm = result.holdout_metrics["confusion_matrix"]
    print("           pred=0  pred=1")
    print(f"  true=0   {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"  true=1   {cm[1][0]:6d}  {cm[1][1]:6d}")
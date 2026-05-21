"""
auto_bootstrap.py — CliMedV2
=============================

Pipeline tự động khi app khởi động:
  1. Split data 80/20 (nếu chưa có)
  2. Compute training statistics (lưu .cache/train_stats.json)
  3. Train predictor model (RandomForest, lưu .cache/predictor.joblib)
  4. Build RAG index (qua rag_engine, lazy)
  5. Load knowledge graph (qua graph_engine, lazy)

Tất cả idempotent — skip step nếu artifact đã có. force_retrain=True để
rebuild toàn bộ.

Trả về BootstrapStatus với chi tiết từng bước để Streamlit hiển thị.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Paths
# ===========================================================================

CACHE_DIR = Path(".cache")
TRAIN_STATS_PATH = CACHE_DIR / "train_stats.json"
PREDICTOR_PATH = CACHE_DIR / "predictor.joblib"

RAW_DATA_PATH = Path("data/NghiencuuHFLC.csv")
SPLIT_DIR = Path("data/splits")
TRAIN_CSV = SPLIT_DIR / "train_80.csv"
TEST_CSV = SPLIT_DIR / "test_20.csv"


# ===========================================================================
# Status tracking
# ===========================================================================

@dataclass
class StepStatus:
    name: str
    ok: bool = False
    skipped: bool = False        # đã có artifact, không cần làm
    elapsed_s: float = 0.0
    message: str = ""
    error: Optional[str] = None


@dataclass
class BootstrapStatus:
    """Tổng kết tất cả các bước."""
    steps: List[StepStatus] = field(default_factory=list)
    total_elapsed_s: float = 0.0
    train_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None
    train_stats: Optional[Dict[str, Any]] = None
    predictor: Any = None

    @property
    def all_ok(self) -> bool:
        return all(s.ok or s.skipped for s in self.steps)

    def summary(self) -> str:
        lines = [f"Bootstrap {'✅' if self.all_ok else '⚠️'} "
                 f"({self.total_elapsed_s:.1f}s)"]
        for s in self.steps:
            icon = "✅" if s.ok else ("⏭️" if s.skipped else "❌")
            lines.append(f"  {icon} {s.name}: {s.message} ({s.elapsed_s:.1f}s)")
        return "\n".join(lines)


# ===========================================================================
# Individual steps
# ===========================================================================

def _step_split_data(force: bool = False) -> StepStatus:
    """Tạo train/test split nếu chưa có."""
    s = StepStatus(name="1. Data split 80/20")
    t0 = time.time()
    try:
        from data_split import create_split_files, split_files_exist

        if split_files_exist() and not force:
            s.skipped = True
            s.ok = True
            s.message = "Đã có split, skip."
            s.elapsed_s = time.time() - t0
            return s

        if not RAW_DATA_PATH.exists():
            s.error = f"Không có {RAW_DATA_PATH}"
            s.message = s.error
            s.elapsed_s = time.time() - t0
            return s

        create_split_files(
            source_csv=RAW_DATA_PATH,
            output_dir=SPLIT_DIR,
            test_size=0.2,
            random_state=42,
        )
        s.ok = True
        s.message = "Đã tạo train_80.csv + test_20.csv"
    except Exception as e:
        s.error = str(e)
        s.message = f"Lỗi: {e}"
        logger.exception("split step fail")
    s.elapsed_s = time.time() - t0
    return s


def _step_compute_stats(train_df: pd.DataFrame,
                        force: bool = False) -> StepStatus:
    """Tính training statistics."""
    s = StepStatus(name="2. Training statistics")
    t0 = time.time()
    try:
        from training_stats import (
            compute_training_stats, save_stats, load_stats,
        )

        if TRAIN_STATS_PATH.exists() and not force:
            s.skipped = True
            s.ok = True
            s.message = f"Đã có {TRAIN_STATS_PATH.name}, skip."
            s.elapsed_s = time.time() - t0
            return s

        stats = compute_training_stats(train_df)
        save_stats(stats, TRAIN_STATS_PATH)
        s.ok = True
        s.message = (f"n_train={stats['n_train']}, classes="
                     f"{stats['class_dist']}")
    except Exception as e:
        s.error = str(e)
        s.message = f"Lỗi: {e}"
        logger.exception("stats step fail")
    s.elapsed_s = time.time() - t0
    return s


def _step_train_predictor(train_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          force: bool = False) -> StepStatus:
    """Train RandomForest predictor (optional, không bắt buộc cho chat)."""
    s = StepStatus(name="3. Train predictor")
    t0 = time.time()
    try:
        import joblib

        if PREDICTOR_PATH.exists() and not force:
            s.skipped = True
            s.ok = True
            s.message = f"Đã có {PREDICTOR_PATH.name}, skip."
            s.elapsed_s = time.time() - t0
            return s

        from feature_engineering import prepare_features
        from predictor import train_and_evaluate

        # Đọc raw CSV, để prepare_features tự split với cùng seed → khớp với
        # train_80/test_20 đã tạo ở bước 1.
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Không có {RAW_DATA_PATH}")
        raw_df = pd.read_csv(RAW_DATA_PATH)

        X, y, meta = prepare_features(
            raw_df,
            target_mode="multiclass",
            test_size=0.2,
            random_state=42,
        )

        result = train_and_evaluate(
            X, y,
            target_spec=meta.get("target_spec"),
            split_indices=meta.get("split_indices"),
            do_cv=False,   # skip CV để nhanh ở bootstrap
        )

        PREDICTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": result.model,
            "feature_names": meta.get("feature_columns", []),
            "target_spec": meta.get("target_spec"),
            "metrics": result.holdout_metrics,
        }, PREDICTOR_PATH)
        s.ok = True
        s.message = (f"F1={result.holdout_metrics.get('f1_macro', 0):.3f}, "
                     f"Acc={result.holdout_metrics.get('accuracy', 0):.3f}")
    except Exception as e:
        s.error = str(e)
        s.message = f"Lỗi (không nghiêm trọng — chat vẫn dùng được): {e}"
        logger.warning("predictor step fail: %s", e)
        # Vẫn cho ok=True vì predictor không bắt buộc
        s.ok = True
    s.elapsed_s = time.time() - t0
    return s


def _step_rag_index(force: bool = False) -> StepStatus:
    """Build/load RAG index (lazy — chỉ trigger 1 lần đầu)."""
    s = StepStatus(name="4. RAG index (PDF guideline)")
    t0 = time.time()
    try:
        from rag_engine import get_rag_engine
        rag = get_rag_engine(force_rebuild=force)
        s.ok = True
        s.message = (f"{len(getattr(rag, 'chunks', []))} chunks "
                     f"từ {len(getattr(rag, 'documents', []))} docs")
    except Exception as e:
        s.error = str(e)
        s.message = f"Lỗi (chat vẫn dùng được, không có RAG): {e}"
        logger.warning("RAG step fail: %s", e)
        # Vẫn ok vì không bắt buộc
        s.ok = True
    s.elapsed_s = time.time() - t0
    return s


def _step_graph_load(force: bool = False) -> StepStatus:
    """Load knowledge graph YAML."""
    s = StepStatus(name="5. Knowledge graph")
    t0 = time.time()
    try:
        from graph_engine import get_graph
        g = get_graph()
        n_nodes = len(g.G.nodes) if hasattr(g, "G") else 0
        n_edges = len(g.G.edges) if hasattr(g, "G") else 0
        s.ok = True
        s.message = f"{n_nodes} nodes, {n_edges} edges"
    except Exception as e:
        s.error = str(e)
        s.message = f"Lỗi (chat vẫn dùng được, không có graph): {e}"
        logger.warning("Graph step fail: %s", e)
        s.ok = True
    s.elapsed_s = time.time() - t0
    return s


# ===========================================================================
# Orchestrator
# ===========================================================================

def bootstrap_all(
    force_retrain: bool = False,
    skip_predictor: bool = False,
    skip_rag: bool = False,
    skip_graph: bool = False,
    on_step: Optional[Callable[[StepStatus, int, int], None]] = None,
) -> BootstrapStatus:
    """
    Chạy toàn bộ pipeline khởi động. Idempotent — chỉ rebuild khi force_retrain.

    on_step: callback(step_status, step_idx, total_steps) cho Streamlit
        progress bar.
    """
    status = BootstrapStatus()
    t_total = time.time()

    n_steps = 5
    step_idx = 0

    # ----- 1. Split -----
    step_idx += 1
    s1 = _step_split_data(force=force_retrain)
    status.steps.append(s1)
    if on_step:
        on_step(s1, step_idx, n_steps)
    if not s1.ok:
        status.total_elapsed_s = time.time() - t_total
        return status

    # Load dataframes
    try:
        from data_split import load_train_df, load_test_df
        status.train_df = load_train_df()
        status.test_df = load_test_df()
    except Exception as e:
        s1.error = f"Load split fail: {e}"
        s1.ok = False
        status.total_elapsed_s = time.time() - t_total
        return status

    # ----- 2. Stats -----
    step_idx += 1
    s2 = _step_compute_stats(status.train_df, force=force_retrain)
    status.steps.append(s2)
    if on_step:
        on_step(s2, step_idx, n_steps)

    if s2.ok:
        try:
            from training_stats import load_stats
            status.train_stats = load_stats(TRAIN_STATS_PATH)
        except Exception as e:
            logger.warning("Load stats fail: %s", e)

    # ----- 3. Predictor (optional) -----
    step_idx += 1
    if skip_predictor:
        s3 = StepStatus(name="3. Train predictor",
                         skipped=True, ok=True,
                         message="Skipped (skip_predictor=True)")
    else:
        s3 = _step_train_predictor(status.train_df, status.test_df,
                                     force=force_retrain)
    status.steps.append(s3)
    if on_step:
        on_step(s3, step_idx, n_steps)

    if s3.ok and not s3.skipped and PREDICTOR_PATH.exists():
        try:
            import joblib
            status.predictor = joblib.load(PREDICTOR_PATH)
        except Exception as e:
            logger.warning("Load predictor fail: %s", e)

    # ----- 4. RAG -----
    step_idx += 1
    if skip_rag:
        s4 = StepStatus(name="4. RAG index", skipped=True, ok=True,
                         message="Skipped (skip_rag=True)")
    else:
        s4 = _step_rag_index(force=force_retrain)
    status.steps.append(s4)
    if on_step:
        on_step(s4, step_idx, n_steps)

    # ----- 5. Graph -----
    step_idx += 1
    if skip_graph:
        s5 = StepStatus(name="5. Knowledge graph", skipped=True, ok=True,
                         message="Skipped (skip_graph=True)")
    else:
        s5 = _step_graph_load(force=force_retrain)
    status.steps.append(s5)
    if on_step:
        on_step(s5, step_idx, n_steps)

    status.total_elapsed_s = time.time() - t_total
    return status


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    force = "--force" in sys.argv
    print(f"Bootstrap (force={force})...")

    def on_step(s, i, n):
        icon = "✅" if s.ok else ("⏭️" if s.skipped else "❌")
        print(f"  [{i}/{n}] {icon} {s.name}: {s.message}")

    status = bootstrap_all(force_retrain=force, on_step=on_step)
    print(f"\n{status.summary()}")
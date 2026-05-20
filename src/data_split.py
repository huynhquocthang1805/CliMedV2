"""
data_split.py — CliMedV2
========================

Chia file CSV gốc thành 2 file riêng `train_XX.csv` và `test_YY.csv` lưu vào
thư mục `data/splits/`. Sau đó các module khác (predictor, eval_chatbot,
patient_rag) đọc trực tiếp từ 2 file này thay vì split lại mỗi lần.

Ưu điểm:
  • Người dùng "thấy" rõ data đã chia ra sao (file vật lý).
  • Reproducible: cùng seed → cùng file → cùng kết quả.
  • Train pipeline và chatbot/RAG dùng CHUNG split → không bị data leakage.

Cách dùng (CLI):
    python src/data_split.py
    python src/data_split.py --test-size 0.2 --seed 42

Cách dùng (Python):
    from data_split import create_split_files, load_train_df, load_test_df
    create_split_files(test_size=0.2, random_state=42)
    train_df = load_train_df()
    test_df = load_test_df()
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Path chuẩn — module khác import constant này để consistent
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SPLIT_DIR = DATA_DIR / "splits"
DEFAULT_SOURCE = DATA_DIR / "NghiencuuHFLC.csv"

# Cột target — phải có để stratify
DEFAULT_TARGET_COL = "benhcanhcuoicunglagi"


# ===========================================================================
# Public API
# ===========================================================================

def create_split_files(
    source_csv: Path = DEFAULT_SOURCE,
    test_size: float = 0.2,
    random_state: int = 42,
    target_col: str = DEFAULT_TARGET_COL,
    output_dir: Path = SPLIT_DIR,
) -> Tuple[Path, Path]:
    """
    Đọc source CSV, split stratified theo target_col, ghi 2 file ra disk.

    Returns:
        (train_path, test_path)

    File output:
        {output_dir}/train_<int>.csv     vd: train_80.csv
        {output_dir}/test_<int>.csv      vd: test_20.csv
        {output_dir}/split_meta.json     thông tin reproducibility
    """
    source_csv = Path(source_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file gốc: {source_csv}")

    df = pd.read_csv(source_csv)
    logger.info("Đọc %s: %d hàng × %d cột", source_csv, *df.shape)

    if target_col not in df.columns:
        raise KeyError(
            f"Cột target '{target_col}' không có trong CSV.\n"
            f"Cột có sẵn: {list(df.columns)[:10]}..."
        )

    # Drop hàng có target NaN trước khi stratify
    valid_mask = df[target_col].notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.warning("Drop %d hàng có target NaN", n_dropped)
        df = df.loc[valid_mask].reset_index(drop=True)

    # Stratified split
    train_df, test_df = train_test_split(
        df, test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    # Reset index cho gọn (BN có ID mới 0..N-1 trong từng file)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Đặt tên file theo % thực tế
    train_pct = int(round((1 - test_size) * 100))
    test_pct = int(round(test_size * 100))
    train_path = output_dir / f"train_{train_pct}.csv"
    test_path = output_dir / f"test_{test_pct}.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Lưu meta để reproducibility
    meta = {
        "source": str(source_csv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_size": test_size,
        "random_state": random_state,
        "target_col": target_col,
        "n_total": int(len(train_df) + len(test_df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_class_distribution": _value_count_dict(train_df[target_col]),
        "test_class_distribution": _value_count_dict(test_df[target_col]),
        "train_file": train_path.name,
        "test_file": test_path.name,
    }
    meta_path = output_dir / "split_meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Đã tạo:")
    logger.info("  Train: %s (%d BN, phân lớp: %s)",
                train_path.name, len(train_df), meta["train_class_distribution"])
    logger.info("  Test:  %s (%d BN, phân lớp: %s)",
                test_path.name, len(test_df), meta["test_class_distribution"])
    logger.info("  Meta:  %s", meta_path.name)

    return train_path, test_path


def load_split_meta(output_dir: Path = SPLIT_DIR) -> Optional[dict]:
    """Load metadata. None nếu chưa split lần nào."""
    meta_path = Path(output_dir) / "split_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_train_df(output_dir: Path = SPLIT_DIR) -> pd.DataFrame:
    """Convenience: đọc file train hiện tại."""
    meta = load_split_meta(output_dir)
    if meta is None:
        raise FileNotFoundError(
            f"Chưa có split. Chạy `create_split_files()` trước "
            f"hoặc bấm nút 'Tạo split' trong app."
        )
    return pd.read_csv(Path(output_dir) / meta["train_file"])


def load_test_df(output_dir: Path = SPLIT_DIR) -> pd.DataFrame:
    """Convenience: đọc file test hiện tại."""
    meta = load_split_meta(output_dir)
    if meta is None:
        raise FileNotFoundError("Chưa có split.")
    return pd.read_csv(Path(output_dir) / meta["test_file"])


def split_files_exist(output_dir: Path = SPLIT_DIR) -> bool:
    """Check nhanh xem đã có split chưa."""
    return load_split_meta(output_dir) is not None


# ===========================================================================
# Helpers
# ===========================================================================

def _value_count_dict(series: pd.Series) -> dict:
    """value_counts() → dict với key str (cho json serializable)."""
    vc = series.value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in vc.items()}


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CSV thành train/test")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help="Đường dẫn file CSV gốc")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Tỷ lệ test (mặc định 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL,
                        help="Cột target để stratify")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    train_path, test_path = create_split_files(
        source_csv=Path(args.source),
        test_size=args.test_size,
        random_state=args.seed,
        target_col=args.target_col,
    )
    print(f"\n✅ Train: {train_path}")
    print(f"✅ Test:  {test_path}")
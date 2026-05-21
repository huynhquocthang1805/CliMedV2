"""
run_eval.py — CliMedV2
=======================

Chạy đánh giá đầy đủ Mini AI Triage Bot trên test holdout, sinh báo cáo
Markdown + CSV + JSON.

Dùng:
    # Full eval trên test_holdout.csv mặc định
    python src/run_eval.py

    # Custom test file + max BN
    python src/run_eval.py --test-csv data/splits/test_25.csv --max 10

    # So sánh 2 cấu hình (vd có/không RAG)
    python src/run_eval.py --no-rag --tag baseline
    python src/run_eval.py --tag with_rag
    # → so 2 file metrics_baseline.json vs metrics_with_rag.json

Output:
    eval_reports/<tag>/
      ├── report.md              ← Markdown report đẹp
      ├── metrics.json           ← raw metrics để diff/A-B test
      ├── per_patient.csv        ← từng BN, dễ xem trong Excel
      └── raw_responses.txt      ← full text response từng BN (debug)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Default paths — repo root assumed parent dir of src/
# ===========================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_CSV = REPO_ROOT / "data" / "eval_test_set" / "test_holdout.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "eval_reports"


# ===========================================================================
# Pretty progress
# ===========================================================================

def _progress(i: int, n: int, result) -> None:
    bar_len = 24
    filled = int(bar_len * i / max(n, 1))
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(f"\r  [{bar}] {i}/{n}  ·  {result.short_summary()[:80]:<80}")
    sys.stdout.flush()
    if i == n:
        sys.stdout.write("\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá CliMedV2 Mini Triage Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test-csv", type=str, default=str(DEFAULT_TEST_CSV),
                        help=f"CSV test (default: {DEFAULT_TEST_CSV})")
    parser.add_argument("--gt-col", type=str, default="_ground_truth_label",
                        help="Tên cột ground truth (default: _ground_truth_label). "
                             "Nếu CSV của bạn chỉ có cột số (vd benhcanhcuoicunglagi 1/2/3), "
                             "dùng --gt-col-int benhcanhcuoicunglagi")
    parser.add_argument("--gt-col-int", type=str, default=None,
                        help="Cột ground truth dạng INT (1/2/3) — sẽ tự map về thuong/canhbao/nang")
    parser.add_argument("--max", type=int, default=None,
                        help="Max BN để eval (debug)")
    parser.add_argument("--tag", type=str, default="default",
                        help="Tên cấu hình, dùng để tách output (default: default)")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "ollama", "hf"])
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name (vd qwen2.5:7b)")
    parser.add_argument("--no-rag", action="store_true", help="Tắt RAG")
    parser.add_argument("--use-graph", action="store_true",
                        help="Bật graph reasoning (chỉ có ý nghĩa nếu BN context phong phú)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--focus-days", type=str, default="3,4,5,6,7",
                        help="Ngày bệnh focus, comma-separated (default: 3-7)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Load test data ---
    test_csv = Path(args.test_csv)
    if not test_csv.exists():
        print(f"❌ Không tìm thấy {test_csv}")
        sys.exit(1)
    df = pd.read_csv(test_csv)
    print(f"📂 Load {len(df)} BN từ {test_csv}")

    # Resolve ground-truth column
    if args.gt_col_int and args.gt_col_int in df.columns:
        mapping = {1: "thuong", 2: "canhbao", 3: "nang"}
        df["_ground_truth_label"] = df[args.gt_col_int].map(mapping)
        gt_col = "_ground_truth_label"
        print(f"  Mapped {args.gt_col_int} (int 1/2/3) → _ground_truth_label")
    elif args.gt_col in df.columns:
        gt_col = args.gt_col
    else:
        print(f"❌ Không có cột {args.gt_col}. Cột có sẵn:")
        print("   " + ", ".join(df.columns.tolist()[:15]) + "...")
        sys.exit(1)

    # Distribution
    dist = df[gt_col].value_counts(dropna=False).to_dict()
    print(f"  Phân phối: {dist}")

    focus_days = [int(d) for d in args.focus_days.split(",") if d.strip()]

    # --- Init bot ---
    print(f"\n🤖 Khởi tạo TriageBot (backend={args.backend}, "
          f"rag={not args.no_rag}, graph={args.use_graph})")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from mini_ai_bot import TriageBot
    bot = TriageBot(
        backend=args.backend,
        model=args.model,
        use_rag=not args.no_rag,
        use_graph=args.use_graph,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(f"  Backend đang dùng: {bot.llm.name}")

    # --- Run batch ---
    n_to_run = min(len(df), args.max) if args.max else len(df)
    print(f"\n⚙️  Chạy phân loại {n_to_run} BN...")
    t0 = time.time()
    records = bot.triage_batch(
        df,
        ground_truth_col=gt_col,
        focus_days=focus_days,
        max_patients=args.max,
        on_progress=_progress,
    )
    elapsed = time.time() - t0
    print(f"\n✅ Hoàn tất trong {elapsed:.1f}s ({elapsed/max(n_to_run,1):.1f}s/BN)\n")

    # --- Compute metrics ---
    from eval_metrics import compute_all_metrics, format_report, export_per_patient_csv

    report_obj = compute_all_metrics(records)
    report_md = format_report(
        report_obj,
        title=f"CliMedV2 Triage Evaluation — tag: `{args.tag}`",
    )

    # --- Save outputs ---
    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Markdown report
    md_path = out_dir / "report.md"
    md_path.write_text(report_md, encoding="utf-8")
    print(f"📄 Report: {md_path}")

    # 2. JSON metrics
    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(report_obj.to_dict(), ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"📊 JSON:   {json_path}")

    # 3. Per-patient CSV
    csv_path = out_dir / "per_patient.csv"
    export_per_patient_csv(records, str(csv_path))
    print(f"📋 CSV:    {csv_path}")

    # 4. Raw responses (debug)
    raw_path = out_dir / "raw_responses.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"\n{'='*70}\nBN #{r['patient_id']}  ·  "
                    f"GT={r['ground_truth']}  ·  Bot={r['bot_label']}  "
                    f"({r.get('latency_ms', 0):.0f}ms)\n{'='*70}\n")
            f.write(r.get("raw_response", "") or "(no response)")
            f.write("\n")
    print(f"🔍 Raw:    {raw_path}")

    # --- Console summary ---
    print("\n" + "=" * 70)
    print(report_md)
    print("=" * 70)
    print(f"\n💾 Tất cả output ở: {out_dir}")
    print(f"\n📌 So sánh các tag: ls {Path(args.out_dir)}/*/metrics.json")


if __name__ == "__main__":
    main()
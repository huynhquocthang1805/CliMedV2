"""
streamlit_panels.py — CliMedV2
================================

UI components mới cho Streamlit app, gọi từ app.py để render 2 tab mới:

  • **Mini Triage Bot** — paste BN text / chọn từ test set → TriageResult có
    cấu trúc: severity badge (1/2/3), gauge confidence, evidence cards,
    red flags, recommended action.

  • **Advanced Eval** — chạy `mini_ai_bot.TriageBot` trên cả test set,
    hiển thị toàn bộ metrics của `eval_metrics`: classification + safety +
    calibration + quality + latency. Hỗ trợ A/B so 2 cấu hình.

Mỗi panel là 1 function `render_*`, không có top-level Streamlit calls →
import nhẹ, không side-effect, dễ test.

Cách dùng từ app.py:
    from streamlit_panels import (
        render_mini_triage_panel, render_advanced_eval_panel
    )
    with triage_tab:
        render_mini_triage_panel(session_mgr)
    with eval_tab:
        render_advanced_eval_panel()
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants & styling helpers
# ===========================================================================

# Màu sắc theo severity — phù hợp Streamlit
SEVERITY_COLOR = {
    "thuong":  "#4caf50",   # xanh lá
    "canhbao": "#ff9800",   # cam
    "nang":    "#e53935",   # đỏ
}
SEVERITY_EMOJI = {
    "thuong":  "🟢",
    "canhbao": "🟡",
    "nang":    "🔴",
}
SEVERITY_VI = {
    "thuong":  "SXHD thông thường (Mức 1)",
    "canhbao": "SXHD có dấu hiệu cảnh báo (Mức 2)",
    "nang":    "SXHD nặng (Mức 3)",
}


def _severity_badge(label: Optional[str], level: Optional[int] = None,
                    big: bool = True) -> str:
    """Render HTML badge cho severity. Trả về HTML string."""
    if not label:
        return ('<span style="background:#9e9e9e;color:white;'
                'padding:6px 14px;border-radius:8px;font-weight:600">'
                'KHÔNG XÁC ĐỊNH</span>')
    color = SEVERITY_COLOR.get(label, "#9e9e9e")
    emoji = SEVERITY_EMOJI.get(label, "❓")
    vi = SEVERITY_VI.get(label, label)
    size = "1.2rem" if big else "0.95rem"
    pad = "10px 18px" if big else "4px 10px"
    return (f'<span style="background:{color};color:white;padding:{pad};'
            f'border-radius:10px;font-weight:700;font-size:{size}">'
            f'{emoji} {vi}</span>')


def _conf_gauge_html(conf: Optional[float]) -> str:
    """Mini progress bar HTML cho confidence."""
    if conf is None:
        return '<span style="color:#9e9e9e">— (no confidence)</span>'
    pct = max(0, min(100, int(conf * 100)))
    if pct >= 80:
        color = "#4caf50"
    elif pct >= 60:
        color = "#ff9800"
    else:
        color = "#e53935"
    return (f'<div style="background:#eee;border-radius:6px;width:100%;'
            f'height:22px;position:relative">'
            f'<div style="background:{color};width:{pct}%;height:22px;'
            f'border-radius:6px"></div>'
            f'<div style="position:absolute;top:0;left:0;width:100%;'
            f'text-align:center;line-height:22px;color:#000;font-weight:600;'
            f'font-size:0.85rem">{conf:.2f}</div></div>')


# ===========================================================================
# Mini Triage Panel
# ===========================================================================

def render_mini_triage_panel(session_mgr=None) -> None:
    """
    Tab phân loại 1 BN — paste text hoặc chọn từ test set sẵn có.

    Hiển thị structured output: severity badge to, confidence gauge, evidence
    cards, red flags, recommended action panel.
    """
    st.markdown(
        '<div style="text-align:center;padding:1rem 0">'
        '<h2 style="color:#1976d2;margin-bottom:0.3rem">'
        '🩺 Mini Triage Bot</h2>'
        '<p style="color:#546e7a;font-style:italic">'
        'Phân loại nhanh 1 BN SXHD theo 3 mức: thường / cảnh báo / nặng. '
        'Kèm bằng chứng có cấu trúc và khuyến nghị xử trí.</p></div>',
        unsafe_allow_html=True,
    )

    # ----- Config row -----
    cfg1, cfg2, cfg3, cfg4 = st.columns([2, 1, 1, 1])
    with cfg1:
        backend = st.selectbox("Backend LLM",
                               ["auto", "ollama", "hf"],
                               key="triage_backend")
    with cfg2:
        use_rag = st.checkbox("📚 RAG", value=True, key="triage_rag",
                              help="Bằng chứng từ guideline Bộ Y tế VN")
    with cfg3:
        temp = st.slider("Temp", 0.0, 1.0, 0.2, 0.05, key="triage_temp",
                         help="Càng thấp càng ổn định")
    with cfg4:
        max_tok = st.number_input("Max tokens", 200, 1500, 500,
                                  step=100, key="triage_maxtok")

    # ----- Input source -----
    st.markdown("##### 1. Chọn nguồn dữ liệu BN")
    source = st.radio("Nguồn",
                      ["✍️ Paste text", "📋 Chọn từ test holdout"],
                      label_visibility="collapsed",
                      horizontal=True,
                      key="triage_source")

    patient_text: Optional[str] = None
    patient_id: Optional[str] = None
    gt_label: Optional[str] = None

    if source.startswith("✍️"):
        default_text = ("--- Bệnh nhân #demo ---\n"
                        "Thông tin: Giới tính: nam | Tuổi: 24 | Cân nặng: 65kg\n"
                        "Ngày bệnh lúc nhập viện: 4\n"
                        "Triệu chứng (+): đau bụng, nôn nhiều, gan to\n"
                        "Lab theo ngày (N3-N7):\n"
                        "  Tiểu cầu (K/uL): N3=120, N4=73, N5=41, N6=22, N7=18\n"
                        "  Hct (%): N3=40, N4=42, N5=46, N6=48, N7=44\n"
                        "  AST: 280, ALT: 195\n")
        patient_text = st.text_area("Text BN (format render_patient_card)",
                                    value=default_text, height=240,
                                    key="triage_text")
        patient_id = "manual"

    else:
        # Load test holdout
        try:
            holdout_path = Path("data/eval_test_set/test_holdout.csv")
            if not holdout_path.exists():
                st.warning(
                    f"⚠️ Chưa có `{holdout_path}`. Vào tab "
                    "**Eval Chatbot → Bước 1** để export trước."
                )
                return
            df = pd.read_csv(holdout_path)
            sel = st.number_input(
                f"Chọn BN (0 → {len(df)-1})", 0, len(df)-1, 0,
                key="triage_select",
            )
            row = df.iloc[sel]
            gt_label = row.get("_ground_truth_label")
            patient_id = str(sel)

            # Render text qua eval_chatbot.render_patient_card
            try:
                from eval_chatbot import render_patient_card
                patient_text = render_patient_card(
                    row, patient_id=sel, focus_days=[3, 4, 5, 6, 7])
            except Exception as e:
                st.error(f"Lỗi render BN: {e}")
                return

            st.caption(
                f"Ground truth (chỉ để so sánh sau): "
                f"`{gt_label}` — {SEVERITY_VI.get(gt_label, '?')}"
            )
            with st.expander("👁 Xem text BN sẽ gửi cho bot"):
                st.code(patient_text, language="text")
        except Exception as e:
            st.error(f"Không load được test_holdout: {e}")
            return

    if not patient_text or len(patient_text.strip()) < 30:
        st.info("Nhập đủ thông tin BN trước khi bấm Triage.")
        return

    # ----- Run button -----
    st.markdown("##### 2. Phân loại")
    if not st.button("🚀 Triage BN này", type="primary",
                     key="triage_run", use_container_width=True):
        return

    # ----- Run TriageBot -----
    with st.spinner("Đang phân loại..."):
        try:
            from mini_ai_bot import TriageBot
            bot = TriageBot(backend=backend, use_rag=use_rag,
                            temperature=temp, max_tokens=int(max_tok))
        except Exception as e:
            st.error(f"❌ Không khởi tạo được LLM: {e}")
            return

        try:
            result = bot.triage(patient_text, patient_id=patient_id)
        except Exception as e:
            st.error(f"❌ Triage fail: {e}")
            return

    # ----- Display result -----
    if not result.parsed_successfully:
        st.error(f"❌ Bot không trả về JSON hợp lệ: {result.parse_error}")
        with st.expander("🔍 Raw response (debug)"):
            st.code(result.raw_response or "(empty)")
        return

    st.markdown("##### 3. Kết quả phân loại")

    # Severity + confidence + ground truth comparison
    res_col1, res_col2 = st.columns([3, 2])
    with res_col1:
        st.markdown(_severity_badge(result.severity_label, big=True),
                    unsafe_allow_html=True)
        if gt_label and gt_label != result.severity_label:
            st.error(
                f"⚠️ Khác ground truth: GT=`{gt_label}` "
                f"({SEVERITY_VI.get(gt_label, '?')}) ↔ "
                f"Bot=`{result.severity_label}`"
            )
        elif gt_label and gt_label == result.severity_label:
            st.success(f"✅ Khớp ground truth `{gt_label}`")
    with res_col2:
        st.markdown("**Confidence**")
        st.markdown(_conf_gauge_html(result.confidence),
                    unsafe_allow_html=True)
        st.caption(f"⏱️ Latency: {result.latency_ms:.0f} ms")

    # Reasoning
    if result.reasoning:
        st.markdown(f"**🧠 Lý do:** {result.reasoning}")

    # Evidence cards
    if result.evidence_lines:
        st.markdown("**📋 Bằng chứng từ BN:**")
        for ev in result.evidence_lines:
            st.markdown(f"  - {ev}")

    # Red flags
    if result.red_flags:
        st.markdown("**🚨 Cờ đỏ phát hiện:**")
        for fl in result.red_flags:
            st.warning(fl, icon="⚠️")

    # Recommended action — emphasized
    st.markdown("---")
    st.markdown("##### 💊 Khuyến nghị xử trí")
    color = SEVERITY_COLOR.get(result.severity_label, "#9e9e9e")
    st.markdown(
        f'<div style="background:{color}22;border-left:6px solid {color};'
        f'padding:14px 18px;border-radius:6px;font-size:1.02rem">'
        f'{result.recommended_action}</div>',
        unsafe_allow_html=True,
    )

    # Optional: save vào chat session
    if session_mgr is not None:
        if st.button("💬 Lưu vào cuộc trò chuyện hiện tại",
                     key="triage_save"):
            sid = st.session_state.get("active_session_id")
            if sid is None:
                sid = session_mgr.create_session()
                st.session_state["active_session_id"] = sid
            user_msg = f"Phân loại BN #{patient_id}:\n```\n{patient_text}\n```"
            asst_msg = (
                f"**Phân loại:** {SEVERITY_VI[result.severity_label]}  \n"
                f"**Confidence:** {result.confidence:.2f}  \n"
                f"**Lý do:** {result.reasoning}  \n\n"
                f"**Bằng chứng:**\n" +
                "\n".join(f"- {e}" for e in result.evidence_lines) +
                f"\n\n**Khuyến nghị:** {result.recommended_action}"
            )
            session_mgr.append_message(sid, "user", user_msg)
            session_mgr.append_message(sid, "assistant", asst_msg)
            st.success("✅ Đã lưu vào cuộc trò chuyện.")

    # Debug
    with st.expander("🔍 Raw LLM response (debug)"):
        st.code(result.raw_response)


# ===========================================================================
# Advanced Eval Panel
# ===========================================================================

def render_advanced_eval_panel(
    load_test_df_fn: Optional[Callable[[], pd.DataFrame]] = None,
    split_files_exist_fn: Optional[Callable[[], bool]] = None,
) -> None:
    """
    Tab eval mới với toàn bộ metrics từ eval_metrics.compute_all_metrics:
      • Classification (Acc, Balanced Acc, F1, Kappa, QWK, MAE)
      • Clinical safety (Under-triage, Critical miss, Severe recall...)
      • Calibration (Brier, ECE, bin histogram)
      • Response quality
      • Latency

    Hỗ trợ A/B compare 2 cấu hình (vd có/không RAG).
    """
    st.markdown(
        '<div style="text-align:center;padding:0.8rem 0">'
        '<h2 style="color:#1976d2">🎯 Advanced Eval — Triage Bot</h2>'
        '<p style="color:#546e7a">Đánh giá đầy đủ trên test holdout với '
        '25+ metrics, focus vào safety y khoa.</p></div>',
        unsafe_allow_html=True,
    )

    # ---- Load test set ----
    holdout_path = Path("data/eval_test_set/test_holdout.csv")
    if not holdout_path.exists():
        st.warning(
            f"⚠️ Chưa có `{holdout_path}`. Trong tab **Eval Chatbot (cũ) → "
            "Bước 1: Xuất test_for_chatbox.txt** để tạo, hoặc dùng "
            "`splits/test_20.csv` với cột `benhcanhcuoicunglagi`."
        )
        # Fallback: cho phép user chọn file CSV khác
        alt = st.text_input(
            "Hoặc nhập đường dẫn CSV khác:",
            value="data/splits/test_20.csv",
            key="adv_eval_alt_path",
        )
        if not alt or not Path(alt).exists():
            return
        csv_path = Path(alt)
        df = pd.read_csv(csv_path)
        # Map cột integer → label nếu cần
        if ("_ground_truth_label" not in df.columns
                and "benhcanhcuoicunglagi" in df.columns):
            mapping = {1: "thuong", 2: "canhbao", 3: "nang"}
            df["_ground_truth_label"] = df["benhcanhcuoicunglagi"].map(mapping)
            st.info("Đã map cột `benhcanhcuoicunglagi` (1/2/3) → "
                    "`_ground_truth_label`.")
    else:
        df = pd.read_csv(holdout_path)
        csv_path = holdout_path

    # ---- Distribution ----
    gt_col = "_ground_truth_label"
    if gt_col not in df.columns:
        st.error(f"❌ CSV không có cột `{gt_col}` và không map được.")
        return

    st.caption(f"📂 Test set: **{len(df)} BN** từ `{csv_path}`")
    dist = df[gt_col].value_counts(dropna=False).to_dict()
    dist_cols = st.columns(len(dist) + 1)
    dist_cols[0].metric("Tổng BN", len(df))
    for i, (lbl, n) in enumerate(dist.items(), start=1):
        dist_cols[i].metric(
            f"{SEVERITY_EMOJI.get(lbl, '❔')} {lbl}", int(n),
            help=SEVERITY_VI.get(lbl, str(lbl)),
        )

    st.markdown("---")

    # ---- Run config ----
    st.markdown("#### ⚙️ Cấu hình run")
    rc1, rc2, rc3, rc4 = st.columns([1, 1, 1, 2])
    with rc1:
        backend = st.selectbox("Backend", ["auto", "ollama", "hf"],
                               key="adv_eval_backend")
    with rc2:
        use_rag = st.checkbox("📚 RAG", value=True, key="adv_eval_rag")
    with rc3:
        temp = st.slider("Temp", 0.0, 1.0, 0.2, 0.05, key="adv_eval_temp")
    with rc4:
        max_n = st.number_input("Max BN (0=all)", 0, len(df), 0,
                                step=1, key="adv_eval_maxn")

    rc5, rc6 = st.columns(2)
    with rc5:
        tag = st.text_input("Tag (label cho run)",
                            value="default", key="adv_eval_tag")
    with rc6:
        focus_days_str = st.text_input("Focus days", value="3,4,5,6,7",
                                       key="adv_eval_focus")

    focus_days = [int(d) for d in focus_days_str.split(",") if d.strip()]
    max_patients = int(max_n) if max_n > 0 else None

    st.markdown("---")

    # ---- Run button ----
    run_col1, run_col2 = st.columns([3, 1])
    with run_col1:
        run_btn = st.button("🚀 Chạy auto-eval", type="primary",
                            key="adv_eval_run", use_container_width=True)
    with run_col2:
        clear_btn = st.button("🗑️ Xóa kết quả",
                              key="adv_eval_clear",
                              use_container_width=True)
    if clear_btn:
        st.session_state.pop(f"adv_eval_result_{tag}", None)
        st.rerun()

    if run_btn:
        _run_advanced_eval(df, gt_col, backend, use_rag, temp,
                            focus_days, max_patients, tag)

    st.markdown("---")

    # ---- Display results ----
    available_tags = [k.replace("adv_eval_result_", "")
                      for k in st.session_state.keys()
                      if k.startswith("adv_eval_result_")]

    if not available_tags:
        st.info("Chưa có kết quả. Bấm 'Chạy auto-eval' để bắt đầu.")
        return

    # ---- Single or A/B view ----
    view_mode = st.radio(
        "Chế độ xem", ["📊 Chi tiết 1 run", "🔀 A/B compare 2 runs"],
        horizontal=True, key="adv_eval_view",
    )

    if view_mode.startswith("📊"):
        sel_tag = st.selectbox("Chọn tag", available_tags,
                               key="adv_eval_sel_single")
        _show_single_eval(sel_tag,
                          st.session_state[f"adv_eval_result_{sel_tag}"])
    else:
        if len(available_tags) < 2:
            st.warning("Cần ít nhất 2 runs để so sánh. Chạy thêm 1 tag nữa.")
            return
        ab1, ab2 = st.columns(2)
        with ab1:
            tag_a = st.selectbox("Run A", available_tags,
                                 key="adv_eval_sel_a")
        with ab2:
            tag_b = st.selectbox("Run B", available_tags,
                                 index=min(1, len(available_tags) - 1),
                                 key="adv_eval_sel_b")
        if tag_a == tag_b:
            st.warning("Chọn 2 tag khác nhau để compare.")
            return
        _show_ab_compare(
            tag_a, st.session_state[f"adv_eval_result_{tag_a}"],
            tag_b, st.session_state[f"adv_eval_result_{tag_b}"],
        )


def _run_advanced_eval(df: pd.DataFrame, gt_col: str,
                       backend: str, use_rag: bool, temp: float,
                       focus_days: List[int],
                       max_patients: Optional[int],
                       tag: str) -> None:
    """Chạy TriageBot trên test set + compute metrics."""
    try:
        from mini_ai_bot import TriageBot
        from eval_metrics import compute_all_metrics
    except Exception as e:
        st.error(f"❌ Import lỗi: {e}")
        return

    try:
        bot = TriageBot(backend=backend, use_rag=use_rag,
                        temperature=temp, max_tokens=500)
        st.info(f"🤖 Backend: `{bot.llm.name}` · RAG: {'on' if use_rag else 'off'}")
    except Exception as e:
        st.error(f"❌ Không khởi tạo được LLM: {e}")
        return

    n_to_run = min(len(df), max_patients) if max_patients else len(df)
    progress = st.progress(0.0, text=f"Bắt đầu eval {n_to_run} BN...")
    log_placeholder = st.empty()
    log_lines: List[str] = []

    def on_prog(i, n, result):
        progress.progress(i / n, text=f"BN {i}/{n}")
        ok = "✅" if result.parsed_successfully else "❌"
        log_lines.append(
            f"{ok} BN#{result.patient_id:>3s} → "
            f"{result.severity_label or '?':>8s} "
            f"({result.latency_ms:.0f}ms)"
        )
        log_placeholder.code("\n".join(log_lines[-10:]))

    t0 = time.time()
    records = bot.triage_batch(
        df, ground_truth_col=gt_col, focus_days=focus_days,
        max_patients=max_patients, on_progress=on_prog,
    )
    elapsed = time.time() - t0
    progress.progress(1.0, text=f"Xong trong {elapsed:.1f}s")

    report = compute_all_metrics(records)
    st.session_state[f"adv_eval_result_{tag}"] = {
        "records": records,
        "report": report.to_dict(),
        "tag": tag,
        "elapsed_s": elapsed,
        "n_run": n_to_run,
        "use_rag": use_rag,
        "temp": temp,
        "backend": backend,
    }
    st.success(f"✅ Hoàn tất tag `{tag}` trong {elapsed:.1f}s "
               f"({elapsed/max(n_to_run,1):.1f}s/BN)")


def _show_single_eval(tag: str, data: Dict[str, Any]) -> None:
    """Show full metrics dashboard for 1 run."""
    report = data["report"]
    records = data["records"]

    # Top-line metrics
    st.markdown(f"### 📊 Kết quả tag `{tag}`")
    cfg_line = (f"Backend: `{data['backend']}` · "
                f"RAG: {'✅' if data['use_rag'] else '❌'} · "
                f"Temp: {data['temp']} · "
                f"N: {data['n_run']} · "
                f"Time: {data['elapsed_s']:.1f}s")
    st.caption(cfg_line)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{report['accuracy']:.1%}")
    m2.metric("Balanced Acc", f"{report['balanced_accuracy']:.1%}")
    m3.metric("Macro F1", f"{report['macro_f1']:.3f}")
    m4.metric("QWK (ordinal)", f"{report['qwk']:.3f}")
    m5.metric("MAE ordinal", f"{report['mae_ordinal']:.2f}")

    # Safety panel (most important)
    s = report.get("safety", {})
    if s:
        st.markdown("#### ⚕️ Clinical safety  *(quan trọng nhất)*")
        sf1, sf2, sf3, sf4 = st.columns(4)
        sf1.metric("Exact match",
                   f"{s.get('exact_match_rate', 0):.1%}")
        sf2.metric("Under-triage ⚠️",
                   f"{s.get('under_triage_rate', 0):.1%}",
                   delta="0% target", delta_color="inverse")
        sf3.metric("Critical miss 🚨",
                   f"{s.get('critical_miss_rate', 0):.1%} "
                   f"({s.get('n_critical_miss', 0)} ca)",
                   delta="0 target", delta_color="inverse")
        sf4.metric("Severe recall",
                   f"{s.get('severe_recall', 0):.1%}",
                   delta="≥90% target",
                   delta_color="off")

        # Deploy readiness check
        ready = (s.get('n_critical_miss', 0) == 0
                 and s.get('under_triage_rate', 1) <= 0.10
                 and s.get('severe_recall', 0) >= 0.90)
        if ready:
            st.success(
                "✅ **Đạt ngưỡng deploy:** critical_miss=0, "
                "under_triage≤10%, severe_recall≥90%. "
                "Bot có thể dùng triage hỗ trợ (vẫn cần bác sĩ duyệt cuối).",
                icon="✅",
            )
        else:
            st.error(
                "⚠️ **Chưa đạt ngưỡng deploy.** Bot chỉ nên dùng **hỗ trợ "
                "tham khảo**, không thay thế bác sĩ triage. Xem chi tiết "
                "các metrics bên dưới để cải thiện.",
                icon="⚠️",
            )

    # Confusion matrix
    if report.get("confusion_matrix"):
        st.markdown("#### 🎯 Confusion matrix")
        classes = report.get("classes", ["thuong", "canhbao", "nang"])
        cm_df = pd.DataFrame(
            report["confusion_matrix"],
            index=[f"true: {c}" for c in classes],
            columns=[f"pred: {c}" for c in classes],
        )
        # Highlight diagonal (correct) vs off-diagonal (errors)
        def _highlight(df_):
            styles = pd.DataFrame("", index=df_.index, columns=df_.columns)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    if i == j:
                        styles.iloc[i, j] = "background-color: #c8e6c9"
                    elif j < i:    # under-triage = below diagonal
                        styles.iloc[i, j] = "background-color: #ffcdd2"
                    else:           # over-triage
                        styles.iloc[i, j] = "background-color: #fff9c4"
            return styles
        st.dataframe(cm_df.style.apply(_highlight, axis=None),
                     use_container_width=True)
        st.caption(
            "🟢 đường chéo = đúng · 🔴 dưới chéo = under-triage (nguy hiểm) "
            "· 🟡 trên chéo = over-triage (an toàn)"
        )

    # Per-class
    if report.get("per_class"):
        st.markdown("#### 📋 Per-class metrics")
        pc_data = []
        for cls, m in report["per_class"].items():
            pc_data.append({
                "Class": f"{SEVERITY_EMOJI.get(cls, '')} {cls}",
                "VN": SEVERITY_VI.get(cls, cls),
                "Precision": f"{m['precision']:.1%}",
                "Recall": f"{m['recall']:.1%}",
                "F1": f"{m['f1']:.3f}",
                "Support": int(m["support"]),
            })
        st.dataframe(pd.DataFrame(pc_data), hide_index=True,
                     use_container_width=True)

    # Calibration
    c = report.get("calibration", {})
    if c and c.get("n_with_conf", 0) > 0:
        st.markdown("#### 📈 Confidence calibration")
        cb1, cb2, cb3 = st.columns(3)
        cb1.metric("Brier score", f"{c['brier']:.3f}",
                   help="Càng thấp càng tốt (0 = hoàn hảo)")
        cb2.metric("ECE",
                   f"{c['ece']:.1%}",
                   help="<5% là tốt, >15% là tệ")
        cb3.metric("N có confidence", c["n_with_conf"])

        if c.get("bins"):
            bin_df = pd.DataFrame([
                {"Range": b["range"],
                 "N": b["n"],
                 "Avg conf": f"{b['avg_conf']:.2f}",
                 "Accuracy": f"{b['accuracy']:.2f}",
                 "Gap (overconfidence)":
                    f"{b['avg_conf'] - b['accuracy']:+.2f}"}
                for b in c["bins"] if b["n"] > 0
            ])
            st.dataframe(bin_df, hide_index=True,
                         use_container_width=True)

    # Quality + latency
    q = report.get("quality", {})
    l = report.get("latency", {})
    if q or l:
        st.markdown("#### 📏 Chất lượng & latency")
        ql1, ql2 = st.columns(2)
        with ql1:
            if q:
                st.markdown("**Chất lượng response**")
                st.write(
                    f"- Độ dài (từ): "
                    f"{q['words']['mean']:.0f} "
                    f"(min={q['words']['min']}, "
                    f"max={q['words']['max']})\n"
                    f"- Citation rate: {q['citation_rate']:.1%}\n"
                    f"- Lab grounding rate: {q['lab_ground_rate']:.1%}\n"
                    f"- JSON parse success: {q['parse_success_rate']:.1%}"
                )
        with ql2:
            if l.get("n", 0) > 0:
                st.markdown("**⏱ Latency**")
                st.write(
                    f"- p50: {l['p50_ms']:.0f} ms\n"
                    f"- p95: {l['p95_ms']:.0f} ms\n"
                    f"- mean: {l['mean_ms']:.0f} ms\n"
                    f"- Tổng: {l['total_s']:.0f}s "
                    f"cho {l['n']} BN"
                )

    # Per-patient table
    st.markdown("#### 📝 Chi tiết từng BN")
    rows = []
    for r in records:
        gt = r["ground_truth"]
        pd_ = r["bot_label"]
        if pd_ is None:
            status = "⚠️ parse fail"
        elif pd_ == gt:
            status = "✅"
        else:
            gt_lvl = {"thuong": 1, "canhbao": 2, "nang": 3}.get(gt, 0)
            pd_lvl = {"thuong": 1, "canhbao": 2, "nang": 3}.get(pd_, 0)
            if pd_lvl == 1 and gt_lvl == 3:
                status = "🚨 CRITICAL MISS"
            elif pd_lvl < gt_lvl:
                status = "❌ under"
            else:
                status = "⚠️ over"
        rows.append({
            "BN": r["patient_id"],
            "GT": gt,
            "Bot": pd_ or "—",
            "Conf": f"{r['bot_confidence']:.2f}" if r['bot_confidence'] else "—",
            "Status": status,
            "Latency": f"{r.get('latency_ms', 0):.0f}ms",
            "Reasoning": (r.get("bot_reasoning") or "")[:100],
        })
    per_df = pd.DataFrame(rows)
    st.dataframe(per_df, hide_index=True, use_container_width=True)

    # Drill-down
    with st.expander("🔍 Drill-down 1 BN"):
        if len(records) > 0:
            sel_idx = st.number_input(
                "BN index", 0, len(records) - 1, 0,
                key=f"adv_eval_drill_{tag}",
            )
            r = records[int(sel_idx)]
            st.markdown(
                f"**GT:** {_severity_badge(r['ground_truth'], big=False)}  "
                f"&nbsp;&nbsp; **Bot:** "
                f"{_severity_badge(r['bot_label'], big=False)}",
                unsafe_allow_html=True,
            )
            if r.get("evidence_lines"):
                st.markdown("**Bằng chứng bot tìm:**")
                for e in r["evidence_lines"]:
                    st.markdown(f"- {e}")
            if r.get("red_flags"):
                st.markdown("**Cờ đỏ:**")
                for f in r["red_flags"]:
                    st.warning(f, icon="⚠️")
            if r.get("recommended_action"):
                st.info(r["recommended_action"], icon="💊")
            with st.expander("Raw LLM response"):
                st.code(r.get("raw_response", "")[:3000])

    # Download buttons
    st.markdown("#### 💾 Tải xuống")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        from eval_metrics import format_report, EvalReport
        # Reconstruct EvalReport from dict for format_report
        rep_obj = EvalReport(**{k: v for k, v in report.items()
                                if k in EvalReport.__dataclass_fields__})
        md = format_report(rep_obj, title=f"Eval `{tag}`")
        st.download_button(
            "⬇️ report.md", md,
            file_name=f"eval_{tag}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "⬇️ metrics.json",
            json.dumps(report, ensure_ascii=False, indent=2),
            file_name=f"metrics_{tag}.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl3:
        per_csv = per_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ per_patient.csv", per_csv,
            file_name=f"per_patient_{tag}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _show_ab_compare(tag_a: str, data_a: Dict[str, Any],
                     tag_b: str, data_b: Dict[str, Any]) -> None:
    """A/B compare 2 eval runs side by side."""
    ra, rb = data_a["report"], data_b["report"]
    st.markdown(f"### 🔀 A/B compare: `{tag_a}` vs `{tag_b}`")

    # Config line
    st.caption(
        f"**A** — backend={data_a['backend']}, RAG={'on' if data_a['use_rag'] else 'off'}, "
        f"temp={data_a['temp']}, n={data_a['n_run']} | "
        f"**B** — backend={data_b['backend']}, RAG={'on' if data_b['use_rag'] else 'off'}, "
        f"temp={data_b['temp']}, n={data_b['n_run']}"
    )

    # Side-by-side metric comparison
    def _delta(va, vb, higher_better=True, pct=False):
        diff = vb - va
        if pct:
            return f"{diff*100:+.1f}pp"
        return f"{diff:+.3f}" if not isinstance(va, int) else f"{diff:+d}"

    def _winner(va, vb, higher_better=True):
        if abs(va - vb) < 1e-9:
            return "tie"
        wins = (vb > va) if higher_better else (vb < va)
        return tag_b if wins else tag_a

    rows = [
        ("Accuracy", ra["accuracy"], rb["accuracy"], True, True),
        ("Balanced acc", ra["balanced_accuracy"], rb["balanced_accuracy"], True, True),
        ("Macro F1", ra["macro_f1"], rb["macro_f1"], True, False),
        ("QWK (ordinal)", ra["qwk"], rb["qwk"], True, False),
        ("MAE ordinal", ra["mae_ordinal"], rb["mae_ordinal"], False, False),
        ("Under-triage ⚠️",
         ra["safety"]["under_triage_rate"],
         rb["safety"]["under_triage_rate"], False, True),
        ("Critical miss 🚨",
         ra["safety"]["critical_miss_rate"],
         rb["safety"]["critical_miss_rate"], False, True),
        ("Severe recall",
         ra["safety"]["severe_recall"],
         rb["safety"]["severe_recall"], True, True),
        ("ECE calibration",
         ra["calibration"].get("ece", 0),
         rb["calibration"].get("ece", 0), False, True),
        ("Citation rate",
         ra["quality"].get("citation_rate", 0),
         rb["quality"].get("citation_rate", 0), True, True),
        ("Lab ground rate",
         ra["quality"].get("lab_ground_rate", 0),
         rb["quality"].get("lab_ground_rate", 0), True, True),
        ("Latency p50 (ms)",
         ra["latency"].get("p50_ms", 0),
         rb["latency"].get("p50_ms", 0), False, False),
    ]
    cmp_df = pd.DataFrame([
        {
            "Metric": name,
            f"A ({tag_a})": f"{va*100:.1f}%" if pct else f"{va:.3f}",
            f"B ({tag_b})": f"{vb*100:.1f}%" if pct else f"{vb:.3f}",
            "Δ (B-A)": _delta(va, vb, hb, pct),
            "Winner": _winner(va, vb, hb),
        }
        for name, va, vb, hb, pct in rows
    ])

    def _color_winner(row):
        if row["Winner"] == tag_a:
            return ["background-color: #fff9c4"] * len(row)
        if row["Winner"] == tag_b:
            return ["background-color: #c8e6c9"] * len(row)
        return [""] * len(row)

    st.dataframe(cmp_df.style.apply(_color_winner, axis=1),
                 hide_index=True, use_container_width=True)
    st.caption(
        f"🟡 = `{tag_a}` thắng · 🟢 = `{tag_b}` thắng · "
        "**Hướng tốt**: với under-triage / critical-miss / MAE / ECE / latency, "
        "**thấp hơn là tốt**; còn lại cao hơn là tốt."
    )

    # Quick verdict
    wins_a = sum(1 for r in cmp_df["Winner"] if r == tag_a)
    wins_b = sum(1 for r in cmp_df["Winner"] if r == tag_b)
    ties = sum(1 for r in cmp_df["Winner"] if r == "tie")
    if wins_b > wins_a:
        st.success(f"🏆 **`{tag_b}` thắng tổng thể** — "
                   f"{wins_b} thắng / {wins_a} thua / {ties} hòa.")
    elif wins_a > wins_b:
        st.success(f"🏆 **`{tag_a}` thắng tổng thể** — "
                   f"{wins_a} thắng / {wins_b} thua / {ties} hòa.")
    else:
        st.info(f"🤝 Hòa: {wins_a}-{wins_b}-{ties}")
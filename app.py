"""
app.py — CliMed Dengue Assistant (v3 — auto-train, 3 tabs)
============================================================

Triết lý: Bệnh nhân-centric, không bắt user click train/split thủ công.

Khi app khởi động (lần đầu):
  1. Tự split data 80/20.
  2. Tự compute training statistics (lookup tables cho positioning).
  3. Tự train predictor (RandomForest).
  4. Tự build RAG index từ PDF guideline.
  5. Tự load knowledge graph.

Sau đó chỉ còn 3 tabs:
  • 💬 Chatbot — TRUNG TÂM: form nhập BN bên trái + chat bên phải.
        Khi user submit form, bot dùng STATS từ 80% train data để định vị
        BN (percentile per class, similar patients), chuyển toàn bộ làm
        context cho Qwen → trả lời CÓ DỮ LIỆU thay vì đoán mò.
  • 📷 OCR — upload ảnh xét nghiệm (autofill vào form bên Chatbot).
  • ⚙️ Settings — system status, retrain button, model info.

Tab cũ (Predictor, Graph viewer, Eval) bị BỎ — đã gói vào Settings hoặc
chạy qua CLI (`run_eval.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="CliMed Dengue Assistant",
    page_icon="🩺",
    layout="wide",
)

# ============================================================================
# CSS
# ============================================================================

st.markdown("""
<style>
.main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
.stChatMessage { background-color: transparent; }
.severity-badge {
    display: inline-block; padding: 6px 14px; border-radius: 10px;
    color: white; font-weight: 700; font-size: 1rem;
}
.evidence-card {
    background-color: #f0f4f8; border-left: 4px solid #1976d2;
    padding: 10px 14px; border-radius: 6px; margin: 8px 0;
}
.warning-banner {
    background-color: #fff3e0; border-left: 4px solid #f57c00;
    padding: 10px 14px; border-radius: 6px; margin: 8px 0;
}
.red-banner {
    background-color: #ffebee; border-left: 4px solid #c62828;
    padding: 10px 14px; border-radius: 6px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Bootstrap (auto-train on first launch)
# ============================================================================

@st.cache_resource(show_spinner=False)
def _bootstrap_once():
    """Chạy 1 lần cho mỗi process. Idempotent — skip nếu đã có artifacts."""
    from auto_bootstrap import bootstrap_all
    return bootstrap_all(force_retrain=False)


def _show_bootstrap_status(status, expander_label="🚀 Trạng thái khởi động"):
    """Render bootstrap status trong expander."""
    icon = "✅" if status.all_ok else "⚠️"
    with st.expander(f"{icon} {expander_label} ({status.total_elapsed_s:.1f}s)",
                     expanded=not status.all_ok):
        for s in status.steps:
            step_icon = "✅" if s.ok else ("⏭️" if s.skipped else "❌")
            st.write(f"{step_icon} **{s.name}**: {s.message} "
                     f"({s.elapsed_s:.1f}s)")
            if s.error:
                st.code(s.error[:300])


# ============================================================================
# Header
# ============================================================================

st.markdown(
    '<div style="text-align:center;padding:0.5rem 0">'
    '<h1 style="color:#1976d2;margin-bottom:0.2rem">'
    '🩺 CliMed Dengue Assistant</h1>'
    '<p style="color:#546e7a;font-style:italic;margin:0">'
    'Trợ lý phân loại SXHD tích hợp Qwen 2.5 + RAG + GraphRAG + Training Stats</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Bootstrap với progress
if "bootstrap_status" not in st.session_state:
    progress_box = st.empty()
    with st.spinner("🔧 Khởi động lần đầu: split data + compute stats + "
                    "train model + build RAG index... (~30-60s)"):
        try:
            status = _bootstrap_once()
            st.session_state["bootstrap_status"] = status
            progress_box.empty()
        except Exception as e:
            st.error(f"❌ Bootstrap failed: {e}")
            st.stop()

bootstrap_status = st.session_state["bootstrap_status"]


# ============================================================================
# Sidebar — chat sessions + LLM config
# ============================================================================

from chat_history import SessionManager


@st.cache_resource
def get_session_manager() -> SessionManager:
    return SessionManager(persist_dir=Path("data/chat_sessions"))


session_mgr = get_session_manager()


with st.sidebar:
    st.markdown("### 🩺 CliMed Dengue")
    st.caption(f"📊 Train: **{len(bootstrap_status.train_df or [])}** BN  ·  "
               f"Test: **{len(bootstrap_status.test_df or [])}** BN")

    st.markdown("---")
    # LLM config
    backend_choice = st.selectbox("🤖 LLM Backend",
                                   ["auto", "ollama", "hf"],
                                   help="auto: ưu tiên Ollama")
    chat_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    chat_use_rag = st.toggle("📚 Bật RAG", value=True)
    chat_use_graph = st.toggle("🕸️ Bật Graph", value=True)
    chat_use_stats = st.toggle("📊 Dùng training stats",
                                value=True,
                                help="Định vị BN vs 80% train data")

    st.markdown("---")
    st.markdown("**💬 Cuộc trò chuyện**")
    if st.button("➕ Cuộc mới", use_container_width=True):
        new_sid = session_mgr.create_session()
        st.session_state["active_session_id"] = new_sid
        st.session_state.pop("intake_patient_data", None)
        st.rerun()

    sessions = session_mgr.list_sessions(sort_by="updated")
    for s in sessions[:15]:
        sid = s["session_id"]
        is_active = (sid == st.session_state.get("active_session_id"))
        label = ("🟢 " if is_active else "💬 ") + s["title"][:30]
        if st.button(label, key=f"sess_{sid}", use_container_width=True):
            st.session_state["active_session_id"] = sid
            st.rerun()


# ============================================================================
# Tabs
# ============================================================================

(chat_tab, ocr_tab, settings_tab) = st.tabs([
    "💬 Chatbot",
    "📷 OCR",
    "⚙️ Settings",
])


# ============================================================================
# Tab 1 — Chatbot (form BN + chat)
# ============================================================================

with chat_tab:

    # ----- Init LLM -----
    @st.cache_resource(show_spinner="Khởi tạo LLM...")
    def _get_llm(_backend):
        from llm_qwen import get_llm_client
        return get_llm_client(backend=_backend)

    try:
        llm_client = _get_llm(backend_choice)
    except Exception as e:
        st.error(f"❌ LLM chưa sẵn sàng: {e}\n\n"
                 "Cài Ollama + pull model: `ollama pull qwen2.5:7b`")
        llm_client = None

    if llm_client is None:
        st.stop()

    # ----- 2 columns layout -----
    col_form, col_chat = st.columns([2, 3], gap="medium")

    # ===== LEFT: Patient intake form + positioning =====
    with col_form:
        st.markdown("### 📋 Thông tin bệnh nhân")
        from patient_intake import render_patient_intake_form, patient_to_text

        patient_data = render_patient_intake_form(key_prefix="intake")

        if patient_data:
            # Position vs training stats
            st.markdown("---")
            st.markdown("### 📊 Định vị BN")

            if not chat_use_stats or bootstrap_status.train_stats is None:
                st.info("Bật toggle 'Dùng training stats' trong sidebar để "
                        "xem định vị BN.")
            else:
                from training_stats import (
                    position_patient_against_stats,
                    find_similar_train_patients,
                    format_positioning_for_chat,
                    SEVERITY_VI,
                )
                positioning = position_patient_against_stats(
                    patient_data, bootstrap_status.train_stats,
                )
                similar = find_similar_train_patients(
                    patient_data, bootstrap_status.train_df, k=5,
                )

                # Best match badge
                best = positioning.get("best_match_class")
                if best:
                    color_map = {"thuong": "#4caf50", "canhbao": "#ff9800",
                                  "nang": "#e53935"}
                    emoji_map = {"thuong": "🟢", "canhbao": "🟡", "nang": "🔴"}
                    score = positioning["score_per_class"][best]
                    st.markdown(
                        f'<div style="text-align:center">'
                        f'<span class="severity-badge" '
                        f'style="background:{color_map[best]};font-size:1.2rem">'
                        f'{emoji_map[best]} Match nhất: {SEVERITY_VI[best]}'
                        f'</span><br><small>score={score:.2f}</small></div>',
                        unsafe_allow_html=True,
                    )

                # 3 scores side by side
                sc_cols = st.columns(3)
                for i, cls in enumerate(["thuong", "canhbao", "nang"]):
                    sc_cols[i].metric(
                        f"{emoji_map.get(cls)} {cls}",
                        f"{positioning['score_per_class'].get(cls, 0):.2f}",
                    )

                # Similar patients
                if similar:
                    st.markdown("**5 BN gần nhất trong train set:**")
                    sim_df = pd.DataFrame([{
                        "BN#": s["patient_idx"],
                        "Nhãn": f"{emoji_map.get(s['label'], '?')} {s['label']}",
                        "Tóm tắt": s["summary"],
                        "Distance": s["distance"],
                    } for s in similar])
                    st.dataframe(sim_df, hide_index=True,
                                 use_container_width=True)
                    # Distribution
                    label_counts = {}
                    for s in similar:
                        label_counts[s["label"]] = label_counts.get(
                            s["label"], 0) + 1
                    st.caption(
                        "Phân phối: " +
                        " · ".join(
                            f"{emoji_map.get(c, '?')} `{c}`: {n}"
                            for c, n in label_counts.items()
                        )
                    )

                with st.expander("📝 Chi tiết vị trí lab"):
                    st.markdown(positioning.get("narrative", "(empty)"))

                # Save to session for chat
                st.session_state["current_positioning"] = positioning
                st.session_state["current_similar"] = similar
                st.session_state["current_positioning_text"] = (
                    format_positioning_for_chat(positioning, similar)
                )

    # ===== RIGHT: Chat =====
    with col_chat:
        st.markdown("### 💬 Trò chuyện")

        # Active session
        if st.session_state.get("active_session_id") is None:
            st.session_state["active_session_id"] = session_mgr.create_session()
        active_sid = st.session_state["active_session_id"]
        active_session = session_mgr.load_session(active_sid)

        # Chat history scrollable
        chat_container = st.container(height=520)
        with chat_container:
            if active_session:
                for msg in active_session.messages:
                    with st.chat_message(msg.role):
                        st.markdown(msg.content)
            if not (active_session and active_session.messages):
                st.info(
                    "👈 Nhập thông tin BN bên trái và bấm "
                    "**'Phân tích BN này'** trước. Sau đó hỏi tôi về BN: "
                    '"BN này có thuộc nhóm nặng không?", '
                    '"Cần làm gì tiếp theo?", '
                    '"Tại sao tiểu cầu giảm như thế này nguy hiểm?"'
                )

        # Chat input
        user_input = st.chat_input("Hỏi về BN, phác đồ, hoặc dấu hiệu...")
        if user_input:
            # Save user message
            session_mgr.append_message(active_sid, "user", user_input)

            # Build context
            from chatbot_engine import chat as run_chat
            patient_data = st.session_state.get("intake_patient_data")
            patient_text = patient_to_text(patient_data) if patient_data else ""

            # Combine context: patient form + positioning + similar
            ctx_parts = []
            if patient_text:
                ctx_parts.append("## NGỮ CẢNH BỆNH NHÂN\n" + patient_text)
            if (chat_use_stats and
                    st.session_state.get("current_positioning_text")):
                ctx_parts.append(st.session_state["current_positioning_text"])
            ctx_block = "\n\n".join(ctx_parts)

            history_for_llm = [
                {"role": m.role, "content": m.content}
                for m in active_session.messages[:-1]
            ] if active_session else []

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    response_gen = run_chat(
                        client=llm_client,
                        user_question=user_input,
                        history=history_for_llm,
                        context=ctx_block,
                        stream=True,
                        temperature=chat_temp,
                        max_tokens=700,
                        use_rag=chat_use_rag,
                        rag_top_k=3,
                        use_graph=chat_use_graph,
                        graph_symptoms=patient_data.get("symptoms") if patient_data else None,
                    )
                    full_response = st.write_stream(response_gen)

            session_mgr.append_message(active_sid, "assistant", full_response)
            st.rerun()


# ============================================================================
# Tab 2 — OCR
# ============================================================================

with ocr_tab:
    st.markdown("### 📷 Upload ảnh xét nghiệm")
    st.caption("Hệ thống tự OCR → parse → bạn copy giá trị sang form BN ở "
               "tab Chatbot.")

    from ocr_engine import extract_text_from_uploaded_file
    from exam_normalizer import parse_exam_text_to_rows, attach_reference_flags

    uploaded = st.file_uploader(
        "PNG / JPG / PDF (chọn nhiều file)",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded:
        all_rows = []
        for idx, f in enumerate(uploaded, 1):
            with st.spinner(f"OCR {f.name}..."):
                res = extract_text_from_uploaded_file(f)
            st.markdown(f"#### 📄 {f.name}")
            cols = st.columns([1, 1, 2])
            cols[0].metric("Method", res.method)
            cols[1].metric("Confidence", f"{res.confidence:.2f}")

            parsed = parse_exam_text_to_rows(res.text)
            if not parsed.empty:
                parsed = attach_reference_flags(parsed)
                parsed["file"] = f.name
                all_rows.append(parsed)
                st.dataframe(parsed, hide_index=True,
                              use_container_width=True)
            else:
                st.info("Không parse được giá trị nào từ ảnh này.")

            with st.expander("Raw OCR text"):
                st.code(res.text[:2000])

        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)
            st.markdown("### 📋 Tổng hợp")
            st.dataframe(combined, hide_index=True, use_container_width=True)
            st.session_state["ocr_records"] = combined


# ============================================================================
# Tab 3 — Settings (status + retrain + tools)
# ============================================================================

with settings_tab:
    st.markdown("### ⚙️ System Status")

    _show_bootstrap_status(bootstrap_status, "Trạng thái khởi động")

    st.markdown("---")
    st.markdown("### 📦 Artifacts")
    arts = {
        "Train CSV": "data/splits/train_80.csv",
        "Test CSV": "data/splits/test_20.csv",
        "Training stats": ".cache/train_stats.json",
        "Predictor model": ".cache/predictor.joblib",
        "Knowledge graph": "data/knowledge/graph_seed.yaml",
        "Guideline PDF": "data/knowledge/Huong-dan-chan-doan-va-dieu-tri(1).pdf",
    }
    art_data = []
    for name, p in arts.items():
        path = Path(p)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            art_data.append({"File": name, "Path": p,
                             "Status": "✅ OK",
                             "Size": f"{size_kb:.1f} KB"})
        else:
            art_data.append({"File": name, "Path": p,
                             "Status": "❌ Missing",
                             "Size": "—"})
    st.dataframe(pd.DataFrame(art_data),
                 hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔄 Maintenance")
    mt1, mt2 = st.columns(2)
    with mt1:
        if st.button("♻️ Force retrain everything",
                      use_container_width=True,
                      help="Xóa cache, split lại, train lại, build RAG lại"):
            st.cache_resource.clear()
            from auto_bootstrap import bootstrap_all
            with st.spinner("Đang retrain toàn bộ..."):
                new_status = bootstrap_all(force_retrain=True)
            st.session_state["bootstrap_status"] = new_status
            st.success("✅ Done. F5 để reload.")
            st.rerun()
    with mt2:
        if st.button("🧹 Clear chat sessions",
                      use_container_width=True):
            for sid in [s["session_id"] for s in session_mgr.list_sessions()]:
                session_mgr.delete_session(sid)
            st.session_state.pop("active_session_id", None)
            st.success("✅ Cleared.")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Predictor metrics (đã train)")
    if bootstrap_status.predictor:
        m = bootstrap_status.predictor.get("metrics", {})
        if m:
            mc = st.columns(3)
            mc[0].metric("F1 macro", f"{m.get('f1_macro', 0):.3f}")
            mc[1].metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
            if m.get("roc_auc") is not None:
                mc[2].metric("ROC AUC", f"{m['roc_auc']:.3f}")
    else:
        st.info("Predictor chưa load được. Bấm 'Force retrain' để rebuild.")

    st.markdown("---")
    st.markdown("### 🛠 Advanced — CLI tools")
    st.code("""
# Eval đầy đủ (25+ metrics) qua CLI
python src/run_eval.py --tag full_opt

# So 2 cấu hình
python src/run_eval.py --no-rag --tag baseline
python src/run_eval.py --tag full_opt
diff eval_reports/baseline/metrics.json eval_reports/full_opt/metrics.json

# Triage 1 BN qua CLI
python src/mini_ai_bot.py "$(cat patient.txt)"

# Test RAG optimizer
python src/rag_optimizer.py "tiểu cầu giảm có nguy hiểm không"
""", language="bash")

"""
patient_intake.py — CliMedV2
=============================

Streamlit form component nhập dữ liệu BN — dùng trong tab Chatbot.

Layout:
  📋 Thông tin BN
  ├── Demographics (tuổi, giới, cân nặng, ngày bệnh khi nhập)
  ├── Triệu chứng (checkboxes các NV* symptoms)
  └── Lab values (input số PLT/Hct/WBC/HFLC theo từng ngày N1-N7)

Output: dict chuẩn cho `training_stats.position_patient_against_stats()`
        và `training_stats.find_similar_train_patients()`.

Cách dùng:
    from patient_intake import render_patient_intake_form
    patient_data = render_patient_intake_form(key_prefix="main")
    if patient_data:
        # ... position vs stats, find similar, build context
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)


# ===========================================================================
# Symptom labels (Vietnamese)
# ===========================================================================

SYMPTOMS_VI_TO_COL = {
    "Nôn ói": "NVnonoi",
    "Tiêu chảy": "NVtieuchay",
    "Đau bụng": "NVdaubung",
    "Đau cơ/khớp": "NVdauco",
    "Đau đầu": "NVdaudau",
    "Đau sau hốc mắt": "NVdausauhocmat",
    "Ho": "NVho",
    "Xuất huyết (chảy máu)": "NVxuathuyet",
    "Gan to (>2cm)": "NVganto",
}

# Dấu hiệu cảnh báo / nặng — extra checkboxes
WARNING_SIGNS = {
    "Đau bụng nhiều/liên tục": "warning_severe_pain",
    "Nôn nhiều/liên tục": "warning_severe_vomit",
    "Lừ đừ/vật vã/li bì": "warning_lethargy",
    "Tay chân lạnh": "warning_cold_extremities",
    "Tiểu ít": "warning_oliguria",
    "Chảy máu niêm mạc": "warning_mucosal_bleed",
}

RED_FLAGS = {
    "🚨 Sốc (mạch nhanh + HA tụt)": "red_shock",
    "🚨 Xuất huyết tiêu hóa": "red_gi_bleed",
    "🚨 Khó thở/suy hô hấp": "red_dyspnea",
    "🚨 Rối loạn ý thức": "red_altered_mental",
}


# ===========================================================================
# Form component
# ===========================================================================

def render_patient_intake_form(key_prefix: str = "intake",
                                compact: bool = False) -> Optional[Dict[str, Any]]:
    """
    Render form. Trả về dict BN nếu user đã bấm nút "Phân tích", ngược lại None.

    State được lưu trong st.session_state[f"{key_prefix}_patient_data"].

    Args:
        key_prefix: prefix cho widget keys (tránh conflict nếu form nhiều nơi).
        compact: True = layout compact 1 cột, False = layout đầy đủ.
    """
    state_key = f"{key_prefix}_patient_data"

    # ===== Demographics =====
    with st.expander("👤 Thông tin BN", expanded=True):
        d1, d2, d3 = st.columns(3)
        with d1:
            age = st.number_input("Tuổi", 0, 120, 25,
                                   key=f"{key_prefix}_age")
        with d2:
            sex = st.radio("Giới", ["nam", "nữ"], horizontal=True,
                            key=f"{key_prefix}_sex")
        with d3:
            weight = st.number_input("Cân nặng (kg)", 5.0, 200.0, 60.0,
                                      step=0.5, key=f"{key_prefix}_weight")

        d4, d5 = st.columns(2)
        with d4:
            fever_day = st.number_input(
                "Ngày bệnh khi nhập viện (N1=ngày đầu sốt)",
                1, 14, 4,
                help="Ngày thứ mấy của bệnh khi đến BV. "
                     "Vd: sốt 4 ngày rồi đến → 4",
                key=f"{key_prefix}_fever_day",
            )
        with d5:
            comorbidity = st.text_input(
                "Bệnh nền (nếu có)",
                placeholder="vd: tăng huyết áp, đái tháo đường…",
                key=f"{key_prefix}_comorbid",
            )

    # ===== Symptoms =====
    with st.expander("🤒 Triệu chứng (tích vào nếu có)", expanded=True):
        sym1, sym2, sym3 = st.columns(3)
        symptoms_dict: Dict[str, int] = {}
        cols = [sym1, sym2, sym3]
        sym_items = list(SYMPTOMS_VI_TO_COL.items())
        for i, (vi, col_name) in enumerate(sym_items):
            with cols[i % 3]:
                v = st.checkbox(vi, key=f"{key_prefix}_sym_{col_name}")
                if v:
                    symptoms_dict[col_name] = 1

        st.markdown("**⚠️ Dấu hiệu cảnh báo:**")
        wcols = st.columns(3)
        warnings_active = []
        for i, (vi, key) in enumerate(WARNING_SIGNS.items()):
            with wcols[i % 3]:
                v = st.checkbox(vi, key=f"{key_prefix}_w_{key}")
                if v:
                    warnings_active.append(vi)

        st.markdown("**🚨 Dấu hiệu nặng:**")
        rcols = st.columns(2)
        red_flags_active = []
        for i, (vi, key) in enumerate(RED_FLAGS.items()):
            with rcols[i % 2]:
                v = st.checkbox(vi, key=f"{key_prefix}_r_{key}")
                if v:
                    red_flags_active.append(vi)

    # ===== Lab values =====
    with st.expander("🧪 Lab values theo ngày bệnh", expanded=True):
        st.caption("Nhập giá trị nếu có XN ngày đó. Bỏ trống = không có XN.")
        n_days = 7
        labs_data: Dict[str, Dict[int, float]] = {}

        # Tabs per lab type cho gọn
        tab_plt, tab_hct, tab_wbc, tab_other = st.tabs(
            ["🩸 Tiểu cầu (K/uL)", "📊 Hct (%)", "⚪ Bạch cầu (K/uL)",
             "🔬 HFLC + %HFLC"])

        with tab_plt:
            labs_data["tieucau"] = _render_lab_row(
                key_prefix, "plt", n_days, "K/uL",
                placeholder="vd: 150")
        with tab_hct:
            labs_data["Hct"] = _render_lab_row(
                key_prefix, "hct", n_days, "%",
                placeholder="vd: 40")
        with tab_wbc:
            labs_data["bachcau"] = _render_lab_row(
                key_prefix, "wbc", n_days, "K/uL",
                placeholder="vd: 5.2")
        with tab_other:
            st.markdown("**HFLC (K/uL):**")
            labs_data["HFLC"] = _render_lab_row(
                key_prefix, "hflc", n_days, "K/uL",
                placeholder="0.1")
            st.markdown("**%HFLC:**")
            labs_data["phantramHFLC"] = _render_lab_row(
                key_prefix, "phflc", n_days, "%",
                placeholder="2.5")

    # ===== Submit =====
    st.markdown("---")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        submit = st.button("🔍 Phân tích BN này",
                            type="primary", use_container_width=True,
                            key=f"{key_prefix}_submit")
    with c2:
        clear = st.button("🗑️ Xóa form", use_container_width=True,
                          key=f"{key_prefix}_clear")
    with c3:
        if state_key in st.session_state:
            st.caption("✅ Đã có data")

    if clear:
        # Reset session keys with key_prefix
        for k in list(st.session_state.keys()):
            if k.startswith(f"{key_prefix}_"):
                del st.session_state[k]
        st.session_state.pop(state_key, None)
        st.rerun()

    if submit:
        patient_data = {
            "age": age,
            "sex": sex,
            "weight": weight,
            "fever_day_at_admission": int(fever_day),
            "comorbidity": comorbidity.strip() if comorbidity else None,
            "symptoms": symptoms_dict,
            "warnings": warnings_active,
            "red_flags": red_flags_active,
            "labs": labs_data,
        }
        st.session_state[state_key] = patient_data
        return patient_data

    return st.session_state.get(state_key)


def _render_lab_row(key_prefix: str, lab_short: str,
                    n_days: int, unit: str,
                    placeholder: str = "") -> Dict[int, float]:
    """Render 1 hàng input số cho lab × N1..Nn. Trả về {day: value} (skip None)."""
    cols = st.columns(n_days)
    out: Dict[int, float] = {}
    for i, c in enumerate(cols, start=1):
        with c:
            v = st.text_input(
                f"N{i}",
                key=f"{key_prefix}_{lab_short}_n{i}",
                placeholder=placeholder if i == 1 else "",
                label_visibility="visible",
            )
            v = v.strip().replace(",", ".") if v else ""
            if v:
                try:
                    out[i] = float(v)
                except ValueError:
                    pass   # bỏ qua input không hợp lệ
    return out


# ===========================================================================
# Format patient → text cho RAG/LLM
# ===========================================================================

def patient_to_text(patient: Dict[str, Any]) -> str:
    """Format BN dict thành text dễ đọc cho LLM, giống render_patient_card."""
    if not patient:
        return ""
    lines = ["--- Bệnh nhân ---"]
    lines.append(f"Thông tin: {patient.get('sex', '?')}, "
                 f"{patient.get('age', '?')} tuổi, "
                 f"{patient.get('weight', '?')}kg")
    lines.append(f"Ngày bệnh khi nhập viện: N{patient.get('fever_day_at_admission', '?')}")

    if patient.get("comorbidity"):
        lines.append(f"Bệnh nền: {patient['comorbidity']}")

    sym_pos = []
    col_to_vi = {v: k for k, v in SYMPTOMS_VI_TO_COL.items()}
    for col, val in (patient.get("symptoms") or {}).items():
        if val:
            sym_pos.append(col_to_vi.get(col, col))
    if sym_pos:
        lines.append(f"Triệu chứng (+): {', '.join(sym_pos)}")

    if patient.get("warnings"):
        lines.append(f"⚠️ Dấu hiệu cảnh báo: {', '.join(patient['warnings'])}")
    if patient.get("red_flags"):
        lines.append(f"🚨 Dấu hiệu nặng: {', '.join(patient['red_flags'])}")

    labs = patient.get("labs", {}) or {}
    lab_label_map = {
        "tieucau": ("Tiểu cầu (K/uL)", "K/uL"),
        "Hct":     ("Hct (%)",          "%"),
        "bachcau": ("Bạch cầu (K/uL)",  "K/uL"),
        "HFLC":    ("HFLC (K/uL)",      "K/uL"),
        "phantramHFLC": ("%HFLC",        "%"),
    }
    has_any = any(v for v in labs.values())
    if has_any:
        lines.append("Lab theo ngày:")
        for lab_key, (lbl, _) in lab_label_map.items():
            day_vals = labs.get(lab_key, {})
            if not day_vals:
                continue
            parts = [f"N{d}={v}" for d, v in sorted(day_vals.items())]
            lines.append(f"  {lbl}: " + ", ".join(parts))

    return "\n".join(lines)
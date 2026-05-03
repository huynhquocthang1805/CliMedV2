# CliMed Refactor - Dengue OCR Assistant

Bản refactor này đi tiếp từ repository `CliMed` hiện có.

## Mục tiêu mới
- OCR ảnh/PDF xét nghiệm
- Parse và chuẩn hóa chỉ số xét nghiệm Dengue
- Rule engine SXHD có guardrail
- Trend + missingness minh bạch
- RAG-lite / Graph reasoning để giải thích
- Clinician mode chỉ hỗ trợ tham khảo, không kê đơn tự động
- Giữ lại tab **Legacy cohort analytics** từ app cũ để tái sử dụng module đã có

## Mapping cũ -> mới
- `src/_common.py`: giữ vai trò normalization + cohort analytics, mở rộng thêm alias xét nghiệm và utility cho OCR
- `src/qa_engine.py`: giữ lại logic legacy cohort QA
- `app.py`: chuyển thành app đa tab, thêm OCR, rule engine, trend, clinician mode, explanation
- Mới thêm:
  - `src/ocr_engine.py`
  - `src/exam_normalizer.py`
  - `src/dengue_rules.py`
  - `src/trend_engine.py`
  - `src/missingness_engine.py`
  - `src/clinician_suggestions.py`
  - `src/knowledge_base.py`

## Chạy
```bash
pip install -r requirements.txt
python src/train_intent_model.py  # tùy chọn cho tab legacy
streamlit run app.py
```

## Guardrail
Ứng dụng chỉ hỗ trợ tham khảo và tổng hợp thông tin lâm sàng, không thay thế bác sĩ và không phải hệ thống kê đơn tự động.


## Bổ sung kho tri thức PDF trong data/knowledge
Project này đã được cập nhật để tự nạp các file hướng dẫn PDF/TXT trong thư mục `data/knowledge/`.
Khi bạn upload ảnh hoặc PDF xét nghiệm, tab **Giải thích RAG/Graph** sẽ dùng:
- dữ kiện OCR vừa trích xuất
- các file hướng dẫn trong `data/knowledge/`
để trả lời ngắn gọn và có citation theo tên file/trang.

Các file đã thêm mặc định:
- `Huong-dan-chan-doan-va-dieu-tri(1).pdf`
- `data missing(1).pdf`
- `Pasted text(4).txt`

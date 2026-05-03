
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
SRC_DIR = Path(__file__).resolve().parent / 'src'
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))
from _common import CRITICAL_SUPPORT_TEXT
from clinician_suggestions import generate_clinician_suggestions
from dengue_rules import evaluate_dengue_rules
from exam_normalizer import attach_reference_flags, parse_exam_text_to_rows
from knowledge_base import answer_question_with_retrieval, build_reasoning_graph, list_loaded_knowledge_sources
from missingness_engine import annotate_missingness
from ocr_engine import extract_text_from_uploaded_file
from qa_engine import HFLCQAEngine
from trend_engine import compute_trend_flags, prepare_trend_table
st.set_page_config(page_title='CliMed Dengue Assistant', page_icon='🩺', layout='wide')
@st.cache_resource
def load_legacy_engine():
    try: return HFLCQAEngine()
    except Exception: return None
engine = load_legacy_engine()
st.title('🩺 CliMed Dengue Assistant')
st.warning(CRITICAL_SUPPORT_TEXT)
# st.markdown('Ứng dụng này được refactor từ repo **CliMed** hiện có, vốn đang có cấu trúc `data/`, `models/`, `src/`, file gốc `app.py` và README; README hiện mô tả đây là bản UI v5 với query router và cohort filters. citeturn983907view0turn811707view0')
with st.sidebar:
    st.header('Thiết lập')
    clinician_mode = st.toggle('Chế độ bác sĩ', value=False)
    st.caption('Clinician mode chỉ hiển thị gợi ý tham khảo có guardrail, không kê đơn tự động.')
    uploaded_files = st.file_uploader('Tải ảnh/PDF xét nghiệm', type=['png','jpg','jpeg','pdf'], accept_multiple_files=True)
    knowledge_files = list_loaded_knowledge_sources()
    st.caption('Kho tri thức đang nạp từ data/knowledge: ' + (', '.join(knowledge_files) if knowledge_files else 'chưa có file'))
if 'ocr_records' not in st.session_state: st.session_state['ocr_records']=[]
if 'ocr_texts' not in st.session_state: st.session_state['ocr_texts']=[]
if uploaded_files:
    all_rows=[]; all_texts=[]
    for idx, f in enumerate(uploaded_files, start=1):
        ocr_result=extract_text_from_uploaded_file(f)
        all_texts.append({'file_name':f.name,'method':ocr_result.method,'confidence':ocr_result.confidence,'text':ocr_result.text})
        parsed=parse_exam_text_to_rows(ocr_result.text)
        if not parsed.empty:
            parsed['report_id']=idx; parsed['report_label']=f.name; parsed['report_time']=pd.Timestamp.now().normalize()+pd.Timedelta(days=idx-1); all_rows.append(parsed)
    if all_rows: st.session_state['ocr_records']=pd.concat(all_rows, ignore_index=True)
    st.session_state['ocr_texts']=all_texts
clinical_tab, trend_tab, explain_tab, clinician_tab, legacy_tab = st.tabs(['1. OCR & Chuẩn hóa','2. Rule engine & Trend','3. Giải thích RAG/Graph','4. Clinician mode','5. Legacy cohort'])
with clinical_tab:
    st.subheader('OCR và bảng chuẩn hóa xét nghiệm')
    if st.session_state['ocr_texts']:
        with st.expander('Văn bản OCR thô'):
            for item in st.session_state['ocr_texts']:
                st.markdown(f"**{item['file_name']}** — method: `{item['method']}` — confidence: `{item['confidence']}`")
                st.text(item['text'][:4000] if item['text'] else 'Không đọc được văn bản, vui lòng nhập tay.')
                st.markdown('---')
    ocr_df = st.session_state['ocr_records'] if isinstance(st.session_state['ocr_records'], pd.DataFrame) else pd.DataFrame()
    if ocr_df.empty:
        st.info('Chưa có dữ liệu OCR. Bạn có thể tải ảnh/PDF ở sidebar rồi chỉnh bảng phía dưới.')
        ocr_df = pd.DataFrame(columns=['exam_name_raw','exam_name_normalized','value','unit','reference_range','source','confidence','report_id','report_label','report_time'])
    edited_df = st.data_editor(attach_reference_flags(ocr_df), use_container_width=True, num_rows='dynamic')
    st.session_state['ocr_records']=edited_df
    st.subheader('Nhập lâm sàng')
    c1,c2,c3=st.columns(3)
    with c1:
        age=st.number_input('Tuổi',0,120,16); sex=st.selectbox('Giới',['nữ','nam']); fever_days=st.number_input('Số ngày sốt / ngày bệnh',0,20,6); fever_now=st.checkbox('Đang sốt',False); can_drink=st.checkbox('Uống được',True)
    with c2:
        abdominal_pain=st.checkbox('Đau bụng'); vomiting=st.checkbox('Nôn'); vomiting_many=st.checkbox('Nôn nhiều / liên tục'); mucosal_bleeding=st.checkbox('Chảy máu niêm mạc'); oliguria=st.checkbox('Tiểu ít')
    with c3:
        lethargy=st.checkbox('Lừ đừ / vật vã / rối loạn ý thức'); dyspnea=st.checkbox('Khó thở'); cold_extremities=st.checkbox('Tay chân lạnh'); myalgia=st.checkbox('Đau cơ / đau khớp'); retro_orbital_pain=st.checkbox('Đau hốc mắt')
    severe_bleeding=st.checkbox('Xuất huyết nặng'); hypotension=st.checkbox('Tụt huyết áp / nghi sốc'); comorbidity_text=st.text_input('Bệnh nền / thông tin thêm')
    st.session_state['clinical_input']={'age':age,'sex':sex,'fever_days':fever_days,'fever_now':fever_now,'can_drink':can_drink,'abdominal_pain':abdominal_pain,'vomiting':vomiting,'vomiting_many':vomiting_many,'mucosal_bleeding':mucosal_bleeding,'oliguria':oliguria,'lethargy':lethargy,'dyspnea':dyspnea,'cold_extremities':cold_extremities,'myalgia':myalgia,'retro_orbital_pain':retro_orbital_pain,'severe_bleeding':severe_bleeding,'hypotension':hypotension,'comorbidity_text':comorbidity_text}
with trend_tab:
    st.subheader('Rule engine, trend và missingness')
    records_df = st.session_state['ocr_records'] if isinstance(st.session_state['ocr_records'], pd.DataFrame) else pd.DataFrame()
    trend_df=prepare_trend_table(records_df); trend_flags=compute_trend_flags(trend_df); rules=evaluate_dengue_rules(records_df, st.session_state.get('clinical_input', {}), trend_flags); st.session_state['rule_output']=rules; st.session_state['trend_flags']=trend_flags
    st.metric('Phân nhóm tham khảo', rules.level); st.write(rules.summary)
    if rules.evidence: st.markdown('**Bằng chứng hỗ trợ**'); [st.write('- ' + x) for x in rules.evidence]
    if rules.warnings: st.markdown('**Dấu hiệu cảnh báo**'); [st.write('- ' + x) for x in rules.warnings]
    if rules.red_flags: st.markdown('**Dấu hiệu đỏ / cần xử trí khẩn**'); [st.error(x) for x in rules.red_flags]
    st.markdown('**Trend flags**'); st.json(trend_flags)
    ann=annotate_missingness(records_df)
    if not ann.empty:
        st.dataframe(ann[['report_label','exam_name_normalized','value','observed_status','feature_mask','time_delta_hours']], use_container_width=True)
        selected_exam=st.selectbox('Chọn xét nghiệm để xem xu hướng', sorted(trend_df['exam_name_normalized'].unique().tolist()) if not trend_df.empty else ['PLT'])
        temp=trend_df[trend_df['exam_name_normalized']==selected_exam][['report_time','numeric_value']]
        if not temp.empty: st.line_chart(temp.set_index('report_time'))
with explain_tab:
    st.subheader('Giải thích theo tri thức nội bộ (RAG-lite + Graph reasoning)')
    rules=st.session_state.get('rule_output'); trend_flags=st.session_state.get('trend_flags', {}); context_facts=[]
    if rules: context_facts.extend(rules.evidence + rules.warnings + rules.red_flags)
    user_q=st.text_input('Câu hỏi giải thích', value='Vì sao kết quả này gợi ý sốt xuất huyết và cần lưu ý gì?')
    ocr_context = '\n'.join([str(x.get('text',''))[:1500] for x in st.session_state.get('ocr_texts', [])])
    retrieval=answer_question_with_retrieval(user_q, context_facts=context_facts, extra_context=ocr_context)
    st.write(retrieval['answer'])
    if retrieval['citations']:
        st.markdown('**Nguồn**'); [st.write('- ' + c) for c in retrieval['citations']]
        with st.expander('Đoạn tri thức truy xuất được'):
            for item in retrieval.get('retrieved', []):
                st.markdown(f"**{item.get('citation','Nguồn')}**")
                st.write(item.get('text','')[:1000])
                st.markdown('---')
        st.caption('Các guardrail này bám theo yêu cầu app: OCR ảnh/PDF xét nghiệm, parse và chuẩn hóa chỉ số, rule engine SXHD, RAG + GraphRAG để giải thích, xử lý missing data minh bạch và clinician suggestions chỉ ở mức clinical decision support. fileciteturn11file8turn11file12turn11file14')
        st.caption('Về mặt y khoa, guideline bệnh viện chia SXHD thành 3 mức: SXHD, SXHD có dấu hiệu cảnh báo, SXHD nặng; phần lớn ca điều trị ngoại trú chủ yếu là điều trị triệu chứng và theo dõi chặt; paracetamol là thuốc hạ nhiệt được dùng, còn aspirin/analgin/ibuprofen cần tránh. fileciteturn11file6turn11file3turn11file1')
        st.caption('Phần missingness dùng mask và time delta, phù hợp với tài liệu bạn tải lên về irregular time series / informative missingness. fileciteturn11file10turn11file9')
    if rules:
        graph=build_reasoning_graph(rules.level, {'plt_low': any('Tiểu cầu giảm' in x for x in rules.evidence), 'wbc_low': any('Bạch cầu' in x for x in rules.evidence), 'hct_up_plt_down': trend_flags.get('hct_up_plt_down', False)})
        st.dataframe(pd.DataFrame([{'source':u,'target':v,'relation':d.get('relation')} for u,v,d in graph.edges(data=True)]), use_container_width=True)
with clinician_tab:
    st.subheader('Clinician-only suggestions')
    if not clinician_mode: st.info('Bật “Chế độ bác sĩ” ở sidebar để xem gợi ý clinician-only.')
    else:
        rules=st.session_state.get('rule_output')
        if rules is None: st.info('Chưa có kết quả rule engine.')
        else:
            suggestions=generate_clinician_suggestions(rules, st.session_state.get('clinical_input', {}))
            for section_key, section_title in [('supportive','1) Gợi ý hỗ trợ triệu chứng'),('monitoring','2) Gợi ý cần bác sĩ cân nhắc tại cơ sở y tế'),('escalation','3) Gợi ý escalation / hồi sức'),('contraindication','4) Chống chỉ định / cần tránh')]:
                st.markdown(f'### {section_title}')
                items=suggestions.get(section_key, [])
                if not items: st.write('Chưa có gợi ý phù hợp.')
                for item in items:
                    with st.container(border=True):
                        st.markdown(f"**{item['name']}**"); st.write('Lý do:', item['reason']); st.write('Điều kiện cần xác nhận:', item['confirm_before_use']); st.write('Citation:', item['citation']); st.write('Confidence:', item['confidence'])
with legacy_tab:
    st.subheader('Legacy cohort analytics từ repo gốc')
    if engine is None: st.info('Chưa nạp được legacy cohort engine. Bạn có thể chạy `python src/train_intent_model.py` để tạo model intent cho tab này.')
    else:
        q=st.text_input('Hỏi trên cohort legacy', value='ở nhóm sốc, triệu chứng nào hay gặp')
        if q: st.write(engine.answer(q)); st.json(engine.debug_parse(q))
        st.dataframe(engine.text_table.head(50), use_container_width=True)

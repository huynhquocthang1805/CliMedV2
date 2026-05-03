
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / 'data' / 'NghiencuuHFLC.csv'
MODEL_DIR = BASE / 'models'

MEASURE_ALIASES = {
    'bạch cầu': 'bachcau', 'bach cau': 'bachcau', 'bachcau': 'bachcau',
    'tiểu cầu': 'tieucau', 'tieu cau': 'tieucau', 'tieucau': 'tieucau',
    'hflc': 'HFLC', '%hflc': 'phantramHFLC', 'phần trăm hflc': 'phantramHFLC', 'phan tram hflc': 'phantramHFLC',
    'hct': 'Hct', 'hematocrit': 'Hct',
    'độ tuổi': 'Tuoi', 'do tuoi': 'Tuoi', 'tuổi': 'Tuoi', 'tuoi': 'Tuoi',
}
SYMPTOM_ALIASES = {
    'nôn ói': 'NVnonoi', 'non oi': 'NVnonoi', 'nonoi': 'NVnonoi',
    'tiêu chảy': 'NVtieuchay', 'tieu chay': 'NVtieuchay', 'tieuchay': 'NVtieuchay',
    'đau bụng': 'NVdaubung', 'dau bung': 'NVdaubung', 'daubung': 'NVdaubung',
    'đau cơ': 'NVdauco', 'dau co': 'NVdauco', 'dauco': 'NVdauco',
    'đau đầu': 'NVdaudau', 'dau dau': 'NVdaudau', 'daudau': 'NVdaudau',
    'đau sau hốc mắt': 'NVdausauhocmat', 'dau sau hoc mat': 'NVdausauhocmat', 'dausauhocmat': 'NVdausauhocmat',
    'ho': 'NVho',
    'xuất huyết': 'NVxuathuyet', 'xuat huyet': 'NVxuathuyet',
    'gan to': 'NVganto', 'gan lớn': 'NVganto',
    'petechia': 'Petechia', 'ban xuat huyet': 'Petechia',
}
CANONICAL_MAP = {
    'petechiae': 'petechia', 'petechia': 'petechia',
    'amdao': 'am dao', 'xhamdao': 'am dao', 'am dao': 'am dao',
    'chaymaumui': 'chay mau mui', 'chay mau mui': 'chay mau mui', 'mui': 'chay mau mui',
    'bammauvetchich': 'vet chich', 'vetchich': 'vet chich', 'vet chich': 'vet chich',
    'chan rang': 'chay mau rang', 'chanrang': 'chay mau rang', 'chaymaurang': 'chay mau rang', 'chay mau rang': 'chay mau rang',
    'tha': 'tang huyet ap', 'tăng huyết áp': 'tang huyet ap', 'tang huyet ap': 'tang huyet ap', 'tanghuyetap': 'tang huyet ap',
    'dtd': 'dai thao duong', 'đái tháo đường': 'dai thao duong', 'dai thao duong': 'dai thao duong', 'daithaoduong': 'dai thao duong',
}
VIETNAMESE_TEXT_LABELS = {
    'petechia': 'ban xuất huyết dạng chấm (petechia)',
    'am dao': 'âm đạo',
    'chay mau mui': 'chảy máu mũi',
    'vet chich': 'vết chích / bầm máu tại vết chích',
    'chay mau rang': 'chảy máu chân răng',
    'tang huyet ap': 'tăng huyết áp',
    'dai thao duong': 'đái tháo đường',
}
SEVERITY_ALIASES = {
    'sốc': 'sốc / nặng', 'soc': 'sốc / nặng', 'nặng': 'sốc / nặng',
    'cảnh báo': 'cảnh báo sốt xuất huyết', 'canh bao': 'cảnh báo sốt xuất huyết',
    'sxh': 'sốt xuất huyết',
}
EXAM_ALIASES = {
    'WBC': ['wbc', 'bc', 'bạch cầu', 'bach cau'],
    'RBC': ['rbc', 'hồng cầu', 'hong cau'],
    'HGB': ['hgb', 'hb'],
    'HCT': ['hct', 'hematocrit', 'dthc'],
    'PLT': ['plt', 'tiểu cầu', 'tieu cau', 'tc'],
    'AST': ['ast', 'ast(got)', 'got'],
    'ALT': ['alt', 'alt(gpt)', 'gpt'],
    'Na': ['na'], 'K': ['k'], 'Cl': ['cl'], 'Creatinine': ['creatinin', 'creatinine'], 'CrCl': ['crcl'],
    'HFLC': ['hflc'], 'NS1': ['dengue ns1ag', 'ns1ag', 'ns1'], 'IgM': ['dengue igm', 'igm'], 'IgG': ['dengue igg', 'igg'],
}
DEFAULT_REFERENCE_RANGES = {
    'WBC': '4-10 K/uL', 'RBC': '3.8-5.5 M/uL', 'HGB': '11-17 g/dL', 'HCT': '35-50 %', 'PLT': '150-400 K/uL',
    'AST': '0-25 U/L', 'ALT': '0-25 U/L', 'Na': '133-143 mmol/L', 'K': '3.5-5.0 mmol/L', 'Cl': '98-106 mmol/L', 'Creatinine': '27-62 umol/L'
}
CRITICAL_SUPPORT_TEXT = 'Ứng dụng chỉ hỗ trợ tham khảo và tổng hợp thông tin lâm sàng, không thay thế bác sĩ và không phải hệ thống kê đơn tự động.'

def simplify_severity(x: Any) -> str:
    x = str(x).strip().lower()
    if x in {'', 'nan'}: return 'không rõ'
    if 'soc' in x or 'sox' in x: return 'sốc / nặng'
    if 'canhbao' in x: return 'cảnh báo sốt xuất huyết'
    if x == 'sxh': return 'sốt xuất huyết'
    return 'khác'

def map_gender(x: Any) -> str:
    s = str(x).strip()
    if s == '1': return 'nam'
    if s == '2': return 'nữ'
    return s if s not in {'', 'nan'} else 'không rõ'

def normalize_daily_measure(base_name: str) -> str:
    return {'bachcau':'bachcau','tieucau':'tieucau','hflc':'HFLC','phantramhflc':'phantramHFLC','hct':'Hct'}.get(base_name.strip().lower(), base_name)

def format_filters_vi(filters: dict) -> str:
    parts = []
    if 'severity_group' in filters: parts.append(f"nhóm {filters['severity_group']}")
    if 'gioitinh_label' in filters: parts.append(f"giới tính {filters['gioitinh_label']}")
    if 'tiencansxh' in filters: parts.append('có tiền căn SXH' if filters['tiencansxh'] == 1 else 'không có tiền căn SXH')
    return ', '.join(parts) if parts else 'toàn bộ bệnh nhân'

def prettify_measure(m: str) -> str:
    return {'bachcau':'bạch cầu','tieucau':'tiểu cầu','HFLC':'HFLC','phantramHFLC':'%HFLC','Hct':'Hct','Tuoi':'tuổi'}.get(m,m)

def prettify_symptom(c: str) -> str:
    return {'NVnonoi':'nôn ói','NVtieuchay':'tiêu chảy','NVdaubung':'đau bụng','NVdauco':'đau cơ','NVdaudau':'đau đầu','NVdausauhocmat':'đau sau hốc mắt','NVho':'ho','NVxuathuyet':'xuất huyết','NVganto':'gan to','Petechia':'ban xuất huyết dạng chấm'}.get(c,c)

def norm_basic_text(x: str) -> str:
    return re.sub(r'\s+', ' ', str(x).strip().lower().replace(';', ',')).strip()

def clean_token(token: str) -> str:
    t = norm_basic_text(token).replace('.', '').replace('_', ' ')
    t = re.sub(r'\s+', ' ', t).strip()
    return CANONICAL_MAP.get(t, t)

def display_token(token: str) -> str:
    return VIETNAMESE_TEXT_LABELS.get(token, token)

def split_multi_value(x: Any) -> list[str]:
    if pd.isna(x): return []
    s = norm_basic_text(x)
    if s in {'', 'nan'}: return []
    return [p.strip() for p in s.split(',') if p.strip()]

def clean_multivalue_cell(x: Any) -> Any:
    if pd.isna(x): return x
    tokens = split_multi_value(x)
    seen, deduped = set(), []
    for tok in [clean_token(t) for t in tokens]:
        if tok not in seen:
            seen.add(tok)
            deduped.append(tok)
    return ', '.join(deduped)

def trust_level(rate: float) -> str:
    if rate >= 0.70: return 'cao'
    if rate >= 0.40: return 'trung bình'
    if rate >= 0.20: return 'thấp'
    return 'rất thấp'

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df['patient_id'] = [f'BN_{i:03d}' for i in range(1, len(df) + 1)]
    df['severity_group'] = df['ghichutinhtrangbenhnang'].apply(simplify_severity) if 'ghichutinhtrangbenhnang' in df.columns else 'không rõ'
    df['gioitinh_label'] = df['gioitinh'].apply(map_gender) if 'gioitinh' in df.columns else 'không rõ'
    return df

def build_cleaned_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ['Vitrixuathuyet','Trieuchungkhac','benhlynen']:
        if col in out.columns: out[col] = out[col].apply(clean_multivalue_cell)
    return out

def build_before_after_table(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for col in ['Vitrixuathuyet','Trieuchungkhac','benhlynen']:
        if col not in df_raw.columns or col not in df_clean.columns: continue
        raw_tokens=[]; clean_tokens=[]
        for x in df_raw[col].dropna(): raw_tokens.extend(split_multi_value(x))
        for x in df_clean[col].dropna(): clean_tokens.extend(split_multi_value(x))
        raw_vc = pd.Series(raw_tokens).value_counts() if raw_tokens else pd.Series(dtype=int)
        clean_vc = pd.Series(clean_tokens).value_counts() if clean_tokens else pd.Series(dtype=int)
        for token in sorted(set(raw_vc.index).union(set(clean_vc.index))):
            rows.append([col, token, display_token(token), int(raw_vc.get(token,0)), int(clean_vc.get(token,0))])
    return pd.DataFrame(rows, columns=['column_name','token','nhan_tieng_viet','count_truoc_lam_sach','count_sau_lam_sach'])

def build_daily_tables(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r'^(.*?)[Nn](\d+)$')
    rows=[]
    for c in df.columns:
        m=pattern.match(c)
        if m:
            ser = pd.to_numeric(df[c], errors='coerce').dropna()
            rows.append([normalize_daily_measure(m.group(1)), int(m.group(2)), c, len(ser), len(df), len(ser)/len(df), ser.quantile(0.25) if len(ser) else None, ser.quantile(0.5) if len(ser) else None, ser.quantile(0.75) if len(ser) else None, ser.median() if len(ser) else None, ser.mean() if len(ser) else None, ser.std() if len(ser)>1 else None, ser.min() if len(ser) else None, ser.max() if len(ser) else None])
    out = pd.DataFrame(rows, columns=['measure','hospital_day','original_column','non_null_count','total_count','coverage_rate','Q1','Q2','Q3','median','average','std','min','max']).sort_values(['measure','hospital_day'])
    if out.empty: return out
    out['trust_level']=out['coverage_rate'].apply(trust_level); out['measure_label']=out['measure'].apply(prettify_measure)
    return out

def build_symptom_tables(df: pd.DataFrame):
    subjective=['NVnonoi','NVtieuchay','NVdaubung','NVdauco','NVdaudau','NVdausauhocmat','NVho']
    objective=['NVxuathuyet','NVganto','Petechia']
    def table(cols, name):
        rows=[]
        for c in cols:
            present = int(df[c].notna().sum()) if c in df.columns else 0
            rows.append([name,c,prettify_symptom(c),present,len(df),present/len(df)])
        return pd.DataFrame(rows, columns=['group','variable','nhan_tieng_viet','present_count','total_patients','lower_bound_rate'])
    return table(subjective,'triệu chứng nhập viện'), table(objective,'dấu hiệu nhập viện')

def build_text_token_tables(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for c in ['Vitrixuathuyet','benhlynen','Trieuchungkhac']:
        if c not in df.columns: continue
        tokens=[]
        for val in df[c]: tokens.extend(split_multi_value(val))
        vc = pd.Series(tokens).value_counts() if tokens else pd.Series(dtype=int)
        for token,count in vc.items(): rows.append([c,token,display_token(token),int(count),count/len(df)])
    return pd.DataFrame(rows, columns=['column_name','token','nhan_tieng_viet','count','rate'])

def build_age_stats(df: pd.DataFrame) -> dict:
    ser=pd.to_numeric(df['Tuoi'], errors='coerce').dropna()
    return {'non_null_count':int(ser.shape[0]),'mean':float(ser.mean()) if len(ser) else None,'median':float(ser.median()) if len(ser) else None,'q1':float(ser.quantile(0.25)) if len(ser) else None,'q2':float(ser.quantile(0.5)) if len(ser) else None,'q3':float(ser.quantile(0.75)) if len(ser) else None,'min':float(ser.min()) if len(ser) else None,'max':float(ser.max()) if len(ser) else None,'std':float(ser.std()) if len(ser)>1 else None,'coverage_rate':float(ser.shape[0]/len(df)) if len(df) else None}

def build_history_tables(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for c in ['tiencansxh','tiencanbenhlykhac','pregnancy']:
        if c in df.columns:
            vc=df[c].value_counts(dropna=False)
            for val,cnt in vc.items(): rows.append([c,str(val),int(cnt),cnt/len(df)])
    return pd.DataFrame(rows, columns=['column_name','value','count','rate'])

def parse_filters(question: str) -> dict:
    q=question.lower(); filters={}
    if 'nữ' in q or ' nu ' in f' {q} ' or q.endswith(' nu') or 'giới tính nữ' in q: filters['gioitinh_label']='nữ'
    elif 'nam' in q or 'giới tính nam' in q: filters['gioitinh_label']='nam'
    q_for=q
    if 'tiền căn sxh' in q or 'tien can sxh' in q:
        filters['tiencansxh']=0 if ('không' in q or 'khong' in q) else 1
        q_for=q_for.replace('tiền căn sxh',' ').replace('tien can sxh',' ')
    for alias, sev in SEVERITY_ALIASES.items():
        if alias in q_for:
            filters['severity_group']=sev; break
    return filters

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out=df.copy()
    for k,v in filters.items():
        if k not in out.columns: continue
        if k=='tiencansxh': out=out[pd.to_numeric(out[k], errors='coerce')==v]
        else: out=out[out[k].astype(str).str.lower()==str(v).lower()]
    return out

def extract_day(question: str):
    q=question.lower(); m=re.search(r'ngày\s*(\d+)',q) or re.search(r'\bn\s*(\d+)\b',q)
    return int(m.group(1)) if m else None

def extract_measure(question: str):
    q=question.lower()
    for alias,canonical in sorted(MEASURE_ALIASES.items(), key=lambda x:-len(x[0])):
        if alias in q: return canonical
    return None

def extract_symptom(question: str):
    q=question.lower()
    for alias,canonical in sorted(SYMPTOM_ALIASES.items(), key=lambda x:-len(x[0])):
        if alias in q: return canonical
    return None

def rule_based_intent(question: str):
    q=question.lower().strip()
    if any(p in q for p in ['vị trí xuất huyết','vi tri xuat huyet','xuất huyết ở vị trí nào','xuat huyet o vi tri nao','chảy máu ở vị trí nào','chay mau o vi tri nao']): return 'top_bleeding_site'
    if any(p in q for p in ['triệu chứng khác','trieu chung khac','các triệu chứng khác','cac trieu chung khac']): return 'top_other_symptoms'
    symptom_markers=['nôn ói','non oi','nonoi','tiêu chảy','tieu chay','tieuchay','đau bụng','dau bung','daubung','đau cơ','dau co','dauco','đau đầu','dau dau','daudau','đau sau hốc mắt','dau sau hoc mat','dausauhocmat','ho','triệu chứng','trieu chung','xuất huyết','xuat huyet']
    has_symptom=any(m in q for m in symptom_markers)
    has_cohort=('ở bệnh nhân' in q or 'o benh nhan' in q or 'ở nhóm' in q or 'o nhom' in q or 'giới tính' in q or 'gioi tinh' in q or 'nữ' in q or ' nam ' in f' {q} ' or 'tiền căn sxh' in q or 'tien can sxh' in q)
    if has_cohort and has_symptom: return 'cohort_query'
    if 'tiền căn sxh' in q or 'tien can sxh' in q: return 'history_stats'
    if 'giới tính' in q or 'gioi tinh' in q or 'nam nữ' in q or 'nam nu' in q: return 'gender_stats'
    if 'độ tuổi' in q or 'do tuoi' in q or 'tuổi trung bình' in q or 'tuoi trung binh' in q: return 'age_stats'
    if 'bệnh lý nền' in q or 'benh ly nen' in q or 'comorbidity' in q: return 'top_comorbidity'
    if 'so sánh' in q or 'so sanh' in q: return 'compare_query'
    if not has_symptom and not has_cohort and any(p in q for p in ['có bao nhiêu bệnh nhân','co bao nhieu benh nhan','tổng số bệnh nhân','tong so benh nhan','bao nhiêu ca','bao nhieu ca']): return 'overview_count'
    return None

def normalize_exam_name(raw_name: str) -> str:
    key=norm_basic_text(raw_name)
    for canonical, aliases in EXAM_ALIASES.items():
        if key == canonical.lower() or key in aliases: return canonical
    for canonical, aliases in EXAM_ALIASES.items():
        if any(alias in key for alias in aliases): return canonical
    return raw_name.strip()

def numeric_or_none(value: Any):
    if value is None: return None
    s = re.sub(r'[^0-9.\-]', '', str(value).strip().replace(',', '.'))
    if s in {'','-','.','-.'}: return None
    try: return float(s)
    except ValueError: return None

def parse_ref_range(ref: str | None):
    if not ref: return None, None
    s=str(ref).replace(',', '.')
    m=re.search(r'(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)', s)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)

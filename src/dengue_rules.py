
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from _common import numeric_or_none
@dataclass
class RuleOutput:
    level: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    data_completeness: str = 'trung bình'

def _latest_numeric(labs: pd.DataFrame, exam: str):
    temp=labs[labs['exam_name_normalized']==exam]
    if temp.empty: return None
    value=temp.iloc[-1]['value']
    return value if isinstance(value,(int,float)) else numeric_or_none(value)

def _lab_positive(labs: pd.DataFrame, exam: str)->bool:
    temp=labs[labs['exam_name_normalized']==exam]
    return False if temp.empty else ('dương' in str(temp.iloc[-1]['value']).lower() or 'pos' in str(temp.iloc[-1]['value']).lower())

def evaluate_dengue_rules(labs: pd.DataFrame, clinical: dict, trend_flags: dict|None=None)->RuleOutput:
    trend_flags=trend_flags or {}; evidence=[]; warnings=[]; red_flags=[]
    plt=_latest_numeric(labs,'PLT'); wbc=_latest_numeric(labs,'WBC'); hct=_latest_numeric(labs,'HCT'); ast=_latest_numeric(labs,'AST'); alt=_latest_numeric(labs,'ALT')
    if plt is not None and plt < 150: evidence.append(f'Tiểu cầu giảm ({plt}).')
    if wbc is not None and wbc <= 4.5: evidence.append(f'Bạch cầu bình thường thấp/giảm ({wbc}).')
    if hct is not None and hct >= 45: evidence.append(f'Hct tăng hoặc cận cao ({hct}).')
    if trend_flags.get('hct_up_plt_down'): warnings.append('Hct tăng kèm tiểu cầu giảm theo thời gian.')
    if (ast is not None and ast >= 400) or (alt is not None and alt >= 400): warnings.append('Men gan tăng >= 400 U/L.')
    if (ast is not None and ast >= 1000) or (alt is not None and alt >= 1000): red_flags.append('Men gan >= 1000 U/L gợi ý tổn thương gan nặng.')
    if _lab_positive(labs,'NS1') or _lab_positive(labs,'IgM') or _lab_positive(labs,'IgG'): evidence.append('Có bằng chứng huyết thanh học/kháng nguyên Dengue dương tính.')
    if clinical.get('abdominal_pain'): warnings.append('Đau bụng nhiều.')
    if clinical.get('vomiting_many'): warnings.append('Nôn nhiều hoặc nôn liên tục.')
    if clinical.get('mucosal_bleeding'): warnings.append('Chảy máu niêm mạc.')
    if clinical.get('oliguria'): warnings.append('Tiểu ít.')
    if clinical.get('lethargy'): warnings.append('Lừ đừ / vật vã / rối loạn ý thức.')
    if clinical.get('cold_extremities'): red_flags.append('Tay chân lạnh, cần loại trừ sốc.')
    if clinical.get('dyspnea'): red_flags.append('Khó thở hoặc nghi tràn dịch / suy hô hấp.')
    if clinical.get('hypotension'): red_flags.append('Hạ huyết áp / nghi sốc.')
    if clinical.get('severe_bleeding'): red_flags.append('Xuất huyết nặng.')
    completeness = sum(v is not None for v in [plt,wbc,hct,ast,alt]) + int(any([_lab_positive(labs,'NS1'),_lab_positive(labs,'IgM'),_lab_positive(labs,'IgG')]))
    dc='cao' if completeness>=5 else 'trung bình' if completeness>=3 else 'thấp'
    if red_flags: return RuleOutput('SXHD nặng / cần khám hoặc xử trí khẩn','Có dữ kiện gợi ý SXHD nặng hoặc biến chứng cần bác sĩ đánh giá ngay.',evidence,warnings,red_flags,dc)
    if warnings: return RuleOutput('SXHD có dấu hiệu cảnh báo cần chú ý','Có các dấu hiệu cảnh báo, nên theo dõi sát và cân nhắc đánh giá tại cơ sở y tế.',evidence,warnings,[],dc)
    if evidence or clinical.get('myalgia') or clinical.get('retro_orbital_pain'): return RuleOutput('Có yếu tố gợi ý SXHD','Dữ liệu hiện tại có các đặc điểm phù hợp gợi ý sốt xuất huyết Dengue nhưng không thay thế chẩn đoán bác sĩ.',evidence,[],[],dc)
    return RuleOutput('Chưa đủ dữ liệu','Chưa đủ dữ liệu xét nghiệm và lâm sàng để gợi ý đáng tin cậy.',evidence,warnings,red_flags,dc)


from __future__ import annotations
import re
from typing import Any
import pandas as pd
from _common import DEFAULT_REFERENCE_RANGES, EXAM_ALIASES, numeric_or_none, parse_ref_range
POSITIVE_WORDS={'duong tinh','dương tính','pos','positive'}
NEGATIVE_WORDS={'am tinh','âm tính','neg','negative'}

def _parse_numeric_lab_lines(text: str):
    rows=[]
    for raw_line in text.splitlines():
        line=re.sub(r'\s+',' ', raw_line).strip()
        if not line: continue
        for canonical, aliases in EXAM_ALIASES.items():
            if not any(alias in line.lower() for alias in aliases): continue
            nums=re.findall(r'-?\d+[\.,]?\d*', line)
            value=numeric_or_none(nums[0]) if nums else None
            if value is None and any(word in line.lower() for word in POSITIVE_WORDS|NEGATIVE_WORDS):
                value='Dương tính' if any(word in line.lower() for word in POSITIVE_WORDS) else 'Âm tính'
            unit_match=re.search(r'(K/uL|K/µL|M/uL|M/µL|g/dL|%|fL|pg|mmol/L|U/L|umol/L|µmol/L|ml/phút)', line, flags=re.I)
            rows.append({'exam_name_raw': line.split()[0], 'exam_name_normalized': canonical, 'value': value, 'unit': unit_match.group(1) if unit_match else None, 'reference_range': DEFAULT_REFERENCE_RANGES.get(canonical), 'source': 'ocr', 'confidence': 0.8})
            break
    return rows

def parse_exam_text_to_rows(text: str) -> pd.DataFrame:
    df=pd.DataFrame(_parse_numeric_lab_lines(text))
    if df.empty:
        return pd.DataFrame(columns=['exam_name_raw','exam_name_normalized','value','unit','reference_range','source','confidence'])
    return df.drop_duplicates(subset=['exam_name_normalized','value','unit']).reset_index(drop=True)

def attach_reference_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out=df.copy(); flags=[]
    for _,row in out.iterrows():
        value=row['value'] if isinstance(row['value'], (int,float)) else numeric_or_none(row['value'])
        lo, hi = parse_ref_range(row.get('reference_range'))
        if value is None or (lo is None and hi is None): flags.append('unknown')
        elif lo is not None and value < lo: flags.append('low')
        elif hi is not None and value > hi: flags.append('high')
        else: flags.append('normal')
    out['range_flag']=flags
    return out

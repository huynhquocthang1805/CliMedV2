
from __future__ import annotations
import pandas as pd
from _common import numeric_or_none

def annotate_missingness(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out=df.copy(); out['numeric_value']=out['value'].apply(lambda x: x if isinstance(x,(int,float)) else numeric_or_none(x)); out['observed_status']=out['numeric_value'].apply(lambda x: 'observed' if x is not None else 'unknown'); out['feature_mask']=out['numeric_value'].apply(lambda x: 1 if x is not None else 0); out['report_time']=pd.to_datetime(out.get('report_time'), errors='coerce'); out=out.sort_values(['exam_name_normalized','report_time']); out['time_delta_hours']=out.groupby('exam_name_normalized')['report_time'].diff().dt.total_seconds().div(3600); return out

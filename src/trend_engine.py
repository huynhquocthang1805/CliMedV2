
from __future__ import annotations
import pandas as pd
from _common import numeric_or_none

def prepare_trend_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out=df.copy(); out['numeric_value']=out['value'].apply(lambda x: x if isinstance(x,(int,float)) else numeric_or_none(x)); out=out.dropna(subset=['numeric_value']); out['report_time']=pd.to_datetime(out.get('report_time'), errors='coerce'); return out.sort_values(['exam_name_normalized','report_time'])

def compute_trend_flags(df: pd.DataFrame) -> dict:
    flags={'plt_falling_consecutive':False,'hct_rising':False,'hct_up_plt_down':False,'ast_alt_up':False}
    if df.empty: return flags
    pivot=df.pivot_table(index='report_time', columns='exam_name_normalized', values='numeric_value', aggfunc='last').sort_index()
    if 'PLT' in pivot.columns and pivot['PLT'].dropna().shape[0]>=2:
        diffs=pivot['PLT'].dropna().diff().dropna(); flags['plt_falling_consecutive']=bool((diffs<0).tail(2).all()) if len(diffs)>=2 else bool((diffs<0).all())
    if 'HCT' in pivot.columns and pivot['HCT'].dropna().shape[0]>=2:
        diffs=pivot['HCT'].dropna().diff().dropna(); flags['hct_rising']=bool((diffs>0).any())
    flags['hct_up_plt_down']=flags['plt_falling_consecutive'] and flags['hct_rising']
    for exam in ['AST','ALT']:
        if exam in pivot.columns and pivot[exam].dropna().shape[0]>=2:
            flags['ast_alt_up']=flags['ast_alt_up'] or bool((pivot[exam].dropna().diff().dropna()>0).any())
    return flags

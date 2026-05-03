
from __future__ import annotations
import joblib, pandas as pd
from _common import MODEL_DIR, load_data, build_cleaned_df, build_before_after_table, build_daily_tables, build_symptom_tables, build_text_token_tables, build_age_stats, build_history_tables, parse_filters, apply_filters, rule_based_intent, extract_day, extract_measure, extract_symptom, prettify_measure, prettify_symptom, format_filters_vi
class HFLCQAEngine:
    def __init__(self):
        self.df_raw=load_data(); self.df=build_cleaned_df(self.df_raw); self.before_after_table=build_before_after_table(self.df_raw,self.df); self.daily_stats=build_daily_tables(self.df); self.symptoms_table,self.signs_table=build_symptom_tables(self.df); self.text_table=build_text_token_tables(self.df); self.age_stats=build_age_stats(self.df); self.history_table=build_history_tables(self.df); mp=MODEL_DIR/'intent_model.joblib'; self.intent_model=joblib.load(mp) if mp.exists() else None
    def predict_intent(self, question:str)->str:
        rb=rule_based_intent(question)
        if rb is not None: return rb
        if self.intent_model is None: return 'fallback'
        return self.intent_model.predict([question])[0]
    @staticmethod
    def _route_name(intent:str)->str:
        if intent=='overview_count': return 'rule/statistics'
        if intent in {'daily_stat','trust_query'}: return 'daily_stats_table'
        if intent in {'symptom_count','top_symptoms'}: return 'symptom_table'
        if intent in {'top_bleeding_site','top_comorbidity','top_other_symptoms'}: return 'cleaned_text_token_table'
        if intent in {'cohort_query','compare_query'}: return 'graph-lite cohort routing'
        if intent in {'age_stats','gender_stats','history_stats'}: return 'demographic/history table'
        return 'fallback'
    def debug_parse(self, question:str)->dict:
        intent=self.predict_intent(question); filters=parse_filters(question); subset=apply_filters(self.df, filters) if filters else self.df
        return {'question':question,'intent':intent,'filters':filters,'cohort_size':int(len(subset)),'symptom_extracted':extract_symptom(question),'measure_extracted':extract_measure(question),'day_extracted':extract_day(question),'route':self._route_name(intent)}
    def _answer_on_subset(self, subset:pd.DataFrame, question:str, filters:dict)->str:
        label=format_filters_vi(filters)
        if subset.empty: return f'Không có bệnh nhân nào khớp bộ lọc: {label}.'
        symptom_col=extract_symptom(question)
        if symptom_col and symptom_col in subset.columns:
            present=int(subset[symptom_col].notna().sum())
            return f'Trong {label}, {prettify_symptom(symptom_col)} được ghi nhận ở {present}/{len(subset)} bệnh nhân ({present/len(subset):.1%}).'
        symptoms_table,_=build_symptom_tables(subset); temp=symptoms_table.sort_values('present_count', ascending=False).head(5); parts=[f"{r['nhan_tieng_viet']}: {int(r['present_count'])} ca" for _,r in temp.iterrows()]
        return f'Trong {label}, các triệu chứng nổi bật là: ' + '; '.join(parts) + '.'
    def answer(self, question:str)->str:
        intent=self.predict_intent(question)
        if intent=='overview_count': return f'Có {len(self.df)} bệnh nhân trong file CSV gốc.'
        if intent=='top_symptoms':
            temp=self.symptoms_table.sort_values('present_count', ascending=False).head(7); parts=[f"{r['nhan_tieng_viet']}: {int(r['present_count'])} ca" for _,r in temp.iterrows()]
            return 'Các triệu chứng nhập viện được ghi nhận là: ' + '; '.join(parts) + '. Lưu ý: các cột NV là presence-only.'
        if intent=='symptom_count':
            symptom_col=extract_symptom(question)
            if not symptom_col: return 'Mình chưa nhận ra triệu chứng bạn hỏi.'
            temp=self.symptoms_table[self.symptoms_table['variable']==symptom_col]
            if temp.empty: temp=self.signs_table[self.signs_table['variable']==symptom_col]
            if temp.empty: return 'Không tìm thấy triệu chứng/dấu hiệu đó trong dữ liệu.'
            row=temp.iloc[0]
            return f"{row['nhan_tieng_viet']} được ghi nhận hiện diện ở {int(row['present_count'])}/{int(row['total_patients'])} bệnh nhân (tỷ lệ tối thiểu {row['lower_bound_rate']:.1%})."
        if intent=='daily_stat':
            measure=extract_measure(question); day=extract_day(question)
            if not measure or day is None: return 'Mình cần biết chỉ số và ngày.'
            row=self.daily_stats[(self.daily_stats['measure']==measure)&(self.daily_stats['hospital_day']==day)]
            if row.empty: return f'Không tìm thấy dữ liệu cho {prettify_measure(measure)} ngày {day}.'
            row=row.iloc[0]; q=question.lower()
            if 'q1' in q: stat_name,value='Q1',row['Q1']
            elif 'q3' in q: stat_name,value='Q3',row['Q3']
            elif 'q2' in q or 'median' in q or 'trung vị' in q: stat_name,value='median',row['median']
            elif 'average' in q or 'trung bình' in q or 'mean' in q or 'avr' in q: stat_name,value='average',row['average']
            else: stat_name,value='median',row['median']
            return f"{prettify_measure(measure)} ngày {day}: {stat_name} = {value:.4f} | n = {int(row['non_null_count'])}/{int(row['total_count'])} | coverage = {row['coverage_rate']:.1%} | độ tin cậy = {row['trust_level']}."
        if intent=='trust_query':
            measure=extract_measure(question); day=extract_day(question)
            if not measure or day is None: return 'Mình cần biết chỉ số và ngày.'
            row=self.daily_stats[(self.daily_stats['measure']==measure)&(self.daily_stats['hospital_day']==day)]
            if row.empty: return f'Không tìm thấy dữ liệu cho {prettify_measure(measure)} ngày {day}.'
            row=row.iloc[0]
            return f"{prettify_measure(measure)} ngày {day}: n = {int(row['non_null_count'])}/{int(row['total_count'])}, coverage = {row['coverage_rate']:.1%}, độ tin cậy = {row['trust_level']}."
        if intent=='top_bleeding_site':
            temp=self.text_table[self.text_table['column_name']=='Vitrixuathuyet'].head(5); parts=[f"{r['nhan_tieng_viet']}: {int(r['count'])} ca" for _,r in temp.iterrows()]
            return 'Các vị trí xuất huyết được ghi nhận nhiều nhất là: ' + '; '.join(parts) + '.'
        if intent=='top_comorbidity':
            temp=self.text_table[self.text_table['column_name']=='benhlynen'].head(8)
            if temp.empty: return 'Không có dữ liệu bệnh lý nền đủ rõ để tổng hợp.'
            parts=[f"{r['nhan_tieng_viet']}: {int(r['count'])} ca" for _,r in temp.iterrows()]
            return 'Các bệnh lý nền được ghi nhận nhiều nhất là: ' + '; '.join(parts) + '.'
        if intent=='top_other_symptoms':
            temp=self.text_table[self.text_table['column_name']=='Trieuchungkhac'].head(8)
            if temp.empty: return 'Không có dữ liệu triệu chứng khác đủ rõ để tổng hợp.'
            parts=[f"{r['nhan_tieng_viet']}: {int(r['count'])} ca" for _,r in temp.iterrows()]
            return 'Các triệu chứng khác được ghi nhận nhiều nhất là: ' + '; '.join(parts) + '.'
        if intent=='age_stats':
            s=self.age_stats; return f"Thống kê độ tuổi: n = {s['non_null_count']}/{len(self.df)}, coverage = {s['coverage_rate']:.1%}, mean = {s['mean']:.2f}, median = {s['median']:.2f}."
        if intent=='history_stats':
            temp=self.history_table[self.history_table['column_name']=='tiencansxh'].sort_values('count', ascending=False)
            if temp.empty: return 'Không có dữ liệu tiền căn SXH để tổng hợp.'
            parts=[f"{r['value']}: {int(r['count'])} ca ({r['rate']:.1%})" for _,r in temp.iterrows()]
            return 'Thống kê tiền căn SXH: ' + '; '.join(parts) + '.'
        if intent=='gender_stats':
            vc=self.df['gioitinh_label'].value_counts(dropna=False); parts=[f'{idx}: {int(cnt)} ca ({cnt/len(self.df):.1%})' for idx,cnt in vc.items()]
            return 'Phân bố giới tính (đã ánh xạ 1=nam, 2=nữ): ' + '; '.join(parts) + '.'
        if intent=='cohort_query': return self._answer_on_subset(apply_filters(self.df, parse_filters(question)), question, parse_filters(question))
        if intent=='compare_query':
            male=self.df[self.df['gioitinh_label']=='nam']; female=self.df[self.df['gioitinh_label']=='nữ']
            return f'So sánh nam và nữ: nam = {len(male)} ca ({len(male)/len(self.df):.1%}), nữ = {len(female)} ca ({len(female)/len(self.df):.1%}).'
        return 'Mình chưa hiểu câu hỏi này.'

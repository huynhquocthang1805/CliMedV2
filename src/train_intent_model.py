
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from _common import MODEL_DIR
TRAIN_DATA=[('có bao nhiêu bệnh nhân','overview_count'),('triệu chứng nào lúc nhập viện gặp nhiều nhất','top_symptoms'),('có bao nhiêu bệnh nhân triệu chứng nôn ói','symptom_count'),('tiêu chảy có bao nhiêu bệnh nhân','symptom_count'),('HFLC ngày 5 trung vị bao nhiêu','daily_stat'),('HFLC ngày 5 có đáng tin không','trust_query'),('vị trí xuất huyết nào gặp nhiều nhất','top_bleeding_site'),('bệnh lý nền nào hay gặp nhất','top_comorbidity'),('triệu chứng khác nào gặp nhiều nhất','top_other_symptoms'),('độ tuổi trung bình là bao nhiêu','age_stats'),('tiền căn sxh thế nào','history_stats'),('giới tính phân bố thế nào','gender_stats'),('ở bệnh nhân nữ, nonoi có bao nhiêu','cohort_query'),('ở nhóm sốc, triệu chứng nào hay gặp','cohort_query'),('so sánh nam và nữ','compare_query')]
X=[x for x,y in TRAIN_DATA]; y=[y for x,y in TRAIN_DATA]
p=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),lowercase=True)),('clf',LogisticRegression(max_iter=2000))]); p.fit(X,y); MODEL_DIR.mkdir(exist_ok=True); joblib.dump(p, MODEL_DIR/'intent_model.joblib'); print('Saved model')

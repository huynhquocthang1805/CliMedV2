from __future__ import annotations
from typing import Any
from pathlib import Path
from functools import lru_cache
import re
import networkx as nx
import pdfplumber

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / 'data' / 'knowledge'

KNOWLEDGE_SNIPPETS=[
    {
        'id':'guideline_symptomatic',
        'title':'Điều trị triệu chứng ngoại trú',
        'text':'Điều trị ngoại trú chủ yếu là điều trị triệu chứng và theo dõi sát. Khuyến khích uống nhiều nước, nước cam, chanh, Oresol. Chỉ dùng Paracetamol đơn chất khi sốt.',
        'citation':'BV Bệnh Nhiệt Đới 2023 – điều trị SXHD và chăm sóc người bệnh.',
        'keywords':{'paracetamol','oresol','uống','nước','ngoại trú','sốt'}
    },
    {
        'id':'guideline_avoid',
        'title':'Chống chỉ định cần tránh',
        'text':'Không dùng Aspirin, Ibuprofen, Analgin trong bối cảnh SXHD vì có thể gây xuất huyết hoặc toan máu.',
        'citation':'BV Bệnh Nhiệt Đới 2023 – điều trị triệu chứng SXHD.',
        'keywords':{'aspirin','ibuprofen','analgin','tránh','không dùng'}
    },
    {
        'id':'guideline_warning',
        'title':'Dấu hiệu cảnh báo',
        'text':'Dấu hiệu cảnh báo gồm đau bụng nhiều, nôn nhiều, chảy máu niêm mạc, tiểu ít, lừ đừ/vật vã, tay chân lạnh, khó thở; Hct tăng kèm tiểu cầu giảm là pattern đáng chú ý.',
        'citation':'BV Bệnh Nhiệt Đới 2023 – phân độ và theo dõi SXHD.',
        'keywords':{'đau bụng','nôn','tiểu ít','lừ đừ','tay chân lạnh','khó thở','hct','tiểu cầu'}
    },
    {
        'id':'missingness_mask',
        'title':'Missingness minh bạch',
        'text':'Với dữ liệu thời gian không đều, nên lưu Mask và Time Delta. Có thể giữ NaN cho tree-based model; chỉ nội suy khoảng ngắn và cần gắn cờ imputed.',
        'citation':'Irregular time series / informative missingness notes.',
        'keywords':{'mask','time','delta','nan','imputed','missing'}
    }
]

def _normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()

def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

@lru_cache(maxsize=1)
def load_pdf_knowledge_chunks() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    if not KNOWLEDGE_DIR.exists():
        return docs

    for file_path in sorted(KNOWLEDGE_DIR.iterdir()):
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_idx, page in enumerate(pdf.pages, start=1):
                        page_text = _normalize_text(page.extract_text() or '')
                        if not page_text:
                            continue
                        for chunk_idx, chunk in enumerate(_chunk_text(page_text), start=1):
                            docs.append({
                                'id': f'{file_path.stem}_p{page_idx}_c{chunk_idx}',
                                'title': file_path.name,
                                'text': chunk,
                                'citation': f'{file_path.name} - trang {page_idx}',
                                'source_file': file_path.name,
                                'page': page_idx,
                            })
            except Exception:
                continue
        elif suffix == '.txt':
            try:
                text = _normalize_text(file_path.read_text(encoding='utf-8', errors='ignore'))
                for chunk_idx, chunk in enumerate(_chunk_text(text), start=1):
                    docs.append({
                        'id': f'{file_path.stem}_c{chunk_idx}',
                        'title': file_path.name,
                        'text': chunk,
                        'citation': file_path.name,
                        'source_file': file_path.name,
                        'page': None,
                    })
            except Exception:
                continue
    return docs


def list_loaded_knowledge_sources() -> list[str]:
    return sorted({doc['source_file'] for doc in load_pdf_knowledge_chunks()})


def retrieve_knowledge(query: str, top_k: int = 5, extra_context: str | None = None):
    q = _normalize_text(' '.join([query or '', extra_context or '']))
    q_terms = set(re.findall(r'\w+', q.lower()))
    scored = []
    for item in KNOWLEDGE_SNIPPETS:
        score = len(q_terms & item['keywords'])
        if score > 0:
            scored.append((score, item))
    for item in load_pdf_knowledge_chunks():
        text_terms = set(re.findall(r'\w+', item['text'].lower()))
        score = len(q_terms & text_terms)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    results = []
    for score, item in scored:
        if item['id'] in seen:
            continue
        seen.add(item['id'])
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def answer_question_with_retrieval(query: str, context_facts: list[str] | None = None, extra_context: str | None = None) -> dict[str, Any]:
    retrieved = retrieve_knowledge(query, top_k=5, extra_context=extra_context)
    context_facts = context_facts or []
    if not retrieved:
        return {
            'answer': 'Chưa tìm thấy tri thức phù hợp trong kho nội bộ. Hãy kiểm tra lại dữ liệu OCR, file hướng dẫn trong thư mục data/knowledge và thông tin lâm sàng.',
            'citations': [],
            'retrieved': []
        }
    parts = []
    citations = []
    if context_facts:
        parts.append('Dựa trên dữ kiện hiện có: ' + '; '.join(context_facts) + '.')
    if extra_context:
        parts.append('Ngữ cảnh OCR/tài liệu vừa tải lên đã được dùng để đối chiếu.')
    for item in retrieved:
        snippet = item['text'][:320].strip()
        parts.append(snippet)
        citations.append(item['citation'])
    return {
        'answer': ' '.join(parts),
        'citations': citations,
        'retrieved': retrieved,
    }


def build_reasoning_graph(rule_level: str, context_flags: dict[str, bool]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node('dengue_suspicion', label='nghi ngờ SXHD')
    g.add_node('warning_pattern', label='pattern cảnh báo')
    g.add_node('supportive_care', label='chăm sóc hỗ trợ')
    g.add_node('avoid_nsaids', label='tránh NSAIDs/aspirin/analgin')
    if context_flags.get('plt_low'):
        g.add_edge('platelet_low', 'dengue_suspicion', relation='supports')
    if context_flags.get('wbc_low'):
        g.add_edge('wbc_low', 'dengue_suspicion', relation='supports')
    if context_flags.get('hct_up_plt_down'):
        g.add_edge('hct_rising_plus_plt_falling', 'warning_pattern', relation='supports')
    if 'cảnh báo' in rule_level.lower() or 'nặng' in rule_level.lower() or 'khẩn' in rule_level.lower():
        g.add_edge('warning_pattern', 'supportive_care', relation='escalates_to')
    g.add_edge('avoid_nsaids', 'dengue_suspicion', relation='contraindicated_in_context')
    return g

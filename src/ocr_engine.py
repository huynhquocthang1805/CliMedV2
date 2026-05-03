
from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Any
from PIL import Image
@dataclass
class OCRResult:
    text: str
    source: str
    confidence: float | None = None
    method: str | None = None

def _extract_pdf_text(uploaded_file: Any) -> OCRResult:
    try:
        import pdfplumber
    except Exception:
        return OCRResult(text='', source='ocr', confidence=None, method='pdfplumber_not_installed')
    texts=[]
    with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ''
            if txt.strip(): texts.append(txt)
    return OCRResult(text='\n\n'.join(texts), source='ocr', confidence=0.9 if texts else 0.0, method='pdfplumber')

def _extract_image_text(uploaded_file: Any) -> OCRResult:
    img = Image.open(io.BytesIO(uploaded_file.getvalue())).convert('RGB')
    try:
        import pytesseract
        txt = pytesseract.image_to_string(img, lang='vie+eng')
        return OCRResult(text=txt, source='ocr', confidence=0.65 if txt.strip() else 0.0, method='pytesseract')
    except Exception:
        return OCRResult(text='', source='ocr', confidence=0.0, method='ocr_not_available')

def extract_text_from_uploaded_file(uploaded_file: Any) -> OCRResult:
    return _extract_pdf_text(uploaded_file) if (uploaded_file.name or '').lower().endswith('.pdf') else _extract_image_text(uploaded_file)

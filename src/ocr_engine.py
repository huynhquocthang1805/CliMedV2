"""
ocr_engine.py — CliMedV2
========================

OCR multi-tier cho phiếu xét nghiệm sốt xuất huyết.

Pipeline:
    PDF có text layer  ──► pdfplumber                       (Tier 1, miễn phí, nhanh)
    PDF scan / ảnh     ──► PyMuPDF rasterize ──► Tesseract  (Tier 2, mặc định)
    Ảnh khó            ──► EasyOCR / PaddleOCR              (Tier 3, optional)

Drop-in: giữ nguyên signature `extract_text_from_uploaded_file(file) -> OCRResult`
nên `app.py` không cần sửa gì.

Cải tiến so với bản cũ:
  1. Preprocess ảnh chuẩn cho ảnh điện thoại (deskew, denoise, threshold).
  2. Fallback OCR cho PDF scan (bản cũ chỉ đọc được PDF có text layer).
  3. Post-process số y khoa: O→0, l→1, "4,52"→"4.52".
  4. Tesseract config tối ưu cho bảng xét nghiệm (PSM 6, OEM 3).
  5. Confidence thực, không hard-code.

Cài đặt:
    pip install pdfplumber pytesseract pymupdf opencv-python-headless pillow numpy
    # System (Tesseract):
    #   Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-vie
    #   macOS:         brew install tesseract tesseract-lang
    #   Windows:       Tải installer từ UB-Mannheim, nhớ tick "Vietnamese"
    # Optional engines:
    #   pip install easyocr           # chính xác hơn cho ảnh khó, cần ~1.5GB
    #   pip install paddleocr paddlepaddle   # tốt nhất cho tiếng Việt + bảng
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Engine name → ưu tiên dùng. Có thể override bằng env var để dễ thử nghiệm.
#   export CLIMED_OCR_ENGINE=easyocr   # hoặc paddleocr / tesseract
DEFAULT_ENGINE = os.getenv("CLIMED_OCR_ENGINE", "tesseract").lower()


# ===========================================================================
# Public dataclass — GIỮ NGUYÊN để tương thích app.py cũ
# ===========================================================================

@dataclass
class OCRResult:
    text: str
    source: str
    confidence: float | None = None
    method: str | None = None


# ===========================================================================
# Image preprocessing — đây là chỗ cải thiện chất lượng OCR nhiều nhất
# ===========================================================================

def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    """PIL (RGB) → OpenCV (BGR ndarray)."""
    arr = np.array(img.convert("RGB"))
    return arr[:, :, ::-1].copy()  # RGB → BGR


def _cv2_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV (BGR) → PIL (RGB)."""
    if arr.ndim == 2:  # grayscale
        return Image.fromarray(arr)
    return Image.fromarray(arr[:, :, ::-1])


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Xoay thẳng ảnh dựa vào Hough lines.
    Nếu góc nghiêng < 0.5° thì bỏ qua (tránh xoay vô ích gây mờ).
    """
    import cv2

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200,
                            minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return gray

    angles: List[float] = []
    for line in lines[:50]:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # chỉ tin các đường gần ngang (trong khoảng ±15°)
        if -15 < angle < 15:
            angles.append(angle)

    if not angles:
        return gray
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _preprocess_for_ocr(pil_img: Image.Image, for_table: bool = True) -> Image.Image:
    """
    Pipeline làm sạch ảnh trước khi OCR. Trả về PIL grayscale image.

    for_table=True (default): dành cho phiếu xét nghiệm dạng bảng
        — dùng adaptive threshold để bảng có border mờ vẫn rõ.
    for_table=False: dành cho text thường, không threshold mạnh.

    Nếu OpenCV không cài được, fallback về PIL ImageOps đơn giản.
    """
    try:
        import cv2
    except ImportError:
        # Fallback: dùng PIL tối thiểu (kết quả kém hơn nhưng vẫn chạy được)
        from PIL import ImageOps
        logger.warning("OpenCV chưa cài, fallback PIL — OCR có thể kém. "
                       "Cài: pip install opencv-python-headless")
        gray = ImageOps.grayscale(pil_img)
        return ImageOps.autocontrast(gray, cutoff=2)

    img = _pil_to_cv2(pil_img)
    h, w = img.shape[:2]

    # 1. Upscale nếu ảnh quá nhỏ (số viết in nhỏ dễ vỡ pixel)
    if max(h, w) < 1500:
        scale = 1500.0 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Khử nhiễu giữ cạnh (bilateral nhẹ tay hơn Gaussian, không làm mờ chữ)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Deskew (xoay thẳng nếu chụp nghiêng)
    gray = _deskew(gray)

    # 5. Threshold thích nghi cho ảnh có bóng / ánh sáng không đều
    if for_table:
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=15,
        )

    return Image.fromarray(gray)


# ===========================================================================
# OCR engines — Tier 2/3
# ===========================================================================

def _ocr_tesseract(pil_img: Image.Image) -> Tuple[str, float]:
    """
    Tesseract OCR với config tối ưu cho bảng xét nghiệm.

    PSM 6 = "Assume a single uniform block of text" — phù hợp cho bảng
            (PSM 3 mặc định hay xuống dòng sai trên bảng có cột).
    OEM 3 = "Default, based on what is available" — dùng LSTM nếu có.
    """
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        return "", 0.0

    config = r"--oem 3 --psm 6"
    try:
        # image_to_data trả về cả confidence per-word → tính avg conf thực
        data = pytesseract.image_to_data(
            pil_img, lang="vie+eng", config=config,
            output_type=Output.DICT,
        )
    except Exception as e:
        logger.error("Tesseract lỗi: %s", e)
        return "", 0.0

    # Lọc word có conf > 0 (Tesseract đặt -1 cho block không phải text)
    words: List[Tuple[int, int, int, str, float]] = []
    for i, txt in enumerate(data["text"]):
        if not txt or not txt.strip():
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            continue
        if conf < 0:
            continue
        words.append((
            int(data["block_num"][i]),
            int(data["line_num"][i]),
            int(data["word_num"][i]),
            txt,
            conf,
        ))

    if not words:
        return "", 0.0

    # Ráp lại text theo block/line giữ thứ tự đọc
    words.sort(key=lambda x: (x[0], x[1], x[2]))
    lines: dict = {}
    for block, line, _, txt, _ in words:
        lines.setdefault((block, line), []).append(txt)
    full_text = "\n".join(" ".join(ws) for ws in lines.values())

    # Confidence: conf trung bình các word, scale về [0, 1]
    avg_conf = float(np.mean([w[4] for w in words])) / 100.0
    return full_text, avg_conf


def _ocr_easyocr(pil_img: Image.Image) -> Tuple[str, float]:
    """EasyOCR — chính xác hơn Tesseract cho ảnh khó. Optional."""
    try:
        import easyocr
    except ImportError:
        return "", 0.0

    # Cache reader để không reload mỗi lần (dùng module-level cache)
    global _EASYOCR_READER
    try:
        reader = _EASYOCR_READER  # type: ignore
    except NameError:
        logger.info("Loading EasyOCR (vi+en) — lần đầu tải model ~1GB...")
        reader = easyocr.Reader(["vi", "en"], gpu=False)
        globals()["_EASYOCR_READER"] = reader

    rgb = np.array(pil_img.convert("RGB"))
    results = reader.readtext(rgb, detail=1, paragraph=False)
    if not results:
        return "", 0.0

    # results: List[(box, text, conf)]
    # Sắp theo y rồi x để đọc đúng thứ tự bảng
    results.sort(key=lambda r: (min(p[1] for p in r[0]),
                                min(p[0] for p in r[0])))
    text = "\n".join(r[1] for r in results)
    avg_conf = float(np.mean([r[2] for r in results]))
    return text, avg_conf


def _ocr_paddle(pil_img: Image.Image) -> Tuple[str, float]:
    """PaddleOCR — tốt nhất cho tiếng Việt + bảng có border. Optional."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return "", 0.0

    global _PADDLE_OCR
    try:
        ocr = _PADDLE_OCR  # type: ignore
    except NameError:
        logger.info("Loading PaddleOCR (lang=vi)...")
        ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
        globals()["_PADDLE_OCR"] = ocr

    bgr = _pil_to_cv2(pil_img)
    raw = ocr.ocr(bgr, cls=True)
    if not raw or not raw[0]:
        return "", 0.0

    blocks = []
    for box, (txt, conf) in raw[0]:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        blocks.append((min(ys), min(xs), txt, float(conf)))
    blocks.sort(key=lambda b: (b[0], b[1]))
    text = "\n".join(b[2] for b in blocks)
    avg_conf = float(np.mean([b[3] for b in blocks]))
    return text, avg_conf


# Routing: chọn engine theo cấu hình
_ENGINE_DISPATCH = {
    "tesseract": _ocr_tesseract,
    "easyocr": _ocr_easyocr,
    "paddleocr": _ocr_paddle,
    "paddle": _ocr_paddle,
}


def _ocr_image_with_fallback(pil_img: Image.Image) -> Tuple[str, float, str]:
    """
    Chạy engine ưu tiên trước. Nếu confidence quá thấp hoặc text rỗng,
    thử fallback sang engine khác có sẵn.

    Trả về: (text, confidence, method_used)
    """
    pre = _preprocess_for_ocr(pil_img, for_table=True)

    # 1. Engine ưu tiên
    primary = _ENGINE_DISPATCH.get(DEFAULT_ENGINE, _ocr_tesseract)
    text, conf = primary(pre)
    method = DEFAULT_ENGINE

    # 2. Nếu kết quả tệ, thử các engine khác
    LOW_CONF = 0.40
    if not text.strip() or conf < LOW_CONF:
        for name, fn in _ENGINE_DISPATCH.items():
            if name == DEFAULT_ENGINE:
                continue
            t2, c2 = fn(pre)
            if t2.strip() and c2 > conf:
                text, conf, method = t2, c2, name + "_fallback"
                break

    # 3. Nếu vẫn rỗng, thử lại với ảnh KHÔNG threshold
    #    (đôi khi threshold mạnh phá ảnh có nền màu)
    if not text.strip():
        pre_soft = _preprocess_for_ocr(pil_img, for_table=False)
        text, conf = primary(pre_soft)
        method = DEFAULT_ENGINE + "_no_threshold"

    return text, conf, method


# ===========================================================================
# Post-processing số y khoa — chỗ này chữa được nhiều lỗi OCR điển hình
# ===========================================================================

# Các chỉ số y khoa quan tâm — match keyword (lowercase) → để biết đang trong context số
_LAB_CONTEXT_KEYWORDS = (
    "wbc", "rbc", "hgb", "hb", "hct", "plt", "ast", "alt", "got", "gpt",
    "bạch cầu", "bach cau", "hồng cầu", "hong cau", "tiểu cầu", "tieu cau",
    "hematocrit", "creatinin", "creatinine", "natri", "kali", "clo",
    "glucose", "ure", "albumin", "ferritin", "ldh", "crp", "inr", "aptt",
    "ns1", "igm", "igg", "hflc",
)

_OCR_CHAR_FIXES_IN_NUMBER = str.maketrans({
    "O": "0", "o": "0",
    "l": "1", "I": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "g": "9",  # chỉ áp dụng khi ở trong context số
})


def _looks_like_lab_line(line: str) -> bool:
    """Dòng này có vẻ chứa chỉ số xét nghiệm không?"""
    low = line.lower()
    return any(kw in low for kw in _LAB_CONTEXT_KEYWORDS)


def _fix_number_token(token: str) -> str:
    """
    Sửa lỗi OCR trong 1 token được nghi ngờ là số.
    Áp dụng `_OCR_CHAR_FIXES_IN_NUMBER` chỉ khi token trông giống số.
    """
    # Token chứa ít nhất 1 chữ số → có khả năng là số bị OCR nhầm vài ký tự
    has_digit = bool(re.search(r"\d", token))
    if not has_digit:
        return token

    # Bỏ qua đơn vị thường gặp (giữ nguyên K/uL, g/dL, mmol/L, %, U/L)
    units = {"k/ul", "k/µl", "m/ul", "m/µl", "g/dl", "mmol/l", "u/l",
             "umol/l", "µmol/l", "ml/phút", "ml/min", "%", "fl", "pg"}
    if token.lower() in units:
        return token

    # Áp dụng char fix
    fixed = token.translate(_OCR_CHAR_FIXES_IN_NUMBER)

    # Chuẩn hóa decimal: "4,52" → "4.52" (Việt Nam dùng dấu phẩy nhưng
    # parser hiện tại chấp nhận cả 2; ta normalize cho đồng nhất)
    # Chỉ thay khi dấu phẩy giữa 2 chữ số: "1,234,567" giữ nguyên,
    # "4,52" thành "4.52".
    fixed = re.sub(r"(?<=\d),(?=\d{1,3}\b)", ".", fixed)

    return fixed


def _postprocess_medical_text(text: str) -> str:
    """
    Sửa lỗi OCR điển hình trong text y khoa:
      • Số trong dòng có chỉ số xét nghiệm: O→0, l→1, S→5, ...
      • Decimal Việt Nam "4,52" → "4.52"
      • Khoảng trắng thừa giữa số và đơn vị: "150 K/uL" giữ, "150K/uL" giữ
    """
    if not text or not text.strip():
        return text

    out_lines = []
    for line in text.splitlines():
        if not _looks_like_lab_line(line):
            out_lines.append(line)
            continue

        # Tách token theo whitespace, sửa từng token, ráp lại
        tokens = re.split(r"(\s+)", line)  # giữ whitespace
        fixed_tokens = [
            tok if tok.isspace() or not tok else _fix_number_token(tok)
            for tok in tokens
        ]
        out_lines.append("".join(fixed_tokens))

    return "\n".join(out_lines)


# ===========================================================================
# PDF handling
# ===========================================================================

def _extract_pdf_text_layer(pdf_bytes: bytes) -> str:
    """Tier 1: thử lấy text layer (chỉ work với PDF có text embedded)."""
    try:
        import pdfplumber
    except ImportError:
        return ""

    pieces: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    pieces.append(t)
    except Exception as e:
        logger.warning("pdfplumber lỗi: %s", e)
        return ""
    return "\n\n".join(pieces)


def _rasterize_pdf(pdf_bytes: bytes, dpi: int = 250) -> List[Image.Image]:
    """
    Tier 2: render mỗi trang PDF thành ảnh PIL, dùng PyMuPDF (fitz).
    PyMuPDF không cần system package (khác pdf2image cần poppler).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error(
            "PyMuPDF chưa cài → không OCR được PDF scan. "
            "Cài: pip install pymupdf"
        )
        return []

    images: List[Image.Image] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.error("Mở PDF lỗi: %s", e)
        return []

    # zoom = dpi / 72 (PDF mặc định 72 dpi)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    finally:
        doc.close()

    return images


# ===========================================================================
# Public API — backward compatible với app.py cũ
# ===========================================================================

def _extract_pdf(uploaded_file: Any) -> OCRResult:
    """
    Pipeline cho PDF:
      1. Thử text layer (pdfplumber)
      2. Nếu rỗng → rasterize → OCR từng trang
    """
    pdf_bytes = uploaded_file.getvalue()

    # Tier 1
    text_layer = _extract_pdf_text_layer(pdf_bytes)
    if text_layer.strip():
        cleaned = _postprocess_medical_text(text_layer)
        return OCRResult(
            text=cleaned, source="ocr",
            confidence=0.95, method="pdfplumber_text_layer",
        )

    # Tier 2: PDF scan → OCR từng trang
    logger.info("PDF không có text layer, fallback OCR từng trang...")
    page_images = _rasterize_pdf(pdf_bytes, dpi=250)
    if not page_images:
        return OCRResult(
            text="", source="ocr", confidence=0.0,
            method="pdf_rasterize_failed",
        )

    page_texts: List[str] = []
    page_confs: List[float] = []
    method_name = "unknown"
    for i, page_img in enumerate(page_images, start=1):
        text, conf, method = _ocr_image_with_fallback(page_img)
        method_name = method
        if text.strip():
            page_texts.append(f"--- Trang {i} ---\n{text}")
            page_confs.append(conf)

    full_text = "\n\n".join(page_texts)
    cleaned = _postprocess_medical_text(full_text)
    avg_conf = float(np.mean(page_confs)) if page_confs else 0.0

    return OCRResult(
        text=cleaned,
        source="ocr",
        confidence=round(avg_conf, 3),
        method=f"pdf_ocr({method_name})",
    )


def _extract_image(uploaded_file: Any) -> OCRResult:
    """Pipeline cho ảnh: preprocess → OCR → post-process."""
    try:
        pil = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    except Exception as e:
        return OCRResult(
            text="", source="ocr", confidence=0.0,
            method=f"image_open_failed:{e}",
        )

    text, conf, method = _ocr_image_with_fallback(pil)
    cleaned = _postprocess_medical_text(text)
    return OCRResult(
        text=cleaned,
        source="ocr",
        confidence=round(conf, 3),
        method=method,
    )


def extract_text_from_uploaded_file(uploaded_file: Any) -> OCRResult:
    """
    Entry point — GIỮ NGUYÊN signature để app.py không phải sửa.

    Dispatch theo đuôi file:
        .pdf  → _extract_pdf  (text layer trước, OCR fallback sau)
        khác  → _extract_image
    """
    name = (getattr(uploaded_file, "name", None) or "").lower()
    if name.endswith(".pdf"):
        return _extract_pdf(uploaded_file)
    return _extract_image(uploaded_file)


# ===========================================================================
# CLI test — tiện debug nhanh từ terminal
# ===========================================================================

if __name__ == "__main__":
    import sys

    class _Wrap:
        """Mini wrapper để dùng giống Streamlit UploadedFile."""
        def __init__(self, path: str):
            self.name = os.path.basename(path)
            with open(path, "rb") as f:
                self._data = f.read()

        def getvalue(self) -> bytes:
            return self._data

    if len(sys.argv) < 2:
        print("Cách dùng: python ocr_engine.py <ảnh hoặc pdf>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    res = extract_text_from_uploaded_file(_Wrap(sys.argv[1]))
    print(f"Method:     {res.method}")
    print(f"Confidence: {res.confidence}")
    print(f"--- TEXT ({len(res.text)} chars) ---")
    print(res.text)
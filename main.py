import io
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import cv2
import easyocr
import fitz  # PyMuPDF
import numpy as np
from docx import Document
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

UPLOAD_DIR = Path("uploads")
OCR_DIR = Path("ocr_results")
UPLOAD_DIR.mkdir(exist_ok=True)
OCR_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".txt"}
MAX_FILE_SIZE_MB = 20

ocr_reader: easyocr.Reader = None


@asynccontextmanager
async def lifespan(app):
    global ocr_reader
    print("Loading EasyOCR model...")
    ocr_reader = easyocr.Reader(["en"], gpu=False)
    print("EasyOCR ready.")
    yield


class ExtractRequest(BaseModel):
    text: str
    fields: List[str]


app = FastAPI(title="Document Upload & OCR API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ── Image preprocessing ──────────────────────────────────────────────────────

def preprocess_for_ocr(img_array: np.ndarray) -> np.ndarray:
    """Enhance image quality before OCR: grayscale → denoise → CLAHE → sharpen."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if img_array.ndim == 3 else img_array.copy()

    # Upscale small images so fine characters are readable
    h, w = gray.shape
    if h < 1200 or w < 900:
        scale = max(1200 / h, 900 / w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Light denoising — preserves edges
    gray = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)

    # CLAHE: local contrast normalisation (fixes uneven lighting/shadows)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Unsharp mask — makes character edges crisper
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


# ── OCR text post-processing ─────────────────────────────────────────────────

def postprocess_ocr_text(text: str) -> str:
    """Fix common OCR misreads: ₹ symbol, O/0 confusion, email dots, etc."""

    # ₹ misread as '<' or '& ' before a digit
    text = re.sub(r'[<&]\s*(?=\d)', '₹', text)

    # ₹ misread as '8' at the very start of a price (e.g. 811,428 → ₹11,428)
    # Pattern: 8 followed immediately by digits + comma + digits (price shape)
    text = re.sub(r'\b8(\d{1,2},\d{3})', r'₹\1', text)

    # 'Rs.' or 'Rs ' or 'INR ' → ₹
    text = re.sub(r'\b(Rs\.?|INR)\s*', '₹', text, flags=re.IGNORECASE)

    # Capital O between digits → 0  (e.g. 12O00 → 12000)
    text = re.sub(r'(?<=\d)[Oo](?=\d)', '0', text)

    # Lowercase l between digits → 1  (e.g. l2,000 → 12,000)
    text = re.sub(r'(?<=\d)l(?=\d)', '1', text)
    text = re.sub(r'\bl(?=\d{2,})', '1', text)

    # Missing dot in email address domains  (@yoactiv com → @yoactiv.com)
    text = re.sub(r'(@[\w-]+)\s+(com|in|org|net|io|co)\b', r'\1.\2', text, flags=re.IGNORECASE)

    # Missing dot in standalone domain-like tokens  (yoactiv com → yoactiv.com)
    text = re.sub(r'\b([\w-]{3,})\s+(com|in|org|net|io)\b(?![\w.])', r'\1.\2', text, flags=re.IGNORECASE)

    # Collapse accidental double spaces
    text = re.sub(r'  +', ' ', text)

    return text


# ── Core OCR helpers ─────────────────────────────────────────────────────────

def ocr_image_array(img_array: np.ndarray) -> str:
    processed = preprocess_for_ocr(img_array)
    results = ocr_reader.readtext(processed)
    raw = "\n".join(text for _, text, conf in results if conf > 0.3)
    return postprocess_ocr_text(raw)


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)           # up from 200 → sharper symbols
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            text = ocr_image_array(np.array(img))
            pages_text.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(pages_text)

    elif ext in {".png", ".jpg", ".jpeg"}:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return ocr_image_array(np.array(img))

    elif ext in {".docx", ".doc"}:
        doc = Document(io.BytesIO(file_bytes))
        raw = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return postprocess_ocr_text(raw)

    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")

    return ""


def save_ocr_result(filename: str, text: str) -> Path:
    stem = Path(filename).stem
    out_path = OCR_DIR / f"{stem}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/upload", summary="Upload & OCR a document")
async def upload_file(file: UploadFile = File(...)):
    if not is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    # Save original file
    (UPLOAD_DIR / file.filename).write_bytes(contents)

    # Run OCR / text extraction
    extracted_text = extract_text(contents, file.filename)

    # Save OCR result to .txt
    ocr_path = save_ocr_result(file.filename, extracted_text)

    return JSONResponse(status_code=200, content={
        "filename": file.filename,
        "size_bytes": len(contents),
        "ocr_saved_to": str(ocr_path),
        "extracted_text": extracted_text,
    })


@app.post("/upload/multiple", summary="Upload & OCR multiple documents")
async def upload_multiple(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        if not is_allowed_file(file.filename):
            results.append({"filename": file.filename, "status": "rejected", "reason": "File type not allowed"})
            continue

        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            results.append({"filename": file.filename, "status": "rejected", "reason": "File too large"})
            continue

        (UPLOAD_DIR / file.filename).write_bytes(contents)
        extracted_text = extract_text(contents, file.filename)
        ocr_path = save_ocr_result(file.filename, extracted_text)

        results.append({
            "filename": file.filename,
            "status": "processed",
            "size_bytes": len(contents),
            "ocr_saved_to": str(ocr_path),
            "extracted_text": extracted_text,
        })

    return JSONResponse(status_code=200, content={"results": results})


@app.get("/files", summary="List uploaded files and their OCR results")
def list_files():
    files = [
        {
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "ocr_result": str(OCR_DIR / f"{f.stem}.txt") if (OCR_DIR / f"{f.stem}.txt").exists() else None,
        }
        for f in UPLOAD_DIR.iterdir() if f.is_file()
    ]
    return {"files": files, "count": len(files)}


def extract_single_field(lines: list[str], field: str) -> dict:
    field_lower = field.lower()
    field_words = [w for w in field_lower.split() if len(w) > 2]

    best_value, best_conf = None, 0.0

    for i, line in enumerate(lines):
        line_lower = line.lower()

        if field_lower in line_lower:
            conf_base = 0.93
        elif field_words and all(w in line_lower for w in field_words):
            conf_base = 0.80
        elif field_words and len(field_words) > 1:
            hit_ratio = sum(1 for w in field_words if w in line_lower) / len(field_words)
            if hit_ratio >= 0.6:
                conf_base = hit_ratio * 0.68
            else:
                continue
        else:
            continue

        same_line_split = re.split(r'[:=]\s*', line, maxsplit=1)
        if len(same_line_split) > 1 and same_line_split[1].strip():
            candidate = same_line_split[1].strip()
            conf = conf_base
        else:
            next_idx = i + 1
            while next_idx < len(lines) and not lines[next_idx].strip():
                next_idx += 1
            if next_idx < len(lines) and len(lines[next_idx]) < 200:
                candidate = lines[next_idx].strip()
                conf = conf_base * 0.87
            else:
                continue

        if conf > best_conf:
            best_value, best_conf = candidate, conf

    if best_value:
        return {"field": field, "value": best_value, "confidence": round(min(best_conf * 100, 99.0), 1)}
    return {"field": field, "value": "Not detected", "confidence": 0.0}


@app.post("/extract", summary="Extract specific fields from OCR text with confidence scores")
async def extract_fields(req: ExtractRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    if not req.fields:
        raise HTTPException(status_code=400, detail="fields list cannot be empty")

    lines = [l.strip() for l in req.text.splitlines()]
    results = [extract_single_field(lines, f) for f in req.fields]

    found = sum(1 for r in results if r["value"] != "Not detected")
    overall = round(sum(r["confidence"] for r in results) / len(results), 1) if results else 0

    return {
        "extractions": results,
        "overall_accuracy": overall,
        "fields_found": found,
        "fields_total": len(results),
    }


@app.get("/health")
def health():
    return {"status": "ok", "ocr_ready": ocr_reader is not None}

# DocScan — Intelligent Document Processing (IDP)

> Upload any document, define the fields you want, and get structured, validated data with confidence scores — fully offline. No cloud, no API keys, no cost.

DocScan is a locally-running **OCR + field extraction** web app. It turns scanned invoices, receipts, IDs and forms into clean, structured data — and now **validates** every extracted value so bad data gets flagged instead of silently trusted.

---

## 1. Project Summary

| | |
|---|---|
| **What it is** | An offline Intelligent Document Processing (IDP) tool |
| **What it does** | Extracts text from documents, pulls out user-defined fields, scores confidence, and validates results |
| **Who it's for** | Anyone processing invoices, receipts, ID cards, and structured forms |
| **Key promise** | 100% offline — no cloud, no API keys, no per-page cost, full data privacy |
| **Built with** | Python · FastAPI · EasyOCR · OpenCV · vanilla JS frontend |

---

## 2. Key Features

- 📤 **Multi-format upload** — PDF, PNG, JPG, DOCX/DOC, XLSX/XLS, TXT
- 🖼️ **Smart image preprocessing** — 5-step OpenCV pipeline for clean OCR input
- 🔍 **Offline OCR** — EasyOCR (CRAFT detector + CNN+RNN recognizer), runs on CPU, no internet needed
- 🧹 **Auto-correction** — regex fixes for common OCR misreads (₹, 0/O, 1/l, emails)
- 🎯 **Custom field extraction** — define any fields; keyword-proximity search finds them
- 📊 **Confidence scoring** — every value gets a 0–99% reliability score
- ✅ **Validation layer** — format + confidence checks flag values as Valid / Review / Invalid / Missing
- 📦 **Batch processing** — upload many documents, processed one-by-one with a live status pipeline
- ⚡ **MD5 caching** — the same file is never OCR'd twice
- 🪵 **Full diagnostic logging** — frontend console + backend logs to trace any error

---

## 3. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend API** | FastAPI | REST endpoints, serves the UI |
| **Server** | Uvicorn (ASGI) | Runs the FastAPI app |
| **OCR Engine** | EasyOCR | CRAFT (CNN) detection + CRNN (CNN+RNN+CTC) recognition, offline on CPU |
| **PDF Handling** | PyMuPDF (fitz) | Renders PDF pages to 300 DPI images |
| **Image Processing** | OpenCV | Grayscale, denoise, contrast, sharpening |
| **Image I/O** | Pillow (PIL) | Image loading & format conversion |
| **Numerics** | NumPy | Image array operations |
| **Office Files** | python-docx, openpyxl | Read Word & Excel files (no OCR needed) |
| **Frontend** | HTML + CSS + Vanilla JS | Single-page UI, no framework |
| **Validation** | Python regex | Format checks for dates, emails, phones, amounts |

---

## 4. System Architecture

```
┌─────────────┐     HTTP/JSON      ┌──────────────────────────────┐
│   Browser   │  ───────────────▶  │        FastAPI Backend        │
│  (index.html)│                    │                               │
│             │                    │  /upload   → OCR pipeline      │
│  - Upload   │  ◀───────────────  │  /extract  → field + validate  │
│  - Pipeline │                    │  /files    → list documents    │
│  - Results  │                    │  /health   → status check      │
│  - Badges   │                    └──────────────────────────────┘
└─────────────┘                                  │
                                                 ▼
                          ┌──────────────────────────────────────┐
                          │  OCR Pipeline (per document)           │
                          │  Preprocess → EasyOCR → Post-process   │
                          └──────────────────────────────────────┘
                                                 │
                                  uploads/   ◀───┴───▶   ocr_results/
                               (raw files)            (extracted text)
```

---

## 5. The Processing Pipeline (Step by Step)

The UI shows four live stages for every document: **Uploaded → Preprocess → OCR → Result**.

### Stage 1 — Upload & Validation
- File checked against allowed types and 20 MB size limit
- Valid files saved to `uploads/`; invalid files rejected with HTTP 400

### Stage 2 — Format Detection
Each format takes a different extraction path:

| Format | Method |
|---|---|
| PDF | Render each page → image (PyMuPDF, 300 DPI) → OCR |
| PNG / JPG | Load image → OCR |
| DOCX / DOC | Read paragraphs + tables (python-docx) — no OCR |
| XLSX / XLS | Read all sheets & rows (openpyxl) — no OCR |
| TXT | Decode UTF-8 directly — no OCR |

### Stage 3 — Image Preprocessing (OpenCV, 5 steps) — Deep Dive

> **Core idea:** OCR accuracy is decided *before* the OCR engine even runs. A neural network can only read what it can clearly see. Real-world scans are blurry, noisy, unevenly lit, skewed, or low-resolution. This 5-step OpenCV pipeline repairs image quality so EasyOCR receives the cleanest possible input. *(Code: `preprocess_for_ocr()` in `main.py`.)*

```
RGB image ─▶ Grayscale ─▶ Upscale ─▶ Denoise ─▶ CLAHE ─▶ Unsharp Mask ─▶ clean image to OCR
```

#### Step 1 — Grayscale Conversion
- **Code:** `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
- **What:** collapses 3 colour channels (R, G, B) into 1 brightness channel.
- **Why:** text is defined by *contrast* (dark ink on light paper), not colour. Colour adds 3× the data and random colour noise with zero benefit to reading.
- **Effect:** 3× less data to process, faster and cleaner OCR.

#### Step 2 — Adaptive Upscaling (Cubic Interpolation)
- **Code:** if `height < 1200` or `width < 900`, resize with `INTER_CUBIC` by `scale = max(1200/h, 900/w)`
- **What:** enlarges small images to a minimum working resolution using cubic interpolation (smooth 16-pixel sampling).
- **Why:** tiny characters and fine symbols (₹, commas, decimal dots) collapse into a few pixels and blur together. OCR needs **enough pixels per character** to recognise its shape.
- **Why cubic (not nearest/linear):** cubic produces smooth edges instead of jagged "staircase" pixels, which the OCR network reads far more reliably.
- **Effect:** small/low-DPI scans become legible; large images are left untouched (no wasted work).

#### Step 3 — Denoising (Non-Local Means)
- **Code:** `cv2.fastNlMeansDenoising(h=8, templateWindowSize=7, searchWindowSize=21)`
- **What:** removes random speckle/grain from the scan.
- **How (Non-Local Means):** for each pixel it searches a **21×21 window** for **7×7 patches** that look similar, and averages them. Because it averages *similar regions* (not just neighbours), it smooths noise **without blurring character edges**.
- **`h=8`** is the filter strength — tuned to clean background grain while preserving thin strokes.
- **Why it matters:** noise creates fake "specks" the OCR can mistake for punctuation or merge into letters.

#### Step 4 — CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Code:** `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`
- **What:** normalises contrast **locally**, tile by tile, across an **8×8 grid** of the image.
- **Why local, not global:** a scanned page often has a bright corner and a shadowed corner. Global contrast stretching would blow out the bright area while leaving the shadow unreadable. CLAHE equalises each tile independently, so faint text in shadows becomes readable *and* bright text isn't overexposed.
- **`clipLimit=2.0`:** caps how aggressively contrast is boosted in any tile, preventing noise from being amplified into artefacts.
- **Effect:** even, readable contrast across the whole page regardless of lighting.

#### Step 5 — Unsharp Mask (Edge Sharpening)
- **Code:** `blur = GaussianBlur(gray,(0,0),3)` then `addWeighted(gray, 1.5, blur, -0.5, 0)`
- **What:** sharpens character edges using the classic *unsharp mask* technique.
- **How:** it creates a blurred copy, then computes `1.5 × original − 0.5 × blurred`. Subtracting the blur amplifies exactly the high-frequency detail — the **edges** of each stroke.
- **Why:** crisp edges let the OCR network separate look-alike glyphs: `0`vs`O`, `1`vs`l`, `₹`vs`8`, `.`vs`,`.
- **Final step:** `np.clip(0,255)` keeps pixel values valid after the maths.

#### PDFs get extra resolution first
> Before this pipeline runs, **PDF pages are rendered at 300 DPI** by PyMuPDF (`page.get_pixmap(dpi=300)`). A PDF has no inherent pixels — DPI decides how many it gets. At 72 DPI the ₹ symbol is a smudge; at 300 DPI it's a sharp, readable glyph. Higher DPI = more pixels per character = better OCR.

#### Summary table

| Step | OpenCV function | Real-world problem it fixes | Key parameter |
|---|---|---|---|
| Grayscale | `cvtColor` | Colour noise & wasted data | RGB → GRAY |
| Upscale | `resize` | Tiny, blurry characters | min 1200×900, INTER_CUBIC |
| Denoise | `fastNlMeansDenoising` | Scanner speckle/grain | h=8, 7×7 / 21×21 windows |
| CLAHE | `createCLAHE` | Uneven lighting / shadows | clipLimit=2.0, 8×8 tiles |
| Unsharp Mask | `GaussianBlur` + `addWeighted` | Soft, fused character edges | 1.5 / −0.5 weights |

### Stage 4 — EasyOCR Recognition — Deep Dive

> **Key insight:** "OCR" is really **two different problems** — *finding* text on the page, and *reading* it. EasyOCR uses a separate, specialised deep-learning model for each. *(Code: `ocr_reader.readtext(processed)`; reader loaded once at startup as `easyocr.Reader(["en"], gpu=False)`.)*

```
Preprocessed image
        │
        ▼
 ┌──────────────────┐        ┌─────────────────────────────────────────┐
 │  DETECTION       │        │  RECOGNITION (per detected text box)      │
 │  CRAFT  (CNN)    │ ─────▶ │  CRNN = CNN ─▶ RNN (BiLSTM) ─▶ CTC        │
 │  "where is text?"│ boxes  │  "what does it say?"                      │
 └──────────────────┘        └─────────────────────────────────────────┘
        │                                        │
        ▼                                        ▼
  word bounding boxes                  text strings + confidence
```

| Stage | Model | Network type | Question it answers |
|---|---|---|---|
| **1. Detection** | **CRAFT** | CNN | *Where* is the text? → bounding boxes |
| **2. Recognition** | **CRNN** | CNN → RNN → CTC | *What* does each box say? → characters |

---

#### Part 1 — CRAFT: the text **detector** (a CNN)

**CRAFT** = **C**haracter **R**egion **A**wareness **F**or **T**ext detection. It's the deep-learning model EasyOCR uses to locate text anywhere on the image.

- **Type:** a fully-convolutional neural network (CNN) built on a **VGG-16** backbone (with batch normalization).
- **What makes it special — it works at the *character* level.** Instead of trying to draw one big box around a whole line, CRAFT predicts **two heat maps** for every pixel:
  1. **Region score** — "how likely is this pixel the *centre of a character*?"
  2. **Affinity score** — "how likely do *two neighbouring characters belong to the same word*?"
- **How boxes are formed:** high region scores mark individual characters; the affinity score then *glues* adjacent characters into words. This bottom-up approach is why CRAFT handles **curved text, rotated text, different fonts and spacing** much better than older box-based detectors.
- **Output:** clean word/line **bounding boxes**, which are cropped and passed one-by-one to the recognizer.

> **In one line for the slide:** *CRAFT is a CNN that finds text by detecting each character and the links between them, so it works even on curved or irregular layouts.*

> **Package note:** CRAFT (detector) and CRNN (recognizer) are **bundled inside the `easyocr` package** — there's no separate install. On first run, EasyOCR automatically downloads these pre-trained model weights (`craft_mlt_25k` for detection + the English recognition model) and caches them locally for fully offline use afterwards.

---

#### Part 2 — CRNN: the text **recognizer** (CNN + RNN + CTC)

Each cropped text box goes through a **CRNN** (Convolutional **R**ecurrent Neural Network) — three sub-stages chained together:

**① CNN — ResNet feature extractor (the "eyes")**
- Slides across the cropped text image and extracts **visual features** — strokes, curves, loops, edges.
- Converts the image into a sequence of feature columns reading left → right.
- *Job: turn pixels into meaningful shape-features.*

**② RNN — Bidirectional LSTM (the "context / memory")**
- An RNN processes data **as a sequence**, remembering what came before and after.
- **Bidirectional** = it reads the word **both left-to-right and right-to-left**, so each character is understood *in context of its neighbours*.
- This is what lets it resolve ambiguous glyphs: an `l` next to other digits is probably `1`; `rn` together vs a single `m`; `cl` vs `d`.
- *Job: understand the order and context of characters, not just isolated shapes.*

**③ CTC — Connectionist Temporal Classification (the "decoder")**
- **Problem it solves:** the CNN produces, say, 50 feature columns, but the word has only 8 letters — and we never told the model *which column maps to which letter* (the alignment problem).
- **CTC** handles this automatically: it allows repeats and "blank" markers, then collapses them (`hh-e-l-ll-o → hello`) to produce the final string.
- It also yields the **confidence score** for the prediction.
- *Job: turn the variable-length feature/letter sequence into clean final text + confidence.*

```
 Cropped text box  "₹1,800"
        │
        ▼
 ┌──────────┐   features   ┌────────────────┐  contextual  ┌─────────┐
 │ CNN      │ ───────────▶ │ RNN  (BiLSTM)  │ ───────────▶ │  CTC    │ ─▶ "₹1,800"
 │ ResNet   │  per column  │ reads sequence │  characters  │ decode  │    + confidence
 └──────────┘              └────────────────┘              └─────────┘
   the eyes                   the memory                     the speller
```

---

#### Why this CNN + RNN combination wins

| Network | Strength | If used alone… |
|---|---|---|
| **CNN** | Recognises *shapes* of characters | …would misread look-alikes with no context (`l`/`1`, `O`/`0`) |
| **RNN (LSTM)** | Understands *sequence & context* | …has no way to see the image |
| **CTC** | Aligns features to text of unknown length | …needed because we can't hand-label every pixel column |

Together: **CNN sees → RNN reasons about order → CTC writes the answer.** This is the standard, battle-tested architecture for reading text "in the wild."

---

#### In DocScan specifically
- Runs with `gpu=False` → **pure CPU, fully offline**, no GPU or internet required.
- The English model loads **once at server startup** (in a background thread) so requests aren't blocked.
- After recognition, DocScan keeps only regions with **confidence > 0.3 (30%)** and joins them into text lines, which then go to post-processing (Stage 5).

### Stage 5 — OCR Post-Processing (Regex Fixes)
Common, predictable OCR mistakes are auto-corrected:

| OCR Misread | Corrected | Rule |
|---|---|---|
| `< 285.71` | `₹285.71` | `<`/`&` before a digit → ₹ |
| `811,428` | `₹11,428` | leading `8` in price shape → ₹ |
| `Rs. 500` / `INR 500` | `₹500` | currency words → ₹ |
| `12O00` | `12000` | `O` between digits → `0` |
| `l2,000` | `12,000` | leading `l` before digits → `1` |
| `@gmail com` | `@gmail.com` | missing dot in email domain |

### Stage 6 — Save & Return
- Cleaned text saved to `ocr_results/<filename>.txt` (audit trail)
- Text returned to the browser in the upload JSON response

---

## 6. Field Extraction (Keyword-Proximity Algorithm)

Instead of a heavy LLM, DocScan uses a fast, offline keyword-proximity search — instant, no model download, accurate on structured documents.

**How it works:**
1. For each field name, scan every OCR line
2. Match the field as **whole words only** (so "ID" won't match inside "paid")
3. Pull the value from after `:` or `=` on the same line
4. If no value on that line, use the next non-empty line
5. Pick the best candidate by confidence

**Smart rule for totals:** numeric fields (total, amount, tax, GST, price…) prefer the **last** strong match — because grand totals sit at the *bottom* of invoices, not the top.

### Confidence Scoring

| Match type | Base confidence |
|---|---|
| Exact label match (`ID:` for field "ID") | 98% |
| Field phrase found in the label | 95% |
| Field phrase found anywhere on line | 90% |
| All field words present | 80% |
| 60%+ of field words present | partial (≈40–68%) |
| Value on next line (not same line) | ×0.87 penalty |
| Not found | 0% → "Not detected" |

*(Confidence is capped at 99%.)*

---

## 7. Validation Layer ✅ (Quality Gate)

After extraction, **every value is validated** so wrong data is caught — even when OCR was confident.

**Two independent signals:**
1. **Format check** — does the value match the field's expected type?
2. **Confidence check** — is the OCR sure enough (≥ 60%)?

**Expected type is inferred from the field name:**

| Field name contains… | Expected type | Check |
|---|---|---|
| email, e-mail | Email | matches `x@y.z` |
| date, dob, expiry, issued | Date | matches a date pattern |
| phone, mobile, contact | Phone | 7–15 digits |
| total, amount, tax, GST, price… | Amount | contains a number |
| anything else | Text | no format rule |

**Four possible verdicts:**

| Status | Badge | Meaning |
|---|---|---|
| **Valid** | ✓ green | Found, correct format, high confidence |
| **Review** | ⚠ amber | Correct format but low confidence — please verify |
| **Invalid** | ✕ red | Wrong format for the field |
| **Missing** | — grey | Nothing was extracted |

**Why two signals matter:**
- `"Slam Fitness"` for a *date* field at 95% confidence → **Invalid** (format catches what confidence can't)
- `"Niranjan"` for a name at 45% confidence → **Review** (looks fine, but unsure)

---

## 8. User Interface Highlights

- **Drag-and-drop upload** with a live 4-step pipeline (Uploaded → Preprocess → OCR → Result)
- **Batch queue** — multiple documents shown with per-document status
- **Stats panel** — word count, character count, page count
- **Raw Text tab** — full OCR output, page by page
- **Extracted tab** — field cards with confidence bars + validation badges
- **Accuracy ring** — overall confidence at a glance, plus "N need review"
- **Copy / Download** — export results (value + confidence + validation status)

---

## 9. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/upload` | Upload & OCR a single document |
| `POST` | `/upload/multiple` | Upload & OCR several documents at once |
| `POST` | `/extract` | Extract & validate fields from OCR text |
| `GET` | `/files` | List uploaded files and their OCR results |
| `GET` | `/health` | Service status + OCR readiness + cache size |

---

## 10. Performance & Reliability

- **MD5 caching** — identical files return instantly, never re-OCR'd
- **Non-blocking OCR** — CPU-heavy work runs in a thread pool, server stays responsive
- **Startup model load** — EasyOCR loads once at boot (English, CPU mode)
- **Diagnostic logging** —
  - *Frontend:* tagged console logs (`[DocScan:doc]`, `[DocScan:extract]`…) + `DocScanLog.dump()`
  - *Backend:* timestamped Python logs with per-page OCR tracing and full stack traces on failure
- **Graceful errors** — failed pages/documents are flagged, not crashed

---

## 11. Running the App

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
uvicorn main:app --reload

# 3. Open in browser
http://localhost:8000
```

---

## 12. Project Structure

```
IDP-Preparation/
├── main.py            # FastAPI backend — OCR, extraction & validation logic
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── workflow.mmd       # Mermaid flowchart of the pipeline
├── static/
│   └── index.html     # Frontend UI (HTML + CSS + JS)
├── uploads/           # Saved uploaded documents
└── ocr_results/       # Saved OCR text output
```

---

## 13. Why DocScan? (Design Choices)

| Choice | Reason |
|---|---|
| **Offline OCR (EasyOCR)** | Privacy, zero cost, no API limits |
| **Keyword extraction (not LLM)** | Instant, no model download, accurate on structured docs |
| **OpenCV preprocessing** | Turns poor scans into OCR-ready images |
| **Validation layer** | Catches errors confidence alone would miss |
| **Vanilla JS frontend** | Lightweight, no build step, easy to host |
| **Single-file backend** | Easy to read, run, and present |

---

## 14. Future Enhancements

- Dedicated **Validate** stage in the pipeline stepper
- Cross-field rules (e.g. `subtotal + tax ≈ total`)
- Multi-language OCR (EasyOCR supports 80+ languages)
- Export to CSV / JSON / Excel
- LLM-assisted extraction for unstructured documents
- User-saved field templates per document type

---

### One-line pitch
**DocScan turns any scanned document into clean, validated, structured data — instantly and entirely offline.**

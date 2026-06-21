# DocScan — Intelligent Document Processing

DocScan is a locally-running document OCR and field extraction application built for a hackathon. Upload any document, define the fields you want, and get structured data with confidence scores — no cloud, no API keys, no cost.

---

## How It Works — Full Flow Explained

### 1. Upload Document

**What:** The user uploads a file through the browser UI (drag and drop or file picker).

**Why:** The system needs to receive the raw document before any processing can begin. Supported formats are PDF, PNG, JPG, DOCX, DOC, and TXT — covering the most common document types used in real-world workflows like invoices, receipts, and ID cards.

**What happens:**
- File type is checked against the allowed list
- File size is checked (max 20MB)
- If invalid → HTTP 400 error is returned immediately
- If valid → file is saved to the `uploads/` folder on disk

---

### 2. File Type Detection

**What:** The system checks the file extension to decide how to extract text from it.

**Why:** Different file formats need completely different approaches. A PDF cannot be read the same way as a plain text file. Choosing the wrong method would either crash or return garbage.

| Format | Method used |
|---|---|
| PDF | Convert each page to an image using PyMuPDF, then run OCR |
| PNG / JPG | Load image directly, then run OCR |
| DOCX / DOC | Use python-docx to read text from paragraphs — no OCR needed |
| TXT | Decode UTF-8 bytes directly — no OCR needed |

---

### 3. Image Preprocessing (OpenCV Pipeline)

**What:** Before OCR runs, the image goes through a 5-step enhancement pipeline using OpenCV.

**Why:** EasyOCR works best on clean, sharp, evenly-lit images. Real-world scanned documents are often blurry, noisy, unevenly lit, or too small. Feeding a bad image into OCR produces bad text. Preprocessing fixes the image quality so the OCR engine has the best possible input.

#### Step 1 — Grayscale Conversion
- Converts the colour (RGB) image to black and white
- EasyOCR only needs brightness information to read text, not colour
- Colour channels add unnecessary noise and slow down processing

#### Step 2 — Upscale (INTER_CUBIC)
- If the image is smaller than 1200×900 pixels, it is enlarged using cubic interpolation
- Small images make fine characters like ₹, commas, and dots blur together
- Enlarging gives EasyOCR more pixels per character to work with, improving accuracy on small print

#### Step 3 — Denoising (fastNlMeans)
- Scanned documents have random speckle and grain (noise) across the page
- fastNlMeans smooths out that background noise without blurring the edges of letters
- Result: clean background, sharp character edges

#### Step 4 — CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Scanned pages often have uneven lighting — one corner is brighter, another is darker
- CLAHE fixes this by normalising contrast in small local 8×8 tile blocks rather than the whole image
- This makes faint or shadow-covered text readable without overexposing the bright areas

#### Step 5 — Unsharp Mask
- Makes character edges crisper by subtracting a blurred version of the image from the original
- This amplifies the edges of each character
- Helps EasyOCR distinguish similar-looking glyphs: `0` vs `O`, `1` vs `l`, `₹` vs `8`

> PDFs specifically are rendered at **300 DPI** (high resolution) before this pipeline runs, using PyMuPDF. Higher DPI means more pixels per character — fine symbols like ₹ that are invisible at 72 DPI become clearly readable at 300 DPI.

---

### 4. EasyOCR — Text Extraction

**What:** EasyOCR runs on the preprocessed image and extracts all visible text.

**Why:** EasyOCR is a free, open-source, offline OCR engine based on a CNN (Convolutional Neural Network). It does not require a GPU, does not call any external API, and supports 80+ languages. It returns each detected text region along with a confidence score.

**What happens:**
- EasyOCR scans the image region by region
- Returns a list of (bounding box, text, confidence) for each detected region
- Only regions with confidence above 30% are kept
- All text regions are joined into lines

---

### 5. OCR Post-Processing (Regex Corrections)

**What:** After OCR runs, common misreads are automatically corrected using regex rules.

**Why:** Even with preprocessing, OCR engines make predictable mistakes on certain characters — especially Indian currency symbols, digits, and email addresses. Post-processing catches and fixes these known errors automatically.

| OCR Misread | Corrected To | Rule |
|---|---|---|
| `< 285.71` | `₹285.71` | `<` before a digit → ₹ |
| `811,428.57` | `₹11,428.57` | `8` at start of price shape → ₹ |
| `Rs. 500` or `INR 500` | `₹500` | Rs / INR → ₹ |
| `12O00` | `12000` | Capital O between digits → 0 |
| `l2,000` | `12,000` | Lowercase l before digits → 1 |
| `@gmail com` | `@gmail.com` | Missing dot in email domain |

---

### 6. Save OCR Result

**What:** The cleaned OCR text is saved to a `.txt` file in the `ocr_results/` folder.

**Why:** Saving the result means you can review the raw OCR output at any time without re-uploading the document. It also serves as an audit trail — you can compare what OCR produced against what field extraction picked up.

---

### 7. Return Text to Browser

**What:** The cleaned text is sent back to the browser as a JSON response.

**Why:** The frontend needs the OCR text to display in the Raw Text tab and to send to the `/extract` endpoint for field extraction. Returning it in the upload response means only one network call is needed.

---

### 8. Field Extraction — Keyword Proximity Search

**What:** The user defines field names in the UI (e.g. Invoice Number, Customer Name, GST No). The extractor scans each line of the OCR text to find a matching line and pulls the value.

**Why:** Instead of using a heavy LLM or ML model for extraction, a lightweight keyword proximity algorithm is used. This runs instantly, works offline, needs no model download, and is accurate enough for structured documents like invoices and receipts.

**How the algorithm works:**
1. Each OCR text line is checked against the field name
2. If the field name is found on a line, the value is pulled from after the `:` or `=` on the same line
3. If no value is on the same line, the next non-empty line is used as the value

---

### 9. Confidence Scoring

**What:** Every extracted field gets a confidence score between 0% and 99%.

**Why:** Confidence scores tell the user how reliable each extracted value is, so they know which fields to verify manually and which to trust automatically.

| Match type | Base confidence |
|---|---|
| Exact field phrase found on line | 93% |
| All words of field name found on line | 80% |
| 60% or more of words found | 60–68% |
| Value on same line after `:` | no change |
| Value on next line | −13% penalty |
| Field not found at all | 0% — shown as "Not detected" |

---

### 10. Display Results in UI

**What:** The browser shows extracted field values alongside colour-coded confidence bars.

**Why:** Visual confidence indicators let users instantly judge result quality without reading numbers.

| Confidence | Bar colour | Meaning |
|---|---|---|
| 70% and above | Green | High confidence — reliable |
| 40% to 70% | Amber | Medium — worth checking |
| Below 40% | Red | Low — verify manually |

The overall accuracy shown at the top is the average confidence across all fields.

---

## Tech Stack

| Library | Purpose |
|---|---|
| FastAPI | REST API — handles upload, extraction, and serves the UI |
| EasyOCR | CNN-based OCR engine — runs fully offline on CPU |
| PyMuPDF | Converts PDF pages to high-resolution images |
| OpenCV | Image preprocessing pipeline |
| Pillow | Image loading and format conversion |
| python-docx | Reads text from DOCX files |
| Uvicorn | ASGI server to run FastAPI |

---

## Running the App

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Project Structure

```
IDP-Preparation/
├── main.py           # FastAPI backend — all OCR and extraction logic
├── requirements.txt  # Python dependencies
├── workflow.mmd      # Mermaid flowchart of the full pipeline
├── workflow.md       # Same flowchart wrapped in markdown
├── static/
│   └── index.html    # Frontend UI
├── uploads/          # Saved uploaded documents
└── ocr_results/      # Saved OCR text output files
```

# DocScan — Intelligent Document Processing (IDP)

> Upload any document, define the fields you want, and get structured, validated data with confidence scores. **Offline-first** OCR + extraction, with an optional **GPT-4.1 AI fallback** that understands context when a simple label match isn't enough.

DocScan is a locally-running **OCR + field extraction** web app. It turns scanned invoices, receipts, IDs and forms into clean, structured data — **validates** every extracted value so bad data gets flagged, and uses a **two-tier extraction engine** that self-corrects: fast offline matching first, then a GPT-4.1 fallback for anything the offline tier misses or gets wrong.

---

## 1. Project Summary

| | |
|---|---|
| **What it is** | An Intelligent Document Processing (IDP) tool with a two-tier extraction engine |
| **What it does** | Extracts text, pulls out user-defined fields, scores confidence, validates results, and recovers/corrects hard fields with GPT-4.1 |
| **Who it's for** | Anyone processing invoices, receipts, ID cards, and structured forms |
| **Key promise** | Offline-first (OCR + matching run locally); GPT-4.1 is an optional fallback only for the fields the offline tier can't handle |
| **Built with** | Python · FastAPI · EasyOCR · OpenCV · Azure OpenAI (GPT-4.1) · vanilla JS frontend |

---

## 2. Key Features

- 📤 **Multi-format upload** — PDF, PNG, JPG, DOCX/DOC, XLSX/XLS, TXT
- 🖼️ **Smart image preprocessing** — 5-step OpenCV pipeline for clean OCR input
- 🔍 **Offline OCR** — EasyOCR (CRAFT detector + CNN+RNN recognizer), runs on CPU, no internet needed
- 🧹 **Auto-correction** — regex fixes for common OCR misreads (₹, 0/O, 1/l, emails)
- 🎯 **Two-tier field extraction** — fast offline keyword-proximity first, **GPT-4.1 AI fallback** second
- 🧠 **Context-aware AI recovery** — GPT-4.1 finds values with no label ("Mr. Ravi is the patient" → name = "Mr. Ravi")
- 🔁 **Self-correcting** — a low-confidence or suspicious offline match is re-checked by GPT-4.1, which **overrides** it if wrong
- 📊 **Confidence scoring** — every value gets a 0–99% score, with a **before → after** transition when AI corrects it
- ✅ **Validation layer** — format + confidence checks flag values as Valid / Review / Invalid / Missing
- 🔬 **Live two-tier flow** — the UI shows "Label matching → GPT-4.1 fallback" happening in real time, with `by label` / `by AI` tags per field
- 📦 **Batch processing** — upload many documents, processed one-by-one with a live status pipeline
- ⚡ **MD5 caching + per-doc result cache** — same file never re-OCR'd; revisiting a done doc is instant
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
| **AI Fallback** | Azure OpenAI — GPT-4.1 | Context-aware extraction for fields the offline tier misses or gets wrong |
| **Config** | python-dotenv | Loads API keys from a git-ignored `.env` (never hardcoded) |
| **Frontend** | HTML + CSS + Vanilla JS | Single-page UI, no framework |
| **Validation** | Python regex | Format checks for dates, emails, phones, amounts |

---

## 4. System Architecture

```
┌─────────────┐     HTTP/JSON      ┌──────────────────────────────────┐
│   Browser   │  ───────────────▶  │          FastAPI Backend          │
│ (index.html)│                    │                                   │
│  - Upload   │                    │  /upload      → OCR pipeline       │
│  - Pipeline │  ◀───────────────  │  /extract     → Tier 1: keyword    │
│  - 2-tier   │                    │  /extract/llm → Tier 2: GPT-4.1    │
│    flow UI  │                    │  /files · /health                  │
│  - Badges   │                    └──────────────────────────────────┘
└─────────────┘                            │                  │
                                           ▼                  ▼
                  ┌────────────────────────────────┐   ┌──────────────────────┐
                  │  OCR Pipeline (per document)     │   │  GPT-4.1 (Azure)      │
                  │  Preprocess → EasyOCR → Post-fix │   │  only for missing /   │
                  └────────────────────────────────┘   │  suspicious fields    │
                                  │                      └──────────────────────┘
                   uploads/   ◀───┴───▶   ocr_results/        (optional fallback)
                (raw files)            (extracted text)
```

**Offline-first by design:** OCR and Tier-1 keyword matching run entirely locally. The GPT-4.1 call (`/extract/llm`) fires **only** for fields the offline tier couldn't find or matched with low confidence — minimizing what leaves the machine. With no API key configured, DocScan runs fully offline (Tier 1 only).

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

## 6. Two-Tier Field Extraction Engine ⭐

DocScan extracts fields in **two tiers**. The fast, free, offline tier handles the easy cases; the AI tier handles only what the first tier can't — so you get the best of both: **speed + privacy by default, intelligence when needed.**

```
                 USER FIELDS  e.g. [Customer Name, GST No, Address]
                              │
                              ▼
        ┌──────────────────────────────────────────────────────┐
        │  TIER 1 — Keyword-Proximity  (offline · instant · free) │
        │  scans OCR lines for the field label, grabs the value   │
        └──────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴────────────────┐
        found, high confidence            missing  OR  low-confidence / suspicious
              │                                 │
              ▼                                 ▼
         keep it (tag: by label)    ┌──────────────────────────────────────────┐
                                    │  TIER 2 — GPT-4.1 fallback (Azure OpenAI)  │
                                    │  understands meaning & context, no label   │
                                    │  needed — its answer OVERRIDES a wrong grab │
                                    └──────────────────────────────────────────┘
                                                  │
                                                  ▼
                                       recovered/corrected (tag: by AI)
                              │
                              ▼
                   VALIDATION  →  Valid / Review / Invalid / Missing  →  UI cards
```

### Tier 1 — Keyword-Proximity (offline, instant, free)

For structured documents (invoices, forms) where fields have labels:
1. For each field name, scan every OCR line
2. Match the field as **whole words only** (so "ID" won't match inside "paid")
3. Pull the value from after `:` or `=` on the same line
4. If no value on that line, use the next non-empty line
5. Pick the best candidate by confidence

**Smart rule for totals:** numeric fields (total, amount, tax, GST, price…) prefer the **last** strong match — grand totals sit at the *bottom* of invoices.

| Match type | Base confidence |
|---|---|
| Exact label match (`ID:` for field "ID") | 98% |
| Field phrase found in the label | 95% |
| Field phrase found anywhere on line | 90% |
| All field words present | 80% |
| 60%+ of field words present | partial (≈40–68%) |
| Value on next line (not same line) | ×0.87 penalty |
| Not found | 0% → "Not detected" |

### Tier 2 — GPT-4.1 AI Fallback (context-aware)

Keyword matching fails on two kinds of fields, and GPT-4.1 fixes both:

| Problem Tier 1 has | Example | GPT-4.1 result |
|---|---|---|
| **No label present** — value is buried in prose | *"Mr. Ravi is the patient"* → field **Name** | finds **"Ravi"** with no label |
| **Confident but WRONG** — matched the wrong text | field **Name** grabs a table header → returns **"Primary?"** | **overrides** with the real name **"Rodney James Foster"** |

**When Tier 2 runs** (only when needed, to minimize cost & data exposure):
- the field was **Not detected** by Tier 1, **OR**
- the Tier-1 match is **below 90% confidence**, **OR**
- the Tier-1 value **failed validation** (wrong format)

**How it works:**
- All such fields are sent in **one** request to GPT-4.1 (`/extract/llm`) with a carefully engineered system prompt
- The prompt enforces: extract the value only (no labels), correct obvious OCR errors, **never hallucinate** ("a wrong value is worse than Not detected"), and return calibrated 0–100 confidence
- Returns strict JSON `{field: {value, confidence}}`; if GPT-4.1 finds a real value it **replaces** the Tier-1 result

**The self-correcting moment (great demo):** when Tier-1 grabs `"Primary?"` for a Name field at 85%, that's below the 90% threshold → GPT-4.1 re-checks → returns `"Rodney James Foster"` at 99%. The card shows the confidence transition **85% → 99%** and a green **AI** badge.

### Confidence scoring

Tier-1 confidence is rule-based (table above). Tier-2 confidence is GPT-4.1's own self-reported score. **Both are capped at 99%** (no extraction is ever truly 100% certain). *Note: LLM self-reported confidence is an estimate, not a measured accuracy.*

### Privacy & cost model

| | Tier 1 (keyword) | Tier 2 (GPT-4.1) |
|---|---|---|
| Where it runs | Locally | Azure OpenAI |
| Data leaves machine? | Never | Only the fields it can't solve |
| Cost | Free | Per-token (only on fallback) |
| Needs internet? | No | Yes |
| If no API key set | ✅ runs | gracefully skipped — app stays fully offline |

> **Why Azure OpenAI (not OpenAI direct):** Azure keeps data inside the organisation's own tenant and does not train on it — the right choice for internal-confidential documents.

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
- **Batch queue** — multiple documents shown with per-document status; **↻ retry** on any failed doc (no re-upload)
- **Live two-tier flow strip** — watch *"Label matching → GPT-4.1 fallback"* happen in real time, with a spinning/✓ state per tier
- **Method breakdown pills** — *"3 by label · 2 by AI"* shows which engine found each field
- **Per-field source tag** — a green **AI** badge marks values recovered/corrected by GPT-4.1
- **Confidence transition** — when AI overrides a wrong match, the bar animates *85% → 99%* (old struck-through)
- **Stats panel** — word count, character count, page count
- **Raw Text tab** — full OCR output, page by page
- **Extracted tab** — field cards with confidence bars + validation badges
- **Accuracy ring** — overall confidence at a glance, plus "N need review"
- **Instant revisit** — a finished document's result is cached, so switching back to it shows immediately (no re-extraction)
- **Copy / Download** — export results (value + confidence + validation status)

---

## 9. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/upload` | Upload & OCR a single document |
| `POST` | `/upload/multiple` | Upload & OCR several documents at once |
| `POST` | `/extract` | **Tier 1** — keyword extraction + validation; returns which fields need a Tier-2 recheck |
| `POST` | `/extract/llm` | **Tier 2** — GPT-4.1 fallback for the missing/suspicious fields |
| `GET` | `/files` | List uploaded files and their OCR results |
| `GET` | `/health` | Service status + OCR readiness + cache size + **whether GPT-4.1 fallback is enabled** |

> Splitting extraction into two endpoints is what lets the UI **show the two tiers happening live** — Tier 1 returns instantly, then the browser calls Tier 2 only for the fields that need it.

---

## 10. Performance & Reliability

- **MD5 caching** — identical files return instantly, never re-OCR'd
- **Per-document result cache** — revisiting a finished document shows its result instantly (no re-extraction)
- **Thread-safe OCR** — a lock serialises EasyOCR so concurrent requests can't crash the shared reader
- **Non-blocking OCR** — CPU-heavy work runs in a thread pool, server stays responsive
- **Request timeouts** — 5-min ceiling on OCR, 45-sec on the GPT-4.1 call, so a stalled request fails cleanly instead of hanging
- **Retry on failure** — any failed document can be retried from the queue without re-uploading
- **Graceful AI degradation** — if GPT-4.1 is unreachable/timed-out, those fields stay "Not detected"; the app never crashes and the offline result still shows
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

# 2. (Optional) Enable the GPT-4.1 fallback — copy .env.example to .env and fill in
#    your Azure OpenAI keys. Skip this and the app runs fully offline (Tier 1 only).
#    .env is git-ignored — never commit your keys.

# 3. Start the server
python -m uvicorn main:app

# 4. Open in browser
http://localhost:8000
```

**`.env` (optional — for the GPT-4.1 fallback):**
```
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_API_VERSION=2024-10-21
```

> **Tip:** run with plain `python -m uvicorn main:app` (no `--reload`) — `--reload` watches the whole folder and can restart the server mid-OCR (especially in synced folders like OneDrive), interrupting long documents. Without reload, OCR always completes cleanly. Check `GET /health` → `"llm_fallback_enabled": true` to confirm GPT-4.1 is active.

---

## 12. Project Structure

```
IDP-Preparation/
├── main.py            # FastAPI backend — OCR, two-tier extraction, GPT-4.1 fallback, validation
├── requirements.txt   # Python dependencies
├── .env.example       # Template for Azure OpenAI keys (safe to commit)
├── .env               # Your real keys — git-ignored, NEVER committed
├── README.md          # This file
├── workflow.mmd       # Mermaid flowchart of the pipeline
├── static/
│   └── index.html     # Frontend UI (HTML + CSS + JS, two-tier flow visuals)
├── uploads/           # Saved uploaded documents
└── ocr_results/       # Saved OCR text output
```

---

## 13. Why DocScan? (Design Choices)

| Choice | Reason |
|---|---|
| **Offline OCR (EasyOCR)** | Privacy, zero cost, no API limits |
| **Two-tier extraction** | Keyword first (fast/free/private), GPT-4.1 only when needed — best of both |
| **GPT-4.1 as a *fallback*, not primary** | Most data never leaves the machine; AI cost is paid only on hard fields |
| **Azure OpenAI (not OpenAI direct)** | Data stays in the org tenant, not used for training — fit for confidential docs |
| **Keys in `.env`, never in code** | Credentials never committed to git |
| **Two extract endpoints** | Lets the UI show the two tiers running live, and isolates the AI call |
| **OpenCV preprocessing** | Turns poor scans into OCR-ready images |
| **Validation layer** | Catches errors confidence alone would miss; also triggers the AI recheck |
| **Vanilla JS frontend** | Lightweight, no build step, easy to host |
| **Single-file backend** | Easy to read, run, and present |

---

## 14. Future Enhancements

- **Real streaming OCR progress** (Server-Sent Events) — replace simulated stages with true per-page progress
- **Logprobs-based confidence** for the AI tier — mathematically-grounded scores instead of self-reported
- **Table / line-item extraction** — pull repeating rows (invoice items), not just single fields
- **Auto-suggest fields** — GPT-4.1 proposes the right fields based on document type
- Cross-field rules (e.g. `subtotal + tax ≈ total`)
- Multi-language OCR (EasyOCR supports 80+ languages)
- Export to CSV / JSON / Excel
- Persistence — results survive a page refresh (SQLite / localStorage)
- User-saved field templates per document type

---

### One-line pitch
**DocScan turns any scanned document into clean, validated, structured data — offline-first, with a self-correcting GPT-4.1 fallback that catches what simple matching can't.**

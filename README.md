# Agronomic Vision & Advisory Engine

Photograph a diseased leaf, add what you know about your field, and get a treatment plan
that is **grounded in a measurement** rather than in a guess.

Crop diseases spread fast, and generic advice ignores local soil and weather. This system
identifies the disease, **quantifies how much tissue is actually damaged**, retrieves the
relevant agricultural extension material, and synthesises a plan tailored to that exact
severity and those local conditions.

The severity number is the point. It is computed by deterministic OpenCV colour masking —
no neural network — and it maps to an **action tier** that constrains the advisory. A 3 %
lesion ratio forces "monitor and sanitise". A 40 % ratio forces "immediate intervention".
That hard metric is what stops the language model from recommending a fungicide spray for a
nearly healthy plant.

---

## Architecture

```
React PWA ──multipart(image + env JSON)──▶ FastAPI  POST /api/v1/diagnose
                                              │
                        ┌─────────────────────┴──────────────────────┐
                        ▼                                            ▼
              ResNet-50 classifier                        OpenCV severity index
              crop / disease / confidence                 leaf mask → healthy /
                                                          chlorotic / necrotic
                                                          → severity %, lesions,
                                                            action tier
                        └─────────────────┬──────────────────────────┘
                                          ▼
                        LlamaIndex query built from all three signals
                        (disease + severity tier + soil/rainfall/season)
                                          ▼
                        PostgreSQL + pgvector — extension bulletins,
                        IPM guides, pesticide label summaries
                                          ▼
                        LLM synthesis (Groq / Gemini free tier)
                        → severity-constrained, cited TreatmentPlan
```

| Layer | Choice | Why |
|---|---|---|
| Vision — classification | PyTorch `torchvision` ResNet-50, ImageNet → 38-class PlantVillage head | Strong transfer baseline; two-stage fine-tuning fits a free CPU/Colab budget |
| Vision — measurement | OpenCV HSV masking | Deterministic and auditable. Every number is a pixel count under a stated threshold |
| Retrieval | LlamaIndex + `bge-small-en-v1.5` (384-dim, local) | Runs on CPU, no API key, no rate limit |
| Vector store | PostgreSQL + pgvector (HNSW, cosine) | Self-hosted, free, one container |
| LLM | Groq `llama-3.3-70b-versatile` or Gemini `gemini-2.0-flash`, over plain `httpx` | Free tiers, no vendor SDK, swappable in ~15 lines |
| API | FastAPI + pydantic v2 | The schema *is* the contract, shared by API, notebooks and tests |

**Everything degrades, nothing crashes.** No trained checkpoint, no Postgres, no API key,
no internet — the pipeline still returns a complete advisory and tells you exactly which
components were degraded.

---

## Quick start

Requires **Python 3.12** (PyTorch publishes no 3.14 wheels yet). Nothing is installed
globally; everything lives in `.venv`.

```powershell
.\scripts\bootstrap.ps1
```

That creates the venv, installs CPU PyTorch and the rest, writes `.env`, generates sample
leaf images and builds the knowledge index. Then either:

```powershell
# the guided walkthrough
.\.venv\Scripts\jupyter.exe lab notebooks

# or serve the API — http://localhost:8000/docs
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend

# or run the tests
.\.venv\Scripts\python.exe -m pytest -q
```

### Optional: turn on the LLM advisor (free)

Get a free key from [console.groq.com/keys](https://console.groq.com/keys) or
[aistudio.google.com/apikey](https://aistudio.google.com/apikey), put it in `.env`, and
restart. `LLM_PROVIDER=auto` picks up whichever key is present. Without a key you get the
deterministic rule-based advisor, which never names a chemical product.

### Optional: pgvector instead of the file store

```powershell
docker compose up -d postgres
# set DATABASE_URL in .env, then
.\.venv\Scripts\python.exe scripts\ingest_knowledge.py
```

Or bring the whole stack up: `docker compose up --build`.

---

## Notebooks

| Notebook | What it does |
|---|---|
| [`00_end_to_end_pipeline`](notebooks/00_end_to_end_pipeline.ipynb) | **Start here.** Image → severity → classification → retrieval → plan, with the mask figures and the escalation ladder across all five tiers |
| [`01_severity_lab`](notebooks/01_severity_lab.ipynb) | Tune the HSV thresholds against known ground truth; tier confusion matrix; stress tests for colour cast, shadow, exposure and cluttered backgrounds |
| [`02_train_resnet50`](notebooks/02_train_resnet50.ipynb) | Dataset audit, augmentation inspection, two-stage training, learning curves, per-class F1 |
| [`03_rag_pgvector`](notebooks/03_rag_pgvector.ipynb) | Corpus ingestion, chunking, retrieval quality, and whether the severity tier actually changes what is retrieved |

## API

| Route | Purpose |
|---|---|
| `GET /api/v1/health` | Liveness |
| `GET /api/v1/readiness` | Per-component status and every active degradation |
| `POST /api/v1/diagnose` | Photo + `environment` JSON → full advisory. `?debug=1` adds the mask overlay |
| `POST /api/v1/severity` | CV only — fast enough for a live camera preview |
| `POST /api/v1/severity/overlay` | The annotated mask image, for the "show me why" view |
| `GET /api/v1/knowledge/search` | Raw retrieval, for corpus debugging |
| `POST /api/v1/knowledge/ingest` | Re-index `data/knowledge/` |

```bash
curl -X POST http://localhost:8000/api/v1/diagnose \
  -F "image=@data/samples/leaf_moderate.png" \
  -F 'environment={"region":"Nashik","soil_type":"black","season":"kharif","rainfall_last_7d_mm":118,"irrigation":"sprinkler"}'
```

---

## The severity index

| severity % | tier | mandated posture |
|---|---|---|
| < 2 | `negligible` | No action. Record a baseline |
| 2–10 | `early` | Monitoring and cultural control only — no blanket spray |
| 10–25 | `moderate` | Targeted spot treatment, biologicals first |
| 25–50 | `severe` | Full chemical intervention on a label-rate schedule |
| > 50 | `critical` | Intervention plus a salvage/replant assessment |

How it is computed: resize and bilateral-denoise → segment the leaf → white balance *using
the background the segmentation just found* → classify each leaf pixel as healthy green,
chlorotic or necrotic → area ratios, lesion count, largest lesion.

Accuracy against synthetic ground truth over 0–70 % damage: **mean absolute error 0.54 pp**
(max 0.84), and within ~1 pp under a warm colour cast or a soil background.

Two design decisions carry that robustness, and both came out of measurement rather than
intuition:

- **Adaptive threshold + border test.** A single fixed saturation floor let a warm-tinted or
  soil background into the leaf mask, and every off-green background pixel then counted as
  diseased tissue — **+49 pp of severity, two action tiers wrong**. `segment_leaf` now tries
  a fixed floor *and* an Otsu threshold and validates each on plausible area plus a border
  test: the region a candidate excludes must actually lie at the frame edge. That last
  condition is what makes Otsu safe on an all-leaf macro shot, where it would otherwise
  split the leaf and carve the lesions out of the measurement.
- **Balance after segmenting, not before.** Grey-world balancing tints a neutral background
  until the leaf mask swallows the whole frame; white-patch does the same when bright
  chlorotic tissue outranks the background. Both fail because the subject is the strongest
  colour present. Estimating the illuminant from the *non-subject* pixels avoids it.

[`notebooks/01_severity_lab.ipynb`](notebooks/01_severity_lab.ipynb) reproduces all of this,
including the failure modes that remain (blue cast −8.9 pp, overexposure −7.9 pp).

## Dataset

Training data is PlantVillage, from the canonical repository of the original paper:
[spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) —
54,306 leaf images, 14 crops, 26 diseases, 38 classes. No account or API token needed.

```powershell
.\.venv\Scripts\python.exe scripts\download_plantvillage.py --source github
```

This does a shallow, blobless, **sparse** clone of only `raw/color`. The repository stores
three copies of every image (`color`, `grayscale`, `segmented`), so a plain `git clone`
pulls roughly 3x the data plus the full history. Stay on `color` — `segmented` ships with
the background already removed, which would bypass the leaf segmentation the severity index
depends on and flatter the results. Kaggle and manual routes remain available as mirrors.

The script verifies the 38 folder names against `app/vision/labels.py`; they match upstream
exactly and in alphabetical order, which is the order `ImageFolder` assigns label indices in.
`ml/train.py` also saves the class list into the checkpoint and `classifier.py` reads it
back, so a label-order mismatch cannot silently make every prediction confidently wrong.

Cite Mohanty, Hughes & Salathé (2016), *Using Deep Learning for Image-Based Plant Disease
Detection*, Frontiers in Plant Science. The repository carries no LICENSE file, so there is
no explicit grant — worth checking the terms before any use beyond coursework.

## Safety

Pesticide dosage is the one place where a hallucination could cause real harm, so it is
handled in code, not in the prompt:

- Any chemical recommendation containing a number (rate, concentration, interval) with no
  citation to a retrieved source is **deleted**, and the removal is reported.
- Citations pointing at evidence ids that were never retrieved are treated as uncited.
- `organic_only` strips every chemical option outright.
- The measured severity tier always overrides whatever urgency the model claims.
- With no retrieved evidence, the prompt forbids naming any active at all.
- Every response carries a verify-with-your-extension-officer disclaimer.

## Layout

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full tree and
[`docs/PLAN.md`](docs/PLAN.md) for design decisions, phasing and the risk register.

```
backend/app/     FastAPI app — api/ schemas/ vision/ rag/ llm/ services/
ml/              training, evaluation, dataset and augmentation policy
notebooks/       the four walkthroughs above
data/knowledge/  the retrieval corpus (Markdown + front matter)
data/samples/    synthetic sample leaves, one per action tier
infra/           pgvector init SQL
scripts/         bootstrap, sample generation, ingestion, dataset download
tests/           76 tests — severity vs. ground truth, segmentation robustness,
                 advisor safety, API, RAG
```

## Status

Phase 0 complete: scaffold, runnable pipeline, severity engine validated against synthetic
ground truth (MAE 0.54 pp) and stress-tested for colour cast, exposure, shadow, JPEG
artefacts and cluttered backgrounds; RAG with fallbacks; provider abstraction; 76 tests
green; all four notebooks execute end to end. Next: real PlantVillage training, then the
React PWA. See `docs/PLAN.md`.

> Automated advisories are decision support, not a substitute for an agronomist. Always
> verify actives, dosages and pre-harvest intervals against the product label and your
> local agricultural extension service.

# Repository map

```
Minor Project/
├─ .venv/                          Python 3.12 venv — the ONLY install target
├─ .env.example                    copy to .env and fill in free-tier keys
├─ requirements.txt                runtime deps
├─ requirements-dev.txt            + jupyter, pytest, ruff
├─ docker-compose.yml              postgres(pgvector) + api [+ web in Phase 3]
│
├─ docs/
│  ├─ PLAN.md                      ← the design decisions and phasing
│  └─ ARCHITECTURE.md              this file
│
├─ notebooks/
│  ├─ 00_end_to_end_pipeline.ipynb MAIN: image → severity → retrieval → plan
│  ├─ 01_severity_lab.ipynb        tune the OpenCV thresholds, view masks
│  ├─ 02_train_resnet50.ipynb      transfer-learning driver for ml/train.py
│  └─ 03_rag_pgvector.ipynb        corpus ingest + retrieval quality checks
│
├─ backend/
│  ├─ Dockerfile
│  └─ app/
│     ├─ main.py                   FastAPI app factory
│     ├─ config.py                 pydantic-settings, reads .env
│     ├─ api/
│     │  ├─ deps.py                singletons (pipeline, index) via lru_cache
│     │  └─ routes/
│     │     ├─ health.py           readiness matrix
│     │     ├─ diagnose.py         POST /diagnose, POST /severity
│     │     └─ knowledge.py        search + ingest
│     ├─ schemas/
│     │  ├─ environment.py         EnvironmentContext (soil, rainfall, ...)
│     │  └─ diagnosis.py           Diagnosis, SeverityReport, TreatmentPlan,
│     │                            AdvisoryResponse
│     ├─ vision/
│     │  ├─ preprocess.py          decode, resize, white-balance, tensors
│     │  ├─ labels.py              PlantVillage 38-class taxonomy
│     │  ├─ classifier.py          ResNet-50 wrapper (checkpoint optional)
│     │  ├─ severity.py            OpenCV severity index + action tier
│     │  └─ synthetic.py           synthetic leaves, damage known by construction
│     ├─ rag/
│     │  ├─ embeddings.py          local bge-small, hash-embed fallback
│     │  ├─ index.py               PGVectorStore | SimpleVectorStore
│     │  ├─ ingest.py              load + chunk data/knowledge/
│     │  └─ retriever.py           query construction, filter, rerank
│     ├─ llm/
│     │  ├─ provider.py            Groq | Gemini | Template  (httpx only)
│     │  └─ advisor.py             grounded prompt → validated TreatmentPlan
│     ├─ services/
│     │  └─ pipeline.py            the orchestrator both API and notebook use
│     └─ core/
│        └─ logging.py
│
├─ ml/
│  ├─ configs/resnet50_plantvillage.yaml
│  ├─ datasets.py                  ImageFolder + augmentation policy
│  ├─ train.py                     two-stage transfer learning
│  └─ evaluate.py                  confusion matrix, per-class F1
│
├─ data/
│  ├─ raw/                         PlantVillage (gitignored, ~1.6 GB)
│  ├─ samples/                     a few committed leaves for smoke tests
│  └─ knowledge/                   extension bulletins & pesticide labels (.md)
│
├─ models/checkpoints/             trained weights (gitignored)
├─ storage/                        SimpleVectorStore fallback persistence
│
├─ frontend/                       React + Vite PWA (Phase 3 placeholder)
├─ infra/postgres/init/            01_pgvector.sql — extension + HNSW index
├─ scripts/
│  ├─ bootstrap.ps1                venv + deps, idempotent
│  ├─ make_samples.py              synthesise sample leaves (no download needed)
│  ├─ ingest_knowledge.py          CLI wrapper around rag/ingest.py
│  └─ download_plantvillage.py     Phase 1 dataset fetch
├─ tests/
│  ├─ conftest.py                  puts backend/ on sys.path; offline Settings fixture
│  ├─ test_severity.py             severity vs. known truth + segmentation stress
│  ├─ test_advisor.py              prompt guard rails, uncited-dosage stripping
│  ├─ test_rag.py                  front matter, embeddings, query build, retrieval
│  ├─ test_pipeline.py             end-to-end in the fully degraded configuration
│  └─ test_api.py                  TestClient smoke over every route
├─ pytest.ini
└─ ruff.toml
```

## Import contract

`backend/` is on `sys.path` as the package root, so everything imports as
`app.<module>`. Notebooks call `setup()` from `notebooks/_bootstrap.py`, scripts call
`bootstrap()` from `scripts/_path.py`, `ml/` entry points insert it directly, and the test
suite gets it from `tests/conftest.py` — no `pip install -e .` required and no
relative-import gymnastics.

## The one rule for this codebase

**Every external dependency degrades, never crashes.** Missing checkpoint, no
Postgres, no API key, no internet — the pipeline still returns a complete
`AdvisoryResponse` with honest `readiness` flags saying which parts were
degraded. This is what makes the notebook demonstrable at any moment.

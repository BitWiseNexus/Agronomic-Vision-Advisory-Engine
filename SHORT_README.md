# Agronomic Vision & Advisory Engine - Quick Summary

**What is it?**
It's an application designed to give tailored agricultural treatment plans. You upload a photo of a diseased crop leaf along with local environmental data (like soil type and recent rainfall), and it tells you exactly what's wrong and how to fix it.

## How it Works (The 4-Step Pipeline)

1. **Disease Identification (AI):** A machine learning model (ResNet-50 via PyTorch) looks at the leaf and identifies the specific crop and disease.
2. **Damage Measurement (Computer Vision):** Instead of guessing how bad the disease is, it uses OpenCV (color masking) to calculate the exact percentage of damaged tissue. This hard number determines the **Severity Tier** (e.g., *negligible*, *moderate*, *critical*).
3. **Knowledge Retrieval (RAG):** It searches a database (PostgreSQL + pgvector) of agricultural extension guides, manuals, and pesticide labels using the identified disease, severity tier, and your local environment data.
4. **Treatment Plan Synthesis (LLM):** A Large Language Model (Groq or Gemini) reads the retrieved guides and writes a customized treatment plan.

## Why is it built this way?

* **No Hallucinated Chemicals:** The severity is calculated mathematically by counting pixels, not by asking an AI. This prevents the LLM from recommending harsh chemical sprays for a leaf that is only 3% damaged.
* **Safety First:** Any chemical dosage recommendation the LLM makes *must* have a citation from the database. If it hallucinates a chemical or dosage without a source, the system deletes it before showing it to the user.
* **Resilience:** If the database goes down, or the internet cuts out, or the LLM API fails, the system doesn't crash. It falls back to simpler, deterministic rule-based advice.

## Tech Stack Overview

* **Backend:** FastAPI (Python)
* **Frontend:** React PWA (Pending implementation)
* **Machine Learning:** PyTorch & OpenCV
* **Database / Search:** PostgreSQL with pgvector & LlamaIndex
* **LLM Providers:** Groq or Google Gemini

## How to Run It

To get started (requires Python 3.12):
```powershell
# Set up the environment and download dependencies
.\scripts\bootstrap.ps1

# Run the API backend locally
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend
```

> **Note:** For the full deep-dive into the architecture, check out the main [`README.md`](README.md).

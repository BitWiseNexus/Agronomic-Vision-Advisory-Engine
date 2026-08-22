<#
    Idempotent project setup. Safe to re-run.

        .\scripts\bootstrap.ps1

    Creates the venv with Python 3.12 (PyTorch has no 3.14 wheels), installs torch from
    the CPU wheel index, installs the rest, writes .env, generates sample leaves and
    ingests the knowledge corpus. Nothing is ever installed globally.
#>

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"

Write-Host "== Agronomic Vision & Advisory Engine :: bootstrap ==" -ForegroundColor Cyan

# --- 1. venv -----------------------------------------------------------------
if (-not (Test-Path $py)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -eq $launcher) { throw "The 'py' launcher was not found. Install Python 3.12." }

    Write-Host "[1/5] Creating the virtual environment (Python 3.12)..."
    & py -3.12 -m venv $venv
    if (-not (Test-Path $py)) { throw "Python 3.12 is required but was not available to 'py -3.12'." }
} else {
    Write-Host "[1/5] Virtual environment already present."
}

& $py -m pip install --upgrade pip --quiet

# --- 2. dependencies ---------------------------------------------------------
Write-Host "[2/5] Installing dependencies (torch from the CPU index; this takes a few minutes)..."
& $py -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install --quiet -r (Join-Path $root "requirements-dev.txt")
& $py -m pip install --quiet pyyaml

# --- 3. .env -----------------------------------------------------------------
$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $root ".env.example") $envFile
    Write-Host "[3/5] Wrote .env from .env.example — add a free GROQ_API_KEY or GEMINI_API_KEY to enable the LLM advisor."
} else {
    Write-Host "[3/5] .env already exists; leaving it alone."
}

# --- 4. sample images --------------------------------------------------------
Write-Host "[4/5] Generating synthetic sample leaves..."
Push-Location (Join-Path $root "scripts")
try { & $py make_samples.py } finally { Pop-Location }

# --- 5. knowledge index ------------------------------------------------------
Write-Host "[5/5] Ingesting the knowledge corpus (downloads the embedding model once)..."
Push-Location (Join-Path $root "scripts")
try { & $py ingest_knowledge.py --local } finally { Pop-Location }

# --- Jupyter kernel ----------------------------------------------------------
& $py -m ipykernel install --user --name agronomic-engine --display-name "Python (Agronomic Engine)" | Out-Null

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "  Notebook :  .\.venv\Scripts\jupyter.exe lab notebooks"
Write-Host "  API      :  .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend"
Write-Host "  Tests    :  .\.venv\Scripts\python.exe -m pytest -q"

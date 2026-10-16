/* ─── API Client ───────────────────────────────────────────────
   Thin wrapper over fetch for the FastAPI backend.
   Base URL defaults to localhost:8000 for dev, configurable via
   VITE_API_URL environment variable.
   ──────────────────────────────────────────────────────────── */

import type {
  AdvisoryResponse,
  EnvironmentContext,
  HealthResponse,
  OverlayResponse,
  ReadinessResponse,
  SeverityReport,
} from './types';

const API_BASE =
  import.meta.env.VITE_API_URL ??
  import.meta.env.VITE_API_BASE_URL ??
  'http://localhost:8000';
const PREFIX = API_BASE.endsWith('/api/v1') ? API_BASE : `${API_BASE}/api/v1`;

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch { /* ignore parse errors */ }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

/* ── Health / Readiness ─────────────────────────────────────── */

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${PREFIX}/health`);
  return handleResponse<HealthResponse>(res);
}

export async function getReadiness(): Promise<ReadinessResponse> {
  const res = await fetch(`${PREFIX}/readiness`);
  return handleResponse<ReadinessResponse>(res);
}

/* ── Diagnosis ──────────────────────────────────────────────── */

export async function postDiagnose(
  imageFile: File,
  environment: EnvironmentContext,
  debug = true,
): Promise<AdvisoryResponse> {
  const form = new FormData();
  form.append('image', imageFile);
  form.append('environment', JSON.stringify(environment));

  const res = await fetch(`${PREFIX}/diagnose?debug=${debug}`, {
    method: 'POST',
    body: form,
  });
  return handleResponse<AdvisoryResponse>(res);
}

/* ── Severity only (fast, for live preview) ─────────────────── */

export async function postSeverity(imageFile: File): Promise<SeverityReport> {
  const form = new FormData();
  form.append('image', imageFile);

  const res = await fetch(`${PREFIX}/severity`, {
    method: 'POST',
    body: form,
  });
  return handleResponse<SeverityReport>(res);
}

/* ── Severity overlay (show me why view) ────────────────────── */

export async function postSeverityOverlay(imageFile: File): Promise<OverlayResponse> {
  const form = new FormData();
  form.append('image', imageFile);

  const res = await fetch(`${PREFIX}/severity/overlay`, {
    method: 'POST',
    body: form,
  });
  return handleResponse<OverlayResponse>(res);
}

export { ApiError };

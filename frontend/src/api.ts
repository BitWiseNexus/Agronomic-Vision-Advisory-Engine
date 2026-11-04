/* ─── API Client ───────────────────────────────────────────────
   Thin wrapper over fetch for the FastAPI backend.
   In dev the Vite proxy forwards /api to the backend, so the
   default base is same-origin. Set VITE_API_URL to call a backend
   on another origin (e.g. a deployed build).
   ──────────────────────────────────────────────────────────── */

import type {
  AdvisoryResponse,
  EnvironmentContext,
  HealthResponse,
  OverlayResponse,
  ReadinessResponse,
  SeverityReport,
} from './types';

const API_BASE: string = (
  import.meta.env.VITE_API_URL ??
  import.meta.env.VITE_API_BASE_URL ??
  ''
).replace(/\/$/, '');
const PREFIX = API_BASE.endsWith('/api/v1') ? API_BASE : `${API_BASE}/api/v1`;

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    } catch { /* ignore parse errors */ }
    // The Vite proxy answers 5xx with an empty body when the backend is down.
    if (res.status >= 500 && !detail) detail = 'The analysis server is not responding.';
    throw new ApiError(res.status, detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

/** fetch() rejects with an opaque TypeError when the server is unreachable. */
async function request(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch {
    throw new ApiError(0, 'Cannot reach the analysis server. Is the backend running on port 8000?');
  }
}

/* ── Health / Readiness ─────────────────────────────────────── */

export async function getHealth(): Promise<HealthResponse> {
  const res = await request(`${PREFIX}/health`);
  return handleResponse<HealthResponse>(res);
}

export async function getReadiness(): Promise<ReadinessResponse> {
  const res = await request(`${PREFIX}/readiness`);
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

  const res = await request(`${PREFIX}/diagnose?debug=${debug}`, {
    method: 'POST',
    body: form,
  });
  return handleResponse<AdvisoryResponse>(res);
}

/* ── Severity only (fast, for live preview) ─────────────────── */

export async function postSeverity(imageFile: File): Promise<SeverityReport> {
  const form = new FormData();
  form.append('image', imageFile);

  const res = await request(`${PREFIX}/severity`, {
    method: 'POST',
    body: form,
  });
  return handleResponse<SeverityReport>(res);
}

/* ── Severity overlay (show me why view) ────────────────────── */

export async function postSeverityOverlay(imageFile: File): Promise<OverlayResponse> {
  const form = new FormData();
  form.append('image', imageFile);

  const res = await request(`${PREFIX}/severity/overlay`, {
    method: 'POST',
    body: form,
  });
  return handleResponse<OverlayResponse>(res);
}

export { ApiError };

/* ─── LocalStorage persistence for history, environment, theme ─
   localStorage is ~5 MB per origin. A full-resolution photo as a
   data URL is easily 0.5–3 MB, so history images are downscaled
   to JPEG thumbnails and saves degrade (drop oldest) rather than
   throw — a full quota must never turn a successful diagnosis
   into an error screen.
   ──────────────────────────────────────────────────────────── */

import type { EnvironmentContext, HistoryEntry } from './types';

const HISTORY_KEY = 'agrovision_history';
const ENV_KEY = 'agrovision_environment';
const THEME_KEY = 'agrovision_theme';
const MAX_HISTORY = 50;

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

/** Returns false instead of throwing when storage is full or unavailable. */
function writeJson(key: string, value: unknown): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

/* ── History ─────────────────────────────────────────────────── */

export function loadHistory(): HistoryEntry[] {
  const history = readJson<HistoryEntry[]>(HISTORY_KEY, []);
  return Array.isArray(history) ? history : [];
}

/** Saves newest-first, evicting the oldest entries until it fits. */
export function saveHistoryEntry(entry: HistoryEntry): boolean {
  const history = [entry, ...loadHistory()].slice(0, MAX_HISTORY);
  while (history.length > 0) {
    if (writeJson(HISTORY_KEY, history)) return true;
    history.pop();
  }
  return false;
}

export function deleteHistoryEntry(id: string): void {
  writeJson(HISTORY_KEY, loadHistory().filter(e => e.id !== id));
}

export function clearHistory(): void {
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch { /* storage unavailable */ }
}

/* ── Environment (persisted across visits) ───────────────────── */

export function loadEnvironment(): EnvironmentContext {
  return readJson<EnvironmentContext>(ENV_KEY, {});
}

export function saveEnvironment(env: EnvironmentContext): void {
  writeJson(ENV_KEY, env);
}

/* ── Theme ───────────────────────────────────────────────────── */

export type ThemePref = 'light' | 'dark' | 'system';

export function loadTheme(): ThemePref {
  try {
    const t = localStorage.getItem(THEME_KEY);
    return t === 'light' || t === 'dark' ? t : 'system';
  } catch {
    return 'system';
  }
}

export function applyTheme(pref: ThemePref): void {
  const root = document.documentElement;
  if (pref === 'system') delete root.dataset.theme;
  else root.dataset.theme = pref;
  try {
    if (pref === 'system') localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, pref);
  } catch { /* per-viewer convenience only */ }
}

export function resolvedTheme(): 'light' | 'dark' {
  const explicit = document.documentElement.dataset.theme;
  if (explicit === 'light' || explicit === 'dark') return explicit;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/* ── Helpers ─────────────────────────────────────────────────── */

export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/** Downscale an image to a JPEG data URL — small enough for localStorage. */
export async function toThumbnailDataUrl(src: Blob | string, maxSide = 560, quality = 0.8): Promise<string> {
  const url = typeof src === 'string' ? src : URL.createObjectURL(src);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    canvas.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', quality);
  } finally {
    if (typeof src !== 'string') URL.revokeObjectURL(url);
  }
}

export const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
export const MAX_UPLOAD_BYTES = 12 * 1024 * 1024; // mirrors the backend limit

/** Client-side check so the user gets an instant, specific message. */
export function validateImageFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return `“${file.name}” isn't a JPEG, PNG or WebP image.`;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `That photo is ${(file.size / 1024 / 1024).toFixed(1)} MB — the limit is 12 MB.`;
  }
  return null;
}

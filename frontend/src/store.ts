/* ─── LocalStorage persistence for history & environment ───── */

import type { EnvironmentContext, HistoryEntry } from './types';

const HISTORY_KEY = 'agrovision_history';
const ENV_KEY = 'agrovision_environment';
const MAX_HISTORY = 50;

/* ── History ─────────────────────────────────────────────────── */

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as HistoryEntry[];
  } catch {
    return [];
  }
}

export function saveHistoryEntry(entry: HistoryEntry): void {
  const history = loadHistory();
  history.unshift(entry);
  // Keep only the latest entries
  if (history.length > MAX_HISTORY) {
    history.length = MAX_HISTORY;
  }
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

export function deleteHistoryEntry(id: string): void {
  const history = loadHistory().filter(e => e.id !== id);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

export function clearHistory(): void {
  localStorage.removeItem(HISTORY_KEY);
}

/* ── Environment (persisted across visits) ───────────────────── */

export function loadEnvironment(): EnvironmentContext {
  try {
    const raw = localStorage.getItem(ENV_KEY);
    if (!raw) return {};
    return JSON.parse(raw) as EnvironmentContext;
  } catch {
    return {};
  }
}

export function saveEnvironment(env: EnvironmentContext): void {
  localStorage.setItem(ENV_KEY, JSON.stringify(env));
}

/* ── Helpers ─────────────────────────────────────────────────── */

export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

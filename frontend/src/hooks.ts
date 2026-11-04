/* ─── Shared hooks ─────────────────────────────────────────── */

import { useCallback, useEffect, useState } from 'react';
import { getReadiness } from './api';
import type { ReadinessResponse } from './types';
import { applyTheme, loadTheme, resolvedTheme, type ThemePref } from './store';

export type BackendState =
  | { status: 'checking' }
  | { status: 'offline' }
  | { status: 'online'; readiness: ReadinessResponse };

// One in-flight readiness request shared by every component that asks.
let cached: Promise<ReadinessResponse> | null = null;

export function useBackend(): BackendState & { retry: () => void } {
  const [state, setState] = useState<BackendState>({ status: 'checking' });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let alive = true;
    cached ??= getReadiness();
    cached
      .then(readiness => alive && setState({ status: 'online', readiness }))
      .catch(() => {
        cached = null;
        if (alive) setState({ status: 'offline' });
      });
    return () => { alive = false; };
  }, [attempt]);

  const retry = useCallback(() => {
    cached = null;
    setState({ status: 'checking' });
    setAttempt(a => a + 1);
  }, []);

  return { ...state, retry };
}

const darkQuery = () => window.matchMedia('(prefers-color-scheme: dark)');

export function useTheme() {
  const [pref, setPref] = useState<ThemePref>(loadTheme);
  const [osDark, setOsDark] = useState(() => darkQuery().matches);
  const effective: 'light' | 'dark' = pref === 'system' ? (osDark ? 'dark' : 'light') : pref;

  useEffect(() => { applyTheme(pref); }, [pref]);

  useEffect(() => {
    const mq = darkQuery();
    const onChange = () => setOsDark(mq.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const toggle = useCallback(() => {
    // Toggling picks the opposite of what is on screen; matching the OS returns to 'system'.
    const next = resolvedTheme() === 'dark' ? 'light' : 'dark';
    setPref((next === 'dark') === darkQuery().matches ? 'system' : next);
  }, []);

  return { pref, effective, toggle };
}

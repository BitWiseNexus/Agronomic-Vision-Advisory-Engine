# Frontend — React PWA (Phase 3)

Placeholder. The backend contract this will consume is already fixed and served at
`http://localhost:8000/docs`, so the app can be built against a live API today.

## Planned stack

- Vite + React + TypeScript
- `vite-plugin-pwa` for the service worker and install prompt
- No component library — a handful of screens does not justify the bundle

## Screens

1. **Capture** — `<input type="file" capture="environment">` plus a live preview.
   Calls `POST /api/v1/severity` on the still frame for instant feedback before the
   farmer commits to the full analysis.
2. **Context form** — soil type, season, irrigation, rainfall in the last 7 days,
   crop stage, organic-only toggle. Every field optional; mirrors `EnvironmentContext`.
   Persisted in `localStorage` — a farmer's soil type does not change between visits.
3. **Result** — severity gauge, the mask overlay from `POST /api/v1/severity/overlay`
   (this is the "show me why" view), the action tier, then the plan with citations.
4. **History** — past diagnoses for the same field, so severity *trend* is visible.
   Trend is what decides escalation, and one photo cannot show it.

## Offline behaviour

Connectivity in a field is unreliable, so captures queue in IndexedDB and flush when the
network returns. The severity endpoint is the cheap one; the full advisory needs the server.

## Contract notes

- `POST /api/v1/diagnose` takes multipart: `image` (file) + `environment` (JSON **string**).
- Read `readiness.degraded` on every response and surface it. If the advisory came from the
  template provider, or the classifier had no checkpoint, the user must be told — an
  unqualified plan built on a placeholder diagnosis is the one genuinely harmful outcome.
- Always display `disclaimer`.

```bash
npm create vite@latest . -- --template react-ts
npm install && npm run dev
```

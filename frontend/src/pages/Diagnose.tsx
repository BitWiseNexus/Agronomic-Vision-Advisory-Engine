/* ─── Diagnose Page ────────────────────────────────────────────
   Three-step flow:
     1. Photo   — upload, take a photo, or try a sample
     2. Context — field conditions (persisted), plus an instant
                  photo-quality check from the fast /severity route
     3. Report  — the full advisory
   ──────────────────────────────────────────────────────────── */

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { postDiagnose, postSeverity } from '../api';
import {
  generateId,
  loadEnvironment,
  saveEnvironment,
  saveHistoryEntry,
  toThumbnailDataUrl,
  validateImageFile,
} from '../store';
import type { AdvisoryResponse, EnvironmentContext, SeverityReport } from '../types';
import { TIER_CONFIG } from '../types';
import Icon from '../components/Icon';
import { TierPill } from '../components/Severity';
import ResultView from '../components/ResultView';
import { overlaySrc } from '../format';
import './Diagnose.css';

type Step = 'capture' | 'context' | 'analyzing' | 'result';

type QuickCheck =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'done'; report: SeverityReport }
  | { status: 'error'; message: string };

const SAMPLES = [
  { file: 'leaf_healthy.jpg', label: 'Healthy' },
  { file: 'leaf_early_stage.jpg', label: 'Early' },
  { file: 'leaf_moderate.jpg', label: 'Moderate' },
  { file: 'leaf_severe.jpg', label: 'Severe' },
  { file: 'leaf_critical.jpg', label: 'Critical' },
];

const STAGES = [
  { label: 'Measuring damaged tissue', detail: 'OpenCV severity index' },
  { label: 'Identifying the disease', detail: 'ResNet-50 classifier' },
  { label: 'Finding guidance', detail: 'Extension bulletins & labels' },
  { label: 'Writing the treatment plan', detail: 'Severity-constrained advisor' },
];

/* Numeric bounds mirror EnvironmentContext in backend/app/schemas/environment.py */
const NUMERIC_FIELDS = {
  rainfall_last_7d_mm: { min: 0, max: 2000, label: 'Rainfall' },
  temperature_c: { min: -20, max: 60, label: 'Temperature' },
  relative_humidity_pct: { min: 0, max: 100, label: 'Humidity' },
  soil_ph: { min: 2, max: 11, label: 'Soil pH' },
  days_since_last_spray: { min: 0, max: 365, label: 'Days since spray' },
  field_area_ha: { min: 0.01, max: 10000, label: 'Field area' },
} as const;
type NumericKey = keyof typeof NUMERIC_FIELDS;

function fieldError(env: EnvironmentContext, key: NumericKey): string | null {
  const value = env[key];
  if (value === null || value === undefined) return null;
  const { min, max } = NUMERIC_FIELDS[key];
  if (Number.isNaN(value) || value < min || value > max) return `Enter ${min}–${max}`;
  if (key === 'days_since_last_spray' && !Number.isInteger(value)) return 'Whole days only';
  return null;
}

/** Mirrors EnvironmentContext.wetness_note() thresholds on the backend. */
function rainfallHint(mm: number | null | undefined): string | null {
  if (mm === null || mm === undefined || Number.isNaN(mm)) return null;
  if (mm >= 100) return 'Heavy rain — high fungal pressure; contact sprays wash off.';
  if (mm >= 30) return 'Moderate rain — spray only in a 6+ hour dry window.';
  if (mm <= 5) return 'Dry week — low splash spread; check irrigation stress.';
  return null;
}

function countFilled(env: EnvironmentContext): number {
  return Object.entries(env).filter(([k, v]) => {
    if (v === null || v === undefined || v === '' || v === 'unknown') return false;
    if (k === 'organic_only') return v === true;
    return true;
  }).length;
}

export default function Diagnose() {
  const uploadRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);

  const [step, setStep] = useState<Step>('capture');
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [env, setEnv] = useState<EnvironmentContext>(loadEnvironment);
  const [result, setResult] = useState<AdvisoryResponse | null>(null);
  const [resultTime, setResultTime] = useState<string | undefined>();
  const [error, setError] = useState<string | null>(null);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [quick, setQuick] = useState<QuickCheck>({ status: 'idle' });
  const [dragging, setDragging] = useState(false);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const [historyNote, setHistoryNote] = useState<string | null>(null);

  // Free the object URL when the photo changes or the page unmounts.
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);
  useEffect(() => { window.scrollTo({ top: 0 }); }, [step]);

  /* ── Photo intake ──────────────────────────────────────────── */
  const acceptFile = useCallback((f: File) => {
    const problem = validateImageFile(f);
    if (problem) {
      setCaptureError(problem);
      return;
    }
    setCaptureError(null);
    setError(null);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setStep('context');

    setQuick({ status: 'loading' });
    postSeverity(f)
      .then(report => setQuick({ status: 'done', report }))
      .catch((err: unknown) => setQuick({ status: 'error', message: err instanceof Error ? err.message : 'Check failed' }));
  }, []);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = ''; // allow re-selecting the same file
    if (f) acceptFile(f);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) acceptFile(f);
  };

  const trySample = async (name: string) => {
    setLoadingSample(name);
    try {
      const res = await fetch(`/samples/${name}`);
      const blob = await res.blob();
      acceptFile(new File([blob], name, { type: 'image/jpeg' }));
    } catch {
      setCaptureError('Could not load the sample image.');
    } finally {
      setLoadingSample(null);
    }
  };

  /* ── Environment form ──────────────────────────────────────── */
  const updateEnv = <K extends keyof EnvironmentContext>(key: K, value: EnvironmentContext[K]) => {
    setEnv(prev => ({ ...prev, [key]: value }));
  };
  const numberProps = (key: NumericKey) => ({
    id: `input-${key}`,
    className: 'input',
    type: 'number',
    inputMode: 'decimal' as const,
    min: NUMERIC_FIELDS[key].min,
    max: NUMERIC_FIELDS[key].max,
    step: key === 'days_since_last_spray' ? 1 : 'any',
    value: env[key] ?? '',
    'aria-invalid': fieldError(env, key) ? true : undefined,
    onChange: (e: React.ChangeEvent<HTMLInputElement>) =>
      updateEnv(key, e.target.value === '' ? null : Number(e.target.value)),
  });
  const invalid = (Object.keys(NUMERIC_FIELDS) as NumericKey[]).some(k => fieldError(env, k));
  const filled = countFilled(env);

  /* ── Submit ────────────────────────────────────────────────── */
  const handleAnalyze = async () => {
    if (!file || invalid) return;
    saveEnvironment(env);
    setStep('analyzing');
    setError(null);
    setHistoryNote(null);

    let response: AdvisoryResponse;
    try {
      response = await postDiagnose(file, env, true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
      setStep('context');
      return;
    }

    const timestamp = new Date().toISOString();
    setResult(response);
    setResultTime(timestamp);
    setStep('result');

    // History is best-effort: thumbnails keep entries small, and a failure here
    // must never hide a diagnosis the user already has on screen.
    try {
      const [thumb, overlayThumb] = await Promise.all([
        toThumbnailDataUrl(file),
        response.debug_overlay_png_b64 ? toThumbnailDataUrl(overlaySrc(response.debug_overlay_png_b64)!) : null,
      ]);
      const saved = saveHistoryEntry({
        id: generateId(),
        timestamp,
        imageDataUrl: thumb,
        response: { ...response, debug_overlay_png_b64: overlayThumb },
      });
      if (!saved) setHistoryNote('This report could not be saved to History — browser storage is full.');
    } catch {
      setHistoryNote('This report could not be saved to History.');
    }
  };

  const handleReset = () => {
    setStep('capture');
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setQuick({ status: 'idle' });
    setHistoryNote(null);
  };

  /* ── Render ────────────────────────────────────────────────── */
  return (
    <div className="diagnose-page container page">
      {step !== 'result' && <Stepper step={step} />}

      {/* ═══ STEP 1: Photo ═══ */}
      {step === 'capture' && (
        <section className="capture rise" id="capture-step">
          <div className="capture-intro">
            <h1>Photograph the affected leaf</h1>
            <p>One leaf, filling most of the frame, against a plain background.</p>
          </div>

          <div
            className={`dropzone ${dragging ? 'is-dragging' : ''}`}
            id="dropzone"
            onDrop={onDrop}
            onDragOver={e => e.preventDefault()}
            onDragEnter={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={e => { if (e.currentTarget === e.target) setDragging(false); }}
          >
            <div className="dropzone-icon"><Icon name="leaf" size={28} /></div>
            <p className="dropzone-title">Drop a leaf photo here</p>
            <p className="dropzone-hint">JPEG, PNG or WebP · up to 12 MB</p>
            <div className="dropzone-actions">
              <button type="button" className="btn btn-primary btn-lg" onClick={() => cameraRef.current?.click()} id="camera-btn">
                <Icon name="camera" /> Take photo
              </button>
              <button type="button" className="btn btn-secondary btn-lg" onClick={() => uploadRef.current?.click()} id="browse-btn">
                <Icon name="upload" /> Upload image
              </button>
            </div>
          </div>

          {/* Two inputs: `capture` forces the camera on phones, so the upload one must not have it. */}
          <input ref={uploadRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={onInputChange} hidden id="file-input" />
          <input ref={cameraRef} type="file" accept="image/*" capture="environment" onChange={onInputChange} hidden id="camera-input" />

          {captureError && (
            <div className="callout callout-danger capture-error" role="alert">
              <Icon name="alert" size={16} /> <span>{captureError}</span>
            </div>
          )}

          <div className="capture-extras">
            <div className="card card-pad tips">
              <h2 className="card-title"><Icon name="info" /> For an accurate reading</h2>
              <ul>
                <li><Icon name="check" size={15} /> Plain, light background — paper or a tray</li>
                <li><Icon name="check" size={15} /> Leaf fills most of the frame, in focus</li>
                <li><Icon name="check" size={15} /> Even daylight, no harsh shadows</li>
                <li><Icon name="check" size={15} /> One leaf per photo, both sides if unsure</li>
              </ul>
            </div>

            <div className="card card-pad samples">
              <h2 className="card-title"><Icon name="image" /> No leaf handy? Try a sample</h2>
              <div className="sample-grid">
                {SAMPLES.map(s => (
                  <button
                    key={s.file}
                    type="button"
                    className="sample"
                    onClick={() => trySample(s.file)}
                    disabled={loadingSample !== null}
                    aria-label={`Use the ${s.label.toLowerCase()} sample leaf`}
                  >
                    <img src={`/samples/${s.file}`} alt="" loading="lazy" />
                    <span>{loadingSample === s.file ? 'Loading…' : s.label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ═══ STEP 2: Context ═══ */}
      {step === 'context' && (
        <section className="context rise" id="context-step">
          <aside className="context-side">
            <div className="card card-pad">
              <h2 className="card-title">
                <Icon name="image" /> Your photo
                <span className="spacer" />
                <button type="button" className="btn btn-ghost btn-sm" onClick={handleReset} id="change-image-btn">
                  Change
                </button>
              </h2>
              {previewUrl && <img src={previewUrl} alt="Selected leaf" className="context-preview" id="preview-image" />}
              <QuickCheckPanel quick={quick} onRetake={handleReset} />
            </div>
          </aside>

          <div className="card context-form">
            <div className="context-form-head">
              <div>
                <h1>Field conditions</h1>
                <p>Optional, but it sharpens the advice — rain and humidity change what and when to spray. Saved on this device for next time.</p>
              </div>
              {filled > 0 && (
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => setEnv({})}>
                  Clear all
                </button>
              )}
            </div>

            {error && (
              <div className="callout callout-danger" role="alert" id="error-banner">
                <Icon name="alert" size={16} />
                <span><strong>Analysis failed.</strong> {error}</span>
              </div>
            )}

            <FormGroup icon="map-pin" title="Location & soil">
              <div className="form-grid">
                <div className="field">
                  <label className="field-label" htmlFor="input-region">Region</label>
                  <input
                    id="input-region"
                    className="input"
                    type="text"
                    placeholder="e.g. Nashik, Maharashtra"
                    autoComplete="address-level2"
                    value={env.region ?? ''}
                    onChange={e => updateEnv('region', e.target.value || null)}
                  />
                </div>
                <div className="field">
                  <label className="field-label" htmlFor="input-soil-type">Soil type</label>
                  <select
                    id="input-soil-type"
                    className="input"
                    value={env.soil_type ?? 'unknown'}
                    onChange={e => updateEnv('soil_type', e.target.value as EnvironmentContext['soil_type'])}
                  >
                    <option value="unknown">Not sure</option>
                    <option value="alluvial">Alluvial</option>
                    <option value="black">Black (cotton soil)</option>
                    <option value="red">Red</option>
                    <option value="laterite">Laterite</option>
                    <option value="sandy">Sandy</option>
                    <option value="loam">Loam</option>
                    <option value="clay">Clay</option>
                  </select>
                </div>
              </div>
            </FormGroup>

            <FormGroup icon="cloud" title="Weather · last 7 days">
              <div className="form-grid form-grid-3">
                <NumberField label="Rainfall" unit="mm" error={fieldError(env, 'rainfall_last_7d_mm')} hint={rainfallHint(env.rainfall_last_7d_mm)} htmlFor="input-rainfall_last_7d_mm">
                  <input {...numberProps('rainfall_last_7d_mm')} placeholder="e.g. 40" />
                </NumberField>
                <NumberField label="Temperature" unit="°C" error={fieldError(env, 'temperature_c')} htmlFor="input-temperature_c">
                  <input {...numberProps('temperature_c')} placeholder="e.g. 28" />
                </NumberField>
                <NumberField label="Humidity" unit="%" error={fieldError(env, 'relative_humidity_pct')} htmlFor="input-relative_humidity_pct">
                  <input {...numberProps('relative_humidity_pct')} placeholder="e.g. 75" />
                </NumberField>
              </div>
            </FormGroup>

            <FormGroup icon="sprout" title="Crop & management">
              <div className="field">
                <span className="field-label" id="lbl-stage">Crop stage</span>
                <ChipGroup
                  labelledBy="lbl-stage"
                  value={env.crop_stage ?? null}
                  options={['Seedling', 'Vegetative', 'Flowering', 'Fruiting', 'Harvest'].map(v => ({ value: v.toLowerCase(), label: v }))}
                  onChange={v => updateEnv('crop_stage', v)}
                />
              </div>
              <div className="field">
                <span className="field-label" id="lbl-season">Season</span>
                <ChipGroup
                  labelledBy="lbl-season"
                  value={env.season && env.season !== 'unknown' ? env.season : null}
                  options={[
                    { value: 'kharif', label: 'Kharif', sub: 'monsoon' },
                    { value: 'rabi', label: 'Rabi', sub: 'winter' },
                    { value: 'zaid', label: 'Zaid', sub: 'summer' },
                  ]}
                  onChange={v => updateEnv('season', (v ?? 'unknown') as EnvironmentContext['season'])}
                />
              </div>
              <div className="field">
                <span className="field-label" id="lbl-irrigation">Irrigation</span>
                <ChipGroup
                  labelledBy="lbl-irrigation"
                  value={env.irrigation && env.irrigation !== 'unknown' ? env.irrigation : null}
                  options={['Rainfed', 'Flood', 'Drip', 'Sprinkler'].map(v => ({ value: v.toLowerCase(), label: v }))}
                  onChange={v => updateEnv('irrigation', (v ?? 'unknown') as EnvironmentContext['irrigation'])}
                />
              </div>

              <details className="more disclosure">
                <summary>
                  <Icon name="chevron-right" size={16} className="chev" />
                  More details <span className="muted">— soil pH, last spray, field size</span>
                </summary>
                <div className="form-grid form-grid-3 more-grid">
                  <NumberField label="Soil pH" error={fieldError(env, 'soil_ph')} htmlFor="input-soil_ph">
                    <input {...numberProps('soil_ph')} placeholder="e.g. 6.5" />
                  </NumberField>
                  <NumberField label="Days since last spray" error={fieldError(env, 'days_since_last_spray')} htmlFor="input-days_since_last_spray">
                    <input {...numberProps('days_since_last_spray')} placeholder="e.g. 10" />
                  </NumberField>
                  <NumberField label="Field area" unit="ha" error={fieldError(env, 'field_area_ha')} htmlFor="input-field_area_ha">
                    <input {...numberProps('field_area_ha')} placeholder="e.g. 1.5" />
                  </NumberField>
                </div>
              </details>
            </FormGroup>

            <button
              type="button"
              role="switch"
              aria-checked={!!env.organic_only}
              className="switch"
              id="organic-toggle"
              onClick={() => updateEnv('organic_only', !env.organic_only)}
            >
              <span className="switch-track" />
              <span className="switch-text">
                Organic only
                <small>Exclude synthetic chemical actives from the plan</small>
              </span>
            </button>

            <div className="form-actions">
              <button type="button" className="btn btn-ghost" onClick={handleReset}>
                <Icon name="arrow-left" size={16} /> Back
              </button>
              <button type="button" className="btn btn-primary btn-lg" onClick={handleAnalyze} disabled={invalid} id="analyze-btn">
                <Icon name="scan" /> Analyze leaf
              </button>
            </div>
          </div>
        </section>
      )}

      {/* ═══ Analyzing ═══ */}
      {step === 'analyzing' && <Analyzing previewUrl={previewUrl} />}

      {/* ═══ STEP 3: Report ═══ */}
      {step === 'result' && result && (
        <div id="result-step">
          {historyNote && (
            <div className="callout callout-warn history-note no-print">
              <Icon name="alert" size={16} /> <span>{historyNote}</span>
            </div>
          )}
          <ResultView result={result} previewUrl={previewUrl} onReset={handleReset} timestamp={resultTime} />
        </div>
      )}
    </div>
  );
}

/* ─── Sub-components ──────────────────────────────────────── */

function Stepper({ step }: { step: Step }) {
  const steps = [
    { key: 'capture', label: 'Photo' },
    { key: 'context', label: 'Field conditions' },
    { key: 'result', label: 'Report' },
  ];
  const current = step === 'analyzing' ? 2 : steps.findIndex(s => s.key === step);
  return (
    <ol className="stepper no-print" aria-label="Progress">
      {steps.map((s, i) => (
        <li key={s.key} className={i < current ? 'is-done' : i === current ? 'is-current' : ''} aria-current={i === current ? 'step' : undefined}>
          <span className="stepper-dot">{i < current ? <Icon name="check" size={13} strokeWidth={3} /> : i + 1}</span>
          <span className="stepper-label">{s.label}</span>
        </li>
      ))}
    </ol>
  );
}

function QuickCheckPanel({ quick, onRetake }: { quick: QuickCheck; onRetake: () => void }) {
  if (quick.status === 'idle') return null;
  if (quick.status === 'loading') {
    return (
      <div className="quick quick-loading">
        <span className="spinner" /> Checking photo quality…
      </div>
    );
  }
  if (quick.status === 'error') {
    return (
      <div className="callout callout-warn quick-callout">
        <Icon name="alert" size={16} />
        <span>Couldn’t pre-check the photo: {quick.message}</span>
      </div>
    );
  }
  const r = quick.report;
  const ok = r.quality_warnings.length === 0;
  return (
    <div className="quick">
      <div className="quick-row">
        <span className="eyebrow">Quick check</span>
        <span className={`badge ${ok ? 'badge-accent' : 'badge-warn'}`}>
          <Icon name={ok ? 'check' : 'alert'} size={12} /> {ok ? 'Photo looks good' : 'Photo could be better'}
        </span>
      </div>
      <div className="quick-reading">
        <span className="quick-pct num">{r.severity_pct.toFixed(1)}%</span>
        <span className="muted">damaged ·</span>
        <TierPill tier={r.action_tier} />
      </div>
      <p className="quick-note">{TIER_CONFIG[r.action_tier].headline}. The full analysis adds disease ID and a plan.</p>
      {!ok && (
        <div className="callout callout-warn quick-callout">
          <Icon name="camera" size={16} />
          <div>
            <ul>{r.quality_warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
            <button type="button" className="btn btn-secondary btn-sm quick-retake" onClick={onRetake}>
              Retake photo
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function FormGroup({ icon, title, children }: { icon: Parameters<typeof Icon>[0]['name']; title: string; children: ReactNode }) {
  return (
    <fieldset className="form-group">
      <legend><Icon name={icon} size={15} /> {title}</legend>
      {children}
    </fieldset>
  );
}

function NumberField({
  label,
  unit,
  error,
  hint,
  htmlFor,
  children,
}: {
  label: string;
  unit?: string;
  error: string | null;
  hint?: string | null;
  htmlFor: string;
  children: ReactNode;
}) {
  return (
    <div className="field">
      <label className="field-label" htmlFor={htmlFor}>
        {label} {unit && <span className="unit">({unit})</span>}
      </label>
      {children}
      {error ? <span className="field-error">{error}</span> : hint ? <span className="field-hint">{hint}</span> : null}
    </div>
  );
}

function ChipGroup({
  value,
  options,
  onChange,
  labelledBy,
}: {
  value: string | null;
  options: { value: string; label: string; sub?: string }[];
  onChange: (v: string | null) => void;
  labelledBy: string;
}) {
  return (
    <div className="chips" role="radiogroup" aria-labelledby={labelledBy}>
      {options.map(o => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          className="chip"
          // Clicking the selected chip clears it — every field is optional.
          onClick={() => onChange(value === o.value ? null : o.value)}
        >
          {o.label}
          {o.sub && <span className="chip-sub">{o.sub}</span>}
        </button>
      ))}
    </div>
  );
}

function Analyzing({ previewUrl }: { previewUrl: string | null }) {
  // The API is a single request, so stage progress is indicative: it advances on
  // a timer and holds on the last stage until the response lands.
  const [active, setActive] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setActive(a => Math.min(a + 1, STAGES.length - 1)), 900);
    return () => window.clearInterval(id);
  }, []);

  return (
    <section className="analyzing card rise" id="analyzing-step" aria-live="polite" aria-busy="true">
      <div className="analyzing-visual">
        {previewUrl && <img src={previewUrl} alt="" />}
        <div className="scanline" />
      </div>
      <div className="analyzing-text">
        <h1>Analyzing your leaf</h1>
        <p>This usually takes a few seconds.</p>
        <ol className="stages">
          {STAGES.map((s, i) => (
            <li key={s.label} className={i < active ? 'is-done' : i === active ? 'is-active' : ''}>
              <span className="stage-icon">
                {i < active ? <Icon name="check" size={13} strokeWidth={3} /> : i === active ? <span className="spinner" /> : null}
              </span>
              <span>
                <span className="stage-label">{s.label}</span>
                <span className="stage-detail">{s.detail}</span>
              </span>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

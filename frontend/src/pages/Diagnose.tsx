/* ─── Diagnose Page ────────────────────────────────────────────
   Three-step flow:
     1. Capture — upload or take a photo
     2. Context — fill environment fields (persisted in localStorage)
     3. Result — severity gauge, overlay, diagnosis, treatment plan
   ──────────────────────────────────────────────────────────── */

import { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { postDiagnose } from '../api';
import { loadEnvironment, saveEnvironment, saveHistoryEntry, generateId, fileToDataUrl } from '../store';
import type { EnvironmentContext, AdvisoryResponse, ActionTier } from '../types';
import { TIER_CONFIG } from '../types';
import ResultView from '../components/ResultView';
import './Diagnose.css';

type Step = 'capture' | 'context' | 'analyzing' | 'result';

export default function Diagnose() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [step, setStep] = useState<Step>('capture');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [env, setEnv] = useState<EnvironmentContext>(loadEnvironment());
  const [result, setResult] = useState<AdvisoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  /* ── File handling ─────────────────────────────────────────── */
  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setError(null);
    setStep('context');
  }, []);

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      handleFileSelect(file);
    }
  }, [handleFileSelect]);

  /* ── Environment form ──────────────────────────────────────── */
  const updateEnv = <K extends keyof EnvironmentContext>(key: K, value: EnvironmentContext[K]) => {
    setEnv(prev => ({ ...prev, [key]: value }));
  };

  /* ── Submit ─────────────────────────────────────────────────── */
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    // persist env so it's remembered next time
    saveEnvironment(env);
    setStep('analyzing');
    setError(null);

    try {
      const response = await postDiagnose(selectedFile, env, true);
      setResult(response);

      // Save to history
      const imageDataUrl = await fileToDataUrl(selectedFile);
      saveHistoryEntry({
        id: generateId(),
        timestamp: new Date().toISOString(),
        imageDataUrl,
        response,
      });

      setStep('result');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Something went wrong';
      setError(message);
      setStep('context');
    }
  };

  /* ── Reset ──────────────────────────────────────────────────── */
  const handleReset = () => {
    setStep('capture');
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  /* ── Render ─────────────────────────────────────────────────── */
  return (
    <div className="diagnose-page container">
      {/* Step Indicator */}
      <div className="step-indicator" id="step-indicator">
        {(['capture', 'context', 'result'] as const).map((s, i) => (
          <div key={s} className={`step-dot-wrapper ${step === s || (step === 'analyzing' && s === 'result') ? 'active' : ''} ${
            (s === 'capture' && step !== 'capture') || (s === 'context' && (step === 'analyzing' || step === 'result')) ? 'completed' : ''
          }`}>
            <div className="step-dot">{i + 1}</div>
            <span className="step-label">{s === 'capture' ? 'Upload' : s === 'context' ? 'Context' : 'Results'}</span>
          </div>
        ))}
      </div>

      {/* ═══ STEP 1: Capture ═══ */}
      {step === 'capture' && (
        <div className="capture-step animate-scale-in" id="capture-step">
          <h2>Upload a Leaf Photo</h2>
          <p className="step-desc">Take or upload a clear photo of the affected leaf for diagnosis</p>

          <div
            className="dropzone glass-card"
            id="dropzone"
            onClick={() => fileInputRef.current?.click()}
            onDrop={onDrop}
            onDragOver={e => e.preventDefault()}
            onDragEnter={e => { e.preventDefault(); e.currentTarget.classList.add('drag-over'); }}
            onDragLeave={e => { e.preventDefault(); e.currentTarget.classList.remove('drag-over'); }}
          >
            <div className="dropzone-content">
              <div className="dropzone-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
              </div>
              <p className="dropzone-text">
                Drag & drop your leaf image here<br/>
                <span className="dropzone-hint">or click to browse · JPEG, PNG, WebP · Max 12 MB</span>
              </p>
            </div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            capture="environment"
            onChange={onFileChange}
            style={{ display: 'none' }}
            id="file-input"
          />

          <button
            className="btn btn-secondary"
            style={{ marginTop: 16 }}
            onClick={() => fileInputRef.current?.click()}
            id="browse-btn"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
            Take a Photo
          </button>
        </div>
      )}

      {/* ═══ STEP 2: Context ═══ */}
      {step === 'context' && (
        <div className="context-step animate-scale-in" id="context-step">
          <div className="context-layout">
            {/* Image Preview */}
            <div className="preview-panel glass-card">
              <h3>Selected Image</h3>
              {previewUrl && (
                <div className="preview-image-wrapper">
                  <img src={previewUrl} alt="Selected leaf" className="preview-image" id="preview-image" />
                </div>
              )}
              <button className="btn btn-ghost" onClick={handleReset} id="change-image-btn">
                ← Change Image
              </button>
            </div>

            {/* Environment Form */}
            <div className="env-form glass-card">
              <h3>Field Context <span className="optional-tag">(all optional)</span></h3>
              <p className="form-hint">This info improves the advisory. It's saved for your next visit.</p>

              {error && (
                <div className="error-banner" id="error-banner">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
                  </svg>
                  {error}
                </div>
              )}

              <div className="form-grid">
                <div className="input-group">
                  <label className="input-label">Region</label>
                  <input
                    className="input-field"
                    type="text"
                    placeholder="e.g. Nashik, Maharashtra"
                    value={env.region ?? ''}
                    onChange={e => updateEnv('region', e.target.value || null)}
                    id="input-region"
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Soil Type</label>
                  <select
                    className="input-field"
                    value={env.soil_type ?? 'unknown'}
                    onChange={e => updateEnv('soil_type', e.target.value as EnvironmentContext['soil_type'])}
                    id="input-soil-type"
                  >
                    <option value="unknown">Select soil…</option>
                    <option value="alluvial">Alluvial</option>
                    <option value="black">Black (Cotton)</option>
                    <option value="red">Red</option>
                    <option value="laterite">Laterite</option>
                    <option value="sandy">Sandy</option>
                    <option value="loam">Loam</option>
                    <option value="clay">Clay</option>
                  </select>
                </div>

                <div className="input-group">
                  <label className="input-label">Season</label>
                  <select
                    className="input-field"
                    value={env.season ?? 'unknown'}
                    onChange={e => updateEnv('season', e.target.value as EnvironmentContext['season'])}
                    id="input-season"
                  >
                    <option value="unknown">Select season…</option>
                    <option value="kharif">Kharif (Monsoon)</option>
                    <option value="rabi">Rabi (Winter)</option>
                    <option value="zaid">Zaid (Summer)</option>
                  </select>
                </div>

                <div className="input-group">
                  <label className="input-label">Irrigation</label>
                  <select
                    className="input-field"
                    value={env.irrigation ?? 'unknown'}
                    onChange={e => updateEnv('irrigation', e.target.value as EnvironmentContext['irrigation'])}
                    id="input-irrigation"
                  >
                    <option value="unknown">Select type…</option>
                    <option value="rainfed">Rainfed</option>
                    <option value="flood">Flood</option>
                    <option value="drip">Drip</option>
                    <option value="sprinkler">Sprinkler</option>
                  </select>
                </div>

                <div className="input-group">
                  <label className="input-label">Rainfall (last 7d, mm)</label>
                  <input
                    className="input-field"
                    type="number"
                    min="0"
                    max="2000"
                    placeholder="e.g. 118"
                    value={env.rainfall_last_7d_mm ?? ''}
                    onChange={e => updateEnv('rainfall_last_7d_mm', e.target.value ? Number(e.target.value) : null)}
                    id="input-rainfall"
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Temperature (°C)</label>
                  <input
                    className="input-field"
                    type="number"
                    min="-20"
                    max="60"
                    placeholder="e.g. 28"
                    value={env.temperature_c ?? ''}
                    onChange={e => updateEnv('temperature_c', e.target.value ? Number(e.target.value) : null)}
                    id="input-temperature"
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Humidity (%)</label>
                  <input
                    className="input-field"
                    type="number"
                    min="0"
                    max="100"
                    placeholder="e.g. 72"
                    value={env.relative_humidity_pct ?? ''}
                    onChange={e => updateEnv('relative_humidity_pct', e.target.value ? Number(e.target.value) : null)}
                    id="input-humidity"
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Crop Stage</label>
                  <input
                    className="input-field"
                    type="text"
                    placeholder="e.g. flowering, fruiting"
                    value={env.crop_stage ?? ''}
                    onChange={e => updateEnv('crop_stage', e.target.value || null)}
                    id="input-crop-stage"
                  />
                </div>
              </div>

              {/* Organic toggle */}
              <div
                className="toggle-wrapper"
                onClick={() => updateEnv('organic_only', !env.organic_only)}
                id="organic-toggle"
                style={{ marginTop: 16 }}
              >
                <div className={`toggle-track ${env.organic_only ? 'active' : ''}`}>
                  <div className="toggle-thumb" />
                </div>
                <span className="input-label" style={{ cursor: 'pointer' }}>Organic only (no synthetic chemicals)</span>
              </div>

              {/* Submit */}
              <div className="form-actions">
                <button className="btn btn-ghost" onClick={handleReset}>
                  ← Back
                </button>
                <button
                  className="btn btn-primary btn-lg"
                  onClick={handleAnalyze}
                  id="analyze-btn"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                  </svg>
                  Analyze Leaf
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ═══ ANALYZING STATE ═══ */}
      {step === 'analyzing' && (
        <div className="analyzing-step animate-scale-in" id="analyzing-step">
          <div className="analyzing-card glass-card">
            <div className="analyzing-visual">
              <div className="analyzing-ring">
                <div className="spinner spinner-lg" />
              </div>
              {previewUrl && (
                <img src={previewUrl} alt="Analyzing" className="analyzing-thumb" />
              )}
            </div>
            <h3>Analyzing Your Leaf</h3>
            <p className="analyzing-desc">
              Running disease classification, severity measurement,
              knowledge retrieval, and treatment synthesis…
            </p>
            <div className="analyzing-steps stagger">
              <AnalysisStepIndicator label="ResNet-50 Classification" />
              <AnalysisStepIndicator label="OpenCV Severity Index" />
              <AnalysisStepIndicator label="RAG Knowledge Retrieval" />
              <AnalysisStepIndicator label="LLM Treatment Synthesis" />
            </div>
          </div>
        </div>
      )}

      {/* ═══ STEP 3: Result ═══ */}
      {step === 'result' && result && (
        <div className="result-step animate-slide-up" id="result-step">
          <ResultView
            result={result}
            previewUrl={previewUrl}
            onReset={handleReset}
            onNewDiagnosis={() => navigate('/diagnose')}
          />
        </div>
      )}
    </div>
  );
}

/* Small sub-component for the analyzing animation */
function AnalysisStepIndicator({ label }: { label: string }) {
  return (
    <div className="analysis-step-item">
      <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
      <span>{label}</span>
    </div>
  );
}

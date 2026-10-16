/* ─── ResultView — displays the full advisory response ─────── */

import type { AdvisoryResponse, ActionTier } from '../types';
import { TIER_CONFIG } from '../types';
import './ResultView.css';

interface ResultViewProps {
  result: AdvisoryResponse;
  previewUrl: string | null;
  onReset: () => void;
  onNewDiagnosis?: () => void;
  isHistoryView?: boolean;
}

export default function ResultView({ result, previewUrl, onReset, isHistoryView }: ResultViewProps) {
  const { diagnosis, severity, plan, evidence, readiness } = result;
  const tierCfg = TIER_CONFIG[severity.action_tier];

  return (
    <div className="result-view">
      {/* ── Degradation warnings ─────────────────────────────── */}
      {readiness.degraded.length > 0 && (
        <div className="degraded-banner" id="degraded-banner">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          <div>
            <strong>Degraded Components</strong>
            <ul>
              {readiness.degraded.map((d, i) => <li key={i}>{d}</li>)}
            </ul>
          </div>
        </div>
      )}

      {/* ── Top Row: Image + Severity Gauge + Diagnosis ────── */}
      <div className="result-top-row">
        {/* Image & Overlay */}
        <div className="result-images glass-card">
          <div className="result-images-inner">
            {previewUrl && (
              <div className="result-img-wrapper">
                <span className="img-label">Original</span>
                <img src={previewUrl} alt="Uploaded leaf" className="result-img" />
              </div>
            )}
            {result.debug_overlay_png_b64 && (
              <div className="result-img-wrapper">
                <span className="img-label">Severity Mask</span>
                <img
                  src={`data:image/png;base64,${result.debug_overlay_png_b64}`}
                  alt="Severity overlay"
                  className="result-img"
                  id="severity-overlay-img"
                />
                <div className="overlay-legend">
                  <span className="legend-item"><span className="legend-dot" style={{ background: '#22c55e' }} /> Healthy</span>
                  <span className="legend-item"><span className="legend-dot" style={{ background: '#fbbf24' }} /> Chlorotic</span>
                  <span className="legend-item"><span className="legend-dot" style={{ background: '#ef4444' }} /> Necrotic</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Severity Gauge */}
        <div className="severity-panel glass-card" id="severity-panel">
          <h3 className="panel-title">Severity Index</h3>

          <div className="severity-gauge">
            <SeverityRing pct={severity.severity_pct} tier={severity.action_tier} />
          </div>

          <div className="tier-badge-large" style={{
            background: tierCfg.bgColor,
            borderColor: tierCfg.borderColor,
            color: tierCfg.color,
          }}>
            <span className="tier-icon">{tierCfg.icon}</span>
            <span>{tierCfg.headline}</span>
          </div>

          <div className="severity-breakdown stagger">
            <BreakdownRow label="Healthy" pct={severity.healthy_pct} color="#22c55e" />
            <BreakdownRow label="Chlorotic" pct={severity.chlorotic_pct} color="#fbbf24" />
            <BreakdownRow label="Necrotic" pct={severity.necrotic_pct} color="#ef4444" />
          </div>

          <div className="severity-stats">
            <div className="stat-item">
              <span className="stat-value">{severity.lesion_count}</span>
              <span className="stat-label">Lesions</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{severity.largest_lesion_pct.toFixed(1)}%</span>
              <span className="stat-label">Largest</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{severity.leaf_coverage_pct.toFixed(0)}%</span>
              <span className="stat-label">Coverage</span>
            </div>
          </div>

          {severity.quality_warnings.length > 0 && (
            <div className="quality-warnings">
              {severity.quality_warnings.map((w, i) => (
                <span key={i} className="badge badge-amber" style={{ fontSize: '0.7rem' }}>⚠ {w}</span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Diagnosis Card ────────────────────────────────────── */}
      <div className="diagnosis-card glass-card animate-slide-up" style={{ animationDelay: '100ms' }} id="diagnosis-card">
        <h3 className="panel-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          Disease Classification
        </h3>
        <div className="diagnosis-main">
          <div className="diagnosis-top-prediction">
            <div>
              <span className="diagnosis-crop">{diagnosis.top.crop}</span>
              <span className="diagnosis-disease">{diagnosis.top.disease}</span>
            </div>
            <div className="diagnosis-confidence">
              <span className="confidence-value">{(diagnosis.top.confidence * 100).toFixed(1)}%</span>
              <span className="confidence-label">confidence</span>
            </div>
          </div>
          {!diagnosis.checkpoint_loaded && (
            <div className="badge badge-amber" style={{ marginTop: 8 }}>⚠ No trained checkpoint — placeholder prediction</div>
          )}
          {!diagnosis.confidence_reliable && diagnosis.checkpoint_loaded && (
            <div className="badge badge-amber" style={{ marginTop: 8 }}>⚠ Low confidence — treat as provisional</div>
          )}
          {diagnosis.is_healthy && (
            <div className="badge badge-green" style={{ marginTop: 8 }}>✓ Classified as healthy</div>
          )}
        </div>

        {diagnosis.alternatives.length > 0 && (
          <div className="alternatives">
            <span className="alternatives-label">Alternatives:</span>
            {diagnosis.alternatives.slice(0, 3).map((alt, i) => (
              <span key={i} className="alternative-item">
                {alt.crop} — {alt.disease} ({(alt.confidence * 100).toFixed(1)}%)
              </span>
            ))}
          </div>
        )}
      </div>

      {/* ── Treatment Plan ────────────────────────────────────── */}
      <div className="plan-card glass-card animate-slide-up" style={{ animationDelay: '200ms' }} id="treatment-plan">
        <h3 className="panel-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
          </svg>
          Treatment Plan
          <span className="badge badge-cyan" style={{ marginLeft: 8, fontSize: '0.65rem' }}>
            via {plan.generated_by}
          </span>
        </h3>

        <p className="plan-summary">{plan.summary}</p>

        {plan.unsupported_claims_removed > 0 && (
          <div className="badge badge-amber" style={{ marginBottom: 16 }}>
            ⚠ {plan.unsupported_claims_removed} unsupported claim{plan.unsupported_claims_removed > 1 ? 's' : ''} removed
          </div>
        )}

        {/* Immediate Actions */}
        {plan.immediate_actions.length > 0 && (
          <div className="plan-section">
            <h4 className="plan-section-title">⚡ Immediate Actions</h4>
            <div className="plan-steps stagger">
              {plan.immediate_actions.map(step => (
                <PlanStepCard key={step.order} step={step} />
              ))}
            </div>
          </div>
        )}

        {/* Follow-up */}
        {plan.followup_actions.length > 0 && (
          <div className="plan-section">
            <h4 className="plan-section-title">📋 Follow-up Actions</h4>
            <div className="plan-steps stagger">
              {plan.followup_actions.map(step => (
                <PlanStepCard key={step.order} step={step} />
              ))}
            </div>
          </div>
        )}

        {/* Chemical options */}
        {plan.chemical_options.length > 0 && (
          <div className="plan-section">
            <h4 className="plan-section-title">🧪 Chemical Options</h4>
            <ul className="plan-list">
              {plan.chemical_options.map((opt, i) => <li key={i}>{opt}</li>)}
            </ul>
          </div>
        )}

        {/* Cultural practices */}
        {plan.cultural_practices.length > 0 && (
          <div className="plan-section">
            <h4 className="plan-section-title">🌱 Cultural Practices</h4>
            <ul className="plan-list">
              {plan.cultural_practices.map((p, i) => <li key={i}>{p}</li>)}
            </ul>
          </div>
        )}

        {/* Monitoring */}
        {plan.monitoring && (
          <div className="plan-section">
            <h4 className="plan-section-title">👁 Monitoring</h4>
            <p className="plan-monitoring">{plan.monitoring}</p>
          </div>
        )}

        {/* Cautions */}
        {plan.cautions.length > 0 && (
          <div className="plan-section plan-cautions">
            <h4 className="plan-section-title">⚠ Cautions</h4>
            <ul className="plan-list plan-list-caution">
              {plan.cautions.map((c, i) => <li key={i}>{c}</li>)}
            </ul>
          </div>
        )}
      </div>

      {/* ── Evidence / Citations ──────────────────────────────── */}
      {evidence.length > 0 && (
        <div className="evidence-card glass-card animate-slide-up" style={{ animationDelay: '300ms' }} id="evidence-section">
          <h3 className="panel-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
            </svg>
            Supporting Evidence
          </h3>
          <div className="evidence-list stagger">
            {evidence.map(e => (
              <div key={e.id} className="evidence-item">
                <div className="evidence-header">
                  <span className="evidence-id">[{e.id}]</span>
                  {e.source && <span className="evidence-source">{e.source}</span>}
                  {e.doc_type && <span className="badge badge-cyan" style={{ fontSize: '0.65rem' }}>{e.doc_type}</span>}
                </div>
                <p className="evidence-text">{e.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Performance & Disclaimer ─────────────────────────── */}
      <div className="result-footer animate-slide-up" style={{ animationDelay: '400ms' }}>
        {Object.keys(result.elapsed_ms).length > 0 && (
          <div className="perf-stats">
            {Object.entries(result.elapsed_ms).map(([key, ms]) => (
              <span key={key} className="perf-item">
                {key}: <strong>{ms.toFixed(0)}ms</strong>
              </span>
            ))}
          </div>
        )}

        <div className="disclaimer-box" id="disclaimer">
          <p>{result.disclaimer}</p>
        </div>

        <div className="result-actions">
          {!isHistoryView && (
            <button className="btn btn-primary" onClick={onReset} id="new-diagnosis-btn">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10"/>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
              </svg>
              New Diagnosis
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─── Sub-components ──────────────────────────────────────── */

function SeverityRing({ pct, tier }: { pct: number; tier: ActionTier }) {
  const cfg = TIER_CONFIG[tier];
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="severity-ring-wrapper">
      <svg viewBox="0 0 140 140" className="severity-ring-svg">
        {/* Background ring */}
        <circle cx="70" cy="70" r={radius} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="10" />
        {/* Progress ring */}
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke={cfg.color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{
            transform: 'rotate(-90deg)',
            transformOrigin: '50% 50%',
            transition: 'stroke-dashoffset 1.2s cubic-bezier(0.16, 1, 0.3, 1)',
            filter: `drop-shadow(0 0 8px ${cfg.color}50)`,
          }}
        />
      </svg>
      <div className="severity-ring-center">
        <span className="severity-ring-value" style={{ color: cfg.color }}>
          {pct.toFixed(1)}
        </span>
        <span className="severity-ring-unit">%</span>
        <span className="severity-ring-label">damaged</span>
      </div>
    </div>
  );
}

function BreakdownRow({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div className="breakdown-row">
      <div className="breakdown-info">
        <span className="breakdown-dot" style={{ background: color }} />
        <span className="breakdown-label">{label}</span>
        <span className="breakdown-value">{pct.toFixed(1)}%</span>
      </div>
      <div className="breakdown-bar-track">
        <div
          className="breakdown-bar-fill"
          style={{
            width: `${Math.min(pct, 100)}%`,
            background: color,
            boxShadow: `0 0 8px ${color}40`,
          }}
        />
      </div>
    </div>
  );
}

function PlanStepCard({ step }: { step: { order: number; action: string; timing: string; rationale: string; citations: string[] } }) {
  return (
    <div className="plan-step-card">
      <div className="plan-step-header">
        <span className="plan-step-order">{step.order}</span>
        <div className="plan-step-meta">
          <span className="plan-step-timing">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
            {step.timing}
          </span>
          {step.citations.length > 0 && (
            <span className="plan-step-citations">
              {step.citations.map(c => `[${c}]`).join(' ')}
            </span>
          )}
        </div>
      </div>
      <p className="plan-step-action">{step.action}</p>
      {step.rationale && (
        <p className="plan-step-rationale">{step.rationale}</p>
      )}
    </div>
  );
}

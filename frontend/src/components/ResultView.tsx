/* ─── ResultView — the advisory report ─────────────────────────
   Reading order follows the farmer's questions:
     1. How bad is it, and what do I do?   → verdict card
     2. Why should I believe that?         → mask, tissue split, ID
     3. What exactly, and when?            → treatment plan
     4. Says who?                          → cited evidence
   ──────────────────────────────────────────────────────────── */

import { useState, type ReactNode } from 'react';
import type { AdvisoryResponse, Diagnosis, Evidence, PlanStep } from '../types';
import { TIER_CONFIG } from '../types';
import Icon, { type IconName } from './Icon';
import ImageCompare from './ImageCompare';
import { Markdown } from './Markdown';
import { markdownPreview, markdownTitle, overlaySrc } from '../format';
import { TierPill, TierScale, TissueBar } from './Severity';
import './ResultView.css';

interface ResultViewProps {
  result: AdvisoryResponse;
  previewUrl: string | null;
  onReset: () => void;
  isHistoryView?: boolean;
  timestamp?: string;
}

const PROVIDER_LABEL: Record<string, string> = {
  groq: 'Groq · Llama 3.3',
  gemini: 'Google Gemini',
  template: 'Rule-based template',
};

const STAGE_LABEL: Record<string, string> = {
  decode: 'Decode image',
  severity: 'Severity index',
  classify: 'Classification',
  retrieve: 'Retrieval',
  synthesize: 'Plan synthesis',
  overlay: 'Mask render',
};

export default function ResultView({ result, previewUrl, onReset, isHistoryView, timestamp }: ResultViewProps) {
  const { diagnosis, severity, plan, evidence, readiness } = result;
  const tier = TIER_CONFIG[severity.action_tier];
  const [openEvidence, setOpenEvidence] = useState<Set<string>>(new Set());
  const evidenceIds = new Set(evidence.map(e => e.id));

  const jumpToEvidence = (id: string) => {
    setOpenEvidence(prev => new Set(prev).add(id));
    requestAnimationFrame(() => {
      const el = document.getElementById(`evidence-${id}`);
      if (!el) return;
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      el.classList.remove('flash');
      void el.offsetWidth; // restart the highlight animation
      el.classList.add('flash');
    });
  };

  /** Turn inline "[S1]" markers into links to the evidence card. */
  const cite = (text: string): ReactNode[] =>
    text.split(/(\[S\d+\])/g).map((part, i) => {
      const m = /^\[(S\d+)\]$/.exec(part);
      if (!m) return part;
      return <CiteChip key={i} id={m[1]} known={evidenceIds.has(m[1])} onJump={jumpToEvidence} />;
    });

  const when = timestamp ? new Date(timestamp) : null;

  return (
    <article className="report">
      {/* ── 1. Verdict ─────────────────────────────────────────── */}
      <section className="card verdict rise" aria-labelledby="verdict-title">
        <div className="verdict-accent" style={{ background: tier.color }} />
        <div className="verdict-head">
          <div className="eyebrow">
            Field report{when && ` · ${when.toLocaleString('en-IN', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })}`}
          </div>
          <div className="verdict-actions no-print">
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => window.print()}>
              <Icon name="printer" size={16} /> Print / PDF
            </button>
            {!isHistoryView && (
              <button type="button" className="btn btn-primary btn-sm" onClick={onReset} id="new-diagnosis-btn">
                <Icon name="camera" size={16} /> New diagnosis
              </button>
            )}
          </div>
        </div>

        <div className="verdict-body">
          <div className="verdict-stat">
            <div className="verdict-pct num">
              {severity.severity_pct.toFixed(1)}<span>%</span>
            </div>
            <div className="verdict-pct-label">of leaf area damaged</div>
          </div>
          <div className="verdict-text">
            <TierPill tier={severity.action_tier} size="lg" />
            <h1 id="verdict-title">{tier.headline}</h1>
            <p className="verdict-id">
              <DiagnosisLine diagnosis={diagnosis} />
            </p>
          </div>
        </div>

        <TierScale pct={severity.severity_pct} tier={severity.action_tier} />

        <div className="verdict-posture">
          <Icon name="info" size={16} />
          <span><strong>What this tier allows:</strong> {tier.posture}</span>
        </div>
      </section>

      {readiness.degraded.length > 0 && <LimitedModeNotice notes={readiness.degraded} />}

      <div className="report-grid">
        {/* ── 2. Evidence from the photo ───────────────────────── */}
        <aside className="report-side">
          <section className="card card-pad rise" style={{ animationDelay: '60ms' }} aria-labelledby="measure-title">
            <h2 className="card-title" id="measure-title">
              <Icon name="scan" /> What the camera measured
            </h2>
            {previewUrl && (
              <>
                <ImageCompare original={previewUrl} overlay={overlaySrc(result.debug_overlay_png_b64)} maxHeight={360} />
                {result.debug_overlay_png_b64 && (
                  <p className="compare-hint no-print">Drag across the photo to compare it with the tissue mask.</p>
                )}
              </>
            )}

            <div className="measure-block">
              <TissueBar severity={severity} />
            </div>

            <div className="stat-row">
              <Stat value={String(severity.lesion_count)} label="Lesions" hint="discrete damaged spots" />
              <Stat value={`${severity.largest_lesion_pct.toFixed(1)}%`} label="Largest" hint="biggest lesion, % of leaf" />
              <Stat value={`${severity.leaf_coverage_pct.toFixed(0)}%`} label="Leaf in frame" hint="share of photo that is leaf" />
            </div>

            {severity.quality_warnings.length > 0 && (
              <div className="callout callout-warn quality">
                <Icon name="camera" size={16} />
                <div>
                  <strong>Photo quality</strong>
                  <ul>{severity.quality_warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
                </div>
              </div>
            )}
          </section>

          <DiagnosisCard diagnosis={diagnosis} />
        </aside>

        {/* ── 3. Plan & 4. Sources ─────────────────────────────── */}
        <div className="report-main">
          <section className="card card-pad rise" style={{ animationDelay: '120ms' }} id="treatment-plan" aria-labelledby="plan-title">
            <h2 className="card-title" id="plan-title">
              <Icon name="file" /> Treatment plan
              <span className="spacer" />
              <span className="badge" title="Which engine wrote this plan">
                <Icon name="sparkles" size={12} /> {PROVIDER_LABEL[plan.generated_by] ?? plan.generated_by}
              </span>
            </h2>

            <p className="plan-summary">{cite(plan.summary)}</p>

            {plan.unsupported_claims_removed > 0 && (
              <div className="callout callout-info plan-note">
                <Icon name="check" size={16} />
                <span>
                  <strong>Safety filter:</strong> {plan.unsupported_claims_removed} recommendation
                  {plan.unsupported_claims_removed > 1 ? 's were' : ' was'} removed because no retrieved source supported
                  {plan.unsupported_claims_removed > 1 ? ' them' : ' it'}.
                </span>
              </div>
            )}

            {plan.immediate_actions.length > 0 && (
              <PlanSection icon="zap" title="Do now" count={plan.immediate_actions.length}>
                <ol className="timeline">
                  {plan.immediate_actions.map(step => <StepItem key={step.order} step={step} cite={cite} onJump={jumpToEvidence} known={evidenceIds} />)}
                </ol>
              </PlanSection>
            )}

            {plan.followup_actions.length > 0 && (
              <PlanSection icon="clock" title="Follow up" count={plan.followup_actions.length}>
                <ol className="timeline">
                  {plan.followup_actions.map(step => <StepItem key={step.order} step={step} cite={cite} onJump={jumpToEvidence} known={evidenceIds} />)}
                </ol>
              </PlanSection>
            )}

            {plan.chemical_options.length > 0 && (
              <PlanSection icon="flask" title="Chemical options">
                <ul className="plain-list">
                  {plan.chemical_options.map((opt, i) => <li key={i}>{cite(opt)}</li>)}
                </ul>
                <p className="fine-print">Always follow the product label for rate, re-entry and pre-harvest interval.</p>
              </PlanSection>
            )}

            {plan.cultural_practices.length > 0 && (
              <PlanSection icon="sprout" title="Field practices">
                <ul className="plain-list">
                  {plan.cultural_practices.map((p, i) => <li key={i}>{cite(p)}</li>)}
                </ul>
              </PlanSection>
            )}

            {plan.monitoring && (
              <PlanSection icon="eye" title="Monitoring">
                <p className="plan-text">{cite(plan.monitoring)}</p>
              </PlanSection>
            )}

            {plan.cautions.length > 0 && (
              <div className="callout callout-warn plan-cautions">
                <Icon name="alert" size={16} />
                <div>
                  <strong>Cautions</strong>
                  <ul>{plan.cautions.map((c, i) => <li key={i}>{cite(c)}</li>)}</ul>
                </div>
              </div>
            )}
          </section>

          {evidence.length > 0 && (
            <section className="card card-pad rise" style={{ animationDelay: '180ms' }} id="evidence-section" aria-labelledby="evidence-title">
              <h2 className="card-title" id="evidence-title">
                <Icon name="book" /> Sources used
                <span className="spacer" />
                <span className="muted card-title-meta">{evidence.length} retrieved</span>
              </h2>
              <p className="section-intro">
                Passages retrieved from the agricultural knowledge base. Plan steps cite them as <span className="cite cite-static">S1</span>.
              </p>
              <div className="evidence-list">
                {evidence.map(e => (
                  <EvidenceItem
                    key={e.id}
                    evidence={e}
                    open={openEvidence.has(e.id)}
                    onToggle={open =>
                      setOpenEvidence(prev => {
                        const next = new Set(prev);
                        if (open) next.add(e.id);
                        else next.delete(e.id);
                        return next;
                      })
                    }
                  />
                ))}
              </div>
            </section>
          )}
        </div>
      </div>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <footer className="report-footer">
        <div className="callout callout-neutral" id="disclaimer">
          <Icon name="info" size={16} />
          <span>{result.disclaimer}</span>
        </div>
        <TechnicalDetails result={result} />
      </footer>
    </article>
  );
}

/* ─── Sub-components ──────────────────────────────────────── */

function DiagnosisLine({ diagnosis }: { diagnosis: Diagnosis }) {
  const { top } = diagnosis;
  if (!diagnosis.checkpoint_loaded) {
    return <span className="muted">Crop and disease not identified — the classifier model isn’t trained yet.</span>;
  }
  if (diagnosis.is_healthy) {
    return <>Looks like a healthy <strong>{top.crop}</strong> leaf</>;
  }
  return (
    <>
      {diagnosis.confidence_reliable ? 'Identified as ' : 'Possibly '}
      <strong>{top.disease}</strong> on <strong>{top.crop}</strong>
      <span className="muted"> · {(top.confidence * 100).toFixed(0)}% confidence</span>
    </>
  );
}

function DiagnosisCard({ diagnosis }: { diagnosis: Diagnosis }) {
  const { top, alternatives } = diagnosis;
  const candidates = [top, ...alternatives];

  return (
    <section className="card card-pad rise" style={{ animationDelay: '100ms' }} id="diagnosis-card" aria-labelledby="id-title">
      <h2 className="card-title" id="id-title">
        <Icon name="search" /> Disease identification
      </h2>

      {!diagnosis.checkpoint_loaded ? (
        <div className="id-empty">
          <div className="id-empty-icon"><Icon name="cpu" size={20} /></div>
          <div>
            <strong>Model not trained yet</strong>
            <p>
              No ResNet-50 checkpoint is loaded, so any label would be a random guess and is hidden. The severity
              measurement above doesn’t depend on it. Train with <code>ml/train.py</code> to enable identification.
            </p>
          </div>
        </div>
      ) : (
        <>
          <div className="id-top">
            <div>
              <div className="id-crop">{top.crop}</div>
              <div className="id-disease">{diagnosis.is_healthy ? 'Healthy' : top.disease}</div>
            </div>
            {!diagnosis.confidence_reliable && <span className="badge badge-warn">Low confidence</span>}
          </div>
          <ul className="conf-list" aria-label="Model confidence by class">
            {candidates.map((c, i) => (
              <li key={c.label} className={i === 0 ? 'is-top' : ''}>
                <span className="conf-name">{c.crop} · {c.disease}</span>
                <span className="conf-track"><span className="conf-fill" style={{ width: `${Math.max(c.confidence * 100, 1)}%` }} /></span>
                <span className="conf-val num">{(c.confidence * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
          {!diagnosis.confidence_reliable && (
            <p className="fine-print">Below the confidence floor — confirm the identification with an extension officer before treating.</p>
          )}
        </>
      )}
    </section>
  );
}

function Stat({ value, label, hint }: { value: string; label: string; hint: string }) {
  return (
    <div className="stat" title={hint}>
      <span className="stat-value num">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

function PlanSection({ icon, title, count, children }: { icon: IconName; title: string; count?: number; children: ReactNode }) {
  return (
    <div className="plan-section">
      <h3 className="plan-section-title">
        <Icon name={icon} size={15} />
        {title}
        {count !== undefined && <span className="plan-count">{count}</span>}
      </h3>
      {children}
    </div>
  );
}

function StepItem({
  step,
  cite,
  onJump,
  known,
}: {
  step: PlanStep;
  cite: (t: string) => ReactNode[];
  onJump: (id: string) => void;
  known: Set<string>;
}) {
  return (
    <li className="timeline-item">
      <span className="timeline-num num">{step.order}</span>
      <div className="timeline-body">
        <div className="timeline-meta">
          <span className="timing"><Icon name="clock" size={12} /> {step.timing}</span>
          {step.citations.map(c => <CiteChip key={c} id={c} known={known.has(c)} onJump={onJump} />)}
        </div>
        <p className="timeline-action">{cite(step.action)}</p>
        {step.rationale && <p className="timeline-why">{cite(step.rationale)}</p>}
      </div>
    </li>
  );
}

function CiteChip({ id, known, onJump }: { id: string; known: boolean; onJump: (id: string) => void }) {
  if (!known) return <span className="cite cite-static">{id}</span>;
  return (
    <a
      href={`#evidence-${id}`}
      className="cite"
      onClick={e => { e.preventDefault(); onJump(id); }}
      title={`Show source ${id}`}
    >
      {id}
    </a>
  );
}

const DOC_TYPE_LABEL: Record<string, string> = {
  extension_bulletin: 'Extension bulletin',
  ipm_guide: 'IPM guide',
  pesticide_label: 'Pesticide label',
};

function EvidenceItem({ evidence: e, open, onToggle }: { evidence: Evidence; open: boolean; onToggle: (open: boolean) => void }) {
  const title = markdownTitle(e.text) ?? e.source ?? e.id;
  const topic = [e.crop, e.disease].filter(v => v && v !== 'general').join(' · ');
  return (
    <details
      className="evidence disclosure"
      id={`evidence-${e.id}`}
      open={open}
      onToggle={ev => {
        const isOpen = (ev.currentTarget as HTMLDetailsElement).open;
        if (isOpen !== open) onToggle(isOpen);
      }}
    >
      <summary>
        <span className="evidence-id">{e.id}</span>
        <span className="evidence-head">
          <span className="evidence-title">{title}</span>
          <span className="evidence-meta">
            {e.doc_type && <span>{DOC_TYPE_LABEL[e.doc_type] ?? e.doc_type.replace(/_/g, ' ')}</span>}
            {topic && <span>{topic}</span>}
            <span className="num" title="Semantic similarity to the query">{Math.round(e.score * 100)}% match</span>
          </span>
          {!open && <span className="evidence-preview">{markdownPreview(e.text, 180)}</span>}
        </span>
        <Icon name="chevron-right" size={16} className="chev" />
      </summary>
      <div className="evidence-body">
        <Markdown text={e.text} skipTitle />
        {e.source && <p className="evidence-source">Source: {e.source}</p>}
      </div>
    </details>
  );
}

function LimitedModeNotice({ notes }: { notes: string[] }) {
  return (
    <details className="card limited disclosure no-print">
      <summary>
        <Icon name="info" size={16} />
        <span><strong>Running in limited mode</strong> — {notes.length} component{notes.length > 1 ? 's' : ''} using a fallback</span>
        <Icon name="chevron-right" size={16} className="chev" />
      </summary>
      <ul>{notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
    </details>
  );
}

function TechnicalDetails({ result }: { result: AdvisoryResponse }) {
  const entries = Object.entries(result.elapsed_ms);
  const total = entries.reduce((s, [, ms]) => s + ms, 0);
  const max = Math.max(...entries.map(([, ms]) => ms), 1);
  const r = result.readiness;
  return (
    <details className="tech disclosure no-print">
      <summary>
        <Icon name="cpu" size={15} />
        <span>Technical details</span>
        <span className="muted num">· {total.toFixed(0)} ms total</span>
        <Icon name="chevron-right" size={15} className="chev" />
      </summary>
      <div className="tech-grid">
        <div>
          <h4 className="eyebrow">Pipeline timing</h4>
          <ul className="timing-list">
            {entries.map(([k, ms]) => (
              <li key={k}>
                <span>{STAGE_LABEL[k] ?? k}</span>
                <span className="timing-track"><span style={{ width: `${(ms / max) * 100}%` }} /></span>
                <span className="num">{ms.toFixed(0)} ms</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="eyebrow">Components</h4>
          <dl className="kv">
            <dt>Classifier checkpoint</dt><dd>{r.checkpoint_loaded ? 'Loaded' : 'Not loaded'}</dd>
            <dt>Vector store</dt><dd>{r.vector_store}</dd>
            <dt>Documents indexed</dt><dd className="num">{r.documents_indexed}</dd>
            <dt>Plan generator</dt><dd>{PROVIDER_LABEL[r.llm_provider] ?? r.llm_provider}</dd>
          </dl>
        </div>
      </div>
    </details>
  );
}

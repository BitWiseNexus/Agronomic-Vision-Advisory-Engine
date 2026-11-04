/* ─── Severity primitives — tier pill, tier scale, tissue bar ── */

import type { ActionTier, SeverityReport } from '../types';
import { TIER_CONFIG, TIER_ORDER } from '../types';
import Icon from './Icon';
import './Severity.css';

/* Tier pill: colour + icon + label, so colour never carries meaning alone */
export function TierPill({ tier, size = 'md' }: { tier: ActionTier; size?: 'md' | 'lg' }) {
  const cfg = TIER_CONFIG[tier];
  return (
    <span className={`tier-pill ${size === 'lg' ? 'tier-pill-lg' : ''}`}>
      <span className="tier-pill-icon" style={{ background: cfg.color }}>
        <Icon name={cfg.icon} strokeWidth={2.75} />
      </span>
      {cfg.label}
    </span>
  );
}

/* Where a reading sits on the five ordinal tiers. Segments are equal width —
   a linear 0–100 axis would make the 0–2 % tier a sliver — and the marker
   is interpolated inside its own segment. */
export function TierScale({ pct, tier, compact = false }: { pct?: number; tier?: ActionTier; compact?: boolean }) {
  const seg = 100 / TIER_ORDER.length;
  let marker: number | null = null;
  if (pct !== undefined && tier) {
    const index = TIER_ORDER.indexOf(tier);
    const [lo, hi] = TIER_CONFIG[tier].range;
    const within = Math.min(Math.max((pct - lo) / (hi - lo), 0), 1);
    marker = (index + within) * seg;
  }

  return (
    <div className={`tier-scale ${compact ? 'tier-scale-compact' : ''}`}>
      <div className="tier-scale-track" role="img" aria-label={
        pct !== undefined && tier ? `${pct.toFixed(1)}% damaged — ${TIER_CONFIG[tier].label} tier` : 'Severity tiers'
      }>
        <div className="tier-scale-segs">
          {TIER_ORDER.map(t => (
            <div
              key={t}
              className={`tier-scale-seg ${tier && t !== tier ? 'is-dim' : ''}`}
              style={{ background: TIER_CONFIG[t].color }}
            />
          ))}
        </div>
        {marker !== null && (
          <div className="tier-scale-marker" style={{ left: `${marker}%` }}>
            <span className="tier-scale-marker-label num">{pct!.toFixed(1)}%</span>
          </div>
        )}
      </div>
      <div className="tier-scale-labels">
        {TIER_ORDER.map(t => {
          const [lo, hi] = TIER_CONFIG[t].range;
          return (
            <div key={t} className={`tier-scale-label ${tier === t ? 'is-active' : ''}`}>
              <span className="tier-scale-name">{TIER_CONFIG[t].label}</span>
              {!compact && (
                <span className="tier-scale-range num">
                  {hi >= 100 ? `≥ ${lo}%` : lo === 0 ? `< ${hi}%` : `${lo}–${hi}%`}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* Healthy / chlorotic / necrotic share of the leaf, as one stacked bar. */
export function TissueBar({ severity }: { severity: SeverityReport }) {
  const parts = [
    { key: 'healthy', label: 'Healthy', hint: 'green tissue', pct: severity.healthy_pct, color: 'var(--tissue-healthy)' },
    { key: 'chlorotic', label: 'Chlorotic', hint: 'yellowing', pct: severity.chlorotic_pct, color: 'var(--tissue-chlorotic)' },
    { key: 'necrotic', label: 'Necrotic', hint: 'dead / brown', pct: severity.necrotic_pct, color: 'var(--tissue-necrotic)' },
  ];
  const other = Math.max(0, 100 - parts.reduce((sum, p) => sum + p.pct, 0));

  return (
    <div className="tissue">
      <div className="tissue-bar" role="img" aria-label={parts.map(p => `${p.label} ${p.pct.toFixed(1)}%`).join(', ')}>
        {parts.filter(p => p.pct > 0).map(p => (
          <div key={p.key} className="tissue-seg" style={{ flexGrow: p.pct, background: p.color }} title={`${p.label}: ${p.pct.toFixed(1)}%`} />
        ))}
        {other >= 0.5 && <div className="tissue-seg tissue-seg-other" style={{ flexGrow: other }} title={`Unclassified: ${other.toFixed(1)}%`} />}
      </div>
      <dl className="tissue-legend">
        {parts.map(p => (
          <div key={p.key} className="tissue-legend-row">
            <dt>
              <span className="dot" style={{ background: p.color }} />
              {p.label}
              <span className="muted">· {p.hint}</span>
            </dt>
            <dd className="num">{p.pct.toFixed(1)}%</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

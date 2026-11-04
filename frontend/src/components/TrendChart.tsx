/* ─── TrendChart — severity over time across saved reports ─────
   One series, so no legend: the card title names it. Tier
   thresholds are recessive reference lines labelled in muted ink;
   each point is coloured by its tier and the tooltip repeats the
   tier as text, so colour never carries meaning alone.
   ──────────────────────────────────────────────────────────── */

import { useEffect, useMemo, useRef, useState } from 'react';
import type { HistoryEntry } from '../types';
import { TIER_CONFIG } from '../types';
import { TierPill } from './Severity';
import './TrendChart.css';

const HEIGHT = 220;
const PAD = { top: 14, right: 84, bottom: 28, left: 40 };
const THRESHOLDS = [
  { at: 10, label: 'Moderate' },
  { at: 25, label: 'Severe' },
  { at: 50, label: 'Critical' },
];

export default function TrendChart({ entries, onSelect }: { entries: HistoryEntry[]; onSelect: (id: string) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const points = useMemo(() => {
    const sorted = [...entries].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    const times = sorted.map(e => new Date(e.timestamp).getTime());
    const t0 = times[0];
    const span = times[times.length - 1] - t0;
    const innerW = width - PAD.left - PAD.right;
    const yMax = Math.max(60, Math.ceil(Math.max(...sorted.map(e => e.response.severity.severity_pct)) / 10) * 10 + 5);
    const y = (pct: number) => PAD.top + (1 - pct / yMax) * (HEIGHT - PAD.top - PAD.bottom);
    return {
      yMax,
      y,
      list: sorted.map((e, i) => ({
        entry: e,
        // Real time spacing; fall back to even spacing when every report shares a timestamp.
        x: PAD.left + (sorted.length === 1 ? innerW / 2 : span > 0 ? ((times[i] - t0) / span) * innerW : (i / (sorted.length - 1)) * innerW),
        y: y(e.response.severity.severity_pct),
      })),
    };
  }, [entries, width]);

  const { list, y, yMax } = points;
  const path = list.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  const ticks = [0, 25, 50, 75, 100].filter(t => t <= yMax);
  const fmtDate = (iso: string) => new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });

  const nearest = (clientX: number) => {
    const rect = wrapRef.current!.getBoundingClientRect();
    const x = clientX - rect.left;
    let best = 0;
    list.forEach((p, i) => { if (Math.abs(p.x - x) < Math.abs(list[best].x - x)) best = i; });
    return best;
  };

  const active = hover !== null ? list[hover] : null;

  return (
    <div className="trend" ref={wrapRef}>
      <svg
        width={width}
        height={HEIGHT}
        role="img"
        aria-label={`Severity across ${list.length} reports, from ${list[0].entry.response.severity.severity_pct.toFixed(1)}% to ${list[list.length - 1].entry.response.severity.severity_pct.toFixed(1)}%`}
        onMouseMove={e => setHover(nearest(e.clientX))}
        onMouseLeave={() => setHover(null)}
        onClick={e => onSelect(list[nearest(e.clientX)].entry.id)}
      >
        {/* y grid + ticks */}
        {ticks.map(t => (
          <g key={t}>
            <line x1={PAD.left} x2={width - PAD.right} y1={y(t)} y2={y(t)} className="trend-grid" />
            <text x={PAD.left - 8} y={y(t)} className="trend-tick" textAnchor="end" dominantBaseline="middle">{t}%</text>
          </g>
        ))}

        {/* tier thresholds */}
        {THRESHOLDS.filter(t => t.at <= yMax).map(t => (
          <g key={t.at}>
            <line x1={PAD.left} x2={width - PAD.right} y1={y(t.at)} y2={y(t.at)} className="trend-threshold" />
            <text x={width - PAD.right + 8} y={y(t.at)} className="trend-threshold-label" dominantBaseline="middle">
              {t.label} {t.at}%
            </text>
          </g>
        ))}

        {/* x labels: first and last only, to avoid collisions */}
        <text x={list[0].x} y={HEIGHT - 8} className="trend-tick" textAnchor={list.length === 1 ? 'middle' : 'start'}>
          {fmtDate(list[0].entry.timestamp)}
        </text>
        {list.length > 1 && (
          <text x={list[list.length - 1].x} y={HEIGHT - 8} className="trend-tick" textAnchor="end">
            {fmtDate(list[list.length - 1].entry.timestamp)}
          </text>
        )}

        {active && <line x1={active.x} x2={active.x} y1={PAD.top} y2={HEIGHT - PAD.bottom} className="trend-crosshair" />}

        <path d={path} className="trend-line" />

        {list.map((p, i) => (
          <circle
            key={p.entry.id}
            cx={p.x}
            cy={p.y}
            r={hover === i ? 6 : 4.5}
            className="trend-dot"
            style={{ fill: TIER_CONFIG[p.entry.response.severity.action_tier].color }}
            tabIndex={0}
            role="button"
            aria-label={`${fmtDate(p.entry.timestamp)}: ${p.entry.response.severity.severity_pct.toFixed(1)}%, ${TIER_CONFIG[p.entry.response.severity.action_tier].label}. Open report.`}
            onFocus={() => setHover(i)}
            onBlur={() => setHover(null)}
            onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(p.entry.id); } }}
          />
        ))}
      </svg>

      {active && (
        <div
          className="trend-tip"
          style={{
            left: Math.min(Math.max(active.x, 90), width - 90),
            top: active.y,
          }}
        >
          <div className="trend-tip-date">
            {new Date(active.entry.timestamp).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}
          </div>
          <div className="trend-tip-value num">{active.entry.response.severity.severity_pct.toFixed(1)}% damaged</div>
          <TierPill tier={active.entry.response.severity.action_tier} />
        </div>
      )}
    </div>
  );
}

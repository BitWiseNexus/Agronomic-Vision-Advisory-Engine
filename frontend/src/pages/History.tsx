/* ─── History Page ─────────────────────────────────────────── */

import { useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { clearHistory, deleteHistoryEntry, loadHistory } from '../store';
import type { ActionTier, HistoryEntry } from '../types';
import { TIER_CONFIG, TIER_ORDER } from '../types';
import Icon from '../components/Icon';
import ResultView from '../components/ResultView';
import { TierPill } from '../components/Severity';
import TrendChart from '../components/TrendChart';
import './History.css';

function entryTitle(e: HistoryEntry): string {
  const d = e.response.diagnosis;
  if (!d.checkpoint_loaded) return 'Leaf inspection';
  return d.is_healthy ? `${d.top.crop} · Healthy` : `${d.top.crop} · ${d.top.disease}`;
}

export default function History() {
  const [entries, setEntries] = useState<HistoryEntry[]>(loadHistory);
  const [params, setParams] = useSearchParams();
  const [filter, setFilter] = useState<ActionTier | 'all'>('all');
  const [confirmClear, setConfirmClear] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const selectedId = params.get('id');
  const selected = useMemo(() => entries.find(e => e.id === selectedId) ?? null, [entries, selectedId]);
  const select = (id: string | null) => {
    setParams(id ? { id } : {});
    window.scrollTo({ top: 0 });
  };

  const counts = useMemo(() => {
    const c: Partial<Record<ActionTier, number>> = {};
    entries.forEach(e => { c[e.response.severity.action_tier] = (c[e.response.severity.action_tier] ?? 0) + 1; });
    return c;
  }, [entries]);
  const visible = filter === 'all' ? entries : entries.filter(e => e.response.severity.action_tier === filter);

  const handleDelete = (id: string) => {
    deleteHistoryEntry(id);
    setEntries(loadHistory());
    setConfirmDelete(null);
  };

  const handleClearAll = () => {
    clearHistory();
    setEntries([]);
    setConfirmClear(false);
    setFilter('all');
  };

  /* ── Detail ─────────────────────────────────────────────────── */
  if (selected) {
    return (
      <div className="history-page container page">
        <button type="button" className="btn btn-ghost back-btn no-print" onClick={() => select(null)} id="back-to-list-btn">
          <Icon name="arrow-left" size={16} /> All reports
        </button>
        <ResultView
          result={selected.response}
          previewUrl={selected.imageDataUrl}
          onReset={() => select(null)}
          isHistoryView
          timestamp={selected.timestamp}
        />
      </div>
    );
  }

  /* ── Empty ──────────────────────────────────────────────────── */
  if (entries.length === 0) {
    return (
      <div className="history-page container page">
        <div className="empty card rise" id="history-empty">
          <div className="empty-icon"><Icon name="history" size={26} /></div>
          <h1>No reports yet</h1>
          <p>Each diagnosis is saved on this device, so you can re-open the plan and watch whether damage is spreading or under control.</p>
          <Link to="/diagnose" className="btn btn-primary btn-lg">
            <Icon name="camera" /> Diagnose a leaf
          </Link>
        </div>
      </div>
    );
  }

  /* ── List ───────────────────────────────────────────────────── */
  return (
    <div className="history-page container page" id="history-page">
      <header className="page-header">
        <div>
          <h1>History</h1>
          <p>{entries.length} report{entries.length !== 1 ? 's' : ''} saved on this device</p>
        </div>
        <div className="history-actions">
          {confirmClear ? (
            <div className="confirm rise" role="group" aria-label="Confirm clearing history">
              <span>Delete all {entries.length} reports?</span>
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirmClear(false)}>Cancel</button>
              <button type="button" className="btn btn-danger-ghost btn-sm" onClick={handleClearAll} id="confirm-clear-btn">Delete all</button>
            </div>
          ) : (
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirmClear(true)} id="clear-history-btn">
              <Icon name="trash" size={15} /> Clear all
            </button>
          )}
        </div>
      </header>

      {entries.length >= 2 && (
        <section className="card card-pad trend-card rise" aria-labelledby="trend-title">
          <h2 className="card-title" id="trend-title">
            <Icon name="layers" /> Damaged leaf area over time
            <span className="spacer" />
            <span className="muted card-title-meta">Click a point to open its report</span>
          </h2>
          <TrendChart entries={entries} onSelect={select} />
        </section>
      )}

      <div className="chips filter-chips" role="group" aria-label="Filter by tier">
        <button type="button" className="chip" aria-pressed={filter === 'all'} onClick={() => setFilter('all')}>
          All <span className="chip-sub">{entries.length}</span>
        </button>
        {TIER_ORDER.filter(t => counts[t]).map(t => (
          <button key={t} type="button" className="chip" aria-pressed={filter === t} onClick={() => setFilter(t)}>
            <span className="dot" style={{ background: TIER_CONFIG[t].color }} />
            {TIER_CONFIG[t].label} <span className="chip-sub">{counts[t]}</span>
          </button>
        ))}
      </div>

      <div className="history-grid">
        {visible.map(entry => {
          const sev = entry.response.severity;
          const region = entry.response.environment?.region;
          return (
            <article key={entry.id} className="history-card card" id={`history-card-${entry.id}`}>
              <button type="button" className="history-card-open" onClick={() => select(entry.id)} aria-label={`Open report: ${entryTitle(entry)}, ${sev.severity_pct.toFixed(1)}% damaged`}>
                <div className="history-thumb">
                  <img src={entry.imageDataUrl} alt="" loading="lazy" />
                  <div className="history-thumb-tier"><TierPill tier={sev.action_tier} /></div>
                </div>
                <div className="history-info">
                  <div className="history-row">
                    <span className="history-title">{entryTitle(entry)}</span>
                    <span className="history-pct num">{sev.severity_pct.toFixed(1)}%</span>
                  </div>
                  <div className="history-meta">
                    <span>{new Date(entry.timestamp).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}</span>
                    {region && <span><Icon name="map-pin" size={12} /> {region}</span>}
                  </div>
                </div>
              </button>
              <div className="history-card-tools">
                {confirmDelete === entry.id ? (
                  <>
                    <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirmDelete(null)}>Keep</button>
                    <button type="button" className="btn btn-danger-ghost btn-sm" onClick={() => handleDelete(entry.id)} id={`delete-${entry.id}`}>Delete</button>
                  </>
                ) : (
                  <button
                    type="button"
                    className="btn btn-ghost btn-icon btn-sm"
                    onClick={() => setConfirmDelete(entry.id)}
                    aria-label="Delete this report"
                    title="Delete"
                  >
                    <Icon name="trash" size={15} />
                  </button>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

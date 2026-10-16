/* ─── History Page ─────────────────────────────────────────── */

import { useState, useMemo } from 'react';
import { loadHistory, deleteHistoryEntry, clearHistory } from '../store';
import type { HistoryEntry } from '../types';
import { TIER_CONFIG } from '../types';
import ResultView from '../components/ResultView';
import './History.css';

export default function History() {
  const [entries, setEntries] = useState<HistoryEntry[]>(loadHistory());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showConfirmClear, setShowConfirmClear] = useState(false);

  const selected = useMemo(
    () => entries.find(e => e.id === selectedId) ?? null,
    [entries, selectedId],
  );

  const handleDelete = (id: string) => {
    deleteHistoryEntry(id);
    setEntries(loadHistory());
    if (selectedId === id) setSelectedId(null);
  };

  const handleClearAll = () => {
    clearHistory();
    setEntries([]);
    setSelectedId(null);
    setShowConfirmClear(false);
  };

  if (entries.length === 0) {
    return (
      <div className="history-page container">
        <div className="history-empty animate-scale-in" id="history-empty">
          <div className="empty-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
          </div>
          <h2>No Diagnoses Yet</h2>
          <p>Your past diagnoses will appear here, so you can track severity trends over time.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="history-page container" id="history-page">
      <div className="history-header">
        <div>
          <h2>Diagnosis History</h2>
          <p className="history-subtitle">{entries.length} diagnosis{entries.length !== 1 ? 'es' : ''} saved</p>
        </div>
        <div className="history-actions">
          {showConfirmClear ? (
            <div className="confirm-clear animate-scale-in">
              <span>Delete all?</span>
              <button className="btn btn-ghost" onClick={() => setShowConfirmClear(false)}>Cancel</button>
              <button className="btn btn-ghost" style={{ color: 'var(--red-400)' }} onClick={handleClearAll} id="confirm-clear-btn">
                Yes, clear
              </button>
            </div>
          ) : (
            <button className="btn btn-ghost" onClick={() => setShowConfirmClear(true)} id="clear-history-btn">
              Clear All
            </button>
          )}
        </div>
      </div>

      {/* Back / detail view */}
      {selected ? (
        <div className="history-detail animate-scale-in">
          <button className="btn btn-ghost" onClick={() => setSelectedId(null)} id="back-to-list-btn">
            ← Back to list
          </button>
          <div className="history-detail-meta">
            <span className="history-detail-date">
              {new Date(selected.timestamp).toLocaleDateString('en-IN', {
                day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
              })}
            </span>
          </div>
          <ResultView
            result={selected.response}
            previewUrl={selected.imageDataUrl}
            onReset={() => setSelectedId(null)}
            isHistoryView
          />
        </div>
      ) : (
        /* Card list */
        <div className="history-grid stagger">
          {entries.map(entry => {
            const tier = entry.response.severity.action_tier;
            const cfg = TIER_CONFIG[tier];
            return (
              <div
                key={entry.id}
                className="history-card glass-card"
                onClick={() => setSelectedId(entry.id)}
                id={`history-card-${entry.id}`}
              >
                <div className="history-card-image">
                  <img src={entry.imageDataUrl} alt="Diagnosis" />
                  <div className="history-card-tier" style={{
                    background: cfg.bgColor,
                    color: cfg.color,
                    borderColor: cfg.borderColor,
                  }}>
                    {cfg.icon} {cfg.label}
                  </div>
                </div>
                <div className="history-card-info">
                  <div className="history-card-header">
                    <div>
                      <span className="history-card-crop">{entry.response.diagnosis.top.crop}</span>
                      <span className="history-card-disease">{entry.response.diagnosis.top.disease}</span>
                    </div>
                    <span className="history-card-severity" style={{ color: cfg.color }}>
                      {entry.response.severity.severity_pct.toFixed(1)}%
                    </span>
                  </div>
                  <div className="history-card-footer">
                    <span className="history-card-date">
                      {new Date(entry.timestamp).toLocaleDateString('en-IN', {
                        day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
                      })}
                    </span>
                    <button
                      className="btn btn-ghost btn-icon"
                      onClick={(e) => { e.stopPropagation(); handleDelete(entry.id); }}
                      title="Delete"
                      id={`delete-${entry.id}`}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

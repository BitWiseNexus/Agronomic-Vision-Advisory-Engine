/* ─── Home / Landing Page ──────────────────────────────────── */

import { useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { getReadiness } from '../api';
import type { ReadinessResponse } from '../types';
import './Home.css';

export default function Home() {
  const navigate = useNavigate();
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);

  useEffect(() => {
    getReadiness()
      .then(r => { setReadiness(r); setBackendUp(true); })
      .catch(() => setBackendUp(false));
  }, []);

  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero" id="hero-section">
        {/* Animated background particles */}
        <div className="hero-particles">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className={`particle particle-${i + 1}`} />
          ))}
        </div>

        <div className="hero-content animate-slide-up">
          <div className="hero-badge badge badge-green">
            <span className="badge-dot" />
            AI-Powered Crop Advisory
          </div>

          <h1 className="hero-title">
            Photograph a leaf.<br />
            <span className="gradient-text">Get a treatment plan.</span>
          </h1>

          <p className="hero-subtitle">
            Upload a photo of a diseased crop leaf. Our engine identifies the disease,
            <strong> measures exactly how much damage</strong> using computer vision,
            and generates a severity-grounded treatment plan backed by agricultural extension material.
          </p>

          <div className="hero-actions">
            <button
              className="btn btn-primary btn-lg"
              id="start-diagnosis-btn"
              onClick={() => navigate('/diagnose')}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                <circle cx="12" cy="13" r="4"/>
              </svg>
              Start Diagnosis
            </button>
            <button
              className="btn btn-secondary btn-lg"
              id="view-history-btn"
              onClick={() => navigate('/history')}
            >
              View History
            </button>
          </div>

          {/* Backend Status */}
          <div className="backend-status animate-fade-in" style={{ animationDelay: '400ms' }}>
            {backendUp === null && (
              <div className="status-item">
                <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
                <span>Checking backend…</span>
              </div>
            )}
            {backendUp === false && (
              <div className="status-item status-warning">
                <span className="status-dot status-dot-red" />
                <span>Backend offline — start it with <code>uvicorn</code></span>
              </div>
            )}
            {backendUp === true && readiness && (
              <div className="status-items">
                <div className="status-item status-ok">
                  <span className="status-dot status-dot-green" />
                  <span>Backend v{readiness.version}</span>
                </div>
                {readiness.components.degraded.length > 0 && (
                  <div className="status-item status-warning">
                    <span className="status-dot status-dot-amber" />
                    <span>{readiness.components.degraded.length} degraded</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="how-it-works container" id="how-it-works">
        <h2 className="section-title">
          How It <span className="gradient-text">Works</span>
        </h2>
        <p className="section-subtitle">
          Four pipeline stages — from photograph to actionable treatment plan
        </p>

        <div className="pipeline-grid stagger">
          <div className="pipeline-card glass-card">
            <div className="pipeline-number">01</div>
            <div className="pipeline-icon pipeline-icon-classify">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <polyline points="21 15 16 10 5 21"/>
              </svg>
            </div>
            <h3>Disease Identification</h3>
            <p>ResNet-50 deep learning model classifies across <strong>38 plant diseases</strong> from the PlantVillage dataset with high confidence.</p>
          </div>

          <div className="pipeline-card glass-card">
            <div className="pipeline-number">02</div>
            <div className="pipeline-icon pipeline-icon-severity">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 20V10"/>
                <path d="M18 20V4"/>
                <path d="M6 20v-4"/>
              </svg>
            </div>
            <h3>Severity Measurement</h3>
            <p>OpenCV counts every pixel — <strong>healthy, chlorotic, necrotic</strong> — and computes an exact damage percentage. No guessing.</p>
          </div>

          <div className="pipeline-card glass-card">
            <div className="pipeline-number">03</div>
            <div className="pipeline-icon pipeline-icon-rag">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                <line x1="8" y1="6" x2="16" y2="6"/>
                <line x1="8" y1="10" x2="14" y2="10"/>
              </svg>
            </div>
            <h3>Knowledge Retrieval</h3>
            <p>RAG pipeline searches agricultural extension bulletins and IPM guides via <strong>pgvector</strong> for cited, grounded advice.</p>
          </div>

          <div className="pipeline-card glass-card">
            <div className="pipeline-number">04</div>
            <div className="pipeline-icon pipeline-icon-plan">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
                <polyline points="10 9 9 9 8 9"/>
              </svg>
            </div>
            <h3>Treatment Plan</h3>
            <p>LLM synthesis constrained by the <strong>action tier</strong> — the measured severity decides what the AI can recommend. No hallucinated pesticides.</p>
          </div>
        </div>
      </section>

      {/* Action Tier Reference */}
      <section className="tier-section container" id="tier-reference">
        <h2 className="section-title">
          Action <span className="gradient-text">Tiers</span>
        </h2>
        <p className="section-subtitle">
          The measured severity maps to a hard-coded tier that constrains the advisory
        </p>

        <div className="tier-table-wrapper glass-card">
          <table className="tier-table" id="tier-table">
            <thead>
              <tr>
                <th>Severity</th>
                <th>Tier</th>
                <th>Mandated Posture</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><code>&lt; 2%</code></td>
                <td><span className="badge badge-green">Negligible</span></td>
                <td>No action. Record a baseline</td>
              </tr>
              <tr>
                <td><code>2–10%</code></td>
                <td><span className="badge badge-green">Early</span></td>
                <td>Monitoring and cultural control only</td>
              </tr>
              <tr>
                <td><code>10–25%</code></td>
                <td><span className="badge badge-amber">Moderate</span></td>
                <td>Targeted spot treatment, biologicals first</td>
              </tr>
              <tr>
                <td><code>25–50%</code></td>
                <td><span className="badge badge-red">Severe</span></td>
                <td>Full chemical intervention on a label-rate schedule</td>
              </tr>
              <tr>
                <td><code>&gt; 50%</code></td>
                <td><span className="badge badge-red">Critical</span></td>
                <td>Intervention plus a salvage/replant assessment</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Footer */}
      <footer className="home-footer">
        <div className="container">
          <p className="footer-disclaimer">
            ⚠ Automated advisories are decision support, not a substitute for an agronomist.
            Always verify actives, dosages and pre-harvest intervals against the product label
            and your local agricultural extension service.
          </p>
        </div>
      </footer>
    </div>
  );
}

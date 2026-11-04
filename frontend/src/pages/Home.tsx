/* ─── Home / Landing Page ──────────────────────────────────── */

import { Link } from 'react-router-dom';
import { useBackend } from '../hooks';
import { TIER_CONFIG, TIER_ORDER, type SeverityReport } from '../types';
import Icon, { type IconName } from '../components/Icon';
import ImageCompare from '../components/ImageCompare';
import { TierPill, TierScale, TissueBar } from '../components/Severity';
import './Home.css';

/* Real output of the severity engine on data/samples/leaf_moderate.png */
const SAMPLE_REPORT: SeverityReport = {
  severity_pct: 16.31,
  necrotic_pct: 9.21,
  chlorotic_pct: 7.1,
  healthy_pct: 83.4,
  lesion_count: 11,
  largest_lesion_pct: 3.5,
  leaf_coverage_pct: 38,
  action_tier: 'moderate',
  quality_warnings: [],
};

const STEPS: { icon: IconName; title: string; body: string }[] = [
  {
    icon: 'scan',
    title: 'Measure the damage',
    body: 'OpenCV separates the leaf from its background and classifies every pixel as healthy, yellowing or dead. The result is an exact damaged-area percentage — counted, not guessed.',
  },
  {
    icon: 'search',
    title: 'Identify the disease',
    body: 'A ResNet-50 network fine-tuned on PlantVillage names the crop and disease across 38 classes, and says when it isn’t confident.',
  },
  {
    icon: 'book',
    title: 'Look up the guidance',
    body: 'The diagnosis, severity and your field conditions query a library of extension bulletins, IPM guides and pesticide labels.',
  },
  {
    icon: 'file',
    title: 'Write a grounded plan',
    body: 'A language model drafts the plan inside the limits of the measured tier. Any chemical or dose without a cited source is removed.',
  },
];

export default function Home() {
  const backend = useBackend();

  return (
    <div className="home">
      {/* ── Hero ─────────────────────────────────────────────── */}
      <section className="hero">
        <div className="container hero-inner">
          <div className="hero-copy rise">
            <span className="badge badge-accent hero-badge">
              <Icon name="leaf" size={13} /> Crop disease advisory
            </span>
            <h1>
              Photograph a leaf.<br />
              <span className="hero-accent">Get a treatment plan you can trust.</span>
            </h1>
            <p className="hero-sub">
              AgroVision measures exactly how much of the leaf is damaged, identifies the disease, and writes a
              treatment plan that is only allowed to be as aggressive as the damage justifies.
            </p>
            <div className="hero-actions">
              <Link to="/diagnose" className="btn btn-primary btn-lg" id="start-diagnosis-btn">
                <Icon name="camera" /> Diagnose a leaf
              </Link>
              <Link to="/history" className="btn btn-secondary btn-lg" id="view-history-btn">
                <Icon name="history" /> Past reports
              </Link>
            </div>

            {backend.status === 'offline' && (
              <div className="callout callout-warn hero-offline" role="status">
                <Icon name="alert" size={16} />
                <div>
                  <strong>Analysis server is offline.</strong> Start it from the project root:
                  <code>.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend</code>
                  <button type="button" className="btn btn-secondary btn-sm" onClick={backend.retry}>
                    <Icon name="refresh" size={14} /> Check again
                  </button>
                </div>
              </div>
            )}
          </div>

          <figure className="hero-demo card rise" style={{ animationDelay: '80ms' }}>
            <ImageCompare original="/samples/leaf_moderate.jpg" overlay="/samples/leaf_moderate_mask.jpg" maxHeight={360} />
            <figcaption className="hero-demo-body">
              <div className="hero-demo-head">
                <div>
                  <div className="eyebrow">Sample result</div>
                  <div className="hero-demo-pct num">16.3% <span>damaged</span></div>
                </div>
                <TierPill tier="moderate" />
              </div>
              <TissueBar severity={SAMPLE_REPORT} />
            </figcaption>
          </figure>
        </div>
      </section>

      {/* ── How it works ─────────────────────────────────────── */}
      <section className="container home-section" id="how-it-works">
        <div className="section-head">
          <span className="eyebrow">How it works</span>
          <h2>From one photo to a plan in four steps</h2>
        </div>
        <ol className="steps">
          {STEPS.map((s, i) => (
            <li key={s.title} className="card card-pad step">
              <div className="step-top">
                <span className="step-icon"><Icon name={s.icon} size={20} /></span>
                <span className="step-num num">0{i + 1}</span>
              </div>
              <h3>{s.title}</h3>
              <p>{s.body}</p>
            </li>
          ))}
        </ol>
      </section>

      {/* ── Tiers ────────────────────────────────────────────── */}
      <section className="container home-section" id="tier-reference">
        <div className="section-head">
          <span className="eyebrow">The safeguard</span>
          <h2>The measurement decides how far the advice can go</h2>
          <p>
            The damaged-area percentage maps to one of five fixed action tiers. The plan can never recommend more than
            its tier allows — so a leaf with 3% damage is never told to spray.
          </p>
        </div>
        <div className="card card-pad tiers-card">
          <TierScale />
          <ul className="tier-list" id="tier-table">
            {TIER_ORDER.map(t => (
              <li key={t}>
                <TierPill tier={t} />
                <span className="tier-range num">
                  {t === 'negligible' ? '< 2%' : t === 'critical' ? '≥ 50%' : `${TIER_CONFIG[t].range[0]}–${TIER_CONFIG[t].range[1]}%`}
                </span>
                <span className="tier-posture">{TIER_CONFIG[t].posture}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* ── Trust ────────────────────────────────────────────── */}
      <section className="container home-section">
        <div className="trust">
          <div className="trust-item">
            <Icon name="book" size={20} />
            <h3>Every dose is cited</h3>
            <p>Chemical recommendations must point to a retrieved source. Unsupported claims are stripped before you see them.</p>
          </div>
          <div className="trust-item">
            <Icon name="refresh" size={20} />
            <h3>Degrades, never breaks</h3>
            <p>No internet or API key? A rule-based advisor still produces a safe plan from the measured severity.</p>
          </div>
          <div className="trust-item">
            <Icon name="database" size={20} />
            <h3>Your reports stay with you</h3>
            <p>History is stored on this device only. Print any report to share it with an extension officer.</p>
          </div>
        </div>
      </section>

      <footer className="home-footer">
        <div className="container">
          <p>
            Decision support, not a substitute for an agronomist. Always check actives, doses and pre-harvest intervals
            against the product label and your local agricultural extension service.
          </p>
        </div>
      </footer>
    </div>
  );
}

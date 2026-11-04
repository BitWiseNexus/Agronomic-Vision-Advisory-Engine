/* ─── API Types ────────────────────────────────────────────────
   Mirrors the Pydantic schemas in backend/app/schemas/
   ──────────────────────────────────────────────────────────── */

export type ActionTier = 'negligible' | 'early' | 'moderate' | 'severe' | 'critical';

export interface ClassPrediction {
  label: string;
  crop: string;
  disease: string;
  confidence: number;
}

export interface Diagnosis {
  top: ClassPrediction;
  alternatives: ClassPrediction[];
  checkpoint_loaded: boolean;
  confidence_reliable: boolean;
  is_healthy: boolean;
}

export interface SeverityReport {
  severity_pct: number;
  necrotic_pct: number;
  chlorotic_pct: number;
  healthy_pct: number;
  lesion_count: number;
  largest_lesion_pct: number;
  leaf_coverage_pct: number;
  action_tier: ActionTier;
  quality_warnings: string[];
}

export interface PlanStep {
  order: number;
  action: string;
  timing: string;
  rationale: string;
  citations: string[];
}

export interface TreatmentPlan {
  summary: string;
  urgency: ActionTier;
  immediate_actions: PlanStep[];
  followup_actions: PlanStep[];
  chemical_options: string[];
  cultural_practices: string[];
  monitoring: string;
  cautions: string[];
  generated_by: string;
  unsupported_claims_removed: number;
}

export interface Evidence {
  id: string;
  text: string;
  score: number;
  source: string;
  doc_type: string;
  crop: string;
  disease: string;
}

export interface Readiness {
  checkpoint_loaded: boolean;
  vector_store: string;
  documents_indexed: number;
  llm_provider: string;
  degraded: string[];
}

export interface AdvisoryResponse {
  diagnosis: Diagnosis;
  severity: SeverityReport;
  environment: EnvironmentContext;
  plan: TreatmentPlan;
  evidence: Evidence[];
  readiness: Readiness;
  elapsed_ms: Record<string, number>;
  debug_overlay_png_b64: string | null;
  disclaimer: string;
}

export type SoilType = 'alluvial' | 'black' | 'red' | 'laterite' | 'sandy' | 'loam' | 'clay' | 'unknown';
export type Season = 'kharif' | 'rabi' | 'zaid' | 'unknown';
export type IrrigationType = 'rainfed' | 'flood' | 'drip' | 'sprinkler' | 'unknown';

export interface EnvironmentContext {
  region?: string | null;
  soil_type?: SoilType;
  season?: Season;
  irrigation?: IrrigationType;
  rainfall_last_7d_mm?: number | null;
  temperature_c?: number | null;
  relative_humidity_pct?: number | null;
  soil_ph?: number | null;
  crop_stage?: string | null;
  days_since_last_spray?: number | null;
  field_area_ha?: number | null;
  organic_only?: boolean;
}

/* Overlay endpoint response */
export interface OverlayResponse {
  severity: SeverityReport;
  overlay_png_b64: string;
  legend: Record<string, string>;
}

/* Health & readiness */
export interface HealthResponse {
  status: string;
  version: string;
}

export interface ReadinessResponse {
  status: string;
  version: string;
  components: Readiness;
  device: string;
  embedding_model: string | null;
}

/* ─── History Entry (stored in localStorage) ───────────────── */
export interface HistoryEntry {
  id: string;
  timestamp: string;
  imageDataUrl: string;
  response: AdvisoryResponse;
}

/* ─── Tier visual mapping ──────────────────────────────────── */
export type TierIcon = 'check' | 'eye' | 'target' | 'zap' | 'siren';

export const TIER_ORDER: ActionTier[] = ['negligible', 'early', 'moderate', 'severe', 'critical'];

export const TIER_CONFIG: Record<ActionTier, {
  label: string;
  headline: string;
  /** What the advisory is allowed to recommend at this tier (mirrors the backend mandate). */
  posture: string;
  /** Severity % range, upper bound exclusive — mirrors SeverityConfig.tier_bounds. */
  range: [number, number];
  color: string;
  icon: TierIcon;
}> = {
  negligible: {
    label: 'Negligible',
    headline: 'Healthy — baseline monitoring',
    posture: 'No treatment. Record a baseline and re-inspect in 7–10 days.',
    range: [0, 2],
    color: 'var(--tier-negligible)',
    icon: 'check',
  },
  early: {
    label: 'Early',
    headline: 'Early stage — monitor and sanitise',
    posture: 'Monitoring and cultural control only — no blanket chemical spray.',
    range: [2, 10],
    color: 'var(--tier-early)',
    icon: 'eye',
  },
  moderate: {
    label: 'Moderate',
    headline: 'Moderate — targeted treatment',
    posture: 'Spot-treat affected plants, biologicals first; chemicals only if a cited source supports it.',
    range: [10, 25],
    color: 'var(--tier-moderate)',
    icon: 'target',
  },
  severe: {
    label: 'Severe',
    headline: 'Severe — immediate intervention',
    posture: 'Full chemical intervention on a label-rate schedule across the affected block.',
    range: [25, 50],
    color: 'var(--tier-severe)',
    icon: 'zap',
  },
  critical: {
    label: 'Critical',
    headline: 'Critical — intervene and assess salvage',
    posture: 'Intervene immediately and decide whether to treat or remove and replant.',
    range: [50, 100],
    color: 'var(--tier-critical)',
    icon: 'siren',
  },
};

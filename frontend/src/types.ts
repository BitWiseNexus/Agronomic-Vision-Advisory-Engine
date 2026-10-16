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
export const TIER_CONFIG: Record<ActionTier, {
  label: string;
  headline: string;
  color: string;
  bgColor: string;
  borderColor: string;
  icon: string;
}> = {
  negligible: {
    label: 'Negligible',
    headline: 'Healthy — baseline monitoring',
    color: '#22c55e',
    bgColor: 'rgba(34, 197, 94, 0.12)',
    borderColor: 'rgba(34, 197, 94, 0.3)',
    icon: '✓',
  },
  early: {
    label: 'Early',
    headline: 'Early stage — monitor and sanitise',
    color: '#4ade80',
    bgColor: 'rgba(74, 222, 128, 0.12)',
    borderColor: 'rgba(74, 222, 128, 0.3)',
    icon: '⬆',
  },
  moderate: {
    label: 'Moderate',
    headline: 'Moderate — targeted treatment',
    color: '#fbbf24',
    bgColor: 'rgba(251, 191, 36, 0.12)',
    borderColor: 'rgba(251, 191, 36, 0.3)',
    icon: '⚠',
  },
  severe: {
    label: 'Severe',
    headline: 'Severe — immediate intervention',
    color: '#f87171',
    bgColor: 'rgba(248, 113, 113, 0.12)',
    borderColor: 'rgba(248, 113, 113, 0.3)',
    icon: '⚡',
  },
  critical: {
    label: 'Critical',
    headline: 'Critical — intervene and assess salvage',
    color: '#ef4444',
    bgColor: 'rgba(239, 68, 68, 0.12)',
    borderColor: 'rgba(239, 68, 68, 0.3)',
    icon: '🚨',
  },
};

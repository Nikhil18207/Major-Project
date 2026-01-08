// API Types for XR2Text with HAQT-ARR
// Authors: S. Nikhil, Dadhania Omkumar

export interface GeneratedReport {
  report: string
  findings: string | null
  impression: string | null
  generation_time_ms: number
  confidence_score: number | null
}

// HAQT-ARR Anatomical Regions (Novel)
export type AnatomicalRegion =
  | 'right_lung'
  | 'left_lung'
  | 'heart'
  | 'mediastinum'
  | 'spine'
  | 'diaphragm'
  | 'costophrenic_angles'

// HAQT-ARR Attention Visualization Response (Novel)
export interface AttentionVisualization {
  anatomical_regions: AnatomicalRegion[]
  region_weights: number[]
  spatial_priors: number[][][] | null  // [regions, H, W] - Learnable Gaussian priors
  generation_time_ms: number
}

// Region weight display info
export interface RegionWeightInfo {
  region: AnatomicalRegion
  weight: number
  displayName: string
  color: string
}

export interface ModelInfo {
  model_name: string
  encoder: string
  decoder: string
  projection_type: string  // "HAQT-ARR" or "Standard"
  projection_queries: number
  anatomical_regions: AnatomicalRegion[] | null  // For HAQT-ARR
  total_parameters: number
  trainable_parameters: number
  device: string
  max_length: number
}

export interface ModelStatus {
  model_loaded: boolean
  device: string
  mode: string
  gpu?: {
    name: string
    memory_allocated_gb: number
    memory_reserved_gb: number
    memory_total_gb: number
  }
}

export interface FeedbackRequest {
  original_report: string
  corrected_report: string
  image_id?: string
  feedback_notes?: string
}

export interface FeedbackResponse {
  success: boolean
  message: string
  feedback_id: string
}

export interface HealthCheck {
  status: string
  model_loaded: boolean
  device: string | null
  gpu_available: boolean
  gpu_name: string | null
  cuda_version: string | null
}

// App State Types

export interface ReportHistoryItem {
  id: string
  imageUrl: string
  report: string
  findings: string | null
  impression: string | null
  generatedAt: Date
  edited: boolean
  editedReport?: string
  // FIX: Add generation time for accurate stats
  generationTimeMs?: number
  // HAQT-ARR attention data (optional)
  attentionData?: AttentionVisualization
}

export interface GenerationSettings {
  maxLength: number
  numBeams: number
  temperature: number
  doSample: boolean
}

// HAQT-ARR Display Constants
export const ANATOMICAL_REGION_DISPLAY: Record<AnatomicalRegion, { name: string; color: string }> = {
  right_lung: { name: 'Right Lung', color: '#3B82F6' },      // Blue
  left_lung: { name: 'Left Lung', color: '#10B981' },        // Green
  heart: { name: 'Heart', color: '#EF4444' },                // Red
  mediastinum: { name: 'Mediastinum', color: '#8B5CF6' },    // Purple
  spine: { name: 'Spine', color: '#F59E0B' },                // Amber
  diaphragm: { name: 'Diaphragm', color: '#EC4899' },        // Pink
  costophrenic_angles: { name: 'CP Angles', color: '#06B6D4' }, // Cyan
}

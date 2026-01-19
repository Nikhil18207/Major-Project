/**
 * API Service for XR2Text with HAQT-ARR
 *
 * Handles all backend communication including:
 * - Report generation
 * - HAQT-ARR attention visualization (Novel)
 * - Model status and info
 * - Feedback submission
 *
 * Features:
 * - Automatic retry on rate limiting (429)
 * - Configurable timeouts per request type
 * - Connection error handling with user-friendly messages
 *
 * Authors: S. Nikhil, Dadhania Omkumar
 */

import axios, { AxiosError } from 'axios'
import type {
  GeneratedReport,
  ModelInfo,
  ModelStatus,
  HealthCheck,
  FeedbackRequest,
  FeedbackResponse,
  AttentionVisualization,
} from '../types'

// Use environment variable for API base URL with fallback for development
// In development, Vite proxy handles /api/* -> localhost:8000
// In production, set VITE_API_URL to the backend URL
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

// Log the API URL in development for debugging
if (import.meta.env.DEV) {
  console.log('API Base URL:', API_BASE_URL)
}

// Timeout settings for different request types
const DEFAULT_TIMEOUT = 30000  // 30 seconds for normal requests
const GENERATION_TIMEOUT = 120000  // 2 minutes for report generation (can be slow on RTX 4060)

// Rate limit retry configuration
const MAX_RETRIES = 3
const RETRY_BASE_DELAY = 1000  // 1 second base delay

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: DEFAULT_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Rate limit info from last request
export interface RateLimitInfo {
  limit: number
  remaining: number
  retryAfter?: number
}

let lastRateLimitInfo: RateLimitInfo | null = null

export function getRateLimitInfo(): RateLimitInfo | null {
  return lastRateLimitInfo
}

// ============================================
// Health and Status
// ============================================

export async function checkHealth(): Promise<HealthCheck> {
  const response = await axios.get('/health')
  return response.data
}

export async function getModelInfo(): Promise<ModelInfo> {
  const response = await api.get('/model/info')
  return response.data
}

export async function getModelStatus(): Promise<ModelStatus> {
  const response = await api.get('/model/status')
  return response.data
}

// ============================================
// Report Generation
// ============================================

export async function generateReport(file: File): Promise<GeneratedReport> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/generate', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: GENERATION_TIMEOUT,  // Longer timeout for generation
  })
  return response.data
}

export async function generateReportBase64(
  imageBase64: string,
  options?: {
    maxLength?: number
    numBeams?: number
    temperature?: number
    doSample?: boolean
  }
): Promise<GeneratedReport> {
  const response = await api.post('/generate/base64', {
    image_base64: imageBase64,
    max_length: options?.maxLength ?? 256,
    num_beams: options?.numBeams ?? 4,
    temperature: options?.temperature ?? 1.0,
    do_sample: options?.doSample ?? false,
  }, {
    timeout: GENERATION_TIMEOUT,  // Longer timeout for generation
  })
  return response.data
}

export async function generateReportsBatch(files: File[]): Promise<{
  results: Array<{
    filename: string
    report: string
    findings: string | null
    impression: string | null
  }>
  total_images: number
  total_time_ms: number
  avg_time_per_image_ms: number
}> {
  const formData = new FormData()
  files.forEach((file) => {
    formData.append('files', file)
  })

  const response = await api.post('/generate/batch', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: GENERATION_TIMEOUT * 2,  // Even longer for batch (multiple images)
  })
  return response.data
}

// ============================================
// HAQT-ARR Attention Visualization (Novel)
// ============================================

/**
 * Get HAQT-ARR attention visualization for an X-ray image.
 *
 * Returns anatomical region weights and spatial priors that show
 * which regions the model focused on during report generation.
 *
 * Only available when HAQT-ARR is enabled in the model.
 */
export async function getAttentionVisualization(file: File): Promise<AttentionVisualization> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/attention/visualize', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

/**
 * Check if HAQT-ARR attention visualization is available.
 * Returns true if the model has anatomical attention enabled.
 */
export async function isHAQTARREnabled(): Promise<boolean> {
  try {
    const modelInfo = await getModelInfo()
    return modelInfo.projection_type?.includes('HAQT-ARR') ?? false
  } catch {
    return false
  }
}

// ============================================
// Feedback
// ============================================

export async function submitFeedback(
  feedback: FeedbackRequest
): Promise<FeedbackResponse> {
  const response = await api.post('/feedback', feedback)
  return response.data
}

// ============================================
// Response Interceptor with Rate Limit Handling
// ============================================

api.interceptors.response.use(
  (response) => {
    // Extract rate limit info from headers
    const limitHeader = response.headers['x-ratelimit-limit']
    const remainingHeader = response.headers['x-ratelimit-remaining']

    if (limitHeader && remainingHeader) {
      lastRateLimitInfo = {
        limit: parseInt(limitHeader, 10),
        remaining: parseInt(remainingHeader, 10),
      }
    }

    return response
  },
  async (error: AxiosError) => {
    // Handle rate limiting (429) with retry
    if (error.response?.status === 429) {
      const retryAfter = error.response.headers['retry-after']
      const retrySeconds = retryAfter ? parseInt(retryAfter, 10) : 60

      // Update rate limit info
      lastRateLimitInfo = {
        limit: 0,
        remaining: 0,
        retryAfter: retrySeconds,
      }

      // Auto-retry for small delays (configurable)
      if (retrySeconds <= MAX_RETRIES * RETRY_BASE_DELAY / 1000) {
        console.warn(`Rate limited. Retrying in ${retrySeconds}s...`)
        await new Promise(resolve => setTimeout(resolve, retrySeconds * 1000))

        // Retry the request
        if (error.config) {
          return api.request(error.config)
        }
      }

      throw new Error(`Rate limit exceeded. Please wait ${retrySeconds} seconds before trying again.`)
    }

    // Handle connection errors
    if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
      throw new Error('Cannot connect to server. Please ensure the backend is running.')
    }

    // Handle timeout
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timed out. The server may be busy processing your request.')
    }

    // Generic error handling
    const message =
      (error.response?.data as { detail?: string })?.detail ||
      error.message ||
      'An unexpected error occurred'
    console.error('API Error:', message)
    throw new Error(message)
  }
)

export default api

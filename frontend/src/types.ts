/**
 * Wire types for the Speech Emotion Recognition API.
 *
 * These mirror the Pydantic models in `backend/app/schemas/prediction.py`. They
 * are deliberately hand-written rather than generated so the frontend can be
 * read without running a generator, but they must be kept in sync with the
 * backend contract - the API tests assert the shapes this file assumes.
 */

/** Probability of a single class, in [0, 1]. */
export type Probability = number

/** Emotion label -> probability, always covering every class the model emits. */
export type ProbabilityMap = Record<string, Probability>

export interface RankedEmotion {
  emotion: string
  probability: Probability
}

/** Facts about the audio the server actually analysed. */
export interface AudioInfo {
  /** Duration of the uploaded file before trimming. */
  duration_sec: number
  /** Sample rate the model worked at (16 kHz). */
  sample_rate: number
  /** Channel count of the source file. */
  channels: number
  /** Sample rate of the source file before resampling. */
  native_sample_rate: number
  /** Seconds of leading/trailing silence that were removed. */
  trimmed_sec: number
  /** Peak amplitude before normalisation. */
  peak_amplitude: number
}

/**
 * Identity of the model that produced a prediction.
 *
 * Deliberately small: the API omits the full description on prediction
 * responses because it is constant for the process lifetime. `GET /model`
 * carries the rest - use {@link ModelInfo} for that.
 */
export interface ServedModelRef {
  run_name: string
  parameters: number
  selection_metric: string
  selection_value: number | null
  trained_epoch: number | null
  device: string
}

export interface Prediction {
  emotion: string
  confidence: Probability
  emotion_index: number
  probabilities: ProbabilityMap
  ranked_emotions: RankedEmotion[]
  audio: AudioInfo
  /** Server-side processing time in milliseconds. */
  processing_ms: number
  model: ServedModelRef
  request_id: string | null
}

export interface EmotionInfo {
  index: number
  label: string
  ravdess_code: string | null
}

export interface FeatureExtraction {
  type: string
  n_mfcc: number
  n_mels: number
  n_fft: number
  hop_length: number
  window: number
  sample_rate: number
  cmvn: boolean
  features_per_frame: number
}

export interface Preprocessing {
  window_sec: number
  resample_to_hz: number
  mono: boolean
  trim_silence: boolean
  trim_top_db: number
  peak_normalize: boolean
  accepted_formats: string[]
  max_upload_mb: number
  /** Shortest clip the server will accept, in seconds. */
  min_duration_sec?: number
  /** Longest clip the server will accept, in seconds. */
  max_duration_sec?: number
}

export interface ModelInfo {
  labels: string[]
  num_classes: number
  architecture: string
  parameters: number
  feature_extraction: FeatureExtraction
  preprocessing: Preprocessing
  dataset: string | null
  run_name: string | null
  trained_epoch: number | null
  selection: Record<string, unknown>
  device: string | null
}

export interface Health {
  status: 'ok' | 'degraded'
  model_loaded: boolean
  model_run: string | null
  num_classes: number | null
  version: string
  detail: string | null
}

export interface ApiErrorBody {
  error: {
    code: string
    message: string
    detail?: string | null
  }
  request_id?: string | null
}

/**
 * Stable machine-readable error codes the backend can return. Kept as a union so
 * a typo in UI code that switches on them fails to compile rather than silently
 * falling through to the generic branch.
 */
export type ApiErrorCode =
  | 'NO_FILE'
  | 'UNSUPPORTED_FORMAT'
  | 'FILE_TOO_LARGE'
  | 'EMPTY_FILE'
  | 'INVALID_AUDIO'
  | 'AUDIO_TOO_SHORT'
  | 'AUDIO_TOO_LONG'
  | 'SILENT_AUDIO'
  | 'MODEL_UNAVAILABLE'
  | 'VALIDATION_ERROR'
  | 'INTERNAL_ERROR'

/** Error carrying the backend's own taxonomy through to the UI. */
export class ApiError extends Error {
  readonly code: ApiErrorCode
  readonly status: number
  readonly detail: string | null
  readonly requestId: string | null

  constructor(params: {
    code: ApiErrorCode
    message: string
    status: number
    detail?: string | null
    requestId?: string | null
  }) {
    super(params.message)
    this.name = 'ApiError'
    this.code = params.code
    this.status = params.status
    this.detail = params.detail ?? null
    this.requestId = params.requestId ?? null
  }
}
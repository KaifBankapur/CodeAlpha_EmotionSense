/**
 * Thin wrapper around the Emotion Recognition HTTP API.
 *
 * Every call goes through `request()` so error translation happens in exactly
 * one place: the backend's `{error: {code, message, detail}}` envelope becomes
 * a typed `ApiError`, and a network/parse failure becomes a synthetic
 * `NETWORK_ERROR`. No component ever inspects a raw Response.
 */

import type {
  ApiErrorBody,
  ApiErrorCode,
  EmotionInfo,
  Health,
  ModelInfo,
  Prediction,
} from '../types'
import { ApiError } from '../types'

/**
 * Same-origin by default. `npm run dev` proxies `/api` to FastAPI (see
 * vite.config.ts), and a production build served by the same host needs no
 * configuration at all.
 */
export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(
  /\/+$/,
  '',
)

/** Abort a prediction if the backend takes unreasonably long. */
const REQUEST_TIMEOUT_MS = 120_000

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
    })
  } catch (cause) {
    // Distinguish "we aborted it" from "the network is down"; both are the
    // user's problem but only one of them is worth retrying.
    const aborted = cause instanceof DOMException && cause.name === 'AbortError'
    throw new ApiError({
      code: 'NETWORK_ERROR' as ApiErrorCode,
      status: 0,
      message: aborted
        ? 'The request timed out. The server may be busy or unreachable.'
        : 'Could not reach the server. Check that the backend is running.',
      detail: cause instanceof Error ? cause.message : String(cause),
    })
  } finally {
    window.clearTimeout(timer)
  }

  const requestId = response.headers.get('X-Request-ID')

  if (!response.ok) {
    let body: ApiErrorBody | null = null
    try {
      body = (await response.json()) as ApiErrorBody
    } catch {
      // A non-JSON error body (proxy timeout page, gateway 502) - fall through
      // to the generic message below.
    }

    const code = (body?.error?.code ?? 'INTERNAL_ERROR') as ApiErrorCode
    throw new ApiError({
      code,
      status: response.status,
      message:
        body?.error?.message ??
        `The server returned an unexpected response (HTTP ${response.status}).`,
      detail: body?.error?.detail ?? null,
      requestId: body?.request_id ?? requestId,
    })
  }

  return (await response.json()) as T
}

/** Liveness plus whether a trained model is loaded. */
export function fetchHealth(signal?: AbortSignal): Promise<Health> {
  return request<Health>('/health', { signal })
}

/** Labels the loaded model can output, read from the model itself. */
export function fetchEmotions(signal?: AbortSignal): Promise<EmotionInfo[]> {
  return request<EmotionInfo[]>('/emotions', { signal })
}

/** Architecture, features and preprocessing actually in use. */
export function fetchModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  return request<ModelInfo>('/model', { signal })
}

/**
 * Upload one audio file and get its prediction.
 *
 * The file name is sent through as the multipart filename because the backend
 * uses it only to pick a decoder; content type is intentionally overridden
 * because browsers report inconsistent MIME types for audio containers.
 */
export function predictEmotion(
  file: Blob,
  filename: string,
  signal?: AbortSignal,
): Promise<Prediction> {
  const form = new FormData()
  form.append('file', file, filename)

  return request<Prediction>('/predict', {
    method: 'POST',
    body: form,
    signal,
  })
}

/** Shorthand for reading an `AudioInfo`-shaped object defensively. */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '-'
  if (seconds < 60) return `${seconds.toFixed(2)} s`
  const mins = Math.floor(seconds / 60)
  return `${mins}m ${(seconds - mins * 60).toFixed(1)}s`
}
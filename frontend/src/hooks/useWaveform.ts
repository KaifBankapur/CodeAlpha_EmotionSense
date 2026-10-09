/**
 * Decodes an audio blob so the UI can show a real waveform and duration.
 *
 * This is presentation only: it never affects the prediction, which the server
 * computes from the uploaded bytes with its own decoder. A decode failure here
 * is therefore non-fatal - the user can still upload the file.
 */

import { useEffect, useRef, useState } from 'react'

/** Peaks per rendered column. Fewer columns = cheaper DOM. */
const PEAK_BUCKETS = 320

export interface WaveformData {
  /** Object URL for `<audio src>`; revoked automatically when it changes. */
  objectUrl: string
  /** Normalised 0..1 peak per bucket. */
  peaks: number[]
  /** Duration of the decoded audio, or null when it could not be decoded. */
  durationSec: number | null
  /** Sample rate reported by the browser's decoder. */
  sampleRate: number | null
  /** True while decoding. */
  loading: boolean
}

export interface UseWaveformResult extends WaveformData {
  /** Set when the browser refused to decode the file. */
  error: string | null
}

/**
 * Animates a number from its previous value to `target`.
 *
 * Used for the confidence readout, so the answer counts up rather than snapping
 * in. Counting *from the previous value* rather than from zero matters: a second
 * prediction should appear to revise the first, not replay the animation from
 * nothing.
 *
 * Returns `target` immediately when the user prefers reduced motion - an
 * animated number is exactly the kind of incidental motion that preference is
 * about.
 */
export function useCountUp(target: number, durationMs = 900): number {
  const [value, setValue] = useState(target)
  const fromRef = useRef(target)
  const frameRef = useRef<number | null>(null)

  useEffect(() => {
    if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setValue(target)
      return
    }

    const from = fromRef.current
    const delta = target - from
    // Nothing to animate. Assigning here (rather than only in the frame callback)
    // avoids a frame of the old value lingering on screen.
    if (Math.abs(delta) < 1e-9) {
      setValue(target)
      return
    }

    const start = performance.now()

    const step = (now: number) => {
      const elapsed = now - start
      if (elapsed >= durationMs) {
        fromRef.current = target
        setValue(target)
        frameRef.current = null
        return
      }
      const t = elapsed / durationMs
      // easeOutCubic: fast start, gentle settle. Reads as "arriving at" the
      // number rather than sliding, which suits a measurement.
      const eased = 1 - Math.pow(1 - t, 3)
      setValue(from + delta * eased)
      frameRef.current = requestAnimationFrame(step)
    }

    frameRef.current = requestAnimationFrame(step)

    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
      // Leaving `fromRef` where it was means an interrupted animation resumes
      // from the last painted value rather than jumping.
      fromRef.current = value
    }
    // `value` is intentionally absent: it is read only on cleanup, and depending
    // on it would restart the animation on every frame.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, durationMs])

  return value
}

type AudioContextCtor = typeof AudioContext

function getAudioContextCtor(): AudioContextCtor | null {
  if (typeof window === 'undefined') return null
  const w = window as unknown as {
    AudioContext?: AudioContextCtor
    webkitAudioContext?: AudioContextCtor
  }
  return w.AudioContext ?? w.webkitAudioContext ?? null
}

/** Reduce a channel's Float32 samples to `buckets` normalised peaks. */
function computePeaks(samples: Float32Array, buckets: number): number[] {
  if (samples.length === 0) return []
  const size = Math.max(1, Math.floor(samples.length / buckets))
  const peaks: number[] = []

  // Scan every sample rather than sampling the envelope: speech transients are
  // short, and a strided scan would miss them and produce a misleading shape.
  for (let start = 0; start < samples.length; start += size) {
    const end = Math.min(start + size, samples.length)
    let max = 0
    for (let i = start; i < end; i += 1) {
      const value = Math.abs(samples[i] ?? 0)
      if (value > max) max = value
    }
    peaks.push(max)
  }

  const ceiling = peaks.reduce((acc, value) => Math.max(acc, value), 0)
  if (ceiling <= 0) return peaks.map(() => 0)
  return peaks.map((value) => value / ceiling)
}

export function useWaveform(blob: Blob | null): UseWaveformResult {
  const [data, setData] = useState<WaveformData>({
    objectUrl: '',
    peaks: [],
    durationSec: null,
    sampleRate: null,
    loading: false,
  })
  const [error, setError] = useState<string | null>(null)
  const contextRef = useRef<AudioContext | null>(null)

  useEffect(() => {
    if (!blob) {
      setData({
        objectUrl: '',
        peaks: [],
        durationSec: null,
        sampleRate: null,
        loading: false,
      })
      setError(null)
      return
    }

    let cancelled = false
    const objectUrl = URL.createObjectURL(blob)
    setError(null)
    setData({
      objectUrl,
      peaks: [],
      durationSec: null,
      sampleRate: null,
      loading: true,
    })

    void (async () => {
      const Ctor = getAudioContextCtor()
      if (!Ctor) {
        if (!cancelled) setError('This browser cannot preview audio waveforms.')
        return
      }

      try {
        const buffer = await blob.arrayBuffer()
        // One shared context; browsers cap how many may exist.
        contextRef.current ??= new Ctor()
        const decoded = await contextRef.current.decodeAudioData(buffer)

        if (cancelled) return

        // Mix to mono for display: a summed waveform is more legible than a
        // mean, and this is a preview, not analysis.
        const channels = decoded.numberOfChannels
        const frames = decoded.length
        const mono = new Float32Array(frames)
        for (let c = 0; c < channels; c += 1) {
          const data = decoded.getChannelData(c)
          for (let i = 0; i < frames; i += 1) mono[i] = (mono[i] ?? 0) + (data[i] ?? 0)
        }
        if (channels > 1) {
          for (let i = 0; i < frames; i += 1) mono[i] = (mono[i] ?? 0) / channels
        }

        setData({
          objectUrl,
          peaks: computePeaks(mono, PEAK_BUCKETS),
          durationSec: decoded.duration,
          sampleRate: decoded.sampleRate,
          loading: false,
        })
      } catch {
        if (cancelled) return
        // Playback still works via <audio>, so this is a soft failure.
        setError('The waveform could not be drawn, but the file can still be analyzed.')
        setData((prev) => ({ ...prev, loading: false }))
      }
    })()

    return () => {
      cancelled = true
      URL.revokeObjectURL(objectUrl)
    }
  }, [blob])

  useEffect(
    () => () => {
      void contextRef.current?.close().catch(() => undefined)
      contextRef.current = null
    },
    [],
  )

  return { ...data, error }
}
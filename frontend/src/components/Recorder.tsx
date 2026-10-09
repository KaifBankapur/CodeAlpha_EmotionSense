/**
 * Microphone recording via MediaRecorder.
 *
 * Notes on correctness:
 * - The permission prompt and `getUserMedia` can only be called from a user
 *   gesture, so everything happens inside the click handler.
 * - `MediaRecorder` emits `dataavailable` on a timer while recording; the last
 *   chunk arrives with the `stop` event, so all of them are collected.
 * - If recording starts but never produces audio (mic muted), we surface an
 *   explicit error instead of uploading an empty blob.
 * - Whatever container the browser picks is transcoded to 16 kHz mono WAV
 *   before upload. See `lib/audio.ts` for why: Chrome's only real option is
 *   WebM/Opus, which the API cannot decode, so uploading the raw blob made
 *   recording fail outright with UNSUPPORTED_FORMAT.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { Loader2, Mic, Square } from 'lucide-react'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { transcodeRecordingToWav } from '@/lib/audio'
import { cn } from '@/lib/utils'

interface RecorderProps {
  onRecorded: (blob: Blob, filename: string) => void
  disabled: boolean
  /** Server-side max duration, used to auto-stop. */
  maxSeconds: number
  /** Server-side min duration, checked after decoding. */
  minSeconds: number
}

type RecorderState = 'idle' | 'requesting' | 'recording' | 'stopping' | 'encoding'

interface MicError extends Error {
  name: string
}

/**
 * Tried in order. Whichever the browser accepts is transcoded to WAV afterwards,
 * so this ordering is a preference rather than a correctness requirement -- it
 * just avoids depending on WebM decode support where Ogg or MP4 is available.
 */
const MIME_CANDIDATES = [
  'audio/webm;codecs=opus',
  'audio/ogg;codecs=opus',
  'audio/mp4',
  'audio/webm',
]

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined
  return MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type))
}

function describeMicError(error: MicError): string {
  switch (error.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone access was denied. Allow it in your browser settings, or upload a file instead.'
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No microphone was found on this device.'
    case 'NotReadableError':
      return 'The microphone is already in use by another application.'
    default:
      return error.message || 'The microphone could not be started.'
  }
}

export function Recorder({ onRecorded, disabled, maxSeconds, minSeconds }: RecorderProps) {
  const [state, setState] = useState<RecorderState>('idle')
  const [elapsed, setElapsed] = useState(0)
  const [level, setLevel] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const audioCtxRef = useRef<AudioContext | null>(null)
  const rafRef = useRef<number | null>(null)
  const timerRef = useRef<number | null>(null)
  const startedAtRef = useRef(0)

  const cleanup = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (audioCtxRef.current) {
      void audioCtxRef.current.close().catch(() => undefined)
      audioCtxRef.current = null
    }
    setLevel(0)
  }, [])

  // Stop the microphone if the component unmounts mid-recording, otherwise the
  // browser keeps showing its "recording" indicator and the mic stays hot.
  useEffect(() => cleanup, [cleanup])

  const stop = useCallback(() => {
    const recorder = recorderRef.current
    if (!recorder || recorder.state === 'inactive') return
    setState('stopping')
    recorder.stop()
  }, [])

  const start = useCallback(async () => {
    setError(null)

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('This browser does not support in-page recording. Please upload a file instead.')
      return
    }

    setState('requesting')
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
    } catch (cause) {
      setState('idle')
      setError(describeMicError(cause as MicError))
      return
    }

    streamRef.current = stream
    chunksRef.current = []

    // Live input level, purely as feedback that the mic is actually hearing.
    try {
      const Ctor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
      if (Ctor) {
        const ctx = new Ctor()
        audioCtxRef.current = ctx
        const analyser = ctx.createAnalyser()
        analyser.fftSize = 1024
        ctx.createMediaStreamSource(stream).connect(analyser)
        const buffer = new Float32Array(analyser.fftSize)
        const tick = () => {
          analyser.getFloatTimeDomainData(buffer)
          let sum = 0
          for (let i = 0; i < buffer.length; i += 1) sum += (buffer[i] ?? 0) ** 2
          const rms = Math.sqrt(sum / buffer.length)
          // Map RMS onto a 0..1 bar with a perceptual curve.
          setLevel(Math.min(1, rms * 4))
          rafRef.current = requestAnimationFrame(tick)
        }
        rafRef.current = requestAnimationFrame(tick)
      }
    } catch {
      // The level meter is optional; recording continues without it.
    }

    const mimeType = pickMimeType()
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    recorderRef.current = recorder

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data)
    }

    recorder.onerror = () => {
      setState('idle')
      setError('Recording stopped unexpectedly.')
      cleanup()
    }

    recorder.onstop = () => {
      cleanup()
      const blob = new Blob(chunksRef.current, { type: mimeType ?? 'audio/webm' })
      chunksRef.current = []
      setState('encoding')

      // Transcode to 16 kHz mono WAV. This is async, so it must not be awaited
      // inside the MediaRecorder event handler's synchronous path -- fire it and
      // let the promise settle into state.
      void (async () => {
        try {
          const wav = await transcodeRecordingToWav(blob)
          // Duration from the decoded samples, not the stopwatch: MediaRecorder
          // can buffer or drop chunks, so the two disagree by hundreds of ms.
          if (wav.durationSec < minSeconds) {
            setState('idle')
            setError(
              `That recording was only ${wav.durationSec.toFixed(1)}s. The model needs at least ${minSeconds}s of speech.`,
            )
            return
          }
          const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
          setState('idle')
          onRecorded(wav.blob, `recording-${stamp}.wav`)
        } catch (cause) {
          setState('idle')
          setError(
            cause instanceof Error
              ? cause.message
              : 'The recording could not be encoded. Try uploading a file instead.',
          )
        }
      })()
    }

    recorder.start(250)
    startedAtRef.current = Date.now()
    setElapsed(0)
    setState('recording')

    timerRef.current = window.setInterval(() => {
      const seconds = (Date.now() - startedAtRef.current) / 1000
      setElapsed(seconds)
      // Stop rather than upload a clip the server will reject for length.
      if (seconds >= maxSeconds) stop()
    }, 100)
  }, [cleanup, maxSeconds, minSeconds, onRecorded, stop])

  const supported =
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== 'undefined'

  const recording = state === 'recording' || state === 'requesting'
  const encoding = state === 'encoding' || state === 'stopping'

  return (
    <div className="flex flex-col gap-2">
      {encoding ? (
        /* Distinct from the recording strip: the mic is already off here, and
           showing a live "recording" affordance during transcoding would be a
           lie. Transcoding takes tens of ms, so it needs its own brief state. */
        <div
          className="flex items-center gap-3 rounded-xl border bg-muted/40 px-4 py-3"
          role="status"
          aria-live="polite"
        >
          <Loader2 className="size-4 shrink-0 animate-spin text-primary" aria-hidden="true" />
          <span className="text-sm text-muted-foreground">
            Encoding recording&hellip;
          </span>
        </div>
      ) : recording ? (
        <div className="flex items-center gap-3 rounded-xl border border-destructive/25 bg-destructive/5 px-4 py-3">
          <span
            className={cn(
              'size-2 shrink-0 rounded-full',
              state === 'requesting'
                ? 'bg-muted-foreground/40'
                : 'animate-pulse bg-destructive',
            )}
            aria-hidden="true"
          />
          <span className="tabular min-w-[6.5rem] font-mono text-sm">
            {state === 'requesting'
              ? 'Requesting mic…'
              : `${Math.floor(elapsed / 60)}:${(elapsed % 60).toFixed(1).padStart(4, '0')}`}
          </span>
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-destructive/15">
            <div
              className="h-full rounded-full bg-destructive transition-[width] duration-100 ease-linear"
              style={{ width: `${level * 100}%` }}
            />
          </div>
          <Button type="button" variant="destructive" size="sm" onClick={stop}>
            <Square className="size-3 fill-current" aria-hidden="true" />
            Stop
          </Button>
        </div>
      ) : (
        <Button
          type="button"
          variant="outline"
          size="lg"
          className="w-full"
          onClick={() => void start()}
          disabled={disabled || !supported}
          title={
            supported
              ? 'Record speech with your microphone'
              : 'Recording is not supported in this browser'
          }
        >
          <Mic className="size-4" aria-hidden="true" />
          Record from microphone
        </Button>
      )}

      {error && (
        <Alert variant="destructive" role="alert">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  )
}

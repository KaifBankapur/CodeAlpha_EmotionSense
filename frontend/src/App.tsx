/**
 * Application shell: owns the selected audio, drives the prediction request and
 * renders whichever of the three panels is relevant (idle / busy / result).
 *
 * All server communication lives in `services/api.ts`; this component only
 * manages view state.
 *
 * Layout is a hero band over a two-column split on desktop, stacked on mobile:
 * input on the left is the thing you act on, result on the right is the thing you
 * read, and that ordering should not reflow as the viewport narrows.
 *
 * The distribution stage is mounted outside the result panel on purpose. It
 * renders from the start with placeholder bars, so the page has something alive
 * in it before the user has done anything, and it does not disappear when a
 * result is cleared.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AudioLines, RotateCcw, Sparkles } from 'lucide-react'

import { AudioPreview } from '@/components/AudioPreview'
import { Backdrop } from '@/components/Backdrop'
import { ConfidenceStage } from '@/components/ConfidenceStage'
import { DropZone } from '@/components/DropZone'
import { Hero } from '@/components/Hero'
import { Recorder } from '@/components/Recorder'
import { ResultPanel } from '@/components/ResultPanel'
import { BusyIndicator, EmptyState, StatusBanner } from '@/components/StatusBanner'
import { ThemeToggle } from '@/components/ThemeToggle'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { useWaveform } from '@/hooks/useWaveform'
import { fetchModelInfo, predictEmotion } from '@/services/api'
import { cn } from '@/lib/utils'
import type { ApiError, ModelInfo, Prediction, RankedEmotion } from '@/types'

/** Used only until the model description arrives, so the UI works even if the
 *  `/model` call fails - the API is the authority, this is a placeholder. */
const FALLBACK_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
const FALLBACK_EXTENSIONS = ['.wav', '.flac', '.ogg', '.mp3', '.m4a']
const FALLBACK_MIN_SEC = 1.0
const FALLBACK_MAX_SEC = 60

interface SelectedAudio {
  blob: Blob
  filename: string
  size: number
}

type Phase = 'idle' | 'predicting' | 'done'

function toApiError(cause: unknown): ApiError {
  if (cause instanceof Error && 'code' in cause) return cause as ApiError
  return {
    name: 'Error',
    message: cause instanceof Error ? cause.message : 'An unexpected error occurred.',
    code: 'INTERNAL_ERROR',
    status: 0,
    detail: null,
    requestId: null,
  } as ApiError
}

export default function App() {
  const [selected, setSelected] = useState<SelectedAudio | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [modelError, setModelError] = useState<ApiError | null>(null)

  // Abort an in-flight request if the component unmounts (React 18 StrictMode
  // mounts twice in development, which would otherwise leave a dangling fetch).
  const abortRef = useRef<AbortController | null>(null)

  const waveform = useWaveform(selected?.blob ?? null)

  // Fetch the served model's description once: it supplies the real upload
  // limit, accepted formats and duration bounds, so the client never has to
  // guess and disagree with the server.
  useEffect(() => {
    const controller = new AbortController()
    fetchModelInfo(controller.signal)
      .then(setModelInfo)
      .catch((cause) => {
        if (controller.signal.aborted) return
        setModelError(toApiError(cause))
      })
    return () => controller.abort()
  }, [])

  useEffect(() => () => abortRef.current?.abort(), [])

  const limits = useMemo(() => {
    const pre = modelInfo?.preprocessing
    return {
      maxBytes: pre ? Math.round(pre.max_upload_mb * 1024 * 1024) : FALLBACK_MAX_UPLOAD_BYTES,
      extensions: pre?.accepted_formats?.length
        ? pre.accepted_formats
        : FALLBACK_EXTENSIONS,
      minSeconds: pre?.min_duration_sec ?? FALLBACK_MIN_SEC,
      maxSeconds: pre?.max_duration_sec ?? FALLBACK_MAX_SEC,
    }
  }, [modelInfo])

  const busy = phase === 'predicting'

  // Before the first prediction there is no distribution, so the stage draws
  // placeholders. Their class names come from `GET /model` - the server's own
  // label list - rather than a second hard-coded copy here that could drift.
  // The probabilities are left at 0 and are never rendered as numbers while
  // empty; the scene ignores them entirely in that state.
  const idleRanked = useMemo<RankedEmotion[]>(
    () => (modelInfo?.labels ?? []).map((label) => ({ emotion: label, probability: 0 })),
    [modelInfo],
  )

  // `null` until the class list is known, so the stage does not mount an empty
  // scene and then immediately re-render it with eight bars in it.
  const stageRanked = prediction ? prediction.ranked_emotions : idleRanked.length ? idleRanked : null

  const handleFile = useCallback((blob: Blob, filename: string) => {
    setSelected({ blob, filename, size: blob.size })
    setPrediction(null)
    setError(null)
    setPhase('idle')
  }, [])

  const handleClear = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setSelected(null)
    setPrediction(null)
    setError(null)
    setPhase('idle')
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!selected || busy) return

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setPhase('predicting')
    setError(null)

    try {
      const result = await predictEmotion(selected.blob, selected.filename, controller.signal)
      setPrediction(result)
      setPhase('done')
    } catch (cause) {
      if (controller.signal.aborted) {
        setPhase('idle')
      } else {
        setPrediction(null)
        setError(toApiError(cause))
        setPhase('idle')
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null
    }
  }, [busy, selected])

  return (
    <div className="flex min-h-dvh flex-col">
      <Backdrop />

      {/* ------------------------------------------------------------ header */}
      <header className="sticky top-0 z-20 border-b glass">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <span
              className="relative grid size-9 shrink-0 place-items-center rounded-xl bg-primary/12 text-primary"
              aria-hidden="true"
            >
              {/* Soft halo, so the mark reads as lit rather than pasted on. */}
              <span className="absolute inset-0 rounded-xl bg-primary/20 blur-lg" />
              <AudioLines className="relative size-4.5" />
            </span>
            <div className="min-w-0">
              {/* Not an <h1>: `Hero` owns the page's single heading, so the brand
                  mark stays a label rather than a second top-level heading. */}
              <span className="block truncate text-sm font-semibold leading-tight tracking-tight sm:text-base">
                Speech Emotion Recognition
              </span>
              <p className="hidden truncate text-xs text-muted-foreground sm:block">
                MFCC + CNN-BiLSTM classifier trained on RAVDESS
              </p>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <span
              className={cn(
                'hidden items-center gap-1.5 rounded-full border bg-background/50 px-2.5 py-1 text-xs backdrop-blur-sm sm:inline-flex',
                modelInfo ? 'text-foreground' : 'text-muted-foreground',
              )}
            >
              <span
                className={cn(
                  'size-1.5 rounded-full',
                  modelInfo
                    ? 'bg-success'
                    : modelError
                      ? 'bg-destructive'
                      : 'bg-muted-foreground/40',
                )}
                aria-hidden="true"
              />
              {modelInfo ? (
                <span className="font-mono text-[0.7rem]">{modelInfo.run_name ?? 'active'}</span>
              ) : (
                <span>{modelError ? 'model unavailable' : 'loading model…'}</span>
              )}
            </span>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* -------------------------------------------------------------- main */}
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 pb-10 sm:px-6">
        <Hero modelInfo={modelInfo} />

        <div className="mt-6 grid gap-5 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)] lg:items-start">
        {/* input column */}
        <section className="hover-lift flex flex-col gap-4 rounded-2xl border glass p-5 shadow-sm">
          <header>
            <h2 className="flex items-center gap-2 text-sm font-semibold">
              <span
                className="grid size-5 place-items-center rounded-full bg-primary text-[0.65rem] font-semibold text-primary-foreground"
                aria-hidden="true"
              >
                1
              </span>
              Provide audio
            </h2>
            <p className="mt-1 pl-7 text-xs text-muted-foreground">
              Upload a clip, or record speech directly in the browser.
            </p>
          </header>

          {modelError && <StatusBanner error={modelError} onDismiss={() => setModelError(null)} />}

          <DropZone
            onFile={(file) => handleFile(file, file.name)}
            disabled={busy}
            accept={limits.extensions}
            maxBytes={limits.maxBytes}
            busy={busy}
          />

          <div className="flex items-center gap-3" aria-hidden="true">
            <span className="h-px flex-1 bg-border" />
            <span className="text-[0.7rem] uppercase tracking-wider text-muted-foreground">or</span>
            <span className="h-px flex-1 bg-border" />
          </div>

          <Recorder
            onRecorded={(blob, filename) => handleFile(blob, filename)}
            disabled={busy}
            maxSeconds={limits.maxSeconds}
            minSeconds={limits.minSeconds}
          />

          {selected && (
            <>
              <AudioPreview
                file={selected.blob}
                filename={selected.filename}
                waveform={waveform}
                disabled={busy}
                onClear={handleClear}
              />

              <div className="flex gap-2">
                <Button
                  type="button"
                  size="lg"
                  className="flex-1"
                  onClick={() => void handleAnalyze()}
                  disabled={busy}
                >
                  <Sparkles className="size-4" aria-hidden="true" />
                  {busy ? 'Analyzing…' : 'Analyze emotion'}
                </Button>

                {phase === 'done' && (
                  <Button
                    type="button"
                    variant="outline"
                    size="lg"
                    onClick={() => {
                      setPrediction(null)
                      setPhase('idle')
                    }}
                    disabled={busy}
                  >
                    <RotateCcw className="size-4" aria-hidden="true" />
                    <span className="sr-only sm:not-sr-only">Clear result</span>
                  </Button>
                )}
              </div>
            </>
          )}
        </section>

        {/* output column */}
        <section className="hover-lift flex flex-col gap-4 rounded-2xl border glass p-5 shadow-sm">
          <header>
            <h2 className="flex items-center gap-2 text-sm font-semibold">
              <span
                className="grid size-5 place-items-center rounded-full bg-primary text-[0.65rem] font-semibold text-primary-foreground"
                aria-hidden="true"
              >
                2
              </span>
              Result
            </h2>
            <p className="mt-1 pl-7 text-xs text-muted-foreground">
              Predicted label, confidence and the full probability distribution.
            </p>
          </header>

          {error && <StatusBanner error={error} onDismiss={() => setError(null)} />}

          {/* The stage sits above the result panel so it is present in every
              state - empty, predicting, answered - and does not remount when the
              answer changes. */}
          <ConfidenceStage ranked={stageRanked} predicted={prediction?.emotion ?? null} />

          <Separator />

          {busy && <BusyIndicator label="Analyzing the audio" />}

          {!busy && prediction && <ResultPanel prediction={prediction} modelInfo={modelInfo} />}

          {!busy && !prediction && !error && <EmptyState />}
        </section>
        </div>
      </main>

      {/* ------------------------------------------------------------ footer */}
      <footer className="mt-auto border-t glass">
        <div className="mx-auto flex max-w-6xl flex-col gap-1 px-4 py-5 text-[0.7rem] leading-relaxed text-muted-foreground sm:px-6">
          <p>
            Features: MFCC(40) + &Delta; + &Delta;&Delta; with per-utterance CMVN &rarr; 2-D CNN
            &rarr; BiLSTM &rarr; attention pooling
          </p>
          <p>
            Trained on RAVDESS (speech, 24 actors) &middot; speaker-disjoint splits &middot;
            predictions computed server-side
          </p>
        </div>
      </footer>
    </div>
  )
}
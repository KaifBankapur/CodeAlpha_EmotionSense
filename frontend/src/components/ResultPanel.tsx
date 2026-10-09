/**
 * Prediction result: the headline emotion, its confidence, the full probability
 * distribution and the facts that explain what the server did with the audio.
 *
 * The layout is deliberately "answer first": the emotion and its confidence get
 * the largest type on the page by a wide margin, and everything else is demoted
 * to supporting metadata. Someone glancing at this page should read the emotion
 * and nothing else.
 *
 * Every value below is read from the response - nothing here is hard-coded, and
 * no metric is displayed that the API did not actually send.
 */

import { Activity, Clock, Cpu, Timer } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { useCountUp } from '@/hooks/useWaveform'
import type { ModelInfo, Prediction } from '@/types'

interface ResultPanelProps {
  prediction: Prediction
  modelInfo: ModelInfo | null
}

/** Compact "label / value" row for the metadata blocks. */
function Fact({
  icon,
  label,
  value,
}: {
  icon?: React.ReactNode
  label: string
  value: React.ReactNode
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <dt className="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
        {icon}
        {label}
      </dt>
      <dd className="tabular min-w-0 truncate text-right text-xs font-medium">{value}</dd>
    </div>
  )
}

export function ResultPanel({ prediction, modelInfo }: ResultPanelProps) {
  const { audio, model, ranked_emotions: ranked } = prediction

  // `trimmed_sec` is the number of seconds of leading/trailing silence that were
  // *removed*, so it is not the analysed window - it is subtracted from the
  // duration to get it. The window itself is a constant of the model, read from
  // `/model` rather than assumed, and the pipeline pads or centre-crops to
  // exactly that length.
  const windowSec = modelInfo?.preprocessing.window_sec ?? null
  const speechSec = Math.max(0, audio.duration_sec - audio.trimmed_sec)
  const analysedSec = windowSec === null ? speechSec : Math.min(speechSec, windowSec)
  const analysedPercent = audio.duration_sec > 0 ? (analysedSec / audio.duration_sec) * 100 : 0

  // Confidence counts up rather than snapping in. The value rendered is still
  // exactly what the server sent - the animation is presentation only, and it
  // lands on the real number.
  const animatedConfidence = useCountUp(prediction.confidence * 100)

  // `/predict` returns a deliberately small model reference; `/model` carries the
  // rest. Fall back gracefully rather than showing a blank when it has not loaded.
  const classes = modelInfo?.num_classes ?? ranked.length
  const architecture = modelInfo?.architecture ?? 'CNN-BiLSTM'
  const features = modelInfo?.feature_extraction

  return (
    <section aria-live="polite" className="flex flex-col gap-5">
      {/* ---------------------------------------------------------- headline */}
      <div className="relative overflow-hidden rounded-2xl border glass p-5 shadow-sm">
        {/* Sheen: a soft light source from the upper left, plus a bloom behind the
            answer so the eye lands there first. Purely decorative. */}
        <div
          className="pointer-events-none absolute -top-24 -left-16 size-56 rounded-full bg-primary/18 blur-3xl"
          aria-hidden="true"
        />
        <div
          className="pointer-events-none absolute -right-20 -bottom-24 size-64 rounded-full bg-chart-2/12 blur-3xl"
          aria-hidden="true"
        />

        <div className="relative flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
          <div className="min-w-0">
            <p className="text-[0.7rem] font-medium uppercase tracking-[0.12em] text-muted-foreground">
              Predicted emotion
            </p>
            <p className="text-gradient mt-1 text-5xl font-semibold capitalize leading-none tracking-tight sm:text-6xl">
              {prediction.emotion}
            </p>
          </div>

          <div className="text-right">
            <p className="text-[0.7rem] font-medium uppercase tracking-[0.12em] text-muted-foreground">
              Confidence
            </p>
            <p className="tabular mt-1 text-5xl font-semibold leading-none tracking-tight text-primary sm:text-6xl">
              {animatedConfidence.toFixed(1)}
              <span className="text-2xl font-medium">%</span>
            </p>
          </div>
        </div>

        <p className="relative mt-4 text-xs leading-relaxed text-muted-foreground">
          Softmax over {classes} classes. This figure is the model&rsquo;s belief for this clip,
          not a calibrated guarantee &mdash; see the README for held-out accuracy on unseen
          speakers.
        </p>
      </div>

      {/* ------------------------------------------------------- distribution */}
      {/* Rendered by `ConfidenceStage` above, so it stays on screen while this
          panel is unmounted - the 3-D scene is present before a prediction
          exists. Nothing to do here. */}

      <Separator />

      {/* ------------------------------------------------------------- facts */}
      <div className="grid gap-5 sm:grid-cols-2">
        <div>
          <h3 className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
            <Activity className="size-3.5" aria-hidden="true" />
            Audio as received
          </h3>
          <dl className="divide-y divide-border/70">
            <Fact
              icon={<Clock className="size-3" />}
              label="Duration"
              value={`${audio.duration_sec.toFixed(2)} s`}
            />
            <Fact
              icon={<Timer className="size-3" />}
              label="Silence trimmed"
              value={`${audio.trimmed_sec.toFixed(2)} s`}
            />
            <Fact label="Speech left" value={`${speechSec.toFixed(2)} s`} />
            {windowSec !== null && (
              <Fact label="Analysed window" value={`${windowSec.toFixed(2)} s`} />
            )}
            <Fact label="Native rate" value={`${audio.native_sample_rate.toLocaleString()} Hz`} />
            <Fact label="Model rate" value={`${audio.sample_rate.toLocaleString()} Hz`} />
            <Fact label="Channels" value={audio.channels === 1 ? '1 (mono)' : audio.channels} />
            <Fact label="Peak amplitude" value={audio.peak_amplitude.toFixed(3)} />
          </dl>

          <p className="mt-2.5 text-[0.7rem] leading-relaxed text-muted-foreground">
            {windowSec === null ? (
              <>
                Silence was trimmed from both ends, leaving {speechSec.toFixed(2)} s of speech for
                the classifier.
              </>
            ) : speechSec > windowSec ? (
              <>
                The model reads a fixed {windowSec.toFixed(2)} s window, so it takes the middle of
                the speech rather than the whole file &mdash; {analysedPercent.toFixed(0)}% of this
                clip was actually classified.
              </>
            ) : (
              <>
                The model reads a fixed {windowSec.toFixed(2)} s window. This clip is no longer
                than that, so all of it was classified and no silence needed removing.
              </>
            )}
          </p>
        </div>

        <div>
          <h3 className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
            <Cpu className="size-3.5" aria-hidden="true" />
            Inference
          </h3>
          <dl className="divide-y divide-border/70">
            <Fact label="Server time" value={`${prediction.processing_ms.toLocaleString()} ms`} />
            <Fact label="Model run" value={<span className="font-mono">{model.run_name}</span>} />
            <Fact label="Architecture" value={architecture} />
            <Fact label="Parameters" value={model.parameters.toLocaleString()} />
            {model.trained_epoch !== null && (
              <Fact label="Selected at epoch" value={model.trained_epoch} />
            )}
            <Fact
              label={model.selection_metric}
              value={
                model.selection_value !== null ? model.selection_value.toFixed(4) : '-'
              }
            />
          </dl>

          <div className="mt-3 flex flex-wrap gap-1.5">
            <Badge variant="secondary" className="font-mono text-[0.65rem]">
              {classes} classes
            </Badge>
            <Badge variant="secondary" className="font-mono text-[0.65rem]">
              {model.device.toUpperCase()}
            </Badge>
            {features && (
              <Badge variant="secondary" className="font-mono text-[0.65rem]">
                {features.features_per_frame} feats/frame
              </Badge>
            )}
            {modelInfo?.dataset && (
              <Badge variant="secondary" className="font-mono text-[0.65rem]">
                {modelInfo.dataset}
              </Badge>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
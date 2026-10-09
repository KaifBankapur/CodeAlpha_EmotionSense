/**
 * Page header block.
 *
 * Carries the one sentence that says what this is, plus four figures that come
 * straight from `GET /model`. Those are deliberately model *facts* - class count,
 * parameter count, features per frame, dataset - and not accuracy numbers:
 * accuracy is a property of a held-out evaluation, not of a loaded checkpoint,
 * and showing it here would imply the running model had been measured on this
 * page. The measured numbers live in the README.
 *
 * Every field falls back to an em dash while `/model` is still loading, so the
 * layout is stable and nothing is ever invented to fill a gap.
 */

import { Badge } from '@/components/ui/badge'
import type { ModelInfo } from '@/types'

interface HeroProps {
  modelInfo: ModelInfo | null
}

const VALUE_CLASS = 'tabular text-xl font-semibold leading-tight tracking-tight sm:text-2xl'

export function Hero({ modelInfo }: HeroProps) {
  const features = modelInfo?.feature_extraction

  const stats = [
    { label: 'Classes', value: modelInfo ? String(modelInfo.num_classes) : '—' },
    {
      label: 'Parameters',
      value: modelInfo ? modelInfo.parameters.toLocaleString() : '—',
    },
    {
      label: 'Features / frame',
      value: features ? String(features.features_per_frame) : '—',
    },
    { label: 'Dataset', value: modelInfo?.dataset ?? '—' },
  ]

  return (
    <section className="relative">
      <div className="flex flex-wrap items-end justify-between gap-x-10 gap-y-6">
        <div className="min-w-0 max-w-xl flex-1">
          <Badge
            variant="secondary"
            className="mb-3 gap-1.5 rounded-full py-1 pl-1.5 pr-3 text-[0.7rem] font-normal"
          >
            <span className="size-1.5 rounded-full bg-success" aria-hidden="true" />
            <span className="pl-0.5">
              Speaker-disjoint split &middot; inference served by FastAPI
            </span>
          </Badge>

          <h1 className="text-balance text-3xl font-semibold leading-[1.1] tracking-tight sm:text-4xl lg:text-[2.75rem]">
            Emotion recognition, <span className="text-gradient">straight from the waveform</span>
          </h1>

          <p className="mt-3 max-w-lg text-pretty text-sm leading-relaxed text-muted-foreground">
            MFCC features feed a 2-D CNN and a bidirectional LSTM with attention pooling. The
            model was trained from scratch on RAVDESS and predicts eight classes from a single
            speech clip.
          </p>
        </div>

        <dl className="grid shrink-0 grid-cols-2 gap-x-8 gap-y-4 sm:grid-cols-4 lg:gap-x-10">
          {stats.map((stat) => (
            <div key={stat.label} className="min-w-0">
              <dt className="text-[0.65rem] font-medium uppercase tracking-[0.12em] text-muted-foreground">
                {stat.label}
              </dt>
              <dd className={`mt-1 truncate ${VALUE_CLASS}`}>{stat.value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  )
}
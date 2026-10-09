/**
 * The distribution stage: 3-D landscape above, 2-D bars below.
 *
 * These are two views of the same eight numbers, not two datasets, so they are
 * rendered as one unit with a single hover state - hovering a bar in one view
 * highlights the same class in the other. Split across two components with
 * separate state, they would read as two unrelated charts.
 *
 * Before any prediction exists the stage still renders: the scene draws an even
 * row of short placeholder bars and the 2-D bars are withheld entirely, so an
 * empty state can never be misread as a uniform posterior. `modelInfo.labels`
 * supplies the real class names for the placeholders, which come from the
 * server, not from a hard-coded list here.
 */

import { Radio } from 'lucide-react'

import { ProbabilityBars } from './ProbabilityBars'
import { ConfidenceScene } from './three/ConfidenceScene'
import { useState } from 'react'
import type { RankedEmotion } from '@/types'

interface ConfidenceStageProps {
  /** Real distribution, or `null` before the first prediction. */
  ranked: RankedEmotion[] | null
  /** Label of the top-ranked class, or `null` when there is no prediction. */
  predicted: string | null
}

export function ConfidenceStage({ ranked, predicted }: ConfidenceStageProps) {
  const [hovered, setHovered] = useState<string | null>(null)
  const populated = ranked !== null && predicted !== null

  return (
    <div>
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h3 className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
          Probability distribution
        </h3>

        <span
          className="flex shrink-0 items-center gap-1.5 text-[0.65rem] text-muted-foreground"
          // Announced politely rather than not at all: it flips when the answer
          // arrives, and that is a state change worth having in the accessibility
          // tree even though the numbers themselves are already live there.
          aria-live="polite"
        >
          <span
            className={
              populated
                ? 'size-1.5 rounded-full bg-success'
                : 'size-1.5 rounded-full bg-muted-foreground/40'
            }
            aria-hidden="true"
          />
          {populated ? (
            <span className="font-mono">softmax output</span>
          ) : (
            <span className="flex items-center gap-1">
              <Radio className="size-3" aria-hidden="true" />
              idle
            </span>
          )}
        </span>
      </div>

      {/* The scene renders nothing at all where WebGL is missing, so the bars
          below remain the primary representation. */}
      {ranked && (
        <ConfidenceScene
          ranked={ranked}
          predicted={predicted}
          populated={populated}
          hovered={hovered}
          onHover={setHovered}
        />
      )}

      {ranked && populated && (
        <div className="mt-5">
          <ProbabilityBars
            ranked={ranked}
            topIndex={ranked.findIndex((item) => item.emotion === predicted)}
            hovered={hovered}
            onHover={setHovered}
          />
        </div>
      )}
    </div>
  )
}
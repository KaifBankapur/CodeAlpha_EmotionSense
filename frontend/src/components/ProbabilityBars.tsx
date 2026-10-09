/**
 * Confidence distribution across all emotion classes.
 *
 * Colour discipline: the predicted class gets the accent, everything else gets a
 * neutral that steps down with rank. An earlier version assigned eight distinct
 * hues by rank; it read as a rainbow and made the *ranking* harder to see, which
 * is the one thing this chart exists to communicate. Hue is now used for a single
 * distinction - "this is the answer" - and the ordering carries the rest.
 *
 * Every bar prints its numeric value as well as its length, because bar length
 * alone hides the difference between 4.0% and 4.4%.
 */

import { cn } from '@/lib/utils'
import type { RankedEmotion } from '@/types'

interface ProbabilityBarsProps {
  ranked: RankedEmotion[]
  /** Index into `ranked` of the predicted class; highlighted. */
  topIndex: number
  /** Applies a muted treatment while the result is still animating in. */
  pending?: boolean
  /**
   * Label currently under the pointer (or focused) in either this chart or the
   * 3-D landscape. Shared so the two highlight together.
   */
  hovered?: string | null
  onHover?: (label: string | null) => void
}

export function ProbabilityBars({
  ranked,
  topIndex,
  pending = false,
  hovered = null,
  onHover,
}: ProbabilityBarsProps) {
  if (ranked.length === 0) return null

  return (
    <ul className="flex flex-col gap-3.5">
      {ranked.map((item, index) => {
        const isTop = index === topIndex
        // Anything that is neither the prediction nor the hovered row recedes.
        const emphasised = isTop || hovered === item.emotion
        const pct = item.probability * 100
        return (
          <li
            key={item.emotion}
            className="flex flex-col gap-1.5"
            onMouseEnter={() => onHover?.(item.emotion)}
            onMouseLeave={() => onHover?.(null)}
          >
            <div className="flex items-baseline justify-between gap-3">
              <span
                className={cn(
                  'text-sm capitalize transition-colors',
                  emphasised ? 'font-semibold text-foreground' : 'text-muted-foreground',
                )}
              >
                {item.emotion}
              </span>
              <span
                className={cn(
                  'tabular text-xs tabular-nums',
                  emphasised ? 'font-semibold text-foreground' : 'text-muted-foreground',
                )}
              >
                {pct.toFixed(1)}%
              </span>
            </div>

            <div
              role="meter"
              aria-valuenow={Math.round(pct)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`${item.emotion} probability`}
              className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
            >
              <div
                className={cn(
                  'h-full rounded-full transition-[width] duration-700 ease-out',
                  isTop ? 'bg-primary' : 'bg-muted-foreground/35',
                  hovered === item.emotion && !isTop && 'bg-muted-foreground/60',
                  pending && 'opacity-40',
                )}
                // A floor of 0.4% keeps a 0.0% class visibly a hairline rather
                // than an empty track, which reads as "missing data".
                style={{ width: `${Math.max(pct, 0.4)}%` }}
              />
            </div>
          </li>
        )
      })}
    </ul>
  )
}

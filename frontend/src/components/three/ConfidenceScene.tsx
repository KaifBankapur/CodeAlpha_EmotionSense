/**
 * Loads and guards the 3-D scene.
 *
 * three.js plus React Three Fiber is ~170 kB gzipped - more than the rest of the
 * application combined. It is therefore split into its own chunk behind
 * `React.lazy`, so it never blocks first paint on a page whose primary content is
 * a file input and a result table.
 *
 * The scene is the page's visual centrepiece and is meant to be present from the
 * start, which conflicts with keeping 170 kB off the critical path. The
 * resolution: mount it once the browser is idle (or immediately, if a prediction
 * arrives first and the user is waiting on it). Nothing the user can act on is
 * ever blocked on that chunk.
 *
 * Two layers of defence, in order:
 *   1.  `hasWebGL()` - refuse to mount at all where WebGL is unavailable, so no
 *       context is requested and nothing can throw.
 *   2.  `WebGLBoundary` - catches failures that only appear once the scene is
 *       live (context loss, driver reset).
 *
 * In both cases the scene degrades to nothing and `ProbabilityBars` carries the
 * data instead, which is why this component can safely return `null`.
 */

import { Suspense, lazy, useEffect, useMemo, useState } from 'react'

import { Skeleton } from '@/components/ui/skeleton'
import { WebGLBoundary, hasWebGL } from './WebGLBoundary'
import type { RankedEmotion } from '@/types'

const EmotionScene = lazy(() => import('./EmotionScene'))

/**
 * How long to wait before pulling the scene in on our own. Long enough that a
 * fast connection has already finished the app chunk and the font, so the 3-D
 * payload cannot compete with first paint.
 *
 * A plain timeout rather than `requestIdleCallback`: that API is Chromium-only,
 * and the timer is the more predictable behaviour anyway - it does not depend on
 * whether the browser happens to think the main thread is busy.
 */
const IDLE_DEFER_MS = 1200

interface Props {
  ranked: RankedEmotion[]
  predicted: string | null
  populated: boolean
  hovered: string | null
  onHover: (label: string | null) => void
}

export function ConfidenceScene({ populated, ...props }: Props) {
  // Probed once per mount. `useMemo` with no deps is deliberate: re-probing on
  // every render would allocate a canvas and a context each time.
  const supported = useMemo(() => hasWebGL(), [])

  const [failed, setFailed] = useState(false)
  const [deferred, setDeferred] = useState(populated)

  // Defer only while idle. A prediction arriving during the wait cancels it
  // immediately - the user is now actively looking at this space.
  useEffect(() => {
    if (populated) {
      setDeferred(true)
      return
    }
    const timer = window.setTimeout(() => setDeferred(true), IDLE_DEFER_MS)
    return () => window.clearTimeout(timer)
  }, [populated])

  if (!supported || failed || props.ranked.length === 0 || !deferred) return null

  return (
    <WebGLBoundary
      onError={() => setFailed(true)}
      // The 2-D bars beside it show the same eight numbers, so a plain skeleton
      // is the honest placeholder: it reserves the space without pretending to
      // display data that has not been drawn yet.
      fallback={
        <div className="flex h-72 w-full flex-col justify-end gap-2 lg:h-80">
          <Skeleton className="h-full w-full rounded-xl" />
        </div>
      }
    >
      <Suspense
        fallback={
          <div className="flex h-72 w-full items-center justify-center lg:h-80">
            <Skeleton className="h-full w-full rounded-xl" />
          </div>
        }
      >
        <EmotionScene {...props} populated={populated} />
      </Suspense>
    </WebGLBoundary>
  )
}
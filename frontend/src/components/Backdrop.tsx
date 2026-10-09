/**
 * Decorative background layers.
 *
 * Three stacked pieces: an aurora of slowly drifting colour fields, a faint
 * engineering grid over the top for structure, and a vignette to stop the colour
 * bleeding into the content area.
 *
 * The whole thing is `aria-hidden` and `pointer-events-none`, because none of it
 * is information - it must never be announced, focusable, or intercept a click
 * meant for the drop zone underneath.
 *
 * Rendering cost matters here: a blurred element forces the compositor to
 * rasterise it every frame, so the blobs are promoted to their own layer with
 * `will-change: transform` (transform-only animation, so no repaint) and the
 * motion is long and slow. Any motion at all is disabled under
 * `prefers-reduced-motion`, which the stylesheet enforces globally.
 */

interface Blob {
  top: string
  /** Exactly one of `left` / `right` is set, to pin the field to one corner. */
  left?: string
  right?: string
  size: string
  tone: string
  delay: string
  duration: string
}

/**
 * Position, size, delay and cycle length per blob. Hand-tuned so no two peaks
 * coincide and the composition never repeats on a short cycle.
 */
const BLOBS: Blob[] = [
  { top: '-18%', left: '-8%', size: '46rem', tone: 'bg-primary/22', delay: '0s', duration: '24s' },
  { top: '-6%', right: '-14%', size: '40rem', tone: 'bg-chart-2/18', delay: '-7s', duration: '29s' },
  { top: '38%', left: '24%', size: '38rem', tone: 'bg-chart-3/14', delay: '-13s', duration: '33s' },
]

export function Backdrop() {
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      {/* Base wash. Not pure background: a faint tint stops the page reading as
          a flat sheet and gives the glass panels something to refract. */}
      <div className="absolute inset-0 bg-gradient-to-b from-primary/[0.045] via-transparent to-chart-2/[0.05]" />

      {BLOBS.map((blob) => (
        <div
          key={blob.delay}
          className={`aurora-blob absolute rounded-full blur-3xl ${blob.tone}`}
          style={{
            top: blob.top,
            left: blob.left,
            right: blob.right,
            width: blob.size,
            height: blob.size,
            animationDelay: blob.delay,
            animationDuration: blob.duration,
          }}
        />
      ))}

      <div className="bg-grid absolute inset-0 opacity-60" />

      {/* Vignette. Keeps contrast high where the text actually sits, which is
          what stops the aurora from costing legibility. */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_0%,transparent_35%,var(--background)_100%)]" />
    </div>
  )
}
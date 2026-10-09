/**
 * Resolve a CSS colour token to something three.js can use.
 *
 * The design tokens in `index.css` are authored in `oklch()`, which is the right
 * choice for CSS but is *not* parseable by three.js' `Color.setStyle` - its
 * colour parser predates oklch and silently yields black for unknown syntax.
 * A black scene on dark backgrounds is a genuinely baffling bug to debug, so it
 * is worth handling explicitly rather than hand-maintaining a second palette.
 *
 * Rather than reimplement the oklch -> sRGB matrices (and risk getting the
 * matrices subtly wrong), this hands the string to the browser, which already
 * implements oklch correctly, and reads the resulting pixel back. That keeps one
 * source of truth: change the token in `index.css` and the 3D scene follows.
 */

/** One reusable 1x1 canvas; allocating one per call leaks GPU-backed surfaces. */
let probe: HTMLCanvasElement | null = null

/** Cache keyed by the raw token text, so a re-render does not re-parse. */
const cache = new Map<string, number>()

/** Converts `#rrggbb` to the 0xRRGGBB integer three.js expects. */
function toHexInt(r: number, g: number, b: number): number {
  // Round rather than truncate: truncating a value like 0.5 * 255 = 127.5 to 127
  // shifts a channel by a full level and accumulates into visible banding.
  const r8 = Math.round(r)
  const g8 = Math.round(g)
  const b8 = Math.round(b)
  return (r8 << 16) | (g8 << 8) | b8
}

/**
 * Reads a custom property off `document.documentElement` and resolves it to a
 * three.js colour integer.
 *
 * @param token  CSS custom property name, without the leading `--`.
 * @param fallback  Hex string used when the document is unavailable (SSR/tests)
 *   or the browser cannot resolve the value.
 */
export function tokenToHex(token: string, fallback = '#4f46e5'): number {
  if (typeof document === 'undefined' || typeof window === 'undefined') {
    return parseInt(fallback.slice(1), 16)
  }

  const raw = getComputedStyle(document.documentElement).getPropertyValue(`--${token}`).trim()
  const key = `${token}:${raw}`
  const hit = cache.get(key)
  if (hit !== undefined) return hit

  let result = parseInt(fallback.slice(1), 16)

  if (raw) {
    if (probe === null) {
      probe = document.createElement('canvas')
      probe.width = 1
      probe.height = 1
    }
    const ctx = probe.getContext('2d', { willReadFrequently: true })
    if (ctx) {
      ctx.clearRect(0, 0, 1, 1)
      // Assigning an unparseable value leaves fillStyle unchanged rather than
      // throwing, so compare before trusting the pixel.
      ctx.fillStyle = '#000000'
      ctx.fillStyle = raw
      if (ctx.fillStyle !== '#000000' || raw.toLowerCase() === '#000000') {
        ctx.fillRect(0, 0, 1, 1)
        const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data
        result = toHexInt(r ?? 0, g ?? 0, b ?? 0)
      }
    }
  }

  cache.set(key, result)
  return result
}

/** Drops cached colours. Called when the theme changes so the scene re-reads them. */
export function clearTokenCache(): void {
  cache.clear()
}
/**
 * Waveform preview.
 *
 * Rendered as SVG bars rather than a <canvas> so it scales with the layout,
 * stays crisp on high-DPI screens, and can be themed with plain CSS classes.
 */

interface WaveformProps {
  peaks: number[]
  /** Playback progress 0..1; rendered as a highlighted region. */
  progress: number
  height?: number
  'aria-label'?: string
}

export function Waveform({ peaks, progress, height = 80, ...rest }: WaveformProps) {
  if (peaks.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-md border border-dashed text-xs text-muted-foreground/60"
        style={{ height }}
        aria-hidden="true"
      >
        No waveform
      </div>
    )
  }

  const barWidth = 100 / peaks.length
  const playedIndex = Math.round(progress * peaks.length)

  return (
    <svg
      className="block w-full overflow-visible"
      viewBox={`0 0 100 ${height}`}
      preserveAspectRatio="none"
      height={height}
      role="img"
      aria-label={rest['aria-label'] ?? 'Audio waveform'}
    >
      {peaks.map((peak, index) => {
        // A floor of 1.5% keeps silence visible as a line instead of nothing.
        const magnitude = Math.max(0.015, peak)
        const barHeight = magnitude * height
        const played = index < playedIndex
        return (
          <rect
            key={index}
            className={
              played
                ? 'fill-primary transition-[fill] duration-150'
                : 'fill-muted-foreground/30 transition-[fill] duration-150'
            }
            x={index * barWidth}
            y={(height - barHeight) / 2}
            width={Math.max(barWidth * 0.6, 0.15)}
            height={barHeight}
            rx={0.2}
          />
        )
      })}
    </svg>
  )
}

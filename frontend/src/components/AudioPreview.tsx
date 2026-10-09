/**
 * Audio preview for the selected file: metadata, transport controls and the
 * waveform with playback progress.
 */

import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { FileAudio, Pause, Play, X } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { Waveform } from '@/components/Waveform'
import type { UseWaveformResult } from '@/hooks/useWaveform'

interface AudioPreviewProps {
  file: File | Blob
  filename: string
  waveform: UseWaveformResult
  disabled: boolean
  onClear: () => void
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1048576).toFixed(2)} MB`
}

function formatElapsed(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

export function AudioPreview({ file, filename, waveform, disabled, onClear }: AudioPreviewProps) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)

  // Reset transport state when a different file becomes current, otherwise the
  // progress bar keeps showing the previous file's position.
  useEffect(() => {
    setPlaying(false)
    setProgress(0)
    setDuration(0)
  }, [filename, file])

  const handleTimeUpdate = () => {
    const audio = audioRef.current
    if (!audio) return
    const total = Number.isFinite(audio.duration) ? audio.duration : 0
    setDuration(total)
    setProgress(total > 0 ? audio.currentTime / total : 0)
  }

  const handleToggle = async () => {
    const audio = audioRef.current
    if (!audio) return
    try {
      if (audio.paused) {
        await audio.play()
      } else {
        audio.pause()
      }
    } catch {
      // Autoplay policy or a decode error; nothing actionable for the user.
      setPlaying(false)
    }
  }

  const handleSeek = (event: ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current
    const next = Number(event.target.value)
    setProgress(next)
    if (audio && Number.isFinite(audio.duration)) {
      audio.currentTime = next * audio.duration
    }
  }

  const total = duration || waveform.durationSec || 0

  return (
    <div className="flex flex-col gap-3 rounded-xl border bg-muted/25 p-3">
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2.5">
          <FileAudio className="size-4 shrink-0 text-primary" aria-hidden="true" />
          <div className="min-w-0">
            <p className="truncate text-sm font-medium" title={filename}>
              {filename}
            </p>
            <p className="tabular text-xs text-muted-foreground">
              {formatBytes(file.size)}
              {total > 0 && ` · ${total.toFixed(2)} s`}
              {waveform.sampleRate !== null && ` · ${waveform.sampleRate} Hz`}
            </p>
          </div>
        </div>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              onClick={onClear}
              disabled={disabled}
              aria-label="Remove the selected audio file"
            >
              <X aria-hidden="true" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Remove file</TooltipContent>
        </Tooltip>
      </div>

      <div className="flex items-center gap-3">
        <Button
          type="button"
          size="icon-lg"
          className="shrink-0 rounded-full"
          onClick={() => void handleToggle()}
          disabled={disabled || !waveform.objectUrl}
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? (
            <Pause className="size-4 fill-current" aria-hidden="true" />
          ) : (
            <Play className="size-4 translate-x-px fill-current" aria-hidden="true" />
          )}
        </Button>

        <div className="relative min-w-0 flex-1">
          <Waveform
            peaks={waveform.peaks}
            progress={progress}
            aria-label="Waveform of the selected audio"
          />
          <input
            className="absolute inset-0 size-full cursor-pointer opacity-0"
            type="range"
            min={0}
            max={1}
            step={0.001}
            value={progress}
            onChange={handleSeek}
            disabled={disabled || !waveform.objectUrl}
            aria-label="Playback position"
          />
          <p className="tabular mt-1 text-right text-[0.7rem] text-muted-foreground">
            {formatElapsed(progress * total)} / {formatElapsed(total)}
          </p>
        </div>
      </div>

      {waveform.loading && <Skeleton className="h-3 w-32" />}
      {waveform.error && <p className="text-xs text-destructive">{waveform.error}</p>}

      {waveform.objectUrl && (
        <audio
          ref={audioRef}
          src={waveform.objectUrl}
          preload="metadata"
          onTimeUpdate={handleTimeUpdate}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => {
            setPlaying(false)
            setProgress(0)
          }}
        />
      )}
    </div>
  )
}

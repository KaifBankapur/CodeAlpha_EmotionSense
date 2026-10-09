/**
 * Drag-and-drop / click-to-browse upload target.
 *
 * Accessibility: the drop area is a real <button>, so it is reachable by keyboard
 * and announced correctly, while still accepting drag events. The hidden
 * <input type="file"> is triggered by that button rather than being the focus
 * target, which avoids the "focus the tiny input" problem of the classic
 * label-wrapping-input pattern.
 *
 * Drag depth is tracked rather than toggled on each event: `dragleave` fires when
 * the pointer crosses onto a child element, so a naive boolean flickers whenever
 * the file moves over the icon or the hint text.
 */

import { useCallback, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react'
import { UploadCloud } from 'lucide-react'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { cn } from '@/lib/utils'

interface DropZoneProps {
  onFile: (file: File) => void
  disabled: boolean
  /** Extensions shown in the hint; also enforced in the picker. */
  accept: string[]
  maxBytes: number
  busy: boolean
}

export function DropZone({ onFile, disabled, accept, maxBytes, busy }: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)
  const dragDepth = useRef(0)

  const acceptAttr = accept.map((ext) => (ext.startsWith('.') ? ext : `.${ext}`)).join(',')
  const extensionsLabel = accept.map((ext) => ext.replace(/^\./, '')).join(', ')

  const take = useCallback(
    (file: File | undefined | null) => {
      if (!file) return
      setLocalError(null)

      // Checked here so an oversized file fails instantly, instead of after an
      // upload the server is going to reject anyway.
      if (file.size > maxBytes) {
        setLocalError(
          `"${file.name}" is ${(file.size / 1048576).toFixed(1)} MB. The limit is ${(
            maxBytes / 1048576
          ).toFixed(0)} MB.`,
        )
        return
      }
      if (file.size === 0) {
        setLocalError(`"${file.name}" is empty.`)
        return
      }
      onFile(file)
    },
    [maxBytes, onFile],
  )

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    take(event.target.files?.[0])
    // Reset so re-selecting the same file fires `change` again.
    event.target.value = ''
  }

  const handleDragEnter = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault()
    dragDepth.current += 1
    if (!disabled) setDragging(true)
  }

  const handleDragLeave = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault()
    dragDepth.current -= 1
    if (dragDepth.current <= 0) {
      dragDepth.current = 0
      setDragging(false)
    }
  }

  const handleDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    if (disabled) return
    take(event.dataTransfer.files?.[0])
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      inputRef.current?.click()
    }
  }

  const maxMb = Math.round(maxBytes / 1048576)

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={handleKeyDown}
        onDragEnter={handleDragEnter}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        disabled={disabled}
        aria-describedby="dropzone-hint"
        className={cn(
          'group flex w-full flex-col items-center gap-2 rounded-xl border-2 border-dashed px-6 py-10',
          'text-center transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-50',
          dragging
            ? 'border-primary bg-primary/5 text-primary'
            : 'border-border bg-muted/30 text-muted-foreground hover:border-primary/50 hover:bg-accent/40 hover:text-foreground',
        )}
      >
        <span
          className={cn(
            'rounded-full p-2.5 transition-colors duration-200',
            dragging ? 'bg-primary/10 text-primary' : 'bg-muted text-muted-foreground group-hover:text-primary',
          )}
        >
          <UploadCloud className="size-5" aria-hidden="true" />
        </span>
        <span className={cn('text-sm font-medium', dragging && 'text-primary')}>
          {dragging ? 'Release to select' : busy ? 'Analyzing…' : 'Drop an audio file here'}
        </span>
        <span id="dropzone-hint" className="text-xs text-muted-foreground/80">
          or click to browse · {extensionsLabel} · up to {maxMb} MB
        </span>
      </button>

      <input
        ref={inputRef}
        type="file"
        className="sr-only"
        accept={acceptAttr}
        onChange={handleChange}
        disabled={disabled}
        tabIndex={-1}
        aria-hidden="true"
      />

      {localError && (
        <Alert variant="destructive" role="alert">
          <AlertDescription>{localError}</AlertDescription>
        </Alert>
      )}
    </div>
  )
}

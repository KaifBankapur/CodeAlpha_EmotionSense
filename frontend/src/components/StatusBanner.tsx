/**
 * Status / error surfaces.
 *
 * `ApiError.code` decides the wording and severity, so a 400 caused by silent
 * audio reads differently from a 503 caused by a missing model, instead of both
 * collapsing into "something went wrong".
 *
 * Each banner maps the server's error code to a *next action*, because the raw
 * message ("audio too short") tells the user what happened but not what to do.
 */

import { AlertCircle, Info, X } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { ApiError } from '@/types'

interface StatusBannerProps {
  error: ApiError | null
  onDismiss?: () => void
}

/** Server-side error code -> a message that tells the user what to do next. */
const GUIDANCE: Record<string, string> = {
  NO_FILE: 'Choose an audio file, or record a clip, before analyzing.',
  UNSUPPORTED_FORMAT: 'Convert the file to WAV, FLAC, OGG, MP3 or M4A and try again.',
  FILE_TOO_LARGE: 'Trim the recording or export it at a lower bitrate.',
  EMPTY_FILE: 'The file contains no data. Re-export it and try again.',
  INVALID_AUDIO: 'The file is not valid audio, or its header is corrupt.',
  AUDIO_TOO_SHORT: 'Record at least a couple of seconds of continuous speech.',
  AUDIO_TOO_LONG: 'Only the first few seconds are needed - shorten the clip.',
  SILENT_AUDIO: 'No speech was detected. Move closer to the microphone and check the input level.',
  MODEL_UNAVAILABLE: 'Start the backend with the model checkpoint present, then retry.',
  VALIDATION_ERROR: 'The request was rejected as invalid. Check the file and try again.',
  INTERNAL_ERROR: 'The server hit an unexpected error. Check the backend logs.',
  NETWORK_ERROR: 'Could not reach the API. Confirm the backend is running on port 8000.',
}

/**
 * Codes that describe a problem the user can fix by changing the audio, as
 * opposed to a server-side or connectivity problem. The distinction matters:
 * the first group should feel like a form error, the second like an outage.
 */
const USER_FIXABLE = new Set([
  'NO_FILE',
  'UNSUPPORTED_FORMAT',
  'FILE_TOO_LARGE',
  'EMPTY_FILE',
  'INVALID_AUDIO',
  'AUDIO_TOO_SHORT',
  'AUDIO_TOO_LONG',
  'SILENT_AUDIO',
  'VALIDATION_ERROR',
])

/**
 * Reusable panel chrome for the three non-result states of the output column:
 * in-flight, empty and (via StatusBanner) errored. Centralised so the empty and
 * busy states cannot drift apart in padding or heading size.
 */
function StateShell({
  tone,
  icon,
  title,
  children,
}: {
  tone: 'muted' | 'primary'
  icon?: React.ReactNode
  title: string
  children?: React.ReactNode
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center gap-3 rounded-xl border border-dashed px-6 py-14 text-center',
        tone === 'muted' ? 'bg-muted/20' : 'bg-primary/5 border-primary/25',
      )}
    >
      {icon && (
        <span
          className={cn(
            'rounded-full p-2.5',
            tone === 'muted' ? 'bg-muted text-muted-foreground' : 'bg-primary/10 text-primary',
          )}
          aria-hidden="true"
        >
          {icon}
        </span>
      )}
      <div className="space-y-1">
        <p className={cn('text-sm font-medium', tone === 'muted' && 'text-muted-foreground')}>
          {title}
        </p>
        {children && <div className="text-xs text-muted-foreground">{children}</div>}
      </div>
    </div>
  )
}

/** Shown while a prediction request is in flight. */
export function BusyIndicator({ label }: { label: string }) {
  return (
    <StateShell tone="primary" icon={<Info className="size-5" />} title={label}>
      <span>Running MFCC extraction and a forward pass through the network.</span>
      <span
        className="mt-2 inline-block h-1 w-40 overflow-hidden rounded-full bg-primary/15"
        aria-hidden="true"
      >
        <span className="block h-full w-1/3 animate-pulse rounded-full bg-primary" />
      </span>
    </StateShell>
  )
}

/** Shown before anything has been submitted. */
export function EmptyState() {
  return (
    <StateShell tone="muted" icon={<Info className="size-5" />} title="No result yet">
      <span>
        Provide audio on the left and press <strong>Analyze emotion</strong>. The predicted label
        and its confidence replace this panel, and the bars above settle onto the model&rsquo;s
        actual distribution.
      </span>
    </StateShell>
  )
}

export function StatusBanner({ error, onDismiss }: StatusBannerProps) {
  if (!error) return null

  const userFixable = USER_FIXABLE.has(error.code)
  const guidance = GUIDANCE[error.code]

  return (
    <div
      role="alert"
      aria-live="assertive"
      className={cn(
        'flex items-start gap-3 rounded-lg border px-3.5 py-3 text-sm',
        userFixable
          ? 'border-destructive/25 bg-destructive/5 text-destructive dark:bg-destructive/10'
          : 'border-border bg-muted/40 text-foreground',
      )}
    >
      <AlertCircle
        className={cn('mt-0.5 size-4 shrink-0', userFixable ? 'text-destructive' : 'text-muted-foreground')}
        aria-hidden="true"
      />

      <div className="min-w-0 flex-1 space-y-1">
        <p className={cn('font-medium', !userFixable && 'text-xs uppercase tracking-wide text-muted-foreground')}>
          {userFixable ? error.message : `${error.code.replace(/_/g, ' ')}`}
        </p>
        {guidance && <p className="text-xs text-muted-foreground">{guidance}</p>}
        {error.requestId && (
          <p className="font-mono text-[0.68rem] text-muted-foreground/80">
            request {error.requestId}
          </p>
        )}
      </div>

      {onDismiss && (
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          onClick={onDismiss}
          aria-label="Dismiss this message"
          className="-mr-1 -mt-0.5 shrink-0 text-muted-foreground hover:text-foreground"
        >
          <X aria-hidden="true" />
        </Button>
      )}
    </div>
  )
}
/**
 * Light / dark / system toggle.
 *
 * A three-state segmented control rather than a two-state switch, because
 * "follow my OS" is a real choice people want to make and then forget they
 * made - a two-state toggle cannot express it.
 */

import { Monitor, Moon, Sun } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useTheme, type Theme } from '@/hooks/useTheme'

const OPTIONS: { value: Theme; label: string; icon: typeof Sun }[] = [
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'dark', label: 'Dark', icon: Moon },
  { value: 'system', label: 'System', icon: Monitor },
]

export function ThemeToggle() {
  const { theme, resolved, setTheme } = useTheme()

  return (
    <div
      role="radiogroup"
      aria-label="Colour theme"
      className="inline-flex items-center gap-0.5 rounded-lg border bg-muted/40 p-0.5"
    >
      {OPTIONS.map(({ value, label, icon: Icon }) => {
        const active = theme === value
        return (
          <Tooltip key={value}>
            <TooltipTrigger asChild>
              <Button
                type="button"
                role="radio"
                aria-checked={active}
                aria-label={`${label} theme`}
                variant="ghost"
                size="icon"
                onClick={() => setTheme(value)}
                className={`size-7 rounded-md transition-colors ${
                  active
                    ? 'bg-background text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <Icon className="size-3.5" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {label}
              {value === 'system' && resolved !== theme ? ` (currently ${resolved})` : ''}
            </TooltipContent>
          </Tooltip>
        )
      })}
    </div>
  )
}

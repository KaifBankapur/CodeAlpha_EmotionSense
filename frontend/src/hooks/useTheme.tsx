/**
 * Theme switching for the shadcn tokens.
 *
 * Three states rather than two: `system` follows the OS and keeps following it,
 * while an explicit `light`/`dark` overrides it. The choice is persisted, and
 * the class is applied to `<html>` because that is where the `.dark` selector in
 * `index.css` is scoped.
 *
 * The inline script in `index.html` applies the stored class before React
 * mounts. Without it the page would paint light and then flip, which is visible
 * as a flash on every reload in dark mode.
 */

import * as React from 'react'

export type Theme = 'light' | 'dark' | 'system'

const STORAGE_KEY = 'ser-theme'

/** Reads the effective theme, resolving `system` against the media query. */
function resolve(theme: Theme): 'light' | 'dark' {
  if (theme !== 'system') return theme
  if (typeof window === 'undefined') return 'light'
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function apply(theme: Theme): void {
  const root = document.documentElement
  root.classList.toggle('dark', resolve(theme) === 'dark')
  // Keeps form controls and the scrollbar in step with the page background.
  root.style.colorScheme = resolve(theme)
}

interface ThemeContextValue {
  /** What the user picked, which may be `system`. */
  theme: Theme
  /** What is actually painted right now. */
  resolved: 'light' | 'dark'
  setTheme: (theme: Theme) => void
}

const ThemeContext = React.createContext<ThemeContextValue | null>(null)

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // Lazy initialiser: read the persisted choice once, before first paint.
  const [theme, setThemeState] = React.useState<Theme>(() => {
    if (typeof window === 'undefined') return 'system'
    const stored = window.localStorage.getItem(STORAGE_KEY)
    return stored === 'light' || stored === 'dark' || stored === 'system' ? stored : 'system'
  })

  const [resolved, setResolved] = React.useState<'light' | 'dark'>(() => resolve(theme))

  React.useEffect(() => {
    apply(theme)
    setResolved(resolve(theme))
  }, [theme])

  // While on `system`, follow the OS changing at sunset rather than only at load.
  React.useEffect(() => {
    if (theme !== 'system') return
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => {
      apply('system')
      setResolved(resolve('system'))
    }
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [theme])

  const setTheme = React.useCallback((next: Theme) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // Private browsing or a storage-blocking policy. The theme still applies
      // for this session; it just will not be remembered.
    }
    setThemeState(next)
  }, [])

  const value = React.useMemo<ThemeContextValue>(
    () => ({ theme, resolved, setTheme }),
    [theme, resolved, setTheme],
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme(): ThemeContextValue {
  const context = React.useContext(ThemeContext)
  if (context === null) {
    throw new Error('useTheme must be used inside <ThemeProvider>.')
  }
  return context
}

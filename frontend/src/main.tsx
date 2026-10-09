import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App'
import { TooltipProvider } from './components/ui/tooltip'
import { ThemeProvider } from './hooks/useTheme'
import './index.css'

const container = document.getElementById('root')
if (!container) {
  throw new Error('Root container #root is missing from index.html')
}

createRoot(container).render(
  <StrictMode>
    {/* ThemeProvider must wrap everything: it owns the `dark` class on <html>
        that the whole token set keys off. */}
    <ThemeProvider>
      {/* Single shared tooltip instance so hovering a button does not mount a
          new provider (and lose the open tooltip's timer) each time. */}
      <TooltipProvider delayDuration={200}>
        <App />
      </TooltipProvider>
    </ThemeProvider>
  </StrictMode>,
)
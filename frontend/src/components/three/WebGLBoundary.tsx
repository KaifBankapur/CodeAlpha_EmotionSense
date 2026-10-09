/**
 * Guard around the WebGL scene.
 *
 * Two failure modes have to be contained, because both otherwise take the whole
 * page down rather than just the canvas:
 *
 * 1.  No WebGL at all. Headless CI, blocklisted GPUs, some Linux VMs and browsers
 *     with hardware acceleration disabled all hit this. Three.js throws inside
 *     its own initialisation, which would propagate through React and unmount
 *     the entire application tree.
 * 2.  WebGL present but context creation *or a later frame* fails - driver
 *     resets, a lost context on tab switch. Only an error boundary catches this,
 *     since it happens after mount.
 *
 * When either trips, the caller renders its non-WebGL fallback so the user still
 * gets the prediction, just without the 3D.
 */

import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'

interface Props {
  children: ReactNode
  /** Rendered instead of the scene when WebGL is missing or the scene throws. */
  fallback: ReactNode
  /** Called once if the scene fails, so the failure can be reported. */
  onError?: (error: Error) => void
}

interface State {
  failed: boolean
}

export class WebGLBoundary extends Component<Props, State> {
  state: State = { failed: false }

  static getDerivedStateFromError(): State {
    return { failed: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Kept out of the render path on purpose: a console.error here would print a
    // stack on every dev-mode double-render.
    this.props.onError?.(error)
    if (import.meta.env.DEV) {
      console.warn('[EmotionScene] falling back to the 2D view:', error.message, info.componentStack)
    }
  }

  render(): ReactNode {
    if (this.state.failed) return this.props.fallback
    return this.props.children
  }
}

/**
 * Whether this browser can actually create a WebGL context.
 *
 * Deliberately creates and then discards a throwaway context rather than
 * feature-sniffing: the context is the thing that fails on locked-down GPUs, and
 * probing for the interface name would report true on exactly those machines.
 */
export function hasWebGL(): boolean {
  if (typeof document === 'undefined') return false
  try {
    const canvas = document.createElement('canvas')
    const gl =
      canvas.getContext('webgl2') ??
      canvas.getContext('webgl') ??
      canvas.getContext('experimental-webgl')
    if (!gl) return false
    // Release the context immediately; browsers cap concurrent contexts at around
    // 16, and a leaked probe would eat one of those slots for the page's life.
    const lose = (gl as WebGLRenderingContext).getExtension('WEBGL_lose_context')
    lose?.loseContext()
    return true
  } catch {
    return false
  }
}
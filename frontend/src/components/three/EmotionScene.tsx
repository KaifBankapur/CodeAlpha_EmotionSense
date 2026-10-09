/**
 * 3-D confidence landscape.
 *
 * The eight class probabilities are drawn as bars on a ground plane. This is the
 * same data the 2-D bars in `ProbabilityBars` show - it is not a separate
 * visualisation of anything else, so it cannot drift out of sync with the numbers
 * the API returned.
 *
 * Interaction is deliberately two-way: hovering a bar in the scene highlights the
 * matching row in the HTML legend below it, and hovering a legend row highlights
 * the bar. The legend is a real focusable list, so the same information is
 * reachable by keyboard and readable by a screen reader - the canvas is
 * `aria-hidden` decoration layered on top of it, not the only representation.
 *
 * Labels are HTML rather than 3-D text on purpose: `drei`'s `<Text>` falls back
 * to fetching a default font over the network at runtime, which breaks offline
 * and in an air-gapped deployment. Keeping type in the DOM also means it
 * inherits the theme and the user's font settings for free.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls as ThreeOrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import * as THREE from 'three'

import { clearTokenCache, tokenToHex } from '@/lib/cssColor'
import { cn } from '@/lib/utils'
import type { RankedEmotion } from '@/types'

interface SceneProps {
  ranked: RankedEmotion[]
  /** Label of the top-ranked class; highlighted in the scene. */
  predicted: string | null
  /**
   * False before any prediction exists. The bars are then drawn as an even,
   * deliberately short row and carry no value labels - the scene is a live
   * placeholder, not a flat distribution, and must never be mistakable for one.
   */
  populated: boolean
  /** Label currently hovered or focused, in either this scene or the 2-D bars. */
  hovered: string | null
  onHover: (label: string | null) => void
}

const MAX_HEIGHT = 2.6
const BAR_WIDTH = 0.52
const BAR_DEPTH = 0.52
const SPACING = 0.78

/** Bars grow from the centre so a value near zero is still a visible hairline. */
const MIN_VISIBLE = 0.012

/**
 * Height of every bar while the scene is empty. Low on purpose: an even row of
 * full-height bars would be a lie (it would look like a uniform posterior), and
 * an even row of zero-height bars would look broken.
 */
const IDLE_HEIGHT = 0.17

interface BarProps {
  position: [number, number, number]
  /** Target height in world units. */
  target: number
  color: number
  dimmed: boolean
  active: boolean
  reducedMotion: boolean
  label: string
  onHover: (label: string | null) => void
}

function Bar({
  position,
  target,
  color,
  dimmed,
  active,
  reducedMotion,
  label,
  onHover,
}: BarProps) {
  const mesh = useRef<THREE.Mesh>(null)
  const current = useRef(reducedMotion ? target : MIN_VISIBLE)

  // Ease toward the target rather than snapping, so consecutive predictions read
  // as the landscape changing instead of teleporting.
  useFrame((_, delta) => {
    const node = mesh.current
    if (!node) return
    if (reducedMotion) {
      node.scale.y = target
      node.position.y = target / 2
      return
    }
    // Frame-rate independent: `delta` is seconds, so the same factor applies at
    // 30 fps and 144 fps. Clamped because a backgrounded tab can hand back a
    // huge delta on return, which would otherwise overshoot badly.
    const step = 1 - Math.exp(-9 * Math.min(delta, 0.1))
    current.current += (target - current.current) * step
    node.scale.y = current.current
    node.position.y = current.current / 2
  })

  return (
    <mesh
      ref={mesh}
      position={position}
      // The geometry is unit height and scaled, so the maths above only ever
      // touches scale.y and keeps the bar's base pinned to the ground.
      //
      // scale.y is seeded from the same value `current` starts at. Without it the
      // mesh renders at the default scale.y of 1 for the frame before the first
      // useFrame callback runs, so a 4%-probability bar flashes up to full height
      // on mount.
      scale={[1, reducedMotion ? target : MIN_VISIBLE, 1]}
      onPointerOver={(event) => {
        event.stopPropagation()
        onHover(label)
      }}
      onPointerOut={() => onHover(null)}
    >
      <boxGeometry args={[BAR_WIDTH, 1, BAR_DEPTH]} />
      <meshStandardMaterial
        color={color}
        roughness={0.42}
        metalness={0.05}
        emissive={color}
        // A small emissive lift on the active bar reads as a glow without
        // needing a post-processing pass, which would cost far more than it is
        // worth here.
        emissiveIntensity={active ? 0.45 : dimmed ? 0.02 : 0.1}
        transparent
        opacity={dimmed ? 0.32 : 1}
      />
    </mesh>
  )
}

/**
 * Camera orbit, using three's own OrbitControls rather than drei's wrapper.
 *
 * drei is a large package and this was its only use; importing the controller
 * from `three/examples` instead drops the dependency entirely. (The scene chunk
 * turned out to be the same size either way, because drei was already being
 * tree-shaken away - but one fewer dependency is still one fewer thing to keep
 * patched.) `controls.update()` has to be driven from the render loop because
 * damping integrates across frames rather than snapping to a target.
 *
 * Orbit stays enabled under `prefers-reduced-motion`. That preference is about
 * suppressing *unrequested* motion - the height easing and the label bob, both of
 * which are switched off in `Bar` and `BarValue`. Dragging is direct
 * manipulation, and switching it off there would leave those users staring at a
 * static image with no indication that the scene is interactive at all.
 */
function SceneControls() {
  const camera = useThree((state) => state.camera)
  const domElement = useThree((state) => state.gl.domElement)

  const controls = useMemo(
    () => new ThreeOrbitControls(camera, domElement),
    [camera, domElement],
  )

  useEffect(() => {
    controls.enablePan = false
    controls.enableZoom = false
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.rotateSpeed = 0.55
    // Clamped so the bars never reorder or occlude one another. The legend below
    // is laid out left-to-right by index, and letting the camera swing behind the
    // scene would silently break that correspondence.
    controls.minPolarAngle = Math.PI / 3.2
    controls.maxPolarAngle = Math.PI / 2.1
    controls.minAzimuthAngle = -Math.PI / 4
    controls.maxAzimuthAngle = Math.PI / 4
    return () => controls.dispose()
  }, [controls])

  useFrame(() => controls.update())

  return null
}

/** The grid plus the bars, isolated so `useFrame` never re-runs on hover. */
function Landscape({
  ranked,
  predicted,
  populated,
  hovered,
  onHover,
  reducedMotion,
}: SceneProps & { reducedMotion: boolean }) {
  // Re-read the tokens whenever the theme flips; the cache is keyed by value so
  // this is cheap, but an explicit clear keeps a light->dark->light cycle honest.
  const [themeTick, setThemeTick] = useState(0)
  useEffect(() => {
    clearTokenCache()
    const observer = new MutationObserver(() => {
      clearTokenCache()
      setThemeTick((n) => n + 1)
    })
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [])

  const { primary, muted, ground } = useMemo(
    () => ({
      primary: tokenToHex('primary', '#4f46e5'),
      muted: tokenToHex('muted-foreground', '#64748b'),
      ground: tokenToHex('border', '#e2e8f0'),
    }),
    // themeTick is the dependency that matters: it flips when `.dark` changes.
    [themeTick],
  )

  // Bars are laid out in `ranked` order, so the scene and the 2-D bars below it
  // always agree left-to-right and the legend indexes map onto each other.
  const bars = useMemo(
    () =>
      ranked.map((item, index) => ({
        item,
        x: (index - (ranked.length - 1) / 2) * SPACING,
      })),
    [ranked],
  )

  const anyHovered = hovered !== null

  return (
    <>
      <ambientLight intensity={0.75} />
      <directionalLight position={[4, 7, 5]} intensity={1.1} />
      {/* A dim rim from behind separates the bars from the background without
          needing an outline pass. */}
      <directionalLight position={[-5, 3, -6]} intensity={0.35} color={primary} />

      <group>
        {bars.map(({ item, x }) => {
          const isTop = populated && item.emotion === predicted
          const isHovered = hovered === item.emotion
          // Floor the height so a near-zero class is still a visible hairline
          // rather than an invisible bar the pointer cannot find.
          const height = populated
            ? Math.max(item.probability * MAX_HEIGHT, MIN_VISIBLE)
            : IDLE_HEIGHT
          // With nothing hovered, only the answer is emphasised. On hover, the
          // hovered class joins it and everything else recedes.
          const dimmed = !isTop && !(anyHovered && isHovered)
          return (
            <Bar
              key={item.emotion}
              position={[x, height / 2, 0]}
              target={height}
              color={isTop ? primary : muted}
              dimmed={dimmed}
              active={isTop || isHovered}
              reducedMotion={reducedMotion}
              label={item.emotion}
              onHover={onHover}
            />
          )
        })}

        {/* Ground plane, sized to sit just under the outermost bars. */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.001, 0]} receiveShadow>
          <planeGeometry args={[bars.length * SPACING + 1.6, 2.6]} />
          <meshStandardMaterial color={ground} roughness={1} metalness={0} transparent opacity={0.35} />
        </mesh>

        {/* A soft elliptical pool of light under the tallest bar. Anchors the
            composition and gives the plane a focal point; `AdditiveBlending` so
            it brightens rather than washing the plane out to grey. */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]}>
          <circleGeometry args={[1.5, 48]} />
          <meshBasicMaterial
            color={primary}
            transparent
            opacity={populated ? 0.13 : 0.04}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
          />
        </mesh>

        {/* Value labels standing above each bar. Suppressed while empty: there
            is no number to show, and drawing the placeholder value would invent
            one. */}
        {populated &&
          bars.map(({ item, x }) => (
            <BarValue
              key={`v-${item.emotion}`}
              x={x}
              value={item.probability}
              color={item.emotion === predicted ? primary : muted}
              reducedMotion={reducedMotion}
            />
          ))}
      </group>

      <SceneControls />
    </>
  )
}

/**
 * The numeric value, as 3-D geometry.
 *
 * Drawn into a canvas texture and used as a sprite rather than pulled in as a
 * 3-D text component. `troika-three-text` (which is what `<Text>` in the common
 * three.js React wrappers uses) fetches a default font over the network when no
 * font is supplied, which would make the app depend on outbound access at
 * runtime. Painting the digits ourselves adds no dependency, costs one 256x128
 * canvas, and stays crisp because the texture is sized well above its on-screen
 * footprint.
 */
function BarValue({
  x,
  value,
  color,
  reducedMotion,
}: {
  x: number
  value: number
  color: number
  reducedMotion: boolean
}) {
  const sprite = useRef<THREE.Sprite>(null)
  const texture = useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 256
    canvas.height = 128
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    const result = new THREE.CanvasTexture(canvas)
    result.colorSpace = THREE.SRGBColorSpace
    return result
  }, [])

  const label = `${(value * 100).toFixed(1)}%`

  useEffect(() => {
    if (!texture) return
    const canvas = texture.image as HTMLCanvasElement
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.fillStyle = `#${color.toString(16).padStart(6, '0')}`
    ctx.font = '600 72px ui-monospace, monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, canvas.width / 2, canvas.height / 2)
    texture.needsUpdate = true
  }, [label, color, texture])

  useEffect(() => () => texture?.dispose(), [texture])

  useFrame((state) => {
    if (!sprite.current || reducedMotion) return
    // Gentle bob. Amplitude is tiny so it reads as "alive" rather than as a
    // chart element that is moving on its own.
    sprite.current.position.y = MAX_HEIGHT + 0.34 + Math.sin(state.clock.elapsedTime * 1.6 + x) * 0.02
  })

  if (!texture) return null

  return (
    <sprite ref={sprite} position={[x, MAX_HEIGHT + 0.34, 0]} scale={[0.62, 0.31, 1]}>
      <spriteMaterial map={texture} transparent depthTest={false} />
    </sprite>
  )
}

export default function EmotionScene({
  ranked,
  predicted,
  populated,
  hovered,
  onHover,
}: SceneProps) {
  const [reducedMotion, setReducedMotion] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const apply = () => setReducedMotion(media.matches)
    apply()
    media.addEventListener('change', apply)
    return () => media.removeEventListener('change', apply)
  }, [])

  const clear = useCallback(() => onHover(null), [onHover])

  return (
    <div className="relative">
      <div
        className="h-72 w-full overflow-hidden rounded-xl border bg-background/25 backdrop-blur-[2px] lg:h-80"
        // The legend below carries the same information accessibly, so the
        // canvas is decorative and must not be announced twice.
        aria-hidden="true"
      >
        <Canvas
          dpr={[1, 2]}
          // Cap DPR at 2: a 3x phone would otherwise render 9x the fragments for
          // no visible gain on flat-shaded boxes.
          camera={{ position: [0, 2.5, 5.4], fov: 40 }}
          // `alpha` lets the page's aurora backdrop show through the canvas
          // instead of the scene sitting on an opaque rectangle. It costs a real
          // compositing blend per frame, so the ground plane below carries the
          // contrast that the background would otherwise provide.
          gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
          onPointerMissed={clear}
        >
          <Landscape
            ranked={ranked}
            predicted={predicted}
            populated={populated}
            hovered={hovered}
            onHover={onHover}
            reducedMotion={reducedMotion}
          />
        </Canvas>

        {/* Idle overlay. The bars behind it are a fixed-height placeholder, so
            without this the scene would be readable as a flat posterior. The
            diagonal hatch says "nothing here yet" without hiding the scene. */}
        {!populated && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center">
            <span className="rounded-full border bg-background/80 px-3.5 py-1.5 text-[0.7rem] font-medium tracking-wide text-muted-foreground backdrop-blur-sm">
              Awaiting audio
            </span>
          </div>
        )}
      </div>

      {/* Accessible, keyboard-reachable legend. Doubles as the hover target for
          the 3-D bars, which is what makes the two views feel like one control. */}
      <ul className="mt-1 flex justify-between gap-0.5" role="list">
        {ranked.map((item) => {
          const active = hovered === item.emotion
          const isTop = item.emotion === predicted
          return (
            <li key={item.emotion} className="min-w-0 flex-1 text-center">
              <button
                type="button"
                onMouseEnter={() => onHover(item.emotion)}
                onMouseLeave={clear}
                onFocus={() => onHover(item.emotion)}
                onBlur={clear}
                // Click toggles, so a keyboard user can pin a class as well as
                // sweep across them with Tab.
                onClick={() => onHover(active ? null : item.emotion)}
                aria-pressed={active}
                className={cn(
                  'w-full rounded px-0.5 py-1 transition-colors',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                  active ? 'text-foreground' : 'text-muted-foreground',
                )}
              >
                <span className="block truncate text-[0.6rem] leading-tight capitalize sm:text-[0.65rem]">
                  {item.emotion}
                </span>
                <span
                  className={cn(
                    'tabular block text-[0.6rem] leading-tight font-medium sm:text-[0.65rem]',
                    isTop && !active && 'text-primary',
                    !populated && 'text-muted-foreground/60',
                  )}
                >
                  {populated ? `${(item.probability * 100).toFixed(0)}%` : '—'}
                </span>
              </button>
            </li>
          )
        })}
      </ul>

      <p className="mt-1 text-center text-[0.65rem] text-muted-foreground">
        {populated
          ? 'Drag to orbit · hover or focus a label to isolate a class'
          : 'Drag to orbit · each bar is a class the model can predict'}
      </p>
    </div>
  )
}
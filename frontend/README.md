# Frontend

React + TypeScript + Vite client for the Speech Emotion Recognition API.

## Quick start

The API must be running first (see the [root README](../README.md)):

```bash
cd ..                       # repository root
.venv\Scripts\activate      # or: source .venv/bin/activate
python -m uvicorn backend.app.main:app --port 8000
```

Then, in this directory:

```bash
npm install
npm run dev                  # http://localhost:5173
```

Vite proxies `/api/*` to `http://127.0.0.1:8000`, so the browser only ever talks
to one origin. There is no backend hostname compiled into the bundle, and CORS
is not exercised in development.

## Scripts

| Script              | What it does                                                    |
| ------------------- | --------------------------------------------------------------- |
| `npm run dev`       | Dev server on port 5173 with HMR and the `/api` proxy            |
| `npm run build`     | Type-check, then emit a production bundle to `dist/`            |
| `npm run preview`   | Serve the built bundle on port 4173, proxy intact               |
| `npm run typecheck` | `tsc --noEmit` on its own - what CI would run                    |

## Configuration

Copy `.env.example` to `.env`. All values are optional.

| Variable                | Default                 | Used by            | Meaning                          |
| ----------------------- | ----------------------- | ------------------ | -------------------------------- |
| `VITE_API_BASE_URL`     | `/api`                  | build and dev      | Path the client calls            |
| `VITE_BACKEND_ORIGIN`   | `http://127.0.0.1:8000` | dev and preview    | Proxy target                     |
| `VITE_PORT`             | `5173`                  | dev                | Dev server port                  |
| `VITE_PREVIEW_PORT`     | `4173`                  | preview            | Preview server port              |

Set `VITE_API_BASE_URL` to the full API origin (`https://api.example.com`) when
the API is on a different host. In that case the backend's `SER_CORS_ORIGINS`
must include this app's origin - it defaults to the two localhost dev origins
and nothing else.

Anything prefixed `VITE_` is inlined into the bundle at build time. Do not put a
secret in a `VITE_*` variable; there is nowhere for it to hide.

## How it works

### State machine

`src/App.tsx` owns a single explicit phase, because the app has more states than
it looks like - idle, file chosen, recording, uploading, done, and an error that
can arrive from any of them:

```
idle ──(file chosen)──► ready ──(submit)──► analysing ──(200)──► done
  ▲                        ▲                    │                    │
  │                        │                    └──(4xx/5xx)──► error │
  └──────(reset)───────────┴────────────────────────────────────────┘
```

Guessing at state with a pile of booleans is how you get a spinner that never
stops. One discriminated value is easier to reason about and impossible to
contradict.

### Limits come from the server

Upload size and duration limits are **not** hard-coded in the client. `GET
/api/model` returns `preprocessing.max_upload_mb`, `min_duration_sec` and
`max_duration_sec`, and the UI reads them from there. A client-side copy of
those numbers would drift from the server the first time either changed, and
the user would see a rejection from a rule they were never told about.

### Files

```
src/
  App.tsx                  orchestration, phase machine, limits from /model
  types.ts                 TypeScript mirrors of the Pydantic schemas
  services/api.ts          typed fetch wrappers, ApiError, request timeouts
  hooks/useWaveform.ts     peak-envelope extraction from an AudioBuffer
  components/
    DropZone.tsx           drag-and-drop + file picker
    Recorder.tsx           MediaRecorder with level metering
    AudioPreview.tsx       <audio> element + waveform
    ProbabilityBars.tsx    distribution, top bar highlighted
    ResultPanel.tsx        verdict, probabilities, model provenance
    StatusBanner.tsx       error/warning surface
    Waveform.tsx           canvas renderer for the peak envelope
```

There is no UI component library. Hand-written CSS in `src/index.css` uses
design tokens for colour, spacing and radius, so the palette can be changed in
one place and it respects `prefers-reduced-motion`.

### Recording

`Recorder.tsx` uses `navigator.mediaDevices.getUserMedia` and `MediaRecorder`.
Two things it does not do, both deliberate:

- It never guesses a MIME type. `MediaRecorder` picks per browser, and the
  resulting blob is uploaded under whatever type it reports. If the browser
  produces something the server rejects, the error message says which formats
  are allowed.
- It stops and revokes the track on unmount. A forgotten microphone stream is a
  privacy bug, not a resource leak, and it keeps the browser's recording
  indicator on for the rest of the session.

## Accessibility

- Every control is a real `<button>` or `<input>`, reachable by keyboard.
- The drop zone is also a labelled file input, so drag-and-drop is an
  enhancement rather than the only path.
- Error and status regions carry `role="status"` / `aria-live="polite"`, so
  screen readers announce the outcome of a prediction.
- The waveform canvas is `aria-hidden`; the audio itself is in an `<audio>`
  element with controls.
- `prefers-reduced-motion: reduce` disables the transitions and the pulse
  animation on the processing indicator.

## Troubleshooting

| Symptom                                   | Cause                                                       |
| ----------------------------------------- | ----------------------------------------------------------- |
| `Failed to fetch` / network error          | Backend not running, or `VITE_BACKEND_ORIGIN` is wrong       |
| 503 from `/health`                         | No trained model. Run `python scripts/train.py --set-active` |
| CORS error in the console                  | API on a different host - add this origin to `SER_CORS_ORIGINS` |
| Recording button disabled                  | Page not on `localhost`/`127.0.0.1`, or HTTPS not used (browsers require a secure context for `getUserMedia`) |
| Port 5173 already in use                   | `strictPort` is on by design. Free the port or set `VITE_PORT` |

## Building for production

```bash
npm run build     # -> dist/
```

`dist/` is static. Serve it from any web server; point it at the API with
`VITE_API_BASE_URL` at build time. There is no server-side rendering, so the
whole app is a few hundred kB of JS (about 55 kB gzipped, React being most of
it).
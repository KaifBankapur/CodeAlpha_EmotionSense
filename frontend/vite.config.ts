import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * The dev server proxies `/api/*` to the FastAPI backend so the browser only
 * ever talks to one origin. That keeps cookies/CORS behaviour identical to
 * production and means no hard-coded backend hostname ends up in the bundle.
 *
 * Override the target with VITE_BACKEND_ORIGIN when the API is not on
 * 127.0.0.1:8000.
 */
export default defineConfig(({ mode }) => {
  // '.' rather than process.cwd(): Vite resolves env files relative to this
  // path, and using the literal keeps the config free of Node type
  // dependencies (and of depending on where the command happened to run from).
  const env = loadEnv(mode, '.', '')
  const target = env.VITE_BACKEND_ORIGIN || 'http://127.0.0.1:8000'
  const port = Number(env.VITE_PORT || 5173)

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      // Mirrors the `@/*` path alias in tsconfig.json so shadcn's generated
      // imports resolve in dev, in build and in the type-checker alike.
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
    server: {
      port,
      strictPort: true,
      proxy: {
        '/api': {
          target,
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
    preview: {
      port: Number(env.VITE_PREVIEW_PORT || 4173),
      strictPort: true,
      proxy: {
        '/api': {
          target,
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
      // three.js + react-three-fiber is ~866 kB (232 kB gzipped) and Vite's
      // default 500 kB warning fires on every build because of it. The chunk is
      // behind `React.lazy` in `ConfidenceScene`, so it is only fetched once a
      // prediction exists and never touches first paint - which is the whole
      // reason it is split out in the first place. Raising the limit with that
      // rationale here, rather than leaving a permanent warning that would train
      // everyone to ignore warnings.
      chunkSizeWarningLimit: 900,
      // Keep the vendor chunk separate so app updates do not invalidate the
      // (large, stable) framework chunk in the browser cache.
      rollupOptions: {
        output: {
          manualChunks: {
            react: ['react', 'react-dom'],
          },
        },
      },
    },
  }
})
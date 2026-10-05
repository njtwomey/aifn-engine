import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { aifnDocs } from './plugins/docs.ts'

const src = path.join(import.meta.dirname, 'src')

/**
 * The AIFN Engine site: the front page, the generated pages of aifn-compute and aifn-methods (`plugins/docs.ts`) and
 * the aifn-render gallery.
 */
export default defineConfig({
  root: import.meta.dirname,
  // The path the app is served under: `/` locally, `/<repository>/` on GitHub Pages (set by the deploy workflow).
  base: process.env.EXAMPLES_BASE ?? '/',
  plugins: [
    react(),
    tailwindcss(),
    aifnDocs(),
    // aifn registers each primitive once at import and a second registration throws, so an edit to package source
    // reloads the page instead of hot-updating the module. Only source files count: the gallery
    // thumbnails in public/ are regenerated screenshots, and must not reload an open page on every write.
    {
      name: 'aifn-full-reload',
      handleHotUpdate({ file, server }) {
        if (file.startsWith(path.join(import.meta.dirname, 'public'))) return []
        if (file.includes(`${path.sep}node_modules${path.sep}`) || !/\.(ts|tsx|js|mjs)$/.test(file)) return
        server.ws.send({ type: 'full-reload' })
        return []
      },
    },
  ],
  // aifn-compute, aifn-methods and aifn-render resolve through the workspace (each package's `exports`).
  resolve: {
    alias: [{ find: /^@examples\//, replacement: `${src}/` }],
  },
  // A single-page app with path URLs (/<chapter>/<page>).
  appType: 'spa',
  server: { port: 5192 },
  preview: { port: 5192 },
})

import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const src = path.join(import.meta.dirname, 'src')
const renderSrc = path.join(import.meta.dirname, '..', 'packages', 'render', 'src')

/**
 * aifn-render examples: a tutorial reference for the rendering system. It depends on aifn-render (and aifn where a
 * render feature takes an aifn object) and third-party packages only.
 */
export default defineConfig({
  root: import.meta.dirname,
  plugins: [
    react(),
    tailwindcss(),
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
  resolve: {
    alias: [
      { find: /^@render\//, replacement: `${renderSrc}/` },
      { find: /^aifn-render\//, replacement: `${renderSrc}/` },
      { find: /^aifn-render$/, replacement: `${renderSrc}/index.ts` },
      { find: /^@examples\//, replacement: `${src}/` },
    ],
  },
  // A single-page app with path URLs (/<chapter>/<page>).
  appType: 'spa',
  server: { port: 5192 },
  preview: { port: 5192 },
})

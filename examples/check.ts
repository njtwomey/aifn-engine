/**
 * `make examples-check`: render the gallery and every recipe page to HTML on the server (through Vite, so aliases, TSX
 * and `?raw` work) and report any that throw. Charts render their containers only (ECharts draws in the browser), so
 * this catches import errors, bad props and exceptions in recipe code, not visual problems. It also enforces the
 * import boundary (aifn-render, aifn and packages only; no site or content, and of aifn-methods only `aifn-methods/gym`
 * in the gym section, whose views take an environment and an agent as values) and warns about snippets
 * longer than a recipe should be. `--imports-only` (run by `make lint`) checks the boundary alone.
 */
import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { createElement, type ComponentType, type ReactNode } from 'react'
import { renderToString } from 'react-dom/server'
import { createServer } from 'vite'

const root = import.meta.dirname
const MAX_SNIPPET_LINES = 45

function checkImports(): number {
  const src = path.join(root, 'src')
  const files = (readdirSync(src, { recursive: true }) as string[])
    .filter((f) => /\.(ts|tsx|css)$/.test(f))
    .map((f) => path.join(src, f))
  const pattern = /(?:\bfrom\s*|\bimport\s*\(?\s*|@import\s+)['"]([^'"]+)['"]/g
  const renderRoot = path.resolve(root, '..', 'packages', 'render')
  let errors = 0
  for (const file of files) {
    const text = readFileSync(file, 'utf8')
    for (const match of text.matchAll(pattern)) {
      const spec = match[1]
      const where = `${path.relative(root, file)}:${text.slice(0, match.index).split('\n').length}`
      const target = spec.startsWith('.') ? path.resolve(path.dirname(file), spec) : null
      const gymRecipe =
        file.startsWith(path.join(src, 'recipes', 'gym') + path.sep) && /^aifn-methods\/gym(\/|$)/.test(spec)
      if (/^(@\/|@lab\/|site\/|content\/|aifn-methods)/.test(spec) && !gymRecipe) {
        errors++
        console.error(
          `FAIL  ${where}: imports '${spec}'; examples import aifn-render, aifn and packages only (gym recipes also aifn-methods/gym)`,
        )
      } else if (target && path.relative(root, target).startsWith('..') && !target.startsWith(renderRoot)) {
        errors++
        console.error(`FAIL  ${where}: imports '${spec}', which is outside examples`)
      }
    }
  }
  return errors
}

const importErrors = checkImports()
if (process.argv.includes('--imports-only')) process.exit(importErrors ? 1 : 0)

// The theme provider and figures read the colour-scheme preference and saved sizes; give them inert stand-ins.
const store = new Map<string, string>()
Object.assign(globalThis, {
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
  localStorage: {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => store.set(k, v),
    removeItem: (k: string) => store.delete(k),
  },
})

const server = await createServer({
  configFile: path.join(root, 'vite.config.ts'),
  server: { middlewareMode: true, hmr: false },
  appType: 'custom',
  logLevel: 'error',
})
let failures = 0
let total = 0
try {
  const { Providers } = await server.ssrLoadModule('aifn-render')
  const { Gallery } = await server.ssrLoadModule('/src/shell/Gallery.tsx')
  const { RecipePage } = await server.ssrLoadModule('/src/shell/RecipePage.tsx')
  const { ENTRIES } = await server.ssrLoadModule('/src/shell/registry.ts')
  const render = (name: string, node: () => ReactNode) => {
    total++
    try {
      const html = renderToString(createElement(Providers, { theme: 'light' }, node()))
      const ids = [...html.matchAll(/data-figure-id="([^"]+)"/g)].map((m) => m[1])
      const repeated = [...new Set(ids.filter((id, i) => ids.indexOf(id) !== i))]
      if (repeated.length) {
        failures++
        console.error(`FAIL  ${name}: figure ids repeat within the page: ${repeated.join(', ')}`)
      }
    } catch (e) {
      failures++
      console.error(`FAIL  ${name}: ${(e as Error).message.split('\n')[0]}`)
    }
  }
  render('gallery', () => createElement(Gallery as ComponentType))
  for (const entry of ENTRIES as { path: string; snippet: string }[]) {
    render(entry.path, () => createElement(RecipePage, { entry }))
    const lines = entry.snippet.split('\n').length
    if (lines > MAX_SNIPPET_LINES)
      console.warn(`WARN  ${entry.path}: snippet is ${lines} lines; trim it with // region markers`)
  }
} finally {
  await server.close()
}
console.log(`${total} pages · ${failures} failed · ${importErrors} import boundary violations`)
if (failures || importErrors) process.exit(1)

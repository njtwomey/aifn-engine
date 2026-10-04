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
let examples = 0
let exampleFailures = 0
let functions = 0
let covered = 0
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
  const { Home } = await server.ssrLoadModule('/src/shell/Home.tsx')
  const { PackagePage, NodePage, FilePage } = await server.ssrLoadModule('/src/docs/DocsPages.tsx')
  const { TREE } = await server.ssrLoadModule('/src/docs/data.ts')
  render('home', () => createElement(Home as ComponentType))
  render('render', () => createElement(Gallery as ComponentType))
  type Node = { pkg: string; path: string; children: Node[]; files: { name: string }[] }
  const nodes: Node[] = []
  const walk = (list: Node[]) => list.forEach((n) => (nodes.push(n), walk(n.children)))
  for (const pkg of ['compute', 'methods']) {
    render(pkg, () => createElement(PackagePage, { pkg }))
    walk(TREE[pkg])
  }
  for (const node of nodes) render(`${node.pkg}/${node.path}`, () => createElement(NodePage, { node }))
  for (const node of nodes)
    for (const file of node.files)
      render(`${node.pkg}/${node.path}/${file.name}`, () => createElement(FilePage, { node, file }))

  // Every documentation example runs, in the scope the page gives it.
  const { default: content } = await server.ssrLoadModule('virtual:aifn-docs/content')
  const { scopeOf, runExample } = await server.ssrLoadModule('/src/docs/run.ts')
  type Example = { title: string; code: string }
  type Content = { examples: Example[]; exports: { name: string; kind: string; examples: Example[] }[] }
  for (const node of nodes) {
    const c = (content as Record<string, Content>)[`${node.pkg}/${node.path}`]
    // Coverage: the functions (and classes) with at least one runnable example of their own.
    const callable = c.exports.filter((x) => x.kind === 'function' || x.kind === 'class')
    functions += callable.length
    covered += callable.filter((x) => x.examples.some((e) => e.title !== '')).length
    // Titled examples are runnable cells; untitled ones are illustrative fragments, shown as plain code.
    const all = [
      ...c.examples.map((e) => ({ at: e.title || 'example', e })),
      ...c.exports.flatMap((x) => x.examples.map((e) => ({ at: x.name, e }))),
    ].filter(({ e }) => e.title !== '')
    if (!all.length) continue
    const scope = await scopeOf(node.pkg, node.path)
    for (const { at, e } of all) {
      examples++
      const r = runExample(e.code, scope)
      if (!r.ok) {
        exampleFailures++
        console.error(`FAIL  ${node.pkg}/${node.path} · ${at}: ${r.error}`)
      } else if (process.argv.includes('--show'))
        console.log(`ok    ${node.pkg}/${node.path} · ${at}\n${[...r.output, r.value].join('\n')}\n`)
    }
  }
  for (const entry of ENTRIES as { path: string; snippet: string }[]) {
    render(entry.path, () => createElement(RecipePage, { entry }))
    const lines = entry.snippet.split('\n').length
    if (lines > MAX_SNIPPET_LINES)
      console.warn(`WARN  ${entry.path}: snippet is ${lines} lines; trim it with // region markers`)
  }
} finally {
  await server.close()
}
console.log(
  `${total} pages · ${failures} failed · ${examples} examples · ${exampleFailures} failed · ${importErrors} import boundary violations`,
)
console.log(`example coverage (report only): ${covered} of ${functions} functions have a runnable example`)
if (failures || exampleFailures || importErrors) process.exit(1)

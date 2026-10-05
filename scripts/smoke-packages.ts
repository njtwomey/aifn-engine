/**
 * `make packages-smoke`: install the packed tarballs (`dist/packages/*.tgz`, built with `--local`) into an empty
 * project outside the workspace and use them as a consumer would, so a package that cannot be installed or imported
 * never reaches a release.
 *
 * - Every path of each package's `exports` is imported in plain Node (no bundler, no TypeScript): each module loads,
 *   and each primitive registers once. A second copy of `aifn-compute` inside `aifn-methods` would throw here.
 * - `aifn-compute` is installed once, shared by `aifn-methods`.
 * - A few values are computed, across the packages, and a figure of `aifn-render` renders to HTML on the server.
 * - A TypeScript file using the packages type-checks against the shipped declarations.
 *
 * What it does not cover: a browser. That `aifn-render` draws, takes its styles and starts its worker in a consumer's
 * bundle is checked by hand before a release (see the README).
 */
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const root = path.join(import.meta.dirname, '..')
const packed = path.join(root, 'dist', 'packages')
const tarballs = fs.existsSync(packed) ? fs.readdirSync(packed).filter((f) => f.endsWith('.tgz')) : []
if (tarballs.length === 0) {
  console.error('smoke: no tarballs in dist/packages; run node scripts/package.ts --version x.y.z --local first')
  process.exit(1)
}
const names = tarballs.map((f) => f.replace(/-\d+\.\d+\.\d+.*\.tgz$/, ''))

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'aifn-smoke-'))
const run = (cmd: string, args: string[]) => execFileSync(cmd, args, { cwd: dir, stdio: 'inherit' })
try {
  fs.writeFileSync(
    path.join(dir, 'package.json'),
    JSON.stringify(
      {
        name: 'aifn-smoke',
        private: true,
        type: 'module',
        dependencies: {
          ...Object.fromEntries(tarballs.map((f, i) => [names[i], `file:${path.join(packed, f)}`])),
          // aifn-render's peers, and their types for the type check.
          react: '^19.2.8',
          'react-dom': '^19.2.8',
          '@types/react': '^19.2.18',
          '@types/react-dom': '^19.2.7',
        },
      },
      null,
      2,
    ),
  )
  run('npm', ['install', '--no-audit', '--no-fund', '--silent'])

  for (const holder of ['aifn-methods', 'aifn-render'])
    for (const held of ['aifn-compute', 'aifn-methods', 'react'])
      if (fs.existsSync(path.join(dir, 'node_modules', holder, 'node_modules', held)))
        throw new Error(`${holder} installed its own copy of ${held}`)

  fs.writeFileSync(
    path.join(dir, 'use.mjs'),
    `import assert from 'node:assert/strict'
import fs from 'node:fs'
import { grad, sum, tensor, toArray } from 'aifn-compute'
import { cholesky, det, solve } from 'aifn-compute/numerics/linalg'
import { datasetRegistry } from 'aifn-methods/data'
import { learningModelRegistry } from 'aifn-methods/learning'
import { createElement } from 'react'
import { renderToString } from 'react-dom/server'

// Every importable path of every package loads (and registers its primitives once).
let paths = 0
for (const name of ${JSON.stringify(names)}) {
  const manifest = JSON.parse(fs.readFileSync(new URL('./node_modules/' + name + '/package.json', import.meta.url), 'utf8'))
  for (const sub of Object.keys(manifest.exports)) {
    // Stylesheets are not modules, and the worker's entry only runs inside a worker.
    if (sub.endsWith('.css') || sub.endsWith('.worker')) continue
    await import(sub === '.' ? name : name + sub.slice(1))
    paths++
  }
}

const A = tensor([[4, 1], [1, 3]])
assert.deepEqual(toArray(cholesky(A).L)[0], [2, 0])
assert.equal(Math.round(det(A)), 11)
const g = toArray(grad((b) => sum(solve(A, b)))(tensor([1, 2])))
assert.ok(Math.abs(g[0] - 2 / 11) < 1e-12 && Math.abs(g[1] - 3 / 11) < 1e-12)
assert.ok(Object.keys(datasetRegistry).length > 0 && Object.keys(learningModelRegistry).length > 0)

// A figure renders on the server (charts draw only in a browser, so this is its frame and controls).
const store = new Map()
Object.assign(globalThis, {
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
  localStorage: { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, v), removeItem: (k) => store.delete(k) },
})
const { Curve, Figure, Plot, Providers, useAxis } = await import('aifn-render')
function Example() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  return createElement(
    Figure,
    { title: 'A line', purpose: 'Rendered from the packed aifn-render.' },
    createElement(Plot, { x, y }, createElement(Curve, { name: 'line', x: [0, 1, 2], y: [0, 1, 4] })),
  )
}
const html = renderToString(createElement(Providers, { theme: 'light' }, createElement(Example)))
assert.ok(html.includes('A line') && html.includes('data-figure-id'))
for (const sheet of ['theme.css', 'base.css', 'styles.css']) {
  const manifest = JSON.parse(fs.readFileSync(new URL('./node_modules/aifn-render/package.json', import.meta.url), 'utf8'))
  assert.ok(fs.existsSync(new URL('./node_modules/aifn-render/' + manifest.exports['./' + sheet], import.meta.url)), sheet)
}
console.log('smoke: ' + paths + ' import paths load; values agree; ' + Object.keys(learningModelRegistry).length + ' models registered; a figure renders')
`,
  )
  run('node', ['use.mjs'])

  fs.writeFileSync(
    path.join(dir, 'use.tsx'),
    `import { tensor, type Tensor } from 'aifn-compute'
import { cholesky, det, type Cholesky } from 'aifn-compute/numerics/linalg'
import { datasetRegistry } from 'aifn-methods/data'
import { Curve, Figure, Plot, useAxis, type Vector } from 'aifn-render'

const A: Tensor = tensor([[4, 1], [1, 3]])
const factor: Cholesky = cholesky(A)
const d: number = det(A)
const jitter: number = factor.jitter
export const names: string[] = Object.keys(datasetRegistry)
export { d, jitter }

export const arrow: Vector = { from: [0, 0], to: [1, 1] }
export function Example() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  return (
    <Figure title="A line">
      <Plot x={x} y={y}>
        <Curve name="line" x={[0, 1, 2]} y={[0, 1, 4]} />
      </Plot>
    </Figure>
  )
}
`,
  )
  fs.writeFileSync(
    path.join(dir, 'tsconfig.json'),
    JSON.stringify({
      compilerOptions: {
        target: 'es2023',
        module: 'esnext',
        moduleResolution: 'bundler',
        strict: true,
        noEmit: true,
        skipLibCheck: false,
        jsx: 'react-jsx',
        lib: ['ES2023', 'DOM'],
        types: [],
      },
      files: ['use.tsx'],
    }),
  )
  execFileSync(path.join(root, 'node_modules', '.bin', 'tsc'), ['-p', dir], { cwd: dir, stdio: 'inherit' })
  console.log('smoke: declarations type-check in a consumer (skipLibCheck off)')
} finally {
  fs.rmSync(dir, { recursive: true, force: true })
}

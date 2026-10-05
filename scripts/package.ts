/**
 * Build `aifn-compute`, `aifn-methods` and `aifn-render` as installable packages and pack each into a tarball,
 * without touching the workspace: the repo keeps importing the TypeScript source through the workspace `exports`, and
 * each package's `dist/` holds what a consumer outside the repo installs.
 *
 * - JavaScript: one ES module entry per importable path, bundled by Vite's library mode with shared chunks, so a
 *   consumer that imports several modules loads each primitive (and its registration), or each React context, once.
 *   Another package of the engine stays an import: a second copy of compute would register every primitive twice.
 *   `aifn-render` also leaves every third-party package external (React must be the consumer's one copy).
 * - Declarations: `tsc` with `emitDeclarationOnly` over the source, into `dist/types`. A package reads the built
 *   declarations of the engine packages it uses, so they are built in order.
 * - `dist/package.json`: the source manifest without `private`, with the version, the licence, `sideEffects` and
 *   `exports` mapping each entry to `{ types, import }`. A dependency on another package of the engine becomes the URL
 *   of that package's tarball in the same GitHub release (the packages are not on the npm registry), or a `file:`
 *   path with `--local`.
 * - Tarballs: `npm pack` of each `dist/`, into `dist/packages/<name>-<version>.tgz` at the repository root. A release
 *   attaches these to the GitHub release `v<version>`.
 *
 * What is particular to `aifn-render`:
 * - Its compute worker finds engine modules through `src/state/worker-modules.ts`, which globs the sibling packages'
 *   source. A package cannot do that, so the build replaces that file with one generated here: an
 *   `import('aifn-compute/…')` or `import('aifn-methods/…')` per module.
 * - The worker is its own entry (`state/compute.worker.js`), and the code that starts it is left as
 *   `new Worker(new URL('<path to that entry>', import.meta.url))`, the form a consumer's bundler recognises and
 *   bundles in turn. (Left to itself, Vite's library mode would inline the worker, with imports nothing could resolve.)
 * - Its stylesheets ship with it: `theme.css` and `base.css` for a consumer with Tailwind v4, and `styles.css`, the
 *   same compiled with every class the components use, for one without.
 *
 * Usage: `node scripts/package.ts --version x.y.z [--local]` (or `make packages VERSION=x.y.z`). `--local` is for
 * trying the tarballs before a release exists (`make packages-smoke`).
 */

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { build, type Plugin } from 'vite'

const root = path.join(import.meta.dirname, '..')
const out = path.join(root, 'dist', 'packages')
const REPOSITORY = 'https://github.com/njtwomey/aifn-engine'

const arg = (name: string) => (process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : null)
const version = arg('--version')
const local = process.argv.includes('--local')
if (!version || !/^\d+\.\d+\.\d+(-[\w.]+)?$/.test(version)) {
  console.error('usage: node scripts/package.ts --version x.y.z [--local]')
  process.exit(1)
}

type Package = { name: string; dir: string; uses: string[]; ui?: boolean }

/** The packages, in build order: each may depend only on those before it. */
const PACKAGES: Package[] = [
  { name: 'aifn-compute', dir: 'compute', uses: [] },
  { name: 'aifn-methods', dir: 'methods', uses: ['aifn-compute'] },
  { name: 'aifn-render', dir: 'render', uses: ['aifn-compute', 'aifn-methods'], ui: true },
]

const dirOf = (name: string) => path.join(root, 'packages', name.replace(/^aifn-/, ''))
const tarball = (name: string) => `${name}-${version}.tgz`
/** Where a consumer gets another package of the engine: the same release's tarball, or the local one. */
const sourceOf = (name: string) =>
  local ? `file:${path.join(out, tarball(name))}` : `${REPOSITORY}/releases/download/v${version}/${tarball(name)}`

/**
 * The entries of a numerics package: every directory under `src` with an `index.ts` (its `exports` is the pattern
 * `./*`), and the root where it has one. Names are import paths without the package name (`index` for the root).
 */
function moduleEntries(src: string): Record<string, string> {
  const entries: Record<string, string> = {}
  if (fs.existsSync(path.join(src, 'index.ts'))) entries.index = path.join(src, 'index.ts')
  const walk = (dir: string, rel: string) => {
    for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
      if (!d.isDirectory() || d.name.startsWith('_')) continue
      const at = rel ? `${rel}/${d.name}` : d.name
      const index = path.join(dir, d.name, 'index.ts')
      if (fs.existsSync(index)) entries[`${at}/index`] = index
      walk(path.join(dir, d.name), at)
    }
  }
  walk(src, '')
  return entries
}

/**
 * The entries of a package whose manifest lists its `exports` one by one (`aifn-render`): each source file an export
 * names, a `*` pattern standing for every file that matches it. Stylesheets are not entries.
 */
function listedEntries(pkgDir: string, exportsField: Record<string, string>): Record<string, string> {
  const entries: Record<string, string> = {}
  const add = (file: string) => {
    const rel = path.relative(path.join(pkgDir, 'src'), file).replace(/\.tsx?$/, '')
    entries[rel] = file
  }
  for (const target of Object.values(exportsField)) {
    if (!/\.tsx?$/.test(target)) continue
    if (!target.includes('*')) add(path.join(pkgDir, target))
    else {
      const [before, after] = target.split('*')
      const dir = path.join(pkgDir, before)
      for (const f of fs.readdirSync(dir))
        if (f.endsWith(after) && !f.endsWith(`.test${after}`) && f !== `index${after}`) add(path.join(dir, f))
    }
  }
  return entries
}

/** A consumer's import path for an entry: `.` for the root, `./a/b` for `a/b/index` or the file `a/b`. */
const subpathOf = (entry: string) => (entry === 'index' ? '.' : `./${entry.replace(/\/index$/, '')}`)

// ── What is particular to aifn-render ────────────────────────────────────────────────────────────────────────────────

const WORKER_ENTRY = 'state/compute.worker'
const WORKER_URL = '__AIFN_COMPUTE_WORKER_URL__'

/** The packaged `worker-modules`: one lazy import by package name per module of compute and of methods. */
function workerModules(): string {
  const table = (name: string) =>
    Object.keys(moduleEntries(path.join(dirOf(name), 'src')))
      .filter((e) => e !== 'index')
      .map((e) => e.replace(/\/index$/, ''))
      .map((p) => `  ${JSON.stringify(p)}: () => import(${JSON.stringify(`${name}/${p}`)}),`)
      .join('\n')
  return [
    `const COMPUTE = {\n${table('aifn-compute')}\n}`,
    `const METHODS = {\n${table('aifn-methods')}\n}`,
    `export function loaderOf(path) {\n  return COMPUTE[path] ?? METHODS[path.replace(/^applied\\//, '')]\n}`,
  ].join('\n\n')
}

/** Swaps `worker-modules` for the generated one, and keeps the worker's start as a URL to the worker's own entry. */
function renderWorker(src: string): Plugin {
  const modules = path.join(src, 'state', 'worker-modules.ts')
  const starter = path.join(src, 'state', 'worker.ts')
  return {
    name: 'aifn-render-worker',
    enforce: 'pre',
    load: (id) => (id === modules ? workerModules() : null),
    transform(code, id) {
      if (id !== starter) return null
      const from = "new URL('./compute.worker.ts', import.meta.url)"
      if (!code.includes(from)) this.error(`${id}: the worker is no longer started with ${from}`)
      return { code: code.replace(from, WORKER_URL), map: null }
    },
    renderChunk(code, chunk) {
      if (!code.includes(WORKER_URL)) return null
      let rel = path.posix.relative(path.posix.dirname(chunk.fileName), `${WORKER_ENTRY}.js`)
      if (!rel.startsWith('.')) rel = `./${rel}`
      return { code: code.replaceAll(WORKER_URL, `new URL(${JSON.stringify(rel)}, import.meta.url)`), map: null }
    },
  }
}

/**
 * `styles.css`: the theme and base styles compiled by Tailwind with every class the built components use, for a
 * consumer without Tailwind. It leaves out Tailwind's reset (preflight), which would restyle the consumer's page.
 */
async function compileStyles(dist: string) {
  const work = path.join(dist, '.styles')
  fs.mkdirSync(work, { recursive: true })
  fs.writeFileSync(
    path.join(work, 'styles.css'),
    [
      '@layer theme, base, components, utilities;',
      "@import 'tailwindcss/theme.css' layer(theme);",
      "@import 'tailwindcss/utilities.css' layer(utilities);",
      "@import '../styles/theme.css';",
      "@import '../styles/base.css';",
      "@source '../';",
    ].join('\n'),
  )
  fs.writeFileSync(path.join(work, 'entry.js'), "import './styles.css'\n")
  await build({
    configFile: false,
    logLevel: 'warn',
    root: work,
    plugins: [tailwindcss()],
    build: {
      outDir: path.join(work, 'out'),
      emptyOutDir: true,
      minify: false,
      lib: { entry: path.join(work, 'entry.js'), formats: ['es'], fileName: 'entry', cssFileName: 'styles' },
    },
  })
  fs.copyFileSync(path.join(work, 'out', 'styles.css'), path.join(dist, 'styles.css'))
  fs.rmSync(work, { recursive: true, force: true })
}

// ── Build ────────────────────────────────────────────────────────────────────────────────────────────────────────────

fs.rmSync(out, { recursive: true, force: true })
fs.mkdirSync(out, { recursive: true })

for (const { name, dir, uses, ui } of PACKAGES) {
  const pkgDir = path.join(root, 'packages', dir)
  const src = path.join(pkgDir, 'src')
  const dist = path.join(pkgDir, 'dist')
  const manifest = JSON.parse(fs.readFileSync(path.join(pkgDir, 'package.json'), 'utf8')) as {
    exports: Record<string, string>
    dependencies?: Record<string, string>
    devDependencies?: Record<string, string>
    [field: string]: unknown
  }
  const entries = ui
    ? { ...listedEntries(pkgDir, manifest.exports), [WORKER_ENTRY]: path.join(src, `${WORKER_ENTRY}.ts`) }
    : moduleEntries(src)
  fs.rmSync(dist, { recursive: true, force: true })

  await build({
    configFile: false,
    logLevel: 'warn',
    plugins: ui ? [renderWorker(src), react()] : [],
    build: {
      outDir: dist,
      emptyOutDir: false,
      target: 'es2023',
      minify: false,
      sourcemap: true,
      lib: { entry: entries, formats: ['es'] },
      rollupOptions: {
        // Another package of the engine stays an import, and for the UI package so does every third-party one.
        external: ui
          ? (id) => !id.startsWith('.') && !path.isAbsolute(id) && !id.startsWith('\0')
          : uses.map((u) => new RegExp(`^${u}(/|$)`)),
        output: { entryFileNames: '[name].js', chunkFileNames: 'chunks/[name]-[hash].js' },
      },
    },
  })

  // Declarations from the source, with the workspace's compiler options. Another package of the engine is read from
  // its built declarations, so that its source is not pulled into this program (and emitted again).
  const paths: Record<string, string[]> = {}
  for (const u of uses) {
    const types = path.join(dirOf(u), 'dist', 'types')
    paths[u] = [path.join(types, 'index.d.ts')]
    paths[`${u}/*`] = [path.join(types, '*', 'index.d.ts')]
  }
  const tsconfig = path.join(dist, 'tsconfig.types.json')
  fs.writeFileSync(
    tsconfig,
    JSON.stringify(
      {
        extends: path.join(root, 'tsconfig.base.json'),
        compilerOptions: {
          noEmit: false,
          emitDeclarationOnly: true,
          declaration: true,
          declarationDir: path.join(dist, 'types'),
          rootDir: src,
          tsBuildInfoFile: null,
          incremental: false,
          composite: false,
          paths,
          ...(ui
            ? { lib: ['ES2023', 'DOM'], jsx: 'react-jsx', types: ['vite/client', 'node'], resolveJsonModule: true }
            : {}),
        },
        include: [src],
        exclude: [path.join(src, '**', '*.test.ts'), path.join(src, '**', '*.test.tsx')],
      },
      null,
      2,
    ),
  )
  execFileSync('npx', ['tsc', '-p', tsconfig], { cwd: root, stdio: 'inherit' })
  fs.rmSync(tsconfig)

  const exportsMap: Record<string, string | { types: string; import: string }> = {}
  for (const entry of Object.keys(entries))
    exportsMap[subpathOf(entry)] = { types: `./types/${entry}.d.ts`, import: `./${entry}.js` }

  const { private: _private, dependencies = {}, devDependencies = {}, ...rest } = manifest
  // Third-party dependencies as the source manifest has them, except React, which must be the consumer's own copy.
  const peers = ['react', 'react-dom']
  const third = Object.entries(dependencies).filter(([d]) => !d.startsWith('aifn-') && !peers.includes(d))
  // The declarations name the types of some third-party packages (KaTeX's options): a consumer needs those too.
  // React's types come with the consumer's React.
  const typings = Object.entries(devDependencies).filter(([d]) => d.startsWith('@types/') && !d.includes('react'))
  const engine = uses.map((u) => [u, sourceOf(u)])
  const peerDependencies = Object.fromEntries(Object.entries(dependencies).filter(([d]) => peers.includes(d)))

  if (ui) {
    fs.cpSync(path.join(pkgDir, 'styles'), path.join(dist, 'styles'), { recursive: true })
    await compileStyles(dist)
    exportsMap['./theme.css'] = './styles/theme.css'
    exportsMap['./base.css'] = './styles/base.css'
    exportsMap['./styles.css'] = './styles.css'
  }

  const packed = {
    ...rest,
    version,
    license: 'MIT',
    repository: { type: 'git', url: `git+${REPOSITORY}.git`, directory: `packages/${dir}` },
    homepage: 'https://njtwomey.github.io/aifn-engine/',
    sideEffects: true,
    files: ['**/*.js', '**/*.js.map', 'types', 'styles', 'styles.css', 'LICENSE', 'README.md'],
    ...(third.length + engine.length ? { dependencies: Object.fromEntries([...engine, ...third, ...typings]) } : {}),
    ...(Object.keys(peerDependencies).length ? { peerDependencies } : {}),
    exports: exportsMap,
  }
  fs.writeFileSync(path.join(dist, 'package.json'), JSON.stringify(packed, null, 2) + '\n')
  fs.copyFileSync(path.join(root, 'LICENSE'), path.join(dist, 'LICENSE'))
  if (fs.existsSync(path.join(pkgDir, 'README.md')))
    fs.copyFileSync(path.join(pkgDir, 'README.md'), path.join(dist, 'README.md'))

  execFileSync('npm', ['pack', '--pack-destination', out, '--silent'], { cwd: dist, stdio: 'inherit' })
  const size = (fs.statSync(path.join(out, tarball(name))).size / 1024).toFixed(0)
  console.log(
    `aifn-package: ${name}@${version}: ${Object.keys(entries).length} entries → dist/packages/${tarball(name)} (${size} kB)`,
  )
}

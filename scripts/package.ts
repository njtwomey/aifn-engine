/**
 * Build `aifn-compute` and `aifn-methods` as installable packages and pack each into a tarball, without touching the
 * workspace: the repo keeps importing the TypeScript source through the workspace `exports`, and each package's
 * `dist/` holds what a consumer outside the repo installs.
 *
 * - JavaScript: one ES module entry per importable path (every directory under `src` with an `index.ts`, and the
 *   package root where it has one), bundled by Vite's library mode with shared chunks, so a consumer that imports
 *   several modules loads each primitive (and its registration) once. `aifn-methods` leaves `aifn-compute` external:
 *   a second copy of compute would register every primitive twice.
 * - Declarations: `tsc` with `emitDeclarationOnly` over the source, into `dist/types`. `aifn-methods` reads compute's
 *   built declarations, so compute is built first.
 * - `dist/package.json`: the source manifest without `private`, with the version, the licence, `sideEffects` (importing
 *   a module registers its primitives, so every file counts) and `exports` mapping each entry to `{ types, import }`.
 *   A dependency on another package of the engine becomes the URL of that package's tarball in the same GitHub
 *   release (the packages are not on the npm registry), or a `file:` path with `--local`.
 * - Tarballs: `npm pack` of each `dist/`, into `dist/packages/<name>-<version>.tgz` at the repository root. A release
 *   attaches these to the GitHub release `v<version>`.
 *
 * Usage: `node scripts/package.ts --version x.y.z [--local]` (or `make packages VERSION=x.y.z`). `--local` is for
 * trying the tarballs before a release exists (`make packages-smoke`).
 */

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { build } from 'vite'

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

/** The packages, in build order: each may depend only on those before it. */
const PACKAGES = [
  { name: 'aifn-compute', dir: 'compute', uses: [] as string[] },
  { name: 'aifn-methods', dir: 'methods', uses: ['aifn-compute'] },
]

const tarball = (name: string) => `${name}-${version}.tgz`
/** Where a consumer gets another package of the engine: the same release's tarball, or the local one. */
const sourceOf = (name: string) =>
  local ? `file:${path.join(out, tarball(name))}` : `${REPOSITORY}/releases/download/v${version}/${tarball(name)}`

/** Entry names (import paths without the package name; `index` for the root) to their source files. */
function entriesOf(src: string): Record<string, string> {
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

fs.rmSync(out, { recursive: true, force: true })
fs.mkdirSync(out, { recursive: true })

for (const { name, dir, uses } of PACKAGES) {
  const pkgDir = path.join(root, 'packages', dir)
  const src = path.join(pkgDir, 'src')
  const dist = path.join(pkgDir, 'dist')
  const entries = entriesOf(src)
  fs.rmSync(dist, { recursive: true, force: true })

  await build({
    configFile: false,
    logLevel: 'warn',
    build: {
      outDir: dist,
      emptyOutDir: false,
      target: 'es2023',
      minify: false,
      sourcemap: true,
      lib: { entry: entries, formats: ['es'] },
      rollupOptions: {
        // Another package of the engine stays an import: the consumer loads the one copy both sides share.
        external: uses.map((u) => new RegExp(`^${u}(/|$)`)),
        output: { entryFileNames: '[name].js', chunkFileNames: 'chunks/[name]-[hash].js' },
      },
    },
  })

  // Declarations from the source, with the workspace's compiler options. Another package of the engine is read from
  // its built declarations, so that its source is not pulled into this program (and emitted again).
  const paths: Record<string, string[]> = {}
  for (const u of uses) {
    const types = path.join(root, 'packages', u.replace(/^aifn-/, ''), 'dist', 'types')
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
        },
        include: [src],
        exclude: [],
      },
      null,
      2,
    ),
  )
  execFileSync('npx', ['tsc', '-p', tsconfig], { cwd: root, stdio: 'inherit' })
  fs.rmSync(tsconfig)

  const manifest = JSON.parse(fs.readFileSync(path.join(pkgDir, 'package.json'), 'utf8')) as Record<string, unknown>
  const exportsMap: Record<string, { types: string; import: string }> = {}
  for (const entry of Object.keys(entries)) {
    const sub = entry === 'index' ? '.' : `./${entry.replace(/\/index$/, '')}`
    exportsMap[sub] = { types: `./types/${entry}.d.ts`, import: `./${entry}.js` }
  }
  const { private: _private, dependencies: _dependencies, ...rest } = manifest
  const packed = {
    ...rest,
    version,
    license: 'MIT',
    repository: { type: 'git', url: `git+${REPOSITORY}.git`, directory: `packages/${dir}` },
    homepage: 'https://njtwomey.github.io/aifn-engine/',
    sideEffects: true,
    files: ['**/*.js', '**/*.js.map', 'types', 'LICENSE', 'README.md'],
    ...(uses.length ? { dependencies: Object.fromEntries(uses.map((u) => [u, sourceOf(u)])) } : {}),
    exports: exportsMap,
  }
  fs.writeFileSync(path.join(dist, 'package.json'), JSON.stringify(packed, null, 2) + '\n')
  fs.copyFileSync(path.join(root, 'LICENSE'), path.join(dist, 'LICENSE'))
  fs.copyFileSync(path.join(pkgDir, 'README.md'), path.join(dist, 'README.md'))

  execFileSync('npm', ['pack', '--pack-destination', out, '--silent'], { cwd: dist, stdio: 'inherit' })
  const size = (fs.statSync(path.join(out, tarball(name))).size / 1024).toFixed(0)
  console.log(
    `aifn-package: ${name}@${version}: ${Object.keys(entries).length} entries → dist/packages/${tarball(name)} (${size} kB)`,
  )
}

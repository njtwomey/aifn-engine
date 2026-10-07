/**
 * `make layers` (part of `make lint` and `make test`): checks the packages against the module tree, `modules.json`.
 *
 * - The tree: every directory under `packages/compute/src` and `packages/methods/src` is a declared node (family, group, module or
 *   area) or an alias (an old flat path, kept only during phase 1 step 1); every declared node has a folder with an
 *   `index.ts` (unless its status is "gap"); a module has no child directories; the plain files at a group's root are
 *   exactly its declared shared files.
 * - Compute (package `aifn-compute`): a file imports a node of a strictly lower family tier, or of its own family and a strictly
 *   lower local tier (D5); its ancestors' shared files by relative path; nothing else of compute by relative path. A
 *   family index is imported from outside the family only for its shared names; the package root and aliases never
 *   (aliases warn while they exist). `foundation` (tier 0) therefore imports nothing else in compute. Compute, its tests
 *   included, never imports `aifn-methods`, and nothing outside aifn-compute.
 * - Applications (package `aifn-methods`): any compute node; ancestors' shared files by relative path; another area only
 *   down `dependsOn` (acyclic), through its public path; never a node of its own area (siblings share through the
 *   parent). No React, DOM libraries or third-party packages.
 * - A parent never imports a child, except an `index.ts` re-exporting it.
 * - The tables between the `aifn-layers` and `aifn-areas` markers in `README.md` are generated from the file:
 *   `--write` regenerates them, and the check fails when stale.
 */
import fs from 'node:fs'
import path from 'node:path'

type CoreModule = { module: string; tier: number; status?: 'gap' }
type Family = { family: string; tier: number; shared: string[]; modules: CoreModule[] }
type AppNode = { module: string; status?: 'gap' } | { group: string; shared: string[]; children: AppNode[] }
type Area = {
  area: string
  dependsOn: string[]
  shared: string[]
  children: AppNode[]
}
type Modules = {
  compute: { package: string; families: Family[] }
  applications: { package: string; areas: Area[] }
  aliases: { path: string; until: string }[]
  transitional?: { file: string; imports: string; reason: string }[]
}

const root = path.resolve(import.meta.dirname, '..')
const computeDir = path.join(root, 'packages', 'compute')
const appsDir = path.join(root, 'packages', 'methods')
const computeSrc = path.join(computeDir, 'src')
const appsSrc = path.join(appsDir, 'src')
const write = process.argv.includes('--write')
const errors: string[] = []
const warnings: string[] = []

const spec = JSON.parse(fs.readFileSync(path.join(root, 'modules.json'), 'utf8')) as Modules
const aliases = new Set(spec.aliases.map((a) => a.path))
const transitional = spec.transitional ?? []

// ── The declared tree ────────────────────────────────────────────────────────────────────────────────────────────────

/** A declared node, keyed by its path under src (`numerics/linalg`, `learning/generalised`). */
type Node = {
  pkg: 'compute' | 'apps'
  path: string
  kind: 'family' | 'module' | 'group' | 'area'
  /** Compute: family tier and local tier (a family itself has local -1; a single-module family is a module at 0). */
  tier: number
  local: number
  shared: string[]
  gap: boolean
  children: string[]
}
const nodes = new Map<string, Node>()
const add = (key: string, n: Node) => {
  if (nodes.has(key)) errors.push(`modules.json: ${key} is declared twice`)
  nodes.set(key, n)
}
for (const f of spec.compute.families) {
  const single = f.modules.length === 0
  add(`compute:${f.family}`, {
    pkg: 'compute',
    path: f.family,
    kind: single ? 'module' : 'family',
    tier: f.tier,
    local: single ? 0 : -1,
    shared: f.shared,
    gap: false,
    children: f.modules.map((m) => `${f.family}/${m.module}`),
  })
  for (const m of f.modules)
    add(`compute:${f.family}/${m.module}`, {
      pkg: 'compute',
      path: `${f.family}/${m.module}`,
      kind: 'module',
      tier: f.tier,
      local: m.tier,
      shared: [],
      gap: m.status === 'gap',
      children: [],
    })
}
const addApp = (prefix: string, n: AppNode) => {
  if ('module' in n) {
    add(`apps:${prefix}/${n.module}`, {
      pkg: 'apps',
      path: `${prefix}/${n.module}`,
      kind: 'module',
      tier: 0,
      local: 0,
      shared: [],
      gap: n.status === 'gap',
      children: [],
    })
    return
  }
  const p = `${prefix}/${n.group}`
  add(`apps:${p}`, {
    pkg: 'apps',
    path: p,
    kind: 'group',
    tier: 0,
    local: 0,
    shared: n.shared,
    gap: false,
    children: n.children.map((c) => `${p}/${'module' in c ? c.module : c.group}`),
  })
  for (const c of n.children) addApp(p, c)
}
const areas = new Map<string, Area>()
for (const a of spec.applications.areas) {
  areas.set(a.area, a)
  add(`apps:${a.area}`, {
    pkg: 'apps',
    path: a.area,
    // An area without children is a single module (timeseries), as a compute family without modules is.
    kind: a.children.length ? 'area' : 'module',
    tier: 0,
    local: 0,
    shared: a.shared,
    gap: false,
    children: a.children.map((c) => `${a.area}/${'module' in c ? c.module : c.group}`),
  })
  for (const c of a.children) addApp(a.area, c)
}

// ── The area DAG ─────────────────────────────────────────────────────────────────────────────────────────────────────

for (const a of areas.values())
  for (const d of a.dependsOn) if (!areas.has(d)) errors.push(`modules.json: area ${a.area} depends on unknown ${d}`)

/** The areas below `area` in the DAG (transitively), or null on a cycle through it. */
function below(area: string, seen: string[] = []): Set<string> | null {
  if (seen.includes(area)) return null
  const out = new Set<string>()
  for (const d of areas.get(area)?.dependsOn ?? []) {
    const sub = below(d, [...seen, area])
    if (!sub) return null
    out.add(d)
    for (const s of sub) out.add(s)
  }
  return out
}
const reachable = new Map<string, Set<string>>()
for (const a of areas.keys()) {
  const r = below(a)
  if (!r) errors.push(`modules.json: area ${a} is on a dependency cycle`)
  reachable.set(a, r ?? new Set())
}

// ── Folders against the file ─────────────────────────────────────────────────────────────────────────────────────────

const dirs = (dir: string) =>
  fs.existsSync(dir)
    ? fs
        .readdirSync(dir, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name)
    : []
const plainFiles = (dir: string) =>
  fs.existsSync(dir)
    ? fs
        .readdirSync(dir, { withFileTypes: true })
        .filter((d) => d.isFile() && /\.tsx?$/.test(d.name) && d.name !== 'index.ts')
        .map((d) => d.name.replace(/\.tsx?$/, ''))
    : []

const publicOf = (pkg: 'compute' | 'apps', p: string) => `${pkg === 'compute' ? 'aifn-compute' : 'aifn-methods'}/${p}`
const srcOf = (pkg: 'compute' | 'apps') => (pkg === 'compute' ? computeSrc : appsSrc)

/** Legacy folders: aliases whose folder is not a declared node (their files are checked leniently). */
const legacyDirs = new Set<string>()
/** Files of a legacy alias left at a family root that is also a new node (graph, optim, nn during the move). */
const legacyFiles = new Set<string>()
function checkDir(pkg: 'compute' | 'apps', rel: string) {
  const full = path.join(srcOf(pkg), rel)
  for (const d of dirs(full)) {
    const p = rel ? `${rel}/${d}` : d
    const n = nodes.get(`${pkg}:${p}`)
    if (n) continue
    if (aliases.has(publicOf(pkg, p))) {
      legacyDirs.add(`${pkg}:${p}`)
      continue
    }
    errors.push(
      `packages/${pkg === 'compute' ? 'compute' : 'methods'}/src/${p}: folder is not declared in modules.json`,
    )
  }
  for (const d of dirs(full)) {
    const p = rel ? `${rel}/${d}` : d
    if (nodes.has(`${pkg}:${p}`)) checkDir(pkg, p)
  }
}
checkDir('compute', '')
checkDir('apps', '')
for (const [key, n] of nodes) {
  const full = path.join(srcOf(n.pkg), n.path)
  const where = `packages/${n.pkg === 'compute' ? 'compute' : 'methods'}/src/${n.path}`
  if (!fs.existsSync(full)) {
    if (!n.gap) errors.push(`modules.json: ${key} has no folder`)
    continue
  }
  if (n.gap) errors.push(`modules.json: ${key} is a gap but has a folder; drop its status`)
  if (!fs.existsSync(path.join(full, 'index.ts'))) errors.push(`${where}: no index.ts`)
  if (n.kind === 'module') {
    const sub = dirs(full).filter((d) => !legacyDirs.has(`${n.pkg}:${n.path}/${d}`))
    if (sub.length) errors.push(`${where}: a module has no child folders (found ${sub.join(', ')})`)
    continue
  }
  const files = plainFiles(full)
  const legacy = aliases.has(publicOf(n.pkg, n.path))
  for (const f of files)
    if (!n.shared.includes(f)) {
      if (legacy) legacyFiles.add(path.join(full, `${f}.ts`))
      else errors.push(`${where}/${f}.ts: not a declared shared file of ${n.path}`)
    }
  for (const s of n.shared)
    if (!files.includes(s))
      (aliases.size ? warnings : errors).push(`modules.json: ${key} shared file ${s}.ts is missing`)
}

// ── Imports ──────────────────────────────────────────────────────────────────────────────────────────────────────────

function* files(dir: string, pattern = /\.tsx?$/): Generator<string> {
  if (!fs.existsSync(dir)) return
  for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, d.name)
    if (d.isDirectory()) {
      if (d.name !== 'node_modules' && d.name !== '__pycache__') yield* files(p, pattern)
    } else if (pattern.test(d.name)) yield p
  }
}

// Static imports and re-exports (`from '…'`, with a default binding before the braces too), side-effect imports and
// `import('…')` type queries, with the named bindings when there are any. A specifier has no whitespace, which keeps
// string literals in prose out.
const statements =
  /(?:\b(?:import|export)\s+(?:type\s+)?(?:(?:[\w$]+\s*,\s*)?\{([^}]*)\}\s*from\s*)?|\bfrom\s*|\bimport\s*\(\s*|^\s*import\s+)['"]([^'"\s]+)['"]/gm
/** `import(x)` with a computed specifier, which no static check can follow. */
const dynamicImport = /\bimport\s*\(\s*[^'"\s)]/g

function* imports(file: string): Generator<{ spec: string; names: string[] | null; where: string }> {
  const text = fs.readFileSync(file, 'utf8')
  if (file.startsWith(computeSrc + path.sep) || file.startsWith(appsSrc + path.sep))
    for (const m of text.matchAll(dynamicImport))
      errors.push(
        `${path.relative(root, file)}:${text.slice(0, m.index).split('\n').length}: import() with a computed specifier`,
      )
  for (const match of text.matchAll(statements)) {
    const line = text.slice(0, match.index).split('\n').length
    const names =
      match[1] === undefined
        ? null
        : match[1]
            .split(',')
            .map((s) =>
              s
                .trim()
                .replace(/^type\s+/, '')
                .split(/\s+as\s+/)[0]
                .trim(),
            )
            .filter(Boolean)
    yield { spec: match[2], names, where: `${path.relative(root, file)}:${line}` }
  }
}

/** Where a file sits: its package, the node (or legacy folder) holding it, and whether it is that node's index. */
type Place = {
  pkg: 'compute' | 'apps'
  dir: string
  node: Node | null
  legacy: boolean
  index: boolean
  shared: boolean
}
function place(file: string): Place | null {
  for (const pkg of ['compute', 'apps'] as const) {
    const rel = path.relative(srcOf(pkg), file)
    if (rel.startsWith('..')) continue
    const dir = path.dirname(rel).split(path.sep).join('/')
    const node = nodes.get(`${pkg}:${dir}`) ?? null
    const index = path.basename(file) === 'index.ts'
    const legacy =
      legacyDirs.has(`${pkg}:${dir}`) ||
      legacyFiles.has(file) ||
      [...legacyDirs].some((d) => `${pkg}:${dir}`.startsWith(`${d}/`))
    const shared = !!node && node.kind !== 'module' && !index && !legacy
    return { pkg, dir, node, legacy, index, shared }
  }
  return null
}

/** The node a public specifier names (`aifn-compute/numerics/linalg`), or 'alias', 'root' or null. */
function target(spec: string): { pkg: 'compute' | 'apps'; node: Node } | 'alias' | 'root' | null {
  if (spec === 'aifn-compute' || spec === 'aifn-methods') return 'root'
  if (aliases.has(spec)) return 'alias'
  const m = /^(aifn-compute|aifn-methods)\/(.+)$/.exec(spec)
  if (!m) return null
  const pkg = m[1] === 'aifn-compute' ? 'compute' : 'apps'
  const node = nodes.get(`${pkg}:${m[2]}`)
  return node ? { pkg, node } : null
}

/** The names a group index takes from its children (`export … from './child'`), which must be imported from the child. */
const childNames = new Map<string, Set<string>>()
function namesFromChildren(n: Node): Set<string> {
  const key = `${n.pkg}:${n.path}`
  const known = childNames.get(key)
  if (known) return known
  const out = new Set<string>()
  const index = path.join(srcOf(n.pkg), n.path, 'index.ts')
  if (fs.existsSync(index)) {
    const text = fs.readFileSync(index, 'utf8')
    for (const m of text.matchAll(/export\s+(?:type\s+)?\{([^}]*)\}\s*from\s*'\.\/([^'/]+)'/g)) {
      if (!nodes.has(`${n.pkg}:${n.path}/${m[2]}`)) continue
      for (const s of m[1].split(','))
        out.add(
          s
            .trim()
            .replace(/^type\s+/, '')
            .split(/\s+as\s+/)
            .pop()!
            .trim(),
        )
    }
  }
  childNames.set(key, out)
  return out
}

const familyOf = (p: string) => p.split('/')[0]
let aliasImports = 0

function checkRelative(file: string, at: Place, spec: string, where: string) {
  const resolved = path.resolve(path.dirname(file), spec)
  const rel = path.relative(srcOf(at.pkg), resolved)
  if (rel.startsWith('..')) return errors.push(`${where}: relative import '${spec}' leaves the package`)
  const tdir = path.dirname(rel).split(path.sep).join('/')
  const base = path.basename(rel).replace(/\.tsx?$/, '')
  if (legacyFiles.has(`${resolved}.ts`)) return
  if ([...legacyDirs].some((d) => d === `${at.pkg}:${tdir}`)) {
    // Into an old flat folder whose files wait for 1c: allowed while the alias exists.
    warnings.push(`${where}: relative import '${spec}' into a legacy folder (TODO(tree 1c))`)
    return
  }
  if (tdir === at.dir && base !== 'index' && fs.existsSync(`${resolved}.ts`)) {
    // Own folder: a module's files import each other; a shared file imports only other shared files.
    if (at.shared && !at.node!.shared.includes(base))
      errors.push(`${where}: shared file imports '${spec}', which is not shared (a parent never imports a child)`)
    return
  }
  // An index re-exports its own files and its descendants.
  if (at.index && (path.dirname(resolved) + path.sep).startsWith(path.dirname(file) + path.sep)) return
  if (at.legacy) {
    if (tdir === at.dir || nodes.has(`${at.pkg}:${tdir}`)) return
    return errors.push(`${where}: relative import '${spec}' leaves its folder`)
  }
  // An ancestor's shared file.
  const anc = nodes.get(`${at.pkg}:${tdir}`)
  if (anc && at.dir.startsWith(`${tdir}/`) && anc.shared.includes(base)) return
  errors.push(`${where}: relative import '${spec}' is neither in its module nor an ancestor's shared file`)
}

function checkCore(file: string) {
  if (file === path.join(computeSrc, 'index.ts')) {
    // The package root (D1): foundation's common surface only.
    for (const { spec, where } of imports(file))
      if (spec !== 'aifn-compute' && !spec.startsWith('aifn-compute/foundation/'))
        errors.push(`${where}: the package root re-exports foundation only, not '${spec}'`)
    return
  }
  const at = place(file)!
  const own = at.node
  for (const { spec, names, where } of imports(file)) {
    if (spec.startsWith('.')) {
      checkRelative(file, at, spec, where)
      continue
    }
    if (spec.startsWith('aifn-methods')) {
      errors.push(`${where}: compute imports the application '${spec}'; compute never imports aifn-methods`)
      continue
    }
    const t = target(spec)
    if (t === null) {
      errors.push(
        `${where}: '${spec}' is not a node of aifn-compute; compute imports only aifn-compute/<family>/<module>`,
      )
      continue
    }
    if (t === 'root') {
      errors.push(`${where}: compute imports the package root; import the defining node`)
      continue
    }
    if (t === 'alias') {
      aliasImports++
      if (!at.legacy) warnings.push(`${where}: imports the alias '${spec}' (TODO(tree 1d))`)
      continue
    }
    if (at.legacy || !own) continue
    const excused = transitional.some((x) => where.startsWith(`packages/compute/src/${x.file}`) && x.imports === spec)
    const tn = t.node
    const sameFamily = familyOf(tn.path) === familyOf(own.path)
    const isGroupIndex = tn.kind === 'family'
    if (isGroupIndex) {
      // A family index: only its shared names, and only from outside the family.
      if (sameFamily) errors.push(`${where}: imports its own family's index '${spec}'; use the relative path`)
      else if (names) {
        const children = namesFromChildren(tn)
        for (const n of names)
          if (children.has(n))
            errors.push(`${where}: imports ${n} from the family index '${spec}'; use its module's path`)
      }
      if (!sameFamily && tn.tier >= own.tier && !excused)
        errors.push(`${where}: ${own.path} (tier ${own.tier}) imports ${spec} (tier ${tn.tier}); only lower tiers`)
      continue
    }
    if (!sameFamily) {
      if (tn.tier >= own.tier && !excused)
        errors.push(
          `${where}: ${own.path} (family tier ${own.tier}) imports ${spec} (tier ${tn.tier}); only lower tiers`,
        )
      continue
    }
    if (tn.path === own.path) {
      errors.push(`${where}: imports its own module through '${spec}'; use './…'`)
      continue
    }
    if (at.shared) {
      errors.push(`${where}: a shared file imports the child '${spec}' (a parent never imports a child)`)
      continue
    }
    if (tn.local >= own.local && !excused)
      errors.push(
        `${where}: ${own.path} (local tier ${own.local}) imports ${spec} (local tier ${tn.local}); only lower`,
      )
  }
}

function checkApp(file: string) {
  const at = place(file)!
  const area = at.dir.split('/')[0]
  const allowed = reachable.get(area) ?? new Set()
  for (const { spec, where } of imports(file)) {
    if (spec.startsWith('.')) {
      checkRelative(file, at, spec, where)
      continue
    }
    const t = target(spec)
    if (t === 'alias') {
      aliasImports++
      if (!at.legacy) warnings.push(`${where}: imports the alias '${spec}' (TODO(tree 1d))`)
      continue
    }
    if (t === null || t === 'root') {
      errors.push(`${where}: '${spec}' is not a node of aifn-compute or aifn-methods; applications import only those`)
      continue
    }
    if (t.pkg === 'compute') continue
    const ta = t.node.path.split('/')[0]
    if (ta === area) {
      if (at.legacy) continue
      errors.push(`${where}: imports '${spec}' of its own area; siblings share only through the parent's shared files`)
    } else if (!allowed.has(ta))
      errors.push(`${where}: ${area} imports ${spec}, which is not below it in the area DAG (dependsOn)`)
  }
}

for (const file of files(computeSrc)) checkCore(file)
for (const file of files(appsSrc)) checkApp(file)
// Compute tests: never an application.
for (const file of files(path.join(computeDir, 'test')))
  for (const { spec, where } of imports(file))
    if (spec.startsWith('aifn-methods'))
      errors.push(`${where}: a compute test imports the application '${spec}'; test that combination in applications`)

// Tests and benchmarks (both packages) import the packages by their public paths, not by a relative path into `src`.
// The exceptions test helpers that are deliberately not exported.
const privateTestImports: Record<string, string> = {
  'packages/compute/test/foundation/trace/protocol.test.ts': 'the runners’ shared protocol helpers are internal',
  'packages/methods/test/data/real/hyphenation.test.ts': 'checks the vendored word list against its source',
  'packages/methods/test/data/real/ecg.test.ts': 'checks the vendored ECG samples against what scripts/ecg.py wrote',
}
for (const dir of [path.join(computeDir, 'test'), path.join(computeDir, 'bench'), path.join(appsDir, 'test')]) {
  if (!fs.existsSync(dir)) continue
  for (const file of files(dir)) {
    const rel = path.relative(root, file)
    for (const { spec, where } of imports(file)) {
      if (!spec.startsWith('.')) continue
      const to = path.resolve(path.dirname(file), spec)
      const intoSrc = [computeSrc, appsSrc].some((s) => to === s || to.startsWith(s + path.sep))
      if (intoSrc && !(rel in privateTestImports))
        errors.push(`${where}: '${spec}' reaches into src by a relative path; import the public 'aifn-compute/…' path`)
    }
  }
}

// Tests mirror the tree (module-tree §5.5): a test file sits in `test/<node path>/`. Only checks of the whole package
// sit at `test/` itself (listed here), beside shared helpers such as `fixtures.ts`. Fixtures sit in
// `test/fixtures/<node path>.json`, their generators in `test/fixtures/gen/<node path>.py`.
const packageTests: Record<'compute' | 'apps', string[]> = {
  compute: ['primitives.test.ts', 'root.test.ts'],
  apps: ['names.test.ts', 'functions.test.ts'],
}
for (const [pkg, dir] of [
  ['compute', computeDir],
  ['apps', appsDir],
] as const) {
  const testDir = path.join(dir, 'test')
  for (const file of files(testDir)) {
    const rel = path.relative(testDir, path.dirname(file)).split(path.sep).join('/')
    const where = path.relative(root, file)
    if (rel === 'fixtures' || rel.startsWith('fixtures/')) {
      if (/\.test\.tsx?$/.test(file)) errors.push(`${where}: a test file in fixtures/; tests sit in test/<node path>/`)
      continue
    }
    if (!rel) {
      if (/\.test\.tsx?$/.test(file) && !packageTests[pkg].includes(path.basename(file)))
        errors.push(`${where}: a test at the package's test root; tests sit in test/<node path>/`)
    } else if (!nodes.has(`${pkg}:${rel}`))
      errors.push(`${where}: test folder '${rel}' is not a node of modules.json; tests mirror the tree`)
  }
  for (const gen of files(path.join(testDir, 'fixtures', 'gen'), /\.py$/)) {
    const rel = path
      .relative(path.join(testDir, 'fixtures', 'gen'), gen)
      .replace(/\.py$/, '')
      .split(path.sep)
      .join('/')
    if (!rel.split('/').some((p) => p.startsWith('_')) && !nodes.has(`${pkg}:${rel}`))
      errors.push(`${path.relative(root, gen)}: fixture generator '${rel}' is not a node of modules.json`)
  }
}

// ── Generated tables ─────────────────────────────────────────────────────────────────────────────────────────────────

/** A Markdown table with padded columns, as Prettier writes it. */
function table(head: string[], rows: string[][]): string {
  const widths = head.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i].length)))
  const line = (cells: string[]) => `| ${cells.map((c, i) => c.padEnd(widths[i])).join(' | ')} |`
  return [line(head), line(widths.map((w) => '-'.repeat(w))), ...rows.map(line)].join('\n')
}

const localTiers = (f: Family) => {
  if (!f.modules.length) return '(one module)'
  const by = new Map<number, string[]>()
  for (const m of f.modules) by.set(m.tier, [...(by.get(m.tier) ?? []), m.status === 'gap' ? `${m.module}*` : m.module])
  return [...by]
    .sort((a, b) => a[0] - b[0])
    .map(([, ms]) => ms.join(', '))
    .join(' · ')
}
const tierTable = table(
  ['Tier', 'Family', 'Modules (local tiers, low to high; * gap)', 'Shared'],
  spec.compute.families.map((f) => [String(f.tier), f.family, localTiers(f), f.shared.join(', ')]),
)
const appTree = (list: AppNode[]): string =>
  list
    .map((c) =>
      'module' in c
        ? `${c.module}${c.status === 'gap' ? '*' : ''}`
        : `${c.group}/{${appTree(c.children)}}${c.shared.length ? ` [${c.shared.join(', ')}]` : ''}`,
    )
    .join(', ')
const areaTable = table(
  ['Area', 'Nodes (group/{children} [shared]; * gap)', 'Depends on'],
  spec.applications.areas.map((a) => [
    a.area,
    `${appTree(a.children) || '(one module)'}${a.shared.length ? ` [${a.shared.join(', ')}]` : ''}`,
    a.dependsOn.length === areas.size - 1 ? 'every other area' : a.dependsOn.join(', ') || '',
  ]),
)

const note = '<!-- Generated from modules.json by `node scripts/layers.ts --write`; do not edit. -->'
function block(name: string, body: string): string {
  return `<!-- ${name}:start -->\n\n${note}\n\n${body}\n\n<!-- ${name}:end -->`
}

function sync(file: string, blocks: Record<string, string>) {
  const full = path.join(root, file)
  let text = fs.readFileSync(full, 'utf8')
  const before = text
  for (const [name, body] of Object.entries(blocks)) {
    const re = new RegExp(`<!-- ${name}:start -->[\\s\\S]*?<!-- ${name}:end -->`)
    if (!re.test(text)) {
      errors.push(`${file}: no <!-- ${name}:start/end --> markers`)
      continue
    }
    text = text.replace(re, block(name, body))
  }
  if (text === before) return
  if (write) {
    fs.writeFileSync(full, text)
    console.log(`aifn-layers: wrote ${file}`)
  } else errors.push(`${file}: tables are stale; run node scripts/layers.ts --write`)
}
sync('README.md', { 'aifn-layers': tierTable, 'aifn-areas': areaTable })

if (process.argv.includes('--verbose')) for (const w of warnings) console.warn(`! ${w}`)
if (errors.length) {
  for (const e of errors) console.error(`✗ ${e}`)
  console.error(`aifn-layers: ${errors.length} error${errors.length === 1 ? '' : 's'}`)
  process.exit(1)
}
const count = (k: Node['kind']) => [...nodes.values()].filter((n) => n.kind === k && !n.gap).length
console.log(
  `aifn-layers: ${spec.compute.families.length} families and ${count('module')} modules follow the tree` +
    (aliases.size ? `; ${aliases.size} aliases, ${legacyDirs.size} legacy folders, ${aliasImports} alias imports` : ''),
)

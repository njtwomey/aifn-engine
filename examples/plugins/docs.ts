/**
 * The documentation of `aifn-compute` and `aifn-methods`, read from their source and served to the app as two virtual
 * modules, so the pages under `/compute` and `/methods` are generated and cannot drift from the code.
 *
 * - `virtual:aifn-docs/tree`: every node of both packages (a directory with an `index.ts`), nested as on disk, with
 *   its one-sentence summary and, for a module, its source files and the names each declares. The sidebar and the
 *   overview pages import it eagerly.
 * - `virtual:aifn-docs/content`: per node, the doc comment that opens its `index.ts`, its `@example` blocks, and its
 *   public exports (name, kind, signature, doc comment, examples), found by following the index's relative
 *   re-exports to the declarations. Large; pages import it lazily.
 *
 * A runnable `@example` is a title on the tag's line and code beneath it, in a module's opening comment or in an
 * export's doc comment: the app shows it as an editable cell that runs, and `make examples-check` runs them all. An
 * `@example` without a title is an illustrative fragment (it may name things it does not define) and is shown as
 * plain code.
 */
import fs from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import type { Plugin } from 'vite'

export type DocExample = { title: string; code: string }
export type DocExport = {
  name: string
  kind: 'function' | 'const' | 'class' | 'type'
  signature: string
  doc: string
  examples: DocExample[]
  /** The declaring file, relative to the repository, and its 1-based line. */
  file: string
  line: number
}
export type DocContent = {
  doc: string
  examples: DocExample[]
  exports: DocExport[]
  /** The comment that opens each source file (by file name), where it has one. */
  fileDocs: Record<string, string>
}
export type DocPackage = 'compute' | 'methods'
/** A source file of a module: the public exports it declares, values before types. */
export type DocFile = { name: string; values: string[]; types: string[]; examples: number }
export type DocTreeNode = {
  pkg: DocPackage
  /** The path within the package, e.g. `numerics/linalg`. */
  path: string
  name: string
  summary: string
  /** How many runnable (titled) examples the node holds (its own and its exports'). */
  examples: number
  children: DocTreeNode[]
  /** A leaf module's source files that declare public exports, by name. */
  files: DocFile[]
}
export type DocTree = Record<DocPackage, DocTreeNode[]>

const repo = path.resolve(import.meta.dirname, '..', '..')
const srcOf = (pkg: DocPackage) => path.join(repo, 'packages', pkg, 'src')

// ── Doc comments ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** The text of a `/** … *\/` comment without its frame. */
function unframe(comment: string): string {
  return comment
    .replace(/^\/\*\*/, '')
    .replace(/\*\/$/, '')
    .split('\n')
    .map((line) => line.replace(/^\s*\* ?/, ''))
    .join('\n')
    .trim()
}

/** A doc comment split into its prose and its `@example` blocks (other tags stay in the prose). */
function splitDoc(text: string): { doc: string; examples: DocExample[] } {
  const [doc, ...blocks] = text.split(/^@example[ \t]*/m)
  const examples = blocks.map((block) => {
    const [first, ...rest] = block.split('\n')
    // `@example code…` on one line has no title; otherwise the tag's line is the title.
    const titled = rest.some((l) => l.trim() !== '')
    return titled
      ? { title: first.trim(), code: rest.join('\n').trim() }
      : { title: '', code: [first, ...rest].join('\n').trim() }
  })
  return { doc: doc.trim(), examples: examples.filter((e) => e.code !== '') }
}

/** The first sentence of a doc comment, without the leading "`package/path`:" that module comments open with. */
function summaryOf(doc: string): string {
  const para = doc.split(/\n\s*\n/)[0].replace(/\s+/g, ' ')
  const body = para.replace(/^`[^`]+`:\s*/, '')
  const end = body.search(/[.:](\s|$)/)
  const sentence = end < 0 ? body : body.slice(0, end)
  return sentence.charAt(0).toUpperCase() + sentence.slice(1)
}

// ── Declarations ─────────────────────────────────────────────────────────────────────────────────────────────────────

const parsed = new Map<string, ts.SourceFile>()
function parse(file: string): ts.SourceFile {
  let sf = parsed.get(file)
  if (!sf) {
    sf = ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.ES2023, true)
    parsed.set(file, sf)
  }
  return sf
}

function docOf(sf: ts.SourceFile, node: ts.Node): string {
  const ranges = ts.getLeadingCommentRanges(sf.text, node.getFullStart()) ?? []
  const last = ranges.filter((r) => sf.text.startsWith('/**', r.pos)).at(-1)
  return last ? unframe(sf.text.slice(last.pos, last.end)) : ''
}

const isExported = (node: ts.Node) =>
  ts.canHaveModifiers(node) && (ts.getModifiers(node) ?? []).some((m) => m.kind === ts.SyntaxKind.ExportKeyword)

const squeeze = (s: string) => s.replace(/^export\s+/, '').trim()

/** A class as its head and the signatures of its constructor and public members, without bodies. */
function classSignature(sf: ts.SourceFile, node: ts.ClassDeclaration): string {
  const head = sf.text.slice(node.getStart(), node.members.pos).replace(/\s*\{\s*$/, '')
  const hidden = (m: ts.ClassElement) =>
    ts.canHaveModifiers(m) &&
    (ts.getModifiers(m) ?? []).some(
      (x) => x.kind === ts.SyntaxKind.PrivateKeyword || x.kind === ts.SyntaxKind.ProtectedKeyword,
    )
  const members = node.members
    .filter((m) => !hidden(m) && !(m.name && ts.isPrivateIdentifier(m.name)))
    .map((m) => {
      const body = (m as { body?: ts.Node }).body
      const text = sf.text.slice(m.getStart(), body ? body.getStart() : m.getEnd())
      return `  ${text.trim().replace(/\s*=\s*[^=]*$/, (init) => (body ? init : ''))}`
    })
  return members.length ? `${head} {\n${members.join('\n')}\n}` : head
}

/** The comment that opens a file, when it is the file's own (followed by a blank line or the imports). */
function fileDoc(file: string): string {
  const m = /^\s*(\/\*\*[\s\S]*?\*\/)(\n\s*\n|\nimport\b)/.exec(parse(file).text)
  return m ? splitDoc(unframe(m[1])).doc : ''
}

/** The exported declarations of a file, by name (the first overload of a function stands for it). */
function declarationsOf(file: string): Map<string, DocExport> {
  const sf = parse(file)
  const out = new Map<string, DocExport>()
  const rel = path.relative(repo, file)
  const add = (name: string, kind: DocExport['kind'], node: ts.Node, signature: string) => {
    if (out.has(name)) return
    const { doc, examples } = splitDoc(docOf(sf, node))
    const line = sf.getLineAndCharacterOfPosition(node.getStart()).line + 1
    out.set(name, { name, kind, signature: squeeze(signature), doc, examples, file: rel, line })
  }
  for (const node of sf.statements) {
    if (!isExported(node)) continue
    if (ts.isFunctionDeclaration(node) && node.name) {
      const end = node.body ? node.body.getStart() : node.getEnd()
      add(node.name.text, 'function', node, sf.text.slice(node.getStart(), end))
    } else if (ts.isVariableStatement(node)) {
      for (const d of node.declarationList.declarations) {
        if (!ts.isIdentifier(d.name)) continue
        const init = d.initializer
        const fn = init && (ts.isArrowFunction(init) || ts.isFunctionExpression(init)) ? init : null
        const signature = fn
          ? `const ${d.name.text} = ${sf.text.slice(fn.getStart(), fn.body.getStart()).replace(/\s*=>\s*$/, '')}`
          : `const ${d.name.text}${d.type ? `: ${d.type.getText()}` : ''}`
        add(d.name.text, fn ? 'function' : 'const', node, signature)
      }
    } else if (ts.isClassDeclaration(node) && node.name) {
      add(node.name.text, 'class', node, classSignature(sf, node))
    } else if (ts.isTypeAliasDeclaration(node) || ts.isInterfaceDeclaration(node)) {
      add(node.name.text, 'type', node, node.getText())
    }
  }
  return out
}

/** A relative specifier as a file of the same module, or null when it names a child module or nothing. */
function resolve(from: string, spec: string): string | null {
  const base = path.resolve(path.dirname(from), spec)
  const file = `${base}.ts`
  return fs.existsSync(file) ? file : null
}

/** The public exports of a file: its own declarations and those its relative re-exports lead to. */
function exportsOf(file: string, seen = new Set<string>()): DocExport[] {
  if (seen.has(file)) return []
  seen.add(file)
  const sf = parse(file)
  const out: DocExport[] = [...declarationsOf(file).values()]
  for (const node of sf.statements) {
    if (!ts.isExportDeclaration(node) || !node.moduleSpecifier || !ts.isStringLiteral(node.moduleSpecifier)) continue
    const spec = node.moduleSpecifier.text
    if (!spec.startsWith('.')) continue
    const target = resolve(file, spec)
    if (!target) continue
    const all = exportsOf(target, seen)
    if (!node.exportClause) out.push(...all)
    else if (ts.isNamedExports(node.exportClause)) {
      const byName = new Map(all.map((e) => [e.name, e]))
      for (const el of node.exportClause.elements) {
        const found = byName.get((el.propertyName ?? el.name).text)
        if (found) out.push({ ...found, name: el.name.text })
      }
    }
  }
  const unique = new Map<string, DocExport>()
  for (const e of out) if (!unique.has(e.name)) unique.set(e.name, e)
  return [...unique.values()]
}

// ── The tree ─────────────────────────────────────────────────────────────────────────────────────────────────────────

type Built = { tree: DocTree; content: Record<string, DocContent> }

/** The order of each package's top-level nodes, from the module tree; nodes it does not name sort after, by name. */
function topOrder(): Record<DocPackage, string[]> {
  const spec = JSON.parse(fs.readFileSync(path.join(repo, 'modules.json'), 'utf8')) as {
    compute: { families: { family: string }[] }
    applications: { areas: { area: string }[] }
  }
  return { compute: spec.compute.families.map((f) => f.family), methods: spec.applications.areas.map((a) => a.area) }
}

function build(): Built {
  parsed.clear()
  const content: Record<string, DocContent> = {}
  const walk = (pkg: DocPackage, dir: string, rel: string): DocTreeNode[] =>
    fs
      .readdirSync(dir, { withFileTypes: true })
      .filter((d) => d.isDirectory() && !d.name.startsWith('_') && fs.existsSync(path.join(dir, d.name, 'index.ts')))
      .map((d) => {
        const index = path.join(dir, d.name, 'index.ts')
        const at = rel ? `${rel}/${d.name}` : d.name
        const sf = parse(index)
        const header = /^\s*\/\*\*[\s\S]*?\*\//.exec(sf.text)?.[0] ?? ''
        const { doc, examples } = splitDoc(unframe(header.trim()))
        const children = walk(pkg, path.join(dir, d.name), at)
        // A parent's page lists its children; only a leaf module lists exports.
        const exports = children.length ? [] : exportsOf(index).sort((a, b) => a.name.localeCompare(b.name))
        const fileDocs: Record<string, string> = {}
        for (const e of exports) {
          const name = path.basename(e.file, '.ts')
          if (name !== 'index' && !(name in fileDocs)) fileDocs[name] = fileDoc(path.join(repo, e.file))
        }
        content[`${pkg}/${at}`] = { doc, examples, exports, fileDocs }
        const runnable = (list: DocExample[]) => list.filter((e) => e.title !== '').length
        const count = runnable(examples) + exports.reduce((n, e) => n + runnable(e.examples), 0)
        const byFile = new Map<string, DocFile>()
        for (const e of exports) {
          const name = path.basename(e.file, '.ts')
          const file = byFile.get(name) ?? { name, values: [], types: [], examples: 0 }
          ;(e.kind === 'type' ? file.types : file.values).push(e.name)
          file.examples += runnable(e.examples)
          byFile.set(name, file)
        }
        const files = [...byFile.values()].sort((a, b) => a.name.localeCompare(b.name))
        return { pkg, path: at, name: d.name, summary: summaryOf(doc), examples: count, children, files }
      })
      .sort((a, b) => a.name.localeCompare(b.name))
  const order = topOrder()
  const top = (pkg: DocPackage) => {
    const rank = (n: DocTreeNode) => (order[pkg].indexOf(n.name) + 1 || Infinity) - 1
    return walk(pkg, srcOf(pkg), '').sort((a, b) => rank(a) - rank(b))
  }
  return { tree: { compute: top('compute'), methods: top('methods') }, content }
}

const TREE = 'virtual:aifn-docs/tree'
const CONTENT = 'virtual:aifn-docs/content'

/** Serves the two virtual modules; a source edit rebuilds them (the app then reloads, as for any package edit). */
export function aifnDocs(): Plugin {
  let built: Built | null = null
  const get = () => (built ??= build())
  return {
    name: 'aifn-docs',
    resolveId: (id) => (id === TREE || id === CONTENT ? `\0${id}` : null),
    load(id) {
      if (id === `\0${TREE}`) return `export default ${JSON.stringify(get().tree)}`
      if (id === `\0${CONTENT}`) return `export default ${JSON.stringify(get().content)}`
      return null
    },
    handleHotUpdate({ file, server }) {
      if (!file.startsWith(path.join(repo, 'packages')) || !file.endsWith('.ts')) return
      built = null
      for (const id of [TREE, CONTENT]) {
        const mod = server.moduleGraph.getModuleById(`\0${id}`)
        if (mod) server.moduleGraph.invalidateModule(mod)
      }
    },
  }
}

/** The generated documentation, for scripts (`examples/check.ts`). */
export const buildDocs = build

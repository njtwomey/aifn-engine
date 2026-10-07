/**
 * The documentation of `aifn-compute` and `aifn-methods`, read from their source and served to the app as two virtual
 * modules, so the pages under `/compute` and `/methods` are generated and cannot drift from the code.
 *
 * - `virtual:aifn-docs/tree`: every node of both packages (a directory with an `index.ts`), nested as on disk, with
 *   its one-sentence summary and, for a module, its source files and the names each declares. The sidebar and the
 *   overview pages import it eagerly.
 * - `virtual:aifn-docs/content`: per node, the doc comment that opens its `index.ts`, its `@example` blocks, and its
 *   source files' declarations (name, kind, signature, doc comment, examples): every function, class and type, each
 *   marked public (the index's relative re-exports lead to it), internal (exported by its file only) or local.
 *   Large; pages import it lazily.
 *
 * `@param name text` and `@returns text` describe a function's parameters and result; the page sets them as a table
 * beside the types the signature gives.
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
/** A parameter of a function: from its signature, with the description its `@param` tag gives. */
export type DocParam = { name: string; type: string; optional: boolean; default: string; doc: string }
export type DocExport = {
  name: string
  kind: 'function' | 'const' | 'class' | 'type'
  signature: string
  doc: string
  /** A function's parameters in order; an options object written as a pattern lists each field as `options.<field>`. */
  params: DocParam[]
  /** What the `@returns` tag says. */
  returns: string
  examples: DocExample[]
  /** The declaring file, relative to the repository, and its 1-based line. */
  file: string
  line: number
  /**
   * `public`: exported by the module's index, so importable. `internal`: exported by its file for the module's other
   * files only. `local`: not exported.
   */
  visibility: 'public' | 'internal' | 'local'
}
export type DocContent = {
  doc: string
  examples: DocExample[]
  exports: DocExport[]
  /** The comment that opens each source file (by file name), where it has one. */
  fileDocs: Record<string, string>
}
export type DocPackage = 'compute' | 'methods'
/**
 * A source file of a module, by the names it declares: `key` are the public functions, classes and constants (the ones
 * a reader calls), `supporting` the internal and local ones behind them, `types` every type.
 */
export type DocFile = { name: string; key: string[]; supporting: string[]; types: string[]; examples: number }
export type DocTreeNode = {
  pkg: DocPackage
  /** The path within the package, e.g. `numerics/linalg`. */
  path: string
  name: string
  summary: string
  /** How many runnable (titled) examples the node holds (its own and its exports'). */
  examples: number
  children: DocTreeNode[]
  /** A leaf module's source files, by name. */
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

type Tags = { doc: string; params: Map<string, string>; returns: string; examples: DocExample[] }

/** A doc comment split into its prose, its `@param` and `@returns` descriptions and its `@example` blocks. */
function splitDoc(text: string): Tags {
  const [head, ...blocks] = text.split(/^@example[ \t]*/m)
  // Before the examples: the prose, then `@param name text` and `@returns text`, each running to the next tag.
  const [doc, ...tags] = head.split(/^(?=@(?:param|returns?)\b)/m)
  const params = new Map<string, string>()
  let returns = ''
  for (const tag of tags) {
    const flat = tag.replace(/\s+/g, ' ').trim()
    const param = /^@param (?:\{[^}]*\} )?([\w.$]+) ?-? ?(.*)$/.exec(flat)
    if (param) params.set(param[1], param[2])
    else returns = flat.replace(/^@returns? ?/, '')
  }
  // The code of an example may be fenced (``` or ```js, ```ts, …) and may open with its imports: the fence lines and
  // the import lines are dropped (the names are in scope when it runs, and the page shows the imports it works out).
  const unfenced = (code: string) =>
    code
      .split('\n')
      .filter((line) => !/^\s*```[\w-]*\s*$/.test(line) && !/^\s*import\s.*\sfrom\s+['"][^'"]+['"];?\s*$/.test(line))
      .join('\n')
      .trim()
  const examples = blocks.map((block) => {
    const [first, ...rest] = block.split('\n')
    // `@example code…` on one line has no title; otherwise the tag's line is the title.
    const titled = rest.some((l) => l.trim() !== '')
    return titled
      ? { title: first.trim(), code: unfenced(rest.join('\n')) }
      : { title: '', code: unfenced([first, ...rest].join('\n')) }
  })
  return { doc: doc.trim(), params, returns, examples: examples.filter((e) => e.code !== '') }
}

/** The first sentence of a doc comment, without the leading "`package/path`:" that module comments open with. */
export function summaryOf(doc: string): string {
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

const oneLine = (s: string) => s.replace(/\s+/g, ' ').trim()

/**
 * A function's parameters with their descriptions. A parameter written as an object pattern (`{ n = 20 }: {…} = {}`)
 * is `options`, followed by one row per field (`options.n`), described by `@param options.n`, else by the comment on
 * that field of its type.
 */
function paramsOf(sf: ts.SourceFile, fn: ts.SignatureDeclaration, docs: Map<string, string>): DocParam[] {
  const out: DocParam[] = []
  for (const p of fn.parameters) {
    const type = p.type ? oneLine(p.type.getText()) : ''
    const init = p.initializer ? oneLine(p.initializer.getText()) : ''
    const optional = !!p.questionToken || !!p.initializer
    if (ts.isIdentifier(p.name)) {
      out.push({ name: p.name.text, type, optional, default: init, doc: docs.get(p.name.text) ?? '' })
    } else if (ts.isObjectBindingPattern(p.name)) {
      out.push({ name: 'options', type, optional, default: init, doc: docs.get('options') ?? '' })
      const members = p.type && ts.isTypeLiteralNode(p.type) ? p.type.members : undefined
      for (const el of p.name.elements) {
        if (!ts.isIdentifier(el.name)) continue
        const field = (el.propertyName ?? el.name).getText()
        const member = members?.find((m) => m.name?.getText() === field)
        const memberType = member && ts.isPropertySignature(member) && member.type ? oneLine(member.type.getText()) : ''
        out.push({
          name: `options.${field}`,
          type: memberType,
          optional: true,
          default: el.initializer ? oneLine(el.initializer.getText()) : '',
          doc: docs.get(`options.${field}`) ?? (member ? oneLine(docOf(sf, member)) : ''),
        })
      }
    } else {
      // An array pattern (`[[a, b], [c, d]]: Mat2`): named by its `@param`, in order, else by the pattern.
      const named = [...docs.keys()].filter(
        (k) => !k.includes('.') && !fn.parameters.some((q) => q.name.getText() === k),
      )
      const taken = out.filter((o) => named.includes(o.name)).length
      const name = named[taken] ?? oneLine(p.name.getText())
      out.push({ name, type, optional, default: init, doc: docs.get(name) ?? '' })
    }
  }
  return out
}

/**
 * The top-level declarations of a file, by name (the first overload of a function stands for it): every function,
 * class and type, and the constants that are exported or made by a call (a primitive from `defineOp`, a registry).
 */
function declarationsOf(file: string): Map<string, DocExport & { exported: boolean }> {
  const sf = parse(file)
  const out = new Map<string, DocExport & { exported: boolean }>()
  const rel = path.relative(repo, file)
  let exported = false
  const add = (
    name: string,
    kind: DocExport['kind'],
    node: ts.Node,
    signature: string,
    fn?: ts.SignatureDeclaration,
  ) => {
    if (out.has(name)) return
    const tags = splitDoc(docOf(sf, node))
    const line = sf.getLineAndCharacterOfPosition(node.getStart()).line + 1
    out.set(name, {
      name,
      kind,
      signature: squeeze(signature),
      doc: tags.doc,
      params: fn ? paramsOf(sf, fn, tags.params) : [],
      returns: tags.returns,
      examples: tags.examples,
      file: rel,
      line,
      visibility: exported ? 'internal' : 'local',
      exported,
    })
  }
  for (const node of sf.statements) {
    exported = isExported(node)
    if (ts.isFunctionDeclaration(node) && node.name) {
      const end = node.body ? node.body.getStart() : node.getEnd()
      add(node.name.text, 'function', node, sf.text.slice(node.getStart(), end), node)
    } else if (ts.isVariableStatement(node)) {
      for (const d of node.declarationList.declarations) {
        if (!ts.isIdentifier(d.name)) continue
        const init = d.initializer
        const fn = init && (ts.isArrowFunction(init) || ts.isFunctionExpression(init)) ? init : null
        if (!exported && !fn && !(init && ts.isCallExpression(init))) continue
        const signature = fn
          ? `const ${d.name.text} = ${sf.text.slice(fn.getStart(), fn.body.getStart()).replace(/\s*=>\s*$/, '')}`
          : `const ${d.name.text}${d.type ? `: ${d.type.getText()}` : ''}`
        add(d.name.text, fn ? 'function' : 'const', node, signature, fn ?? undefined)
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
  const out: DocExport[] = [...declarationsOf(file).values()].filter((d) => d.exported)
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
        // A parent's page lists its children; a leaf module lists its source files and everything they declare.
        const exports: DocExport[] = []
        const fileDocs: Record<string, string> = {}
        const files: DocFile[] = []
        const runnable = (list: DocExample[]) => list.filter((e) => e.title !== '').length
        if (!children.length) {
          const isPublic = new Set(exportsOf(index).map((e) => `${e.file}:${e.line}`))
          const moduleDir = path.join(dir, d.name)
          const sources = fs
            .readdirSync(moduleDir)
            .filter((f) => f.endsWith('.ts') && !f.endsWith('.test.ts') && !f.endsWith('.d.ts'))
            .sort()
          for (const source of sources) {
            const name = path.basename(source, '.ts')
            const declared = [...declarationsOf(path.join(moduleDir, source)).values()]
              .map((e): DocExport => (isPublic.has(`${e.file}:${e.line}`) ? { ...e, visibility: 'public' } : e))
              .sort((x, y) => x.line - y.line)
            if (!declared.length) continue
            exports.push(...declared)
            if (name !== 'index') fileDocs[name] = fileDoc(path.join(moduleDir, source))
            const values = declared.filter((e) => e.kind !== 'type')
            files.push({
              name,
              key: values.filter((e) => e.visibility === 'public').map((e) => e.name),
              supporting: values.filter((e) => e.visibility !== 'public').map((e) => e.name),
              types: declared.filter((e) => e.kind === 'type').map((e) => e.name),
              examples: declared.reduce((n, e) => n + runnable(e.examples), 0),
            })
          }
        }
        content[`${pkg}/${at}`] = { doc, examples, exports, fileDocs }
        const count = runnable(examples) + exports.reduce((n, e) => n + runnable(e.examples), 0)
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

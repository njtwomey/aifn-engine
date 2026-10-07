/**
 * `make map`: the packages as a tree, to see where something lives (or would live) before reading code. Every module is
 * listed with the first sentence of its `index.ts` comment, and under it each source file with the first sentence of
 * its opening comment and the names its module exports from it.
 *
 * It is read from the same source as the site's `/compute` and `/methods` pages (`examples/plugins/docs.ts`), so it
 * cannot drift from the code, and a summary is only as good as the docstring it comes from ("(no summary)" marks a gap).
 *
 * ```bash
 * node scripts/map.ts                          # both packages, modules and files
 * node scripts/map.ts compute/optim methods    # only the nodes under these paths
 * node scripts/map.ts --modules                # modules only, no files
 * ```
 */
import { buildDocs, summaryOf, type DocTreeNode } from '../examples/plugins/docs.ts'

const args = process.argv.slice(2)
const modulesOnly = args.includes('--modules')
const prefixes = args.filter((a) => !a.startsWith('--')).map((a) => a.replace(/\/+$/, ''))

const { tree, content } = buildDocs()

const id = (n: DocTreeNode) => `${n.pkg}/${n.path}`
const wanted = (n: DocTreeNode) =>
  !prefixes.length || prefixes.some((p) => id(n) === p || id(n).startsWith(`${p}/`) || p.startsWith(`${id(n)}/`))
const summary = (doc: string) => (doc ? summaryOf(doc) : '(no summary)')

const lines: string[] = []
const visit = (n: DocTreeNode, depth: number) => {
  if (!wanted(n)) return
  const pad = '  '.repeat(depth)
  lines.push(`${pad}${depth ? n.name : id(n)}/ — ${n.summary || '(no summary)'}`)
  for (const c of n.children) visit(c, depth + 1)
  if (modulesOnly) return
  const docs = content[id(n)]?.fileDocs ?? {}
  for (const f of n.files) {
    if (f.name === 'index') continue
    const exported = f.key.length ? ` · ${f.key.join(', ')}` : ''
    lines.push(`${pad}  ${f.name}.ts — ${summary(docs[f.name] ?? '')}${exported}`)
  }
}
for (const pkg of ['compute', 'methods'] as const) for (const n of tree[pkg]) visit(n, 0)
console.log(lines.join('\n'))

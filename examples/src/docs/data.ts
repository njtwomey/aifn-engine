/** The generated documentation (`plugins/docs.ts`): the tree, eagerly; each node's content, on first use. */
import { useEffect, useState } from 'react'
import tree from 'virtual:aifn-docs/tree'
import type { DocContent, DocFile, DocPackage, DocTreeNode } from '../../plugins/docs'

export type { DocContent, DocExample, DocExport, DocFile, DocPackage, DocTreeNode } from '../../plugins/docs'

export const TREE = tree

export const PACKAGES: Record<DocPackage, { name: string; title: string; blurb: string }> = {
  compute: {
    name: 'aifn-compute',
    title: 'Compute',
    blurb:
      'Tensor-native numerics with no dependencies: what each module holds, how it behaves and how to call it. Every page is generated from the source, and its examples run in the page.',
  },
  methods: {
    name: 'aifn-methods',
    title: 'Methods',
    blurb: 'Named models, datasets, environments and agents built on compute. Every page is generated from the source.',
  },
}

/** The URL path of a node: `<package>/<path>`. */
export const nodePath = (n: Pick<DocTreeNode, 'pkg' | 'path'>) => `${n.pkg}/${n.path}`

const index = new Map<string, DocTreeNode>()
const walk = (nodes: readonly DocTreeNode[]) => {
  for (const n of nodes) {
    index.set(nodePath(n), n)
    walk(n.children)
  }
}
walk(tree.compute)
walk(tree.methods)

/** The node at a URL path, if there is one. */
export const nodeAt = (path: string): DocTreeNode | undefined => index.get(path)

/** The source file of a module at a URL path (`<package>/<module path>/<file>`), if there is one. */
export function fileAt(path: string): { node: DocTreeNode; file: DocFile } | undefined {
  const cut = path.lastIndexOf('/')
  const node = index.get(path.slice(0, cut))
  const file = node?.files.find((f) => f.name === path.slice(cut + 1))
  return node && file ? { node, file } : undefined
}

/** The URL path of a module's source file. */
export const filePath = (n: Pick<DocTreeNode, 'pkg' | 'path'>, file: DocFile) => `${nodePath(n)}/${file.name}`

/** A node's ancestors, outermost first. */
export function ancestorsOf(n: DocTreeNode): DocTreeNode[] {
  const parts = n.path.split('/')
  return parts.slice(0, -1).map((_, i) => index.get(`${n.pkg}/${parts.slice(0, i + 1).join('/')}`)!)
}

let loading: Promise<Record<string, DocContent>> | null = null
let loaded: Record<string, DocContent> | null = null
export const loadContent = () => (loading ??= import('virtual:aifn-docs/content').then((m) => (loaded = m.default)))

export type TypeRef = { name: string; path: string; signature: string; summary: string }

let types: Map<string, { module: string; file: string; public: boolean; ref: TypeRef }[]> | null = null

/**
 * The type a name refers to from a source file (`file`, repository-relative) of a module (`<package>/<path>`): the
 * file's own, else the module's (a public one first), else the one public type of that name anywhere; null when there
 * is none or the name is ambiguous. Needs the content to have loaded.
 */
export function typeRef(name: string, module: string, file: string): TypeRef | null {
  if (!loaded) return null
  if (!types) {
    types = new Map()
    for (const [key, c] of Object.entries(loaded))
      for (const e of c.exports) {
        if (e.kind !== 'type' && e.kind !== 'class') continue
        const file = e.file.replace(/^.*\//, '').replace(/\.ts$/, '')
        const summary = e.doc.split(/\n\s*\n/)[0].replace(/\s+/g, ' ')
        const ref = { name: e.name, path: `${key}/${file}`, signature: e.signature, summary }
        const entry = { module: key, file: e.file, public: e.visibility === 'public', ref }
        types.set(e.name, [...(types.get(e.name) ?? []), entry])
      }
  }
  const all = types.get(name) ?? []
  const mine = all.filter((t) => t.module === module)
  const open = all.filter((t) => t.public)
  const found = mine.find((t) => t.file === file) ?? mine.find((t) => t.public) ?? mine[0]
  return found?.ref ?? (open.length === 1 ? open[0].ref : null)
}

/** A node's content, or null while the content chunk loads. */
export function useContent(path: string): DocContent | null {
  const [, setReady] = useState(loaded !== null)
  useEffect(() => {
    let live = true
    if (!loaded) void loadContent().then(() => live && setReady(true))
    return () => {
      live = false
    }
  }, [])
  return loaded?.[path] ?? null
}

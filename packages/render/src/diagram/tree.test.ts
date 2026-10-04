import { binaryTree, treeFromParents, type NestedBinaryTree, type Tree } from 'aifn-compute/graph'
import { describe, expect, it } from 'vitest'
import { treeLayout } from './tree'

/** A deterministic pseudo-random tree of n nodes (each node's parent is an earlier node). */
function randomTree(n: number, seed: number): Tree {
  let s = seed
  const next = () => (s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31
  return treeFromParents(Array.from({ length: n }, (_, i) => (i === 0 ? -1 : Math.floor(next() ** 2 * i))))
}

/** Checks the tidy-tree properties: levels, order, spacing and centred parents. */
function checkTidy(tree: Tree, gap: number) {
  const at = treeLayout(tree, { siblingGap: gap })
  const byLevel = new Map<number, number[]>()
  for (const n of tree.nodes) {
    const p = at[n.id]!
    byLevel.set(p.y, [...(byLevel.get(p.y) ?? []), p.x])
    if (n.parent !== null) expect(p.y).toBeGreaterThan(at[n.parent]!.y)
    if (n.children.length > 1 || (n.children.length === 1 && tree.arity !== 2)) {
      const xs = n.children.map((c) => at[c]!.x)
      expect(p.x).toBeCloseTo((xs[0] + xs[xs.length - 1]) / 2, 9)
    }
    // Siblings in order, at least a node width plus the gap apart.
    for (let i = 1; i < n.children.length; i++)
      expect(at[n.children[i]]!.x - at[n.children[i - 1]]!.x).toBeGreaterThanOrEqual(1 + gap - 1e-9)
  }
  // No two nodes of a level closer than a node width plus the gap.
  for (const xs of byLevel.values()) {
    xs.sort((a, b) => a - b)
    for (let i = 1; i < xs.length; i++) expect(xs[i] - xs[i - 1]).toBeGreaterThanOrEqual(1 + gap - 1e-9)
  }
  return at
}

describe('treeLayout', () => {
  it('is tidy on random trees', () => {
    for (let seed = 1; seed <= 20; seed++) checkTidy(randomTree(40, seed), 0.4)
  })

  it('keeps the side of a lone child in a binary tree', () => {
    const t = binaryTree({ label: 'r', left: { label: 'a', right: { label: 'b' } }, right: null })
    const at = checkTidy(t, 0.4)
    expect(at[1]!.x).toBeLessThan(at[0]!.x)
    expect(at[2]!.x).toBeGreaterThan(at[1]!.x)
  })

  it('is compact on an unbalanced binary tree: a long left spine does not push the right subtree away', () => {
    // A left spine of depth 6 with a leaf hanging right of each spine node.
    let spine: NestedBinaryTree = { label: 'x' }
    for (let i = 0; i < 6; i++) spine = { label: `s${i}`, left: spine, right: { label: `l${i}` } }
    const t = binaryTree(spine)
    const at = checkTidy(t, 0.4)
    const xs = at.map((p) => p!.x)
    // Width grows with the depth, not with the number of leaves squared.
    expect(Math.max(...xs) - Math.min(...xs)).toBeLessThan(6 * 1.4 + 1e-9)
  })

  it('places nodes by height and orients to the right', () => {
    const t = treeFromParents([-1, 0, 0, 1, 1], {
      data: (i) => ({ height: [3, 1, 0, 0, 0][i] }),
    })
    const at = treeLayout(t, { heightAxis: 1, orientation: 'right' })
    expect(at.map((p) => p!.x)).toEqual([0, 2, 3, 3, 3])
    expect(at[0]!.y).toBeCloseTo((at[1]!.y + at[2]!.y) / 2, 9)
    const hidden = treeLayout(t, { collapsed: new Set([1]) })
    expect(hidden[3]).toBeNull()
  })

  it('puts dendrogram leaves of any depth side by side on one row', () => {
    const t = treeFromParents([-1, 0, 0, 1, 1, 4, 4], { data: (i) => ({ height: [3, 2, 0, 0, 1, 0, 0][i] }) })
    const at = treeLayout(t, { heightAxis: true, siblingGap: 0.5 })
    const leaves = [2, 3, 5, 6].map((v) => at[v]!)
    for (const p of leaves) expect(p.y).toBe(leaves[0].y)
    const xs = leaves.map((p) => p.x).sort((a, b) => a - b)
    for (let i = 1; i < xs.length; i++) expect(xs[i] - xs[i - 1]).toBeCloseTo(1.5, 9)
  })
})

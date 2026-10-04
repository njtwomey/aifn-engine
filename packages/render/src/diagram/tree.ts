/**
 * A tidy layout for rooted trees (`Tree` from `aifn-compute/graph`): Walker's algorithm in the linear-time form of Buchheim,
 * Jünger and Leipert (2002, "Improving Walker's algorithm to run in linear time", Graph Drawing, LNCS 2528), after
 * Reingold and Tilford (1981, "Tidier drawings of trees", IEEE Trans. Softw. Eng. 7(2)). Parents are centred over
 * their children, subtrees are pushed apart only as far as their contours require, and smaller subtrees between two
 * big ones are spaced evenly. Nodes may have different widths. In a binary tree (`arity: 2`) a lone child keeps its
 * side: it is laid out beside an invisible sibling, so it hangs to the left or right of its parent.
 */
import type { Tree } from 'aifn-compute/graph'
import type { Point } from './force'

/** Options of `treeLayout`. Distances are in grid units. */
export type TreeLayoutOptions = {
  /** `down`: the root at the top, levels in rows (default). `right`: the root at the left, levels in columns. */
  orientation?: 'down' | 'right'
  /** Gap between neighbouring nodes on one level, beyond their half-widths (default 0.4). */
  siblingGap?: number
  /** Gap between neighbouring subtrees that are not siblings (default: `siblingGap`). */
  subtreeGap?: number
  /** Distance between levels (default 1.4). */
  levelGap?: number
  /** A node's extent across the levels (its width when `down`, its height when `right`); default 1. */
  size?: (id: number) => number
  /**
   * Place nodes along the level axis by their numeric `height` (dendrogram style: leaves at 0 at the bottom, the root
   * highest at the top) instead of by depth. Leaves are then placed side by side in order and each parent midway
   * between its outer children (Walker's contours assume one row per depth). `true` scales heights so the tallest spans `levelGap` × the tree's depth;
   * a number is grid units per unit of height. Nodes without a height count as 0.
   */
  heightAxis?: boolean | number
  /** Nodes drawn as leaves: their descendants get no position (`null`). */
  collapsed?: ReadonlySet<number>
}

/**
 * Positions (grid units, root at x = 0 across) for every node of a tree, indexed by id; `null` for nodes hidden under
 * a collapsed node. Siblings keep their order, left to right (top to bottom when `right`).
 */
export function treeLayout(tree: Tree, options: TreeLayoutOptions = {}): (Point | null)[] {
  const {
    orientation = 'down',
    siblingGap = 0.4,
    levelGap = 1.4,
    size = () => 1,
    heightAxis = false,
    collapsed,
  } = options
  const subtreeGap = options.subtreeGap ?? siblingGap
  const n = tree.nodes.length

  // The layout tree: the visible nodes plus invisible siblings for lone children of binary nodes.
  const kids: number[][] = []
  const parent: number[] = []
  const width: number[] = []
  const real: number[] = [] // layout node → tree node, −1 for a placeholder
  const layoutOf = new Int32Array(n).fill(-1)
  const add = (id: number, p: number, w: number) => {
    const k = kids.length
    kids.push([])
    parent.push(p)
    width.push(w)
    real.push(id)
    if (id >= 0) layoutOf[id] = k
    if (p >= 0) kids[p].push(k)
    return k
  }
  const binary = tree.arity === 2
  // Depth-first so children are added in order.
  const visit = (id: number, p: number) => {
    const k = add(id, p, size(id))
    const node = tree.nodes[id]
    if (collapsed?.has(id)) return
    const cs = node.children
    if (binary && cs.length === 1) {
      const side = tree.nodes[cs[0]].slot ?? 0
      if (side === 1) add(-1, k, 0)
      visit(cs[0], k)
      if (side !== 1) add(-1, k, 0)
    } else for (const c of cs) visit(c, k)
  }
  visit(tree.root, -1)
  const m = kids.length

  // Buchheim, Jünger and Leipert's first walk, apportion and second walk, with a distance per pair of nodes.
  const prelim = new Float64Array(m)
  const mod = new Float64Array(m)
  const shift = new Float64Array(m)
  const change = new Float64Array(m)
  const thread = new Int32Array(m).fill(-1)
  const ancestor = Int32Array.from({ length: m }, (_, i) => i)
  const number = new Int32Array(m) // index among siblings
  for (const cs of kids) cs.forEach((c, i) => (number[c] = i))
  const leftSibling = (v: number) => (parent[v] >= 0 && number[v] > 0 ? kids[parent[v]][number[v] - 1] : -1)
  const leftmostSibling = (v: number) => (parent[v] >= 0 ? kids[parent[v]][0] : v)
  const nextLeft = (v: number) => (kids[v].length ? kids[v][0] : thread[v])
  const nextRight = (v: number) => (kids[v].length ? kids[v][kids[v].length - 1] : thread[v])
  const distance = (a: number, b: number) =>
    (width[a] + width[b]) / 2 + (parent[a] === parent[b] ? siblingGap : subtreeGap)

  const moveSubtree = (wm: number, wp: number, amount: number) => {
    const subtrees = number[wp] - number[wm]
    change[wp] -= amount / subtrees
    shift[wp] += amount
    change[wm] += amount / subtrees
    prelim[wp] += amount
    mod[wp] += amount
  }
  const executeShifts = (v: number) => {
    let s = 0
    let c = 0
    for (let i = kids[v].length - 1; i >= 0; i--) {
      const w = kids[v][i]
      prelim[w] += s
      mod[w] += s
      c += change[w]
      s += shift[w] + c
    }
  }
  const ancestorOf = (vim: number, v: number, fallback: number) =>
    parent[ancestor[vim]] === parent[v] ? ancestor[vim] : fallback
  const apportion = (v: number, defaultAncestor: number): number => {
    const w = leftSibling(v)
    if (w < 0) return defaultAncestor
    let vip = v
    let vop = v
    let vim = w
    let vom = leftmostSibling(vip)
    let sip = mod[vip]
    let sop = mod[vop]
    let sim = mod[vim]
    let som = mod[vom]
    while (nextRight(vim) >= 0 && nextLeft(vip) >= 0) {
      vim = nextRight(vim)
      vip = nextLeft(vip)
      vom = nextLeft(vom)
      vop = nextRight(vop)
      ancestor[vop] = v
      const amount = prelim[vim] + sim - (prelim[vip] + sip) + distance(vim, vip)
      if (amount > 0) {
        moveSubtree(ancestorOf(vim, v, defaultAncestor), v, amount)
        sip += amount
        sop += amount
      }
      sim += mod[vim]
      sip += mod[vip]
      som += mod[vom]
      sop += mod[vop]
    }
    if (nextRight(vim) >= 0 && nextRight(vop) < 0) {
      thread[vop] = nextRight(vim)
      mod[vop] += sim - sop
    }
    if (nextLeft(vip) >= 0 && nextLeft(vom) < 0) {
      thread[vom] = nextLeft(vip)
      mod[vom] += sip - som
      return v
    }
    return defaultAncestor
  }
  // The first walk, as in the paper: each child is walked, then apportioned against its left siblings, before the next
  // child is walked (a leaf is placed relative to its left sibling's shifted position).
  const firstWalk = (v: number) => {
    const w = leftSibling(v)
    if (kids[v].length === 0) {
      prelim[v] = w >= 0 ? prelim[w] + distance(w, v) : 0
      return
    }
    let defaultAncestor = kids[v][0]
    for (const c of kids[v]) {
      firstWalk(c)
      defaultAncestor = apportion(c, defaultAncestor)
    }
    executeShifts(v)
    const mid = (prelim[kids[v][0]] + prelim[kids[v][kids[v].length - 1]]) / 2
    if (w >= 0) {
      prelim[v] = prelim[w] + distance(w, v)
      mod[v] = prelim[v] - mid
    } else prelim[v] = mid
  }
  if (heightAxis === false) firstWalk(0)
  else {
    // By height, leaves of any depth can share a row, so contours per depth no longer separate them: place the leaves
    // side by side in order and each parent midway between its outer children, as dendrograms are drawn.
    let next = 0
    let prev = -1
    const place = (v: number) => {
      if (kids[v].length === 0) {
        prelim[v] = prev < 0 ? 0 : next + width[prev] / 2 + siblingGap + width[v] / 2
        next = prelim[v]
        prev = v
        return
      }
      kids[v].forEach(place)
      prelim[v] = (prelim[kids[v][0]] + prelim[kids[v][kids[v].length - 1]]) / 2
    }
    place(0)
    // Absolute positions: the second walk adds no modifiers.
    mod.fill(0)
  }
  const across = new Float64Array(m)
  const depth = new Int32Array(m)
  const pre: [number, number][] = [[0, 0]]
  while (pre.length) {
    const [v, sum] = pre.pop()!
    across[v] = prelim[v] + sum
    for (const c of kids[v]) {
      depth[c] = depth[v] + 1
      pre.push([c, sum + mod[v]])
    }
  }
  const rootAt = across[0]

  // Along the levels: depth, or the node's height.
  let along = (k: number) => depth[k] * levelGap
  if (heightAxis !== false) {
    const h = (k: number) => tree.nodes[real[k]].height ?? 0
    let maxH = 0
    let maxDepth = 0
    for (let k = 0; k < m; k++)
      if (real[k] >= 0) {
        maxH = Math.max(maxH, h(k))
        maxDepth = Math.max(maxDepth, depth[k])
      }
    const scale = typeof heightAxis === 'number' ? heightAxis : maxH > 0 ? (levelGap * Math.max(maxDepth, 1)) / maxH : 1
    along = (k) => (maxH - h(k)) * scale
  }
  const out: (Point | null)[] = new Array(n).fill(null)
  for (let k = 0; k < m; k++) {
    const id = real[k]
    if (id < 0) continue
    const a = across[k] - rootAt
    out[id] = orientation === 'down' ? { x: a, y: along(k) } : { x: along(k), y: a }
  }
  return out
}

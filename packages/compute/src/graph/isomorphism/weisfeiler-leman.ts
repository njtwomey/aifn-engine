/**
 * Colour refinement, the 1-dimensional Weisfeiler-Leman test, and the 2-dimensional folklore test, on one or several
 * graphs with one shared palette so that colours can be compared across graphs.
 *
 * Round 0 colours come from initial labels (any numbers, one per node) or are all equal. In round $t + 1$ every node's
 * signature is its round-$t$ colour followed by the sorted multiset of its neighbours' round-$t$ colours,
 *
 * $$c^{(t+1)}_v = \operatorname{hash}\big(c^{(t)}_v, \{\!\{ c^{(t)}_u : u \in \Ncal(v) \}\!\}\big),$$
 *
 * and the distinct signatures over all graphs are sorted and numbered $0, 1, 2, \dots$, so the numbering is canonical:
 * independent of the node order and of the order the graphs are given in (Shervashidze et al. 2011, "Weisfeiler-Lehman
 * graph kernels", JMLR 12, §2.1, number them by a hash table instead). The refinement stops at the first round whose
 * partition has no more classes than the round before (the stable colouring), or after a set number of rounds. Two
 * graphs are distinguished at the first round where their colour histograms differ (Weisfeiler and Leman 1968, "The
 * reduction of a graph to canonical form and the algebra which appears therein").
 *
 * The folklore test 2-FWL, as strong as the oblivious 3-WL (Cai, Fürer and Immerman 1992, "An optimal lower bound on
 * the number of variables for graph identification", Combinatorica 12(4), §5), colours ordered pairs $(u, v)$ from
 * their atomic type ($u = v$, adjacent or not, and the node labels) and refines the colour of $(u, v)$ by the
 * multiset over all $w$ of the pair of colours of $(u, w)$ and $(w, v)$. It costs $O(n^3)$ per round and graph.
 *
 * Results are plain data (typed arrays and nested arrays), so both run in a worker as they are.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Status } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { adjacency, type Graph } from '../graph'

/** Options of `weisfeilerLeman`, `weisfeilerLemanSteps` and `folkloreWeisfeilerLeman2`. */
export interface WeisfeilerLemanOptions {
  /** The largest number of refinement rounds (default: until stable, at most the largest node count). */
  rounds?: number
  /** Initial node labels, one array per graph with one number per node (default all equal). */
  initial?: readonly (readonly number[])[]
}

/** The refinement trace of `weisfeilerLeman`, round 0 (the initial colours) first. */
export interface WeisfeilerLemanTrace {
  /** `colours[t][g][v]`: the colour of node $v$ of graph $g$ after round $t$, numbered from 0 in the shared palette. */
  colours: Int32Array[][]
  /** `classes[t]`: the number of distinct colours after round $t$, over all graphs. */
  classes: number[]
  /** `histograms[t][g][c]`: how many nodes of graph $g$ have colour $c$ after round $t$ (length `classes[t]`). */
  histograms: number[][][]
  /** The first round whose partition equals the one before (the stable colouring); the last round run if none. */
  stable: number
  /**
   * For two or more graphs: the first round at which some pair of graphs has different histograms, or $-1$ when every
   * round ties ($-1$ for a single graph).
   */
  distinguishedAt: number
}

/**
 * Canonical numbering of string signatures: the distinct ones sorted, then numbered from 0.
 *
 * @param signatures One array of signatures per graph, one per node (or pair).
 * @returns The colours, laid out as the signatures, and the number of distinct ones.
 */
function numberSignatures(signatures: string[][]): { colours: Int32Array[]; classes: number } {
  const distinct = [...new Set(signatures.flat())].sort()
  const index = new Map(distinct.map((s, i) => [s, i]))
  return { colours: signatures.map((sig) => Int32Array.from(sig, (s) => index.get(s)!)), classes: distinct.length }
}

/**
 * A fixed-width key, so that sorting the strings sorts the numbers.
 *
 * @param c A colour.
 * @returns Its decimal digits, zero-padded to eight.
 */
const key = (c: number) => c.toString().padStart(8, '0')

/**
 * The colour histogram of every graph.
 *
 * @param colours The colours of each graph's nodes.
 * @param classes The size of the palette.
 * @returns `out[g][c]`, the number of nodes of graph $g$ with colour $c$.
 */
function histogramsOf(colours: Int32Array[], classes: number): number[][] {
  return colours.map((cs) => {
    const h = new Array<number>(classes).fill(0)
    for (const c of cs) h[c]++
    return h
  })
}

/**
 * Round-0 colours: the distinct initial labels in increasing order, numbered from 0 (all 0 without labels). Throws
 * `ShapeError` when a graph's labels do not have one entry per node.
 *
 * @param graphs The graphs.
 * @param initial The initial labels, one array per graph, or none.
 * @param where The caller's name, for error messages.
 * @returns The colours of every graph and the number of distinct ones.
 */
function initialColours(
  graphs: readonly Graph[],
  initial: readonly (readonly number[])[] | undefined,
  where: string,
): { colours: Int32Array[]; classes: number } {
  if (initial && initial.length !== graphs.length)
    throw new ShapeError(where, `${where}: ${initial.length} label arrays for ${graphs.length} graphs`)
  const labels = graphs.map((g, k) => {
    const given = initial?.[k]
    if (given && given.length !== g.nodes)
      throw new ShapeError(where, `${where}: the initial labels of graph ${k} need ${g.nodes} values`)
    return Array.from({ length: g.nodes }, (_, v) => (given ? given[v] : 0))
  })
  const distinct = [...new Set(labels.flat())].sort((a, b) => a - b)
  const index = new Map(distinct.map((s, i) => [s, i]))
  return { colours: labels.map((ls) => Int32Array.from(ls, (s) => index.get(s)!)), classes: distinct.length }
}

/**
 * Whether two histograms are equal.
 *
 * @param a A histogram.
 * @param b Another, over the same palette.
 * @returns True when every count agrees.
 */
const sameHistogram = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i])

/**
 * The first round at which some graph's histogram differs from the first graph's.
 *
 * @param histograms The histograms of every round.
 * @param count The number of graphs.
 * @returns The round, or $-1$ (always $-1$ for fewer than two graphs).
 */
const firstDifference = (histograms: number[][][], count: number) =>
  count > 1 ? histograms.findIndex((hs) => hs.some((h) => !sameHistogram(h, hs[0]))) : -1

/**
 * One round of colour refinement on every graph, with one palette.
 *
 * @param lists The neighbours of every node of every graph.
 * @param prev The colours of the round before.
 * @returns The new colours and the number of distinct ones.
 */
function refine(lists: number[][][], prev: Int32Array[]): { colours: Int32Array[]; classes: number } {
  const signatures = lists.map((adj, g) =>
    adj.map((near, v) => {
      const multiset = near.map((u) => prev[g][u]).sort((a, b) => a - b)
      return `${key(prev[g][v])}|${multiset.map(key).join(',')}`
    }),
  )
  return numberSignatures(signatures)
}

/**
 * The neighbour lists of every graph, as `adjacency` orders them.
 *
 * @param graphs The graphs.
 * @returns `lists[g][v]`, the neighbours of node $v$ of graph $g$.
 */
const neighbourLists = (graphs: readonly Graph[]) =>
  graphs.map((g) => adjacency(g).map((arcs) => arcs.map((a) => a.to)))

/**
 * Colour refinement (1-WL) on one or several graphs with one shared palette (see the file comment), every round at
 * once. Throws `ShapeError` for initial labels that do not match the graphs.
 *
 * @param graphs The graphs. An undirected graph uses each edge both ways; a directed one refines by out-neighbours.
 * @param options The largest number of rounds and the initial labels.
 * @returns The colours, class counts and histograms of every round, the stable round and the first distinguishing
 *   round.
 *
 * @example The 4-path and the 3-leaf star differ in their degrees, so one round separates them
 * import { chainGraph, starGraph } from 'aifn-compute/graph/structures'
 * const r = weisfeilerLeman([chainGraph(4), starGraph(3)])
 * print('distinguished at round', r.distinguishedAt)
 * print('path colours after round 1:', r.colours[1][0])
 *
 * @example A 6-cycle and two triangles: every node keeps one colour, so 1-WL cannot tell them apart
 * import { cycleGraph } from 'aifn-compute/graph/structures'
 * const edges = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]].map(([from, to]) => ({ from, to }))
 * const triangles = { kind: 'graph', nodes: 6, edges, directed: false }
 * const r = weisfeilerLeman([cycleGraph(6), triangles])
 * print('distinguished at', r.distinguishedAt, ' stable at', r.stable, ' classes', r.classes)
 */
export function weisfeilerLeman(graphs: readonly Graph[], options: WeisfeilerLemanOptions = {}): WeisfeilerLemanTrace {
  const lists = neighbourLists(graphs)
  const limit = options.rounds ?? Math.max(1, ...graphs.map((g) => g.nodes))
  const first = initialColours(graphs, options.initial, 'weisfeilerLeman')
  const colours: Int32Array[][] = [first.colours]
  const classes: number[] = [first.classes]
  let stable = 0
  for (let t = 1; t <= limit; t++) {
    const next = refine(lists, colours[t - 1])
    colours.push(next.colours)
    classes.push(next.classes)
    stable = t
    // Refinement never merges classes, so an equal class count means an equal partition.
    if (next.classes === classes[t - 1]) break
  }
  const histograms = colours.map((cs, t) => histogramsOf(cs, classes[t]))
  return { colours, classes, histograms, stable, distinguishedAt: firstDifference(histograms, graphs.length) }
}

/** The state of `weisfeilerLemanSteps` after a round. */
export interface WeisfeilerLemanState extends Status {
  /** The rounds run so far (0 before the first); equal to the step count `t`. */
  round: number
  /** `colours[g][v]`: the colour of node $v$ of graph $g$ after this round. */
  colours: Int32Array[]
  /** The number of distinct colours over all graphs. */
  classes: number
  /** `histograms[g][c]`: how many nodes of graph $g$ have colour $c$. */
  histograms: number[][]
  /** Whether some two graphs' histograms differ at this round. */
  distinguished: boolean
  /** True once the partition is stable (this round split no class), or the round limit is reached. */
  done: boolean
}

/**
 * Colour refinement one round per step, for a figure that steps through it: `run(alg, undefined, t)` gives the
 * colours after round $t$, or at the stable round if that comes first. Its states are the rounds of `weisfeilerLeman`.
 * Throws as `weisfeilerLeman` does.
 *
 * @param graphs The graphs, as for `weisfeilerLeman`.
 * @param options The largest number of rounds and the initial labels.
 * @returns The algorithm: its state starts at round 0 (the initial colours).
 *
 * @example Stepping the 4-path and the star round by round
 * import { chainGraph, starGraph } from 'aifn-compute/graph/structures'
 * const alg = weisfeilerLemanSteps([chainGraph(4), starGraph(3)])
 * const s = run(alg, undefined, 1)
 * print('round', s.round, ' colours of the path:', s.colours[0], ' distinguished:', s.distinguished)
 */
export function weisfeilerLemanSteps(
  graphs: readonly Graph[],
  options: WeisfeilerLemanOptions = {},
): Algorithm<void, WeisfeilerLemanState> {
  const lists = neighbourLists(graphs)
  const limit = options.rounds ?? Math.max(1, ...graphs.map((g) => g.nodes))
  const state = (round: number, colours: Int32Array[], classes: number, done: boolean): WeisfeilerLemanState => {
    const histograms = histogramsOf(colours, classes)
    const distinguished = graphs.length > 1 && histograms.some((h) => !sameHistogram(h, histograms[0]))
    return { t: round, round, colours, classes, histograms, distinguished, done }
  }
  return {
    name: 'weisfeiler-leman',
    init: () => {
      const first = initialColours(graphs, options.initial, 'weisfeilerLemanSteps')
      return state(0, first.colours, first.classes, limit === 0)
    },
    step: (s) => {
      const next = refine(lists, s.colours)
      return state(s.round + 1, next.colours, next.classes, next.classes === s.classes || s.round + 1 >= limit)
    },
    done: (s) => s.done,
  }
}

/** The result of `folkloreWeisfeilerLeman2`. */
export interface FolkloreTrace {
  /** `classes[t]`: the number of distinct pair colours after round $t$, over all graphs. */
  classes: number[]
  /** `histograms[t][g][c]`: how many ordered pairs of graph $g$ have colour $c$ after round $t$. */
  histograms: number[][][]
  /** `pairColours[t][g][u * n + v]`: the colour of the ordered pair $(u, v)$ of graph $g$ after round $t$. */
  pairColours: Int32Array[][]
  /** The stable round, as in `weisfeilerLeman`. */
  stable: number
  /** The first round at which two graphs' pair histograms differ, or $-1$. */
  distinguishedAt: number
}

/**
 * The 2-dimensional folklore Weisfeiler-Leman test (2-FWL) on undirected graphs with a shared palette (see the file
 * comment). It costs $O(n^3)$ per round per graph, so it suits graphs of a few dozen nodes. Throws `DomainError` for a
 * directed graph and `ShapeError` for initial labels that do not match the graphs.
 *
 * @param graphs The undirected graphs.
 * @param options The largest number of rounds, and initial node labels (folded into the atomic type of each pair).
 * @returns The pair colours, class counts and histograms of every round, the stable round and the first
 *   distinguishing round.
 *
 * @example 2-FWL counts triangles, so it separates a 6-cycle from two triangles
 * import { cycleGraph } from 'aifn-compute/graph/structures'
 * const edges = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]].map(([from, to]) => ({ from, to }))
 * const triangles = { kind: 'graph', nodes: 6, edges, directed: false }
 * print('distinguished at', folkloreWeisfeilerLeman2([cycleGraph(6), triangles]).distinguishedAt)
 */
export function folkloreWeisfeilerLeman2(
  graphs: readonly Graph[],
  options: WeisfeilerLemanOptions = {},
): FolkloreTrace {
  const where = 'folkloreWeisfeilerLeman2'
  graphs.forEach((g, k) => {
    if (g.directed !== false) throw new DomainError(where, `${where}: graph ${k} must be undirected`)
  })
  initialColours(graphs, options.initial, where)
  const adjacent = graphs.map((g) => {
    const a = new Uint8Array(g.nodes * g.nodes)
    adjacency(g).forEach((arcs, v) => arcs.forEach((arc) => (a[v * g.nodes + arc.to] = 1)))
    return a
  })
  const limit = options.rounds ?? Math.max(1, ...graphs.map((g) => g.nodes))
  const atomic = graphs.map((g, k) => {
    const labels = options.initial?.[k]
    const n = g.nodes
    const sig: string[] = []
    for (let u = 0; u < n; u++)
      for (let v = 0; v < n; v++) {
        const kind = u === v ? 'e' : adjacent[k][u * n + v] ? 'a' : 'n'
        sig.push(`${kind}|${labels ? labels[u] : 0}|${labels ? labels[v] : 0}`)
      }
    return sig
  })
  const first = numberSignatures(atomic)
  const pairColours: Int32Array[][] = [first.colours]
  const classes = [first.classes]
  let stable = 0
  for (let t = 1; t <= limit; t++) {
    const prev = pairColours[t - 1]
    const signatures = graphs.map((g, k) => {
      const n = g.nodes
      const c = prev[k]
      const sig: string[] = []
      for (let u = 0; u < n; u++)
        for (let v = 0; v < n; v++) {
          const pairs: number[] = []
          for (let w = 0; w < n; w++) pairs.push(c[u * n + w] * 1e6 + c[w * n + v])
          pairs.sort((a, b) => a - b)
          sig.push(`${key(c[u * n + v])}|${pairs.join(',')}`)
        }
      return sig
    })
    const next = numberSignatures(signatures)
    pairColours.push(next.colours)
    classes.push(next.classes)
    stable = t
    if (next.classes === classes[t - 1]) break
  }
  const histograms = pairColours.map((cs, t) => histogramsOf(cs, classes[t]))
  return { classes, histograms, pairColours, stable, distinguishedAt: firstDifference(histograms, graphs.length) }
}

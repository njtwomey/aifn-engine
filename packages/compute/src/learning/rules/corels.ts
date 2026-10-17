/**
 * CORELS, certifiably optimal rule lists (Angelino, Larus-Stone, Alabi, Seltzer and Rudin, 2018, "Learning certifiably
 * optimal rule lists for categorical data", JMLR 18(234), §3 and Algorithm 1): branch and bound over rule-list
 * prefixes, with every prefix recorded for a figure to replay.
 *
 * Over lists of distinct pre-mined antecedents (`mineAntecedents`), each rule predicting the majority class of the
 * rows it captures and the default rule the majority of the rest, it minimises the regularised risk
 * $R(d) = \ell(d) + \lambda K$, where $\ell(d)$ is the fraction of training rows misclassified and $K$ the number of
 * rules. The search runs over the tree of prefixes (the root is the empty list; a child appends one unused
 * antecedent), breadth first or best first by lower bound. The bounds:
 *
 * - the hierarchical lower bound $b(d_p) = \ell_p(d_p) + \lambda K$, the errors of the rows the prefix captures plus
 *   the penalty, which holds for every extension (Theorem 1), so a prefix with $b \ge R^c$, the best risk so far, is
 *   pruned with its subtree;
 * - the one-step lookahead (Lemma 2): when $b + \lambda \ge R^c$ no extension can win, so the prefix is evaluated but
 *   not expanded;
 * - the lower bound on antecedent support (Theorem 10): an antecedent that captures less than a fraction $\lambda$ of
 *   the rows in its position cannot be in an optimal list; and on accurate support (Theorem 11): nor can one that
 *   captures less than $\lambda$ correctly classified;
 * - the permutation bound (Corollary 16): of the prefixes that are permutations of one antecedent set, only the one of
 *   smallest $b$ is kept;
 * - the equivalent-points bound (Theorem 20): rows that every antecedent treats alike but that carry different labels
 *   force at least their minority count of errors on whichever rule captures them, so $b(d_p)$ plus the minority
 *   errors among the uncaptured rows bounds the risk of the prefix and of every extension.
 *
 * The search ends when the queue is empty: the best list is then optimal among all lists of the given antecedents,
 * which is the certificate. Differences from CORELS: no prefix trie or incremental computation (each prefix stores
 * its uncaptured rows, one byte per row), and a budget `maxNodes` of evaluated prefixes after which the search stops
 * without a certificate. A plain function of plain data, so it can run in a worker.
 */

import type { Antecedent } from './conditions'

/** Options of `corels`. */
export type CorelsOptions = {
  /** The price $\lambda \ge 0$ of one rule in training error (default 0.01). */
  lambda?: number
  /** The queue order: `breadth` (default) or `best` (smallest lower bound first). */
  order?: 'breadth' | 'best'
  /** Which bounds to use beyond Theorem 1 (default all). */
  bounds?: {
    /** The one-step lookahead (Lemma 2). */
    lookahead?: boolean
    /** The lower bound on antecedent support (Theorem 10). */
    support?: boolean
    /** The lower bound on accurate support (Theorem 11). */
    accurateSupport?: boolean
    /** The permutation bound (Corollary 16). */
    permutation?: boolean
    /** The equivalent-points bound (Theorem 20). */
    equivalentPoints?: boolean
  }
  /** Stop after evaluating this many prefixes (default 200000); the result then carries no certificate. */
  maxNodes?: number
}

/** What became of a generated prefix: still queued, expanded, a leaf, or the bound that discarded it. */
export type NodeStatus =
  | 'queued'
  | 'expanded'
  | 'leaf'
  | 'bound'
  | 'lookahead'
  | 'support'
  | 'accurate-support'
  | 'permutation'
  | 'equivalent-points'

/** A prefix the search generated. */
export type CorelsNode = {
  /** Its index in `nodes`. */
  id: number
  /** The index of its parent ($-1$ for the root). */
  parent: number
  /** The antecedent indices, in order. */
  prefix: number[]
  /** The lower bound $b(d_p)$. */
  bound: number
  /** Once evaluated, the risk $R$ of the list with this prefix and its default rule (NaN before). */
  objective: number
  /** The final status. */
  status: NodeStatus
  /** The search step at which it was generated. */
  generated: number
  /** The search step at which it got its status. */
  decided: number
}

/** One search step: a prefix popped from the queue, and the best list after it. */
export type CorelsStep = {
  /** The node popped. */
  node: number
  /** The best list so far. */
  best: number[]
  /** Its risk. */
  bestObjective: number
  /** The queue's length after the step. */
  queue: number
}

/** What `corels` returns. */
export type CorelsResult = {
  /** The best list found, as antecedent indices. */
  best: number[]
  /** The labels of the best list's rules. */
  labels: number[]
  /** Its default label. */
  defaultLabel: number
  /** Its regularised risk. */
  bestObjective: number
  /** True when the queue was emptied, so `best` is certified optimal. */
  certified: boolean
  /** Every generated prefix. */
  nodes: CorelsNode[]
  /** Every search step. */
  steps: CorelsStep[]
}

/**
 * Branch-and-bound search for the rule list of least regularised risk (see the file comment).
 *
 * @param antecedents The pre-mined antecedents (their masks over the training rows are used).
 * @param y The class of each row, $0, \dots, K - 1$.
 * @param options $\lambda$, the queue order, the bounds and the node budget.
 * @returns The optimal list, its labels and risk, the certificate flag, every generated prefix and every step.
 *
 * @example The optimal list of a toy table
 * const rows = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]
 * const y = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0]
 * const antecedents = mineAntecedents(rows, nominalConditions([2, 2]), { minSupport: 0.1 })
 * const r = corels(antecedents, y, { lambda: 0.05 })
 * print('list:', r.best.map((j) => ruleText(antecedents[j].conditions)), '->', r.labels, 'else', r.defaultLabel)
 * print('risk:', r.bestObjective, ' certified:', r.certified, ' prefixes:', r.nodes.length)
 */
export function corels(
  antecedents: readonly Antecedent[],
  y: ArrayLike<number>,
  options: CorelsOptions = {},
): CorelsResult {
  const lambda = options.lambda ?? 0.01
  const order = options.order ?? 'breadth'
  const use = {
    lookahead: true,
    support: true,
    accurateSupport: true,
    permutation: true,
    equivalentPoints: true,
    ...options.bounds,
  }
  const maxNodes = options.maxNodes ?? 200000
  const n = y.length
  let K = 2
  for (let i = 0; i < n; i++) K = Math.max(K, y[i] + 1)

  // Rows that every antecedent treats alike form an equivalence class; any list misclassifies at least the minority
  // labels of each class among the rows it leaves to one rule (Theorem 20).
  const classOf = new Int32Array(n)
  {
    const ids = new Map<string, number>()
    for (let i = 0; i < n; i++) {
      let k = ''
      for (const a of antecedents) k += a.mask[i] ? '1' : '0'
      if (!ids.has(k)) ids.set(k, ids.size)
      classOf[i] = ids.get(k) as number
    }
  }
  const nClasses = classOf.reduce((a, b) => Math.max(a, b + 1), 0)
  const minorityErrors = (uncaptured: Uint8Array) => {
    if (!use.equivalentPoints) return 0
    const c = new Int32Array(nClasses * K)
    for (let i = 0; i < n; i++) if (uncaptured[i]) c[classOf[i] * K + y[i]]++
    let e = 0
    for (let g = 0; g < nClasses; g++) {
      let total = 0
      let most = 0
      for (let k = 0; k < K; k++) {
        total += c[g * K + k]
        most = Math.max(most, c[g * K + k])
      }
      e += total - most
    }
    return e / n
  }

  type Work = { id: number; uncaptured: Uint8Array; prefixErrors: number; labels: number[]; floor: number }
  const nodes: CorelsNode[] = []
  const steps: CorelsStep[] = []
  const work = new Map<number, Work>()
  const permutation = new Map<string, { bound: number; id: number }>()

  // The majority class and its count among the marked rows.
  const majority = (mask: Uint8Array) => {
    const c = new Array<number>(K).fill(0)
    for (let i = 0; i < n; i++) if (mask[i]) c[y[i]]++
    let best = 0
    for (let k = 1; k < K; k++) if (c[k] > c[best]) best = k
    const total = c.reduce((a, b) => a + b, 0)
    return { label: best, correct: c[best], total }
  }
  const objectiveOf = (w: Work, length: number) => {
    const d = majority(w.uncaptured)
    return { R: (w.prefixErrors + d.total - d.correct) / n + lambda * length, label: d.label }
  }

  const all = new Uint8Array(n).fill(1)
  const root: Work = { id: 0, uncaptured: all, prefixErrors: 0, labels: [], floor: minorityErrors(all) }
  nodes.push({ id: 0, parent: -1, prefix: [], bound: 0, objective: NaN, status: 'queued', generated: 0, decided: 0 })
  work.set(0, root)
  const queue = new Queue(order === 'best' ? (a, b) => nodes[a].bound - nodes[b].bound : null)
  queue.push(0)
  let best: number[] = []
  let bestLabels: number[] = []
  let bestDefault = 0
  let bestObjective = Infinity
  let step = 0
  let evaluated = 0

  while (queue.size > 0 && evaluated < maxNodes) {
    const id = queue.pop()
    const node = nodes[id]
    const w = work.get(id)
    if (!w) continue
    step++
    if (node.bound >= bestObjective) {
      node.status = 'bound'
      node.decided = step
      work.delete(id)
      steps.push({ node: id, best: best.slice(), bestObjective, queue: queue.size })
      continue
    }
    // Its own risk is b plus the default rule's errors, at least the unavoidable ones: no need to evaluate it.
    if (node.bound + w.floor >= bestObjective) {
      node.status = 'equivalent-points'
      node.decided = step
      work.delete(id)
      steps.push({ node: id, best: best.slice(), bestObjective, queue: queue.size })
      continue
    }
    evaluated++
    const ev = objectiveOf(w, node.prefix.length)
    node.objective = ev.R
    if (ev.R < bestObjective) {
      bestObjective = ev.R
      best = node.prefix.slice()
      bestLabels = w.labels.slice()
      bestDefault = ev.label
    }
    if (use.lookahead && node.bound + w.floor + lambda >= bestObjective) {
      node.status = 'lookahead'
      node.decided = step
      work.delete(id)
      steps.push({ node: id, best: best.slice(), bestObjective, queue: queue.size })
      continue
    }
    node.status = 'expanded'
    node.decided = step
    let children = 0
    antecedents.forEach((a, j) => {
      if (node.prefix.includes(j)) return
      const captured = new Uint8Array(n)
      let cap = 0
      for (let i = 0; i < n; i++)
        if (w.uncaptured[i] && a.mask[i]) {
          captured[i] = 1
          cap++
        }
      const d = majority(captured)
      const prefix = [...node.prefix, j]
      const bound = (w.prefixErrors + (d.total - d.correct)) / n + lambda * prefix.length
      const childId = nodes.length
      const record: CorelsNode = {
        id: childId,
        parent: id,
        prefix,
        bound,
        objective: NaN,
        status: 'queued',
        generated: step,
        decided: step,
      }
      nodes.push(record)
      if (use.support && cap / n < lambda) {
        record.status = 'support'
        return
      }
      if (use.accurateSupport && d.correct / n < lambda) {
        record.status = 'accurate-support'
        return
      }
      if (bound >= bestObjective) {
        record.status = 'bound'
        return
      }
      if (use.permutation) {
        const key = prefix
          .slice()
          .sort((p, q) => p - q)
          .join(',')
        const seen = permutation.get(key)
        if (seen && seen.bound <= bound) {
          record.status = 'permutation'
          return
        }
        if (seen) {
          const other = nodes[seen.id]
          if (other.status === 'queued') {
            // The queued permutation is dominated: dropping its work makes the queue skip it when popped.
            other.status = 'permutation'
            other.decided = step
            work.delete(seen.id)
          }
        }
        permutation.set(key, { bound, id: childId })
      }
      const uncaptured = w.uncaptured.slice()
      for (let i = 0; i < n; i++) if (captured[i]) uncaptured[i] = 0
      const floor = minorityErrors(uncaptured)
      if (use.equivalentPoints && bound + floor >= bestObjective) {
        record.status = 'equivalent-points'
        return
      }
      work.set(childId, {
        id: childId,
        uncaptured,
        prefixErrors: w.prefixErrors + (d.total - d.correct),
        labels: [...w.labels, d.label],
        floor,
      })
      queue.push(childId)
      children++
    })
    if (children === 0) node.status = 'leaf'
    work.delete(id)
    steps.push({ node: id, best: best.slice(), bestObjective, queue: queue.size })
  }
  return {
    best,
    labels: bestLabels,
    defaultLabel: bestDefault,
    bestObjective,
    certified: queue.size === 0,
    nodes,
    steps,
  }
}

/** A FIFO queue of node ids, or a binary min-heap when given a comparison. */
class Queue {
  /** The stored ids (a FIFO from `head` on, or a heap). */
  private items: number[] = []
  /** The index of the FIFO's first id. */
  private head = 0
  /** The heap's comparison (negative when the first id comes first), or null for a FIFO. */
  private readonly less: ((a: number, b: number) => number) | null

  /**
   * An empty queue.
   *
   * @param less The heap's comparison, or null for first in, first out.
   */
  constructor(less: ((a: number, b: number) => number) | null) {
    this.less = less
  }

  /** The number of ids queued. */
  get size(): number {
    return this.items.length - this.head
  }

  /**
   * Queue an id.
   *
   * @param v The id.
   */
  push(v: number): void {
    this.items.push(v)
    if (!this.less) return
    let i = this.items.length - 1
    while (i > 0) {
      const p = (i - 1) >> 1
      if (this.less(this.items[i], this.items[p]) >= 0) break
      ;[this.items[i], this.items[p]] = [this.items[p], this.items[i]]
      i = p
    }
  }

  /**
   * Take the next id: the oldest (FIFO) or the smallest (heap).
   *
   * @returns The id.
   */
  pop(): number {
    if (!this.less) return this.items[this.head++]
    const top = this.items[0]
    const last = this.items.pop() as number
    if (this.items.length > 0) {
      this.items[0] = last
      let i = 0
      for (;;) {
        const l = 2 * i + 1
        const r = l + 1
        let m = i
        if (l < this.items.length && this.less(this.items[l], this.items[m]) < 0) m = l
        if (r < this.items.length && this.less(this.items[r], this.items[m]) < 0) m = r
        if (m === i) break
        ;[this.items[i], this.items[m]] = [this.items[m], this.items[i]]
        i = m
      }
    }
    return top
  }
}

/**
 * The terms of RuleFit (Friedman and Popescu, 2008, "Predictive learning via rule ensembles", Annals of Applied
 * Statistics 2(3), §3 and §5): rules read off the nodes of a tree ensemble, as 0/1 columns, and the winsorised linear
 * terms of the inputs. The lasso over them is `ruleFitPath` of `aifn-methods/learning/linear`.
 *
 * Every node of every tree except the root defines a rule $r_k(\xvec) \in \{0, 1\}$, the conjunction of the split
 * conditions on the path from the root to the node ($x_f \le t$ to the left child, $x_f > t$ to the right), so a tree
 * with $t$ leaves gives $2(t - 1)$ rules. Rules with the same coverage of the training rows are merged (the first is
 * kept) and rules that cover no row or every row are dropped. Each input $x_j$ also enters as a linear term,
 * winsorised to its $\beta$ and $1 - \beta$ quantiles, $l_j = \min(\delta_j^+, \max(\delta_j^-, x_j))$, and scaled by
 * $0.4/\mathrm{sd}(l_j)$, where 0.4 is the average standard deviation $\sqrt{s(1 - s)}$ of a rule whose support $s$ is
 * uniform on $(0, 1)$; the rules keep their 0/1 scale, so the penalty falls hardest on rules of very small or very
 * large support. The trees are any binary trees with `feature` and `threshold` at their splits, such as the trees of
 * `aifn-methods/learning/trees-and-ensembles`.
 */

import type { Tree } from 'aifn-compute/foundation/contracts'
import { satisfies, simplifyRule, type Predicate } from 'aifn-compute/learning/explain'
import { quantile } from 'aifn-compute/probability/stats'
import type { RuleRows } from './conditions'

/** A binary tree whose splits send $x_f \le t$ to the first child and $x_f > t$ to the second. */
export type RuleTree = Pick<Tree<{ feature: number; threshold: number }>, 'root' | 'nodes'>

/** A rule of the ensemble: its conditions, the tree it came from and the node it ends at. */
export type EnsembleRule = {
  /** The conditions on the path from the root, one interval per feature. */
  conditions: Predicate[]
  /** The index of the tree in the ensemble. */
  tree: number
  /** The node of that tree the rule describes. */
  node: number
}

/**
 * The rules of every non-root node of every tree: the conjunction of the conditions on the path to the node
 * ($x_f \le t$ to the left child, $x_f > t$ to the right), with the intervals on one feature intersected.
 *
 * @param trees The trees of the ensemble: binary, `children` = `[left, right]` at a split; a node with another number
 *   of children is a leaf.
 * @returns $\sum_m 2(t_m - 1)$ rules for trees of $t_m$ leaves, tree by tree, in depth-first order with the left
 *   child first.
 *
 * @example The four rules of a tree with three leaves
 * const node = (id, parent, children, feature = -1, threshold = 0) => ({ id, parent, children, feature, threshold })
 * const nodes = [node(0, null, [1, 2], 0, 0.5), node(1, 0, []), node(2, 0, [3, 4], 1, 2)]
 * nodes.push(node(3, 2, []), node(4, 2, []))
 * const tree = { root: 0, nodes }
 * for (const r of ensembleRules([tree])) print(ruleText(r.conditions), ' (node', r.node + ')')
 */
export function ensembleRules(trees: readonly RuleTree[]): EnsembleRule[] {
  const out: EnsembleRule[] = []
  trees.forEach((tree, m) => {
    const path = new Map<number, Predicate[]>([[tree.root, []]])
    const stack = [tree.root]
    while (stack.length > 0) {
      const id = stack.pop() as number
      const node = tree.nodes[id]
      const here = path.get(id) as Predicate[]
      if (node.children.length !== 2) continue
      const [left, right] = node.children
      const f = node.feature
      const t = node.threshold
      const l = [...here, { feature: f, lower: -Infinity, upper: t }]
      const r = [...here, { feature: f, lower: t, upper: Infinity }]
      path.set(left, l)
      path.set(right, r)
      out.push({ conditions: simplifyRule(l), tree: m, node: left })
      out.push({ conditions: simplifyRule(r), tree: m, node: right })
      stack.push(right, left)
    }
  })
  return out
}

/** Rules as 0/1 columns on a set of rows, with duplicates and constant columns removed. */
export type RuleColumns = {
  /** The rules kept, in the order given. */
  rules: EnsembleRule[]
  /** Column $k$: 1 where rule $k$ holds, one entry per row. */
  columns: Uint8Array[]
  /** The support $s_k$ of each kept rule on these rows. */
  support: number[]
  /** How many rules were merged into an earlier one with the same coverage. */
  merged: number
  /** How many rules covered no row or every row. */
  constant: number
}

/**
 * Evaluate rules on the training rows and keep one rule per distinct coverage pattern, dropping those that cover no
 * row or every row.
 *
 * @param rows The training rows.
 * @param rules The candidate rules (from `ensembleRules`).
 * @returns The kept rules, their columns and supports, and the counts merged and dropped.
 *
 * @example Supports of four stump rules on eight rows
 * const rows = [1, 2, 3, 4, 5, 6, 7, 8].map((v) => [v])
 * const R = (lower, upper) => ({ conditions: [{ feature: 0, lower, upper }], tree: 0, node: 0 })
 * const rules = [R(-Infinity, 4.5), R(4.5, Infinity), R(-Infinity, 7.5), R(7.5, Infinity), R(-Infinity, 4.5)]
 * const cols = ruleFeatures(rows, rules)
 * print('support:', cols.support, ' merged:', cols.merged)
 */
export function ruleFeatures(rows: RuleRows, rules: readonly EnsembleRule[]): RuleColumns {
  const n = rows.length
  const seen = new Set<string>()
  const kept: EnsembleRule[] = []
  const columns: Uint8Array[] = []
  const support: number[] = []
  let merged = 0
  let constant = 0
  for (const rule of rules) {
    const col = new Uint8Array(n)
    let s = 0
    for (let i = 0; i < n; i++) {
      if (satisfies(rows[i], rule.conditions)) {
        col[i] = 1
        s++
      }
    }
    if (s === 0 || s === n) {
      constant++
      continue
    }
    const key = col.join('')
    if (seen.has(key)) {
      merged++
      continue
    }
    seen.add(key)
    kept.push(rule)
    columns.push(col)
    support.push(s / n)
  }
  return { rules: kept, columns, support, merged, constant }
}

/** The winsorised and scaled linear terms of the inputs. */
export type LinearTerms = {
  /** The lower winsorising quantile $\delta_j^-$ of each feature. */
  lower: number[]
  /** The upper winsorising quantile $\delta_j^+$ of each feature. */
  upper: number[]
  /** The mean of each winsorised feature on the training rows. */
  mean: number[]
  /** The standard deviation (divisor $n$) of each winsorised feature on the training rows. */
  sd: number[]
  /** The factor $0.4/\mathrm{sd}(l_j)$ the design uses (0 for a constant feature). */
  scale: number[]
}

/**
 * The winsorising limits and scales of the linear terms: the $\beta$ and $1 - \beta$ quantiles of each feature
 * (`quantile`, linear interpolation), and the mean, standard deviation and scale $0.4/\mathrm{sd}(l_j)$ of the
 * winsorised feature.
 *
 * @param rows The training rows ($n \times d$).
 * @param beta The winsorising quantile $\beta$.
 * @returns The limits, moments and scales of each feature.
 *
 * @example An outlier is clipped
 * const rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100].map((v) => [v])
 * print(linearTerms(rows, 0.1))
 */
export function linearTerms(rows: RuleRows, beta = 0.025): LinearTerms {
  const d = rows.length > 0 ? rows[0].length : 0
  const out: LinearTerms = { lower: [], upper: [], mean: [], sd: [], scale: [] }
  for (let j = 0; j < d; j++) {
    const v = rows.map((r) => r[j])
    const lo = quantile(v, beta)
    const hi = quantile(v, 1 - beta)
    const w = rows.map((r) => Math.min(hi, Math.max(lo, r[j])))
    const mean = w.reduce((a, b) => a + b, 0) / w.length
    const sd = Math.sqrt(w.reduce((a, b) => a + (b - mean) ** 2, 0) / w.length)
    out.lower.push(lo)
    out.upper.push(hi)
    out.mean.push(mean)
    out.sd.push(sd)
    out.scale.push(sd > 0 ? 0.4 / sd : 0)
  }
  return out
}

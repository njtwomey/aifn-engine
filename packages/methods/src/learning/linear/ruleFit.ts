/**
 * RuleFit (Friedman and Popescu, 2008, "Predictive learning via rule ensembles", Annals of Applied Statistics
 * 2(3), §3, §5 and §6): the lasso path over the rules of a tree ensemble and the winsorised linear terms of the
 * inputs, its predictions and its term importances.
 *
 * The rules and linear terms are those of `aifn-compute/learning/rules` (`ensembleRules`, `ruleFeatures`,
 * `linearTerms`): rules are 0/1 columns, merged when two cover the same training rows, and each input $x_j$ enters as
 * $0.4\, l_j / \mathrm{sd}(l_j)$ with $l_j$ winsorised to its $\beta$ and $1 - \beta$ quantiles. The coefficients
 * solve the lasso
 *
 * $$
 * \min_{a_0, \avec, \bvec} \frac{1}{2n}\sum_i \Big(y_i - a_0 - \sum_k a_k r_k(\xvec_i) - \sum_j b_j l_j(x_{ij})\Big)^2
 * + \alpha \Big(\sum_k \lvert a_k \rvert + \sum_j \lvert b_j \rvert\Big)
 * $$
 *
 * over a path of penalties $\alpha$ by `elasticNetPath` (whose loss is the squared error over $2n$). Importances are
 * $I_k = \lvert a_k \rvert \sqrt{s_k(1 - s_k)}$ for a rule of support $s_k$ and
 * $I_j = \lvert b_j \rvert\,\mathrm{sd}(l_j)$ for a linear term (equations 28 and 29), with $b_j$ on the scale of
 * $l_j$. Differences from the paper: the trees are whatever ensemble is passed in (the paper draws each tree's size
 * from an exponential distribution and fits on subsamples with shrinkage $\nu \approx 0.01$); squared-error loss only.
 */

import { satisfies, type Predicate } from 'aifn-compute/learning/explain'
import {
  linearTerms,
  ruleFeatures,
  type EnsembleRule,
  type LinearTerms,
  type RuleRows,
} from 'aifn-compute/learning/rules'
import { tensor, toArray } from 'aifn-compute/foundation/tensor'
import { elasticNetPath } from './elasticNet'

/** A fitted RuleFit path. */
export type RuleFitPath = {
  /** The rules of the design (after merging), in the order given. */
  rules: EnsembleRule[]
  /** The support of each rule on the training rows. */
  support: number[]
  /** The linear terms, or null when they were left out. */
  linear: LinearTerms | null
  /** The penalties $\alpha$, decreasing. */
  alphas: number[]
  /**
   * The coefficients per penalty: `coef[t][k]` for term $k$ (the rules first, then the linear terms on the scaled
   * design).
   */
  coef: number[][]
  /** The intercept per penalty. */
  intercept: number[]
  /** The number of nonzero coefficients per penalty. */
  nonzero: number[]
  /** Rules merged into an earlier one for equal coverage. */
  merged: number
  /** Rules dropped for covering no row or every row. */
  constant: number
}

/** Options of `ruleFitPath`. */
export type RuleFitOptions = {
  /** Add the linear terms (default true). */
  linear?: boolean
  /** Their winsorising quantile $\beta$ (default 0.025). */
  beta?: number
  /** The number of penalties (default 40). */
  count?: number
  /** The decades the penalties span below the smallest penalty that zeroes every coefficient (default 3). */
  span?: number
  /** The solver's relative tolerance (default $10^{-6}$). */
  tolerance?: number
  /** The solver's most sweeps per penalty (default 2000). */
  maxSteps?: number
}

/**
 * One design row: the rules' 0/1 values, then the scaled winsorised inputs.
 *
 * @param row The input row.
 * @param rules The rules of the design.
 * @param lin The linear terms, or null for none.
 * @returns The row of the design.
 */
function designRow(row: ArrayLike<number>, rules: readonly EnsembleRule[], lin: LinearTerms | null): number[] {
  const out: number[] = rules.map((r) => (satisfies(row, r.conditions) ? 1 : 0))
  if (lin) {
    for (let j = 0; j < lin.scale.length; j++) {
      out.push(lin.scale[j] * Math.min(lin.upper[j], Math.max(lin.lower[j], row[j])))
    }
  }
  return out
}

/**
 * The lasso path over rule and linear terms (see the file comment). A plain function of plain data (about 0.4 s for
 * 300 rules on 400 rows), so a figure can run it in a worker.
 *
 * @param rows The training rows ($n \times d$).
 * @param y The targets, $n$ values.
 * @param rules The candidate rules (from `ensembleRules`); an empty list fits the linear terms alone.
 * @param options The linear terms, the penalty grid and the solver's stopping rule.
 * @returns The path: the rules kept and their supports, the linear terms, and per penalty the coefficients,
 *   intercept and nonzero count.
 *
 * @example Four stump rules on eight rows
 * import { ensembleRules } from 'aifn-compute/learning/rules'
 * const rows = [1, 2, 3, 4, 5, 6, 7, 8].map((v) => [v])
 * const y = [1, 1, 1, 1, 5, 5, 5, 9]
 * const node = (id, parent, children, feature = -1, threshold = 0) => ({ id, parent, children, feature, threshold })
 * const stump = (t) => ({ root: 0, nodes: [node(0, null, [1, 2], 0, t), node(1, 0, []), node(2, 0, [])] })
 * const fit = ruleFitPath(rows, y, ensembleRules([stump(4.5), stump(7.5)]), { linear: false, count: 5 })
 * print('alphas:', fit.alphas)
 * print('nonzero:', fit.nonzero)
 */
export function ruleFitPath(
  rows: RuleRows,
  y: ArrayLike<number>,
  rules: readonly EnsembleRule[],
  options: RuleFitOptions = {},
): RuleFitPath {
  const cols = ruleFeatures(rows, rules)
  const lin = options.linear === false ? null : linearTerms(rows, options.beta ?? 0.025)
  const X = rows.map((r) => designRow(r, cols.rules, lin))
  const path = elasticNetPath(tensor(X), tensor(Array.from(y)), {
    count: options.count ?? 40,
    span: options.span ?? 3,
    tolerance: options.tolerance ?? 1e-6,
    maxSteps: options.maxSteps ?? 2000,
  })
  const C = toArray(path.coef) as number[][]
  const k = path.alphas.length
  const coef = Array.from({ length: k }, (_, t) => C.map((row) => row[t]))
  return {
    rules: cols.rules,
    support: cols.support,
    linear: lin,
    alphas: path.alphas,
    coef,
    intercept: path.intercept,
    nonzero: path.nonzero,
    merged: cols.merged,
    constant: cols.constant,
  }
}

/**
 * The predictions of the fit at one penalty, $\hat a_0 + \sum_k \hat a_k r_k(\xvec) + \sum_j \hat b_j l_j(x_j)$.
 *
 * @param fit The path.
 * @param t The index of the penalty in `fit.alphas`.
 * @param rows The rows to predict.
 * @returns One prediction per row.
 *
 * @example At the largest penalty every prediction is the mean
 * import { ensembleRules } from 'aifn-compute/learning/rules'
 * const rows = [1, 2, 3, 4, 5, 6, 7, 8].map((v) => [v])
 * const y = [1, 1, 1, 1, 5, 5, 5, 9]
 * const node = (id, parent, children, feature = -1, threshold = 0) => ({ id, parent, children, feature, threshold })
 * const stump = { root: 0, nodes: [node(0, null, [1, 2], 0, 4.5), node(1, 0, []), node(2, 0, [])] }
 * const fit = ruleFitPath(rows, y, ensembleRules([stump]), { count: 5 })
 * print('largest penalty:', ruleFitPredict(fit, 0, [[2], [6]]))
 * print('smallest penalty:', ruleFitPredict(fit, 4, [[2], [6]]))
 */
export function ruleFitPredict(fit: RuleFitPath, t: number, rows: RuleRows): number[] {
  const w = fit.coef[t]
  const nz: number[] = []
  w.forEach((v, k) => {
    if (v !== 0) nz.push(k)
  })
  const K = fit.rules.length
  return rows.map((row) => {
    let s = fit.intercept[t]
    for (const k of nz) {
      if (k < K) s += satisfies(row, fit.rules[k].conditions) ? w[k] : 0
      else {
        const j = k - K
        const lin = fit.linear as LinearTerms
        s += w[k] * lin.scale[j] * Math.min(lin.upper[j], Math.max(lin.lower[j], row[j]))
      }
    }
    return s
  })
}

/** One term of a fit with its coefficient and importance. */
export type RuleFitTerm =
  | {
      /** A rule. */
      kind: 'rule'
      /** The index of the term in the design. */
      index: number
      /** The rule's conditions. */
      conditions: Predicate[]
      /** Its support $s_k$ on the training rows. */
      support: number
      /** Its coefficient $a_k$. */
      coefficient: number
      /** Its importance $\lvert a_k \rvert \sqrt{s_k(1 - s_k)}$. */
      importance: number
    }
  | {
      /** A linear term. */
      kind: 'linear'
      /** The index of the term in the design. */
      index: number
      /** The input feature $j$. */
      feature: number
      /** Its coefficient $b_j$ on the winsorised (unscaled) input. */
      coefficient: number
      /** Its importance $\lvert b_j \rvert\,\mathrm{sd}(l_j)$. */
      importance: number
    }

/**
 * The nonzero terms of the fit at one penalty, most important first: $I_k = \lvert a_k \rvert \sqrt{s_k(1 - s_k)}$
 * for a rule, $I_j = \lvert b_j \rvert\,\mathrm{sd}(l_j)$ for a linear term, with $b_j$ the coefficient on the
 * unscaled winsorised input.
 *
 * @param fit The path.
 * @param t The index of the penalty in `fit.alphas`.
 * @returns The terms with a nonzero coefficient, by decreasing importance.
 *
 * @example The terms of the least penalised fit
 * import { ensembleRules } from 'aifn-compute/learning/rules'
 * const rows = [1, 2, 3, 4, 5, 6, 7, 8].map((v) => [v])
 * const y = [1, 1, 1, 1, 5, 5, 5, 9]
 * const node = (id, parent, children, feature = -1, threshold = 0) => ({ id, parent, children, feature, threshold })
 * const stump = (t) => ({ root: 0, nodes: [node(0, null, [1, 2], 0, t), node(1, 0, []), node(2, 0, [])] })
 * const fit = ruleFitPath(rows, y, ensembleRules([stump(4.5), stump(7.5)]), { linear: false, count: 5 })
 * for (const term of ruleImportances(fit, 4)) print(term.kind, term.index, term.coefficient, term.importance)
 */
export function ruleImportances(fit: RuleFitPath, t: number): RuleFitTerm[] {
  const K = fit.rules.length
  const out: RuleFitTerm[] = []
  fit.coef[t].forEach((v, k) => {
    if (v === 0) return
    if (k < K) {
      const s = fit.support[k]
      out.push({
        kind: 'rule',
        index: k,
        conditions: fit.rules[k].conditions,
        support: s,
        coefficient: v,
        importance: Math.abs(v) * Math.sqrt(s * (1 - s)),
      })
    } else {
      const j = k - K
      const lin = fit.linear as LinearTerms
      const b = v * lin.scale[j]
      out.push({ kind: 'linear', index: k, feature: j, coefficient: b, importance: Math.abs(b) * lin.sd[j] })
    }
  })
  return out.sort((a, b) => b.importance - a.importance)
}

/**
 * Bayesian rule lists (Letham, Rudin, McCormick and Madigan, 2015, "Interpretable classifiers using rules and Bayesian
 * analysis", Annals of Applied Statistics 9(3), §2.2 to §2.6): the log posterior of a decision list over pre-mined
 * antecedents, and a Metropolis-Hastings sampler over lists.
 *
 * Antecedents are mined beforehand (`mineAntecedents`). A decision list $d = (a_1, \dots, a_m)$ is an ordered choice
 * of $m$ distinct antecedents; row $i$ is captured by the first antecedent it meets, or by the default rule $j = 0$.
 * The prior is hierarchical: the length $m$ is $\Poisson(\lambda)$ truncated to $0, \dots, \lvert A \rvert$; each
 * antecedent's cardinality $c_j$ is $\Poisson(\eta)$ truncated to the cardinalities that still have unused
 * antecedents; the antecedent is uniform among the unused ones of that cardinality. Each rule $j$ has a class
 * distribution $\thetavec_j \sim \Dir(\alpha)$, integrated out: with $N_{j,l}$ the rows of class $l$ captured by rule
 * $j$, the marginal likelihood is
 *
 * $$
 * \prod_{j=0}^{m} \frac{\Gamma(\sum_l \alpha_l)}{\prod_l \Gamma(\alpha_l)}
 * \frac{\prod_l \Gamma(N_{j,l} + \alpha_l)}{\Gamma(\sum_l N_{j,l} + \alpha_l)}.
 * $$
 *
 * The sampler proposes, uniformly among the kinds possible, to move an antecedent to another position, to add an
 * unused antecedent at a position, or to remove one; the acceptance ratio includes the proposal probabilities of
 * §2.6 and the probability of choosing each kind. A rule's posterior predictive class probability is
 * $(\alpha_l + N_{j,l}) / \sum_k (\alpha_k + N_{j,k})$. Difference from the paper: one chain, started from the empty
 * list (the paper runs three chains from prior draws and checks the Gelman-Rubin statistic).
 */

import { child, integers, uniform, type Stream } from 'aifn-compute/foundation/random'
import { logFactorial, logGamma } from 'aifn-compute/numerics/special'
import type { Antecedent } from './conditions'

/** Hyperparameters of the prior and the likelihood. */
export type BrlPrior = {
  /** The prior expected list length $\lambda$ (default 3). */
  lambda?: number
  /** The prior expected antecedent cardinality $\eta$ (default 1). */
  eta?: number
  /** The Dirichlet pseudo-count per class $\alpha$ (default 1). */
  alpha?: number
}

/** The terms of a list's log posterior and its captured class counts. */
export type BrlScore = {
  /** The log prior of the list. */
  logPrior: number
  /** The log marginal likelihood of the labels. */
  logLikelihood: number
  /** Their sum, the unnormalised log posterior. */
  logPosterior: number
  /** The captured class counts of each rule, the default rule last. */
  counts: number[][]
}

/**
 * The unnormalised log posterior of a decision list (see the file comment).
 *
 * @param list Indices into `antecedents`, in order.
 * @param antecedents The pre-mined antecedents.
 * @param y The class of each row, $0, \dots, K - 1$.
 * @param classes The number of classes $K$.
 * @param prior $\lambda$, $\eta$ and $\alpha$.
 * @returns The log prior, the log likelihood, their sum and the captured counts.
 *
 * @example The empty list and a one-rule list
 * const rows = [[0], [0], [0], [1], [1], [1]]
 * const y = [0, 0, 1, 1, 1, 1]
 * const antecedents = mineAntecedents(rows, nominalConditions([2]))
 * print('empty:', ruleListLogPosterior([], antecedents, y, 2).logPosterior)
 * print('x = 1 first:', ruleListLogPosterior([1], antecedents, y, 2).logPosterior)
 */
export function ruleListLogPosterior(
  list: readonly number[],
  antecedents: readonly Antecedent[],
  y: ArrayLike<number>,
  classes: number,
  prior: BrlPrior = {},
): BrlScore {
  const lambda = prior.lambda ?? 3
  const eta = prior.eta ?? 1
  const alpha = prior.alpha ?? 1
  const A = antecedents.length
  // Length: Poisson(lambda) truncated to 0..A.
  let norm = 0
  for (let k = 0; k <= A; k++) norm += Math.exp(k * Math.log(lambda) - logFactorial(k))
  let logPrior = list.length * Math.log(lambda) - logFactorial(list.length) - Math.log(norm)
  // Cardinalities and antecedents, in order.
  const sizes = antecedents.map((a) => a.conditions.length)
  const remaining = new Map<number, number>()
  for (const c of sizes) remaining.set(c, (remaining.get(c) ?? 0) + 1)
  for (const j of list) {
    const c = sizes[j]
    let z = 0
    for (const [size, count] of remaining) if (count > 0) z += Math.exp(size * Math.log(eta) - logFactorial(size))
    logPrior += c * Math.log(eta) - logFactorial(c) - Math.log(z)
    logPrior -= Math.log(remaining.get(c) ?? 1)
    remaining.set(c, (remaining.get(c) ?? 1) - 1)
  }
  // Likelihood: Dirichlet-multinomial per rule.
  const counts = capturedCounts(list, antecedents, y, classes)
  const constant = logGamma(classes * alpha) - classes * logGamma(alpha)
  let logLikelihood = 0
  for (const c of counts) {
    let total = 0
    let term = constant
    for (let l = 0; l < classes; l++) {
      term += logGamma(c[l] + alpha)
      total += c[l] + alpha
    }
    logLikelihood += term - logGamma(total)
  }
  return { logPrior, logLikelihood, logPosterior: logPrior + logLikelihood, counts }
}

/**
 * The class counts each rule of a list captures (a row goes to the first antecedent it meets), the default rule last.
 *
 * @param list Indices into `antecedents`, in order.
 * @param antecedents The pre-mined antecedents.
 * @param y The class of each row.
 * @param classes The number of classes $K$.
 * @returns $m + 1$ rows of $K$ counts.
 *
 * @example Captured counts of a one-rule list
 * const rows = [[0], [0], [0], [1], [1], [1]]
 * const antecedents = mineAntecedents(rows, nominalConditions([2]))
 * print(capturedCounts([1], antecedents, [0, 0, 1, 1, 1, 1], 2))
 */
export function capturedCounts(
  list: readonly number[],
  antecedents: readonly Antecedent[],
  y: ArrayLike<number>,
  classes: number,
): number[][] {
  const counts = Array.from({ length: list.length + 1 }, () => new Array<number>(classes).fill(0))
  for (let i = 0; i < y.length; i++) {
    let k = list.findIndex((j) => antecedents[j].mask[i])
    if (k < 0) k = list.length
    counts[k][y[i]]++
  }
  return counts
}

/** One state of the sampler. */
export type BrlStep = {
  /** The step number (0 is the start). */
  t: number
  /** The current list after the step. */
  list: number[]
  /** Its log posterior. */
  logPosterior: number
  /** The kind of proposal of this step (`start` at step 0). */
  proposal: 'start' | 'move' | 'add' | 'remove'
  /** The proposed list. */
  proposed: number[]
  /** Its log posterior. */
  proposedLogPosterior: number
  /** Whether the proposal was accepted. */
  accepted: boolean
  /** The best list visited so far. */
  best: number[]
  /** Its log posterior. */
  bestLogPosterior: number
}

/**
 * Metropolis-Hastings over decision lists (see the file comment): every state, the start first. Each step draws from
 * its own child stream. A plain function of plain data and a stream, so it can run in a worker (about 70 ms for 2000
 * steps on 2201 rows).
 *
 * @param antecedents The pre-mined antecedents.
 * @param y The class of each row.
 * @param classes The number of classes $K$.
 * @param s The random stream.
 * @param options The prior, the number of `steps` (default 2000) and the `start` list (default empty).
 * @returns The $\text{steps} + 1$ states.
 *
 * @example A short chain on a toy table
 * const rows = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 0], [1, 0], [0, 1]]
 * const y = [0, 0, 1, 1, 1, 0, 1, 0]
 * const antecedents = mineAntecedents(rows, nominalConditions([2, 2]), { minSupport: 0.2 })
 * const states = bayesianRuleListTrace(antecedents, y, 2, stream(1), { steps: 50 })
 * const last = states[states.length - 1]
 * print('accepted:', states.filter((s) => s.accepted).length - 1, 'of 50')
 * print('best list:', last.best.map((j) => ruleText(antecedents[j].conditions)), last.bestLogPosterior)
 */
export function bayesianRuleListTrace(
  antecedents: readonly Antecedent[],
  y: ArrayLike<number>,
  classes: number,
  s: Stream,
  options: BrlPrior & { steps?: number; start?: number[] } = {},
): BrlStep[] {
  const A = antecedents.length
  const score = (l: number[]) => ruleListLogPosterior(l, antecedents, y, classes, options).logPosterior
  let list = (options.start ?? []).slice()
  let lp = score(list)
  let best = list.slice()
  let bestLp = lp
  const out: BrlStep[] = [
    {
      t: 0,
      list,
      logPosterior: lp,
      proposal: 'start',
      proposed: list,
      proposedLogPosterior: lp,
      accepted: true,
      best,
      bestLogPosterior: bestLp,
    },
  ]
  const kinds = (l: number[]) => {
    const k: ('move' | 'add' | 'remove')[] = []
    if (l.length >= 2) k.push('move')
    if (l.length < A) k.push('add')
    if (l.length >= 1) k.push('remove')
    return k
  }
  const steps = options.steps ?? 2000
  for (let t = 1; t <= steps; t++) {
    const r = child(s, 'step', t)
    const possible = kinds(list)
    const kind = possible[integers(r, possible.length)]
    const proposed = list.slice()
    let logQForward = -Math.log(possible.length)
    let logQBack = 0
    const m = list.length
    if (kind === 'move') {
      const from = integers(r, m)
      let to = integers(r, m - 1)
      if (to >= from) to++
      const [a] = proposed.splice(from, 1)
      proposed.splice(to, 0, a)
      logQForward += -Math.log(m * (m - 1))
      logQBack = -Math.log(kinds(proposed).length) - Math.log(m * (m - 1))
    } else if (kind === 'add') {
      const unused = Array.from({ length: A }, (_, j) => j).filter((j) => !list.includes(j))
      const a = unused[integers(r, unused.length)]
      const at = integers(r, m + 1)
      proposed.splice(at, 0, a)
      logQForward += -Math.log((A - m) * (m + 1))
      logQBack = -Math.log(kinds(proposed).length) - Math.log(m + 1)
    } else {
      const at = integers(r, m)
      proposed.splice(at, 1)
      logQForward += -Math.log(m)
      logQBack = -Math.log(kinds(proposed).length) - Math.log((A - (m - 1)) * m)
    }
    const plp = score(proposed)
    const logRatio = plp - lp + logQBack - logQForward
    const accepted = Math.log(uniform(r) as number) < logRatio
    if (accepted) {
      list = proposed
      lp = plp
      if (lp > bestLp) {
        best = list.slice()
        bestLp = lp
      }
    }
    out.push({
      t,
      list,
      logPosterior: lp,
      proposal: kind,
      proposed,
      proposedLogPosterior: plp,
      accepted,
      best,
      bestLogPosterior: bestLp,
    })
  }
  return out
}

/**
 * The posterior predictive class probabilities of each rule of a list,
 * $(\alpha + N_{j,l}) / \sum_k (\alpha + N_{j,k})$, the default rule last.
 *
 * @param counts The captured class counts of each rule, as `ruleListLogPosterior` returns them.
 * @param alpha The Dirichlet pseudo-count $\alpha$ per class.
 * @returns One probability vector per rule.
 *
 * @example Probabilities from counts
 * print(ruleProbabilities([[1329, 338], [16, 124]]))
 */
export function ruleProbabilities(counts: readonly number[][], alpha = 1): number[][] {
  return counts.map((c) => {
    const total = c.reduce((a, b) => a + b + alpha, 0)
    return c.map((v) => (v + alpha) / total)
  })
}

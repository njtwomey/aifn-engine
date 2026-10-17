/**
 * Heuristics that rank a candidate rule by the counts it covers: $p$ of the $P$ positive and $n$ of the $N$ negative
 * training rows, the positives being the rows of the class the rule predicts.
 *
 * Every measure is a function of $(p, n)$ for fixed $(P, N)$, so it is a field on coverage space, the rectangle
 * $[0, N] \times [0, P]$ of Fürnkranz and Flach (2005), "ROC 'n' rule learning", Machine Learning 58, and its level
 * sets are the measure's isometrics. The measures: coverage $(p + n)/(P + N)$; precision $p/(p + n)$; the Laplace
 * estimate $(p + 1)/(p + n + k)$ for $k$ classes (Clark and Boswell, 1991); the m-estimate
 * $(p + m P/(P + N))/(p + n + m)$ (Cestnik, 1990); rule accuracy $(p + N - n)/(P + N)$, which orders rules as
 * $p - n$; weighted relative accuracy $(pN - nP)/(P + N)^2$ (Lavrač, Flach and Zupan, 1999; the Klösgen measure with
 * $a = 1$ of `aifn-compute/learning/subgroups`); FOIL's information gain
 * $p\,(\log_2 \frac{p}{p + n} - \log_2 \frac{p_0}{p_0 + n_0})$ against a parent rule covering $(p_0, n_0)$, the empty
 * rule $(P, N)$ by default (Quinlan, 1990; `foilGain` of `aifn-compute/logic/induction`); CN2's likelihood-ratio
 * statistic $2\,[p \ln(p/e_p) + n \ln(n/e_n)]$ with expected counts $e_p = (p + n)P/(P + N)$ and
 * $e_n = (p + n)N/(P + N)$ (Clark and Niblett, 1989); and RIPPER's pruning value $(p - n)/(p + n)$ (Cohen, 1995).
 * The two-class forms are given; the Laplace estimate takes the class count as an option. A rule covering nothing
 * scores $-\infty$ on every measure but coverage and the likelihood ratio (both 0). Larger is better for every
 * measure.
 */

import { foilGain } from 'aifn-compute/logic/induction'

/** The counts a rule is scored by: the positives $p$ and negatives $n$ it covers, out of $P$ and $N$. */
export type RuleCounts = {
  /** Positive rows covered, $0 \le p \le P$. */
  p: number
  /** Negative rows covered, $0 \le n \le N$. */
  n: number
  /** Positive rows in all. */
  P: number
  /** Negative rows in all. */
  N: number
}

/** The measures `ruleQuality` knows. */
export type RuleMeasure =
  | 'coverage'
  | 'precision'
  | 'laplace'
  | 'm-estimate'
  | 'accuracy'
  | 'wracc'
  | 'foil-gain'
  | 'likelihood-ratio'
  | 'ripper-prune'

/** Options of the measures that take parameters. */
export type RuleMeasureOptions = {
  /**
   * The m-estimate's pseudo-count $m \ge 0$ (default 2). $m = 0$ is precision; as $m \to \infty$ it tends to the
   * prior $P/(P + N)$.
   */
  m?: number
  /** The Laplace estimate's number of classes $k$ (default 2). */
  classes?: number
  /** FOIL gain's parent rule counts $(p_0, n_0)$ (default the empty rule, $(P, N)$). */
  parent?: { p: number; n: number }
}

/** A measure's key, its display name and its formula in TeX, for controls and tables. */
export type RuleMeasureInfo = {
  /** The key `ruleQuality` takes. */
  key: RuleMeasure
  /** The display name, plain text. */
  name: string
  /** The formula in TeX, as a function of $p$, $n$, $P$ and $N$. */
  tex: string
}

/** The measures of `ruleQuality` in a fixed order, each with a display name and its formula in TeX. */
export const RULE_MEASURES: readonly RuleMeasureInfo[] = [
  { key: 'coverage', name: 'coverage', tex: '\\frac{p + n}{P + N}' },
  { key: 'precision', name: 'precision', tex: '\\frac{p}{p + n}' },
  { key: 'laplace', name: 'Laplace estimate', tex: '\\frac{p + 1}{p + n + 2}' },
  { key: 'm-estimate', name: 'm-estimate', tex: '\\frac{p + m P/(P + N)}{p + n + m}' },
  { key: 'accuracy', name: 'rule accuracy', tex: '\\frac{p + N - n}{P + N}' },
  {
    key: 'wracc',
    name: 'weighted relative accuracy',
    tex: '\\frac{p + n}{P + N} \\left(\\frac{p}{p + n} - \\frac{P}{P + N}\\right)',
  },
  {
    key: 'foil-gain',
    name: 'FOIL gain',
    tex: 'p \\left(\\log_2 \\frac{p}{p + n} - \\log_2 \\frac{P}{P + N}\\right)',
  },
  {
    key: 'likelihood-ratio',
    name: 'likelihood ratio (CN2)',
    tex: '2 \\left[p \\ln \\frac{p}{e_p} + n \\ln \\frac{n}{e_n}\\right]',
  },
  { key: 'ripper-prune', name: 'RIPPER pruning value', tex: '\\frac{p - n}{p + n}' },
]

/**
 * The Laplace estimate $(p + 1)/(p + n + k)$: precision with one pseudo-count per class; $-\infty$ when nothing is
 * covered.
 *
 * @param options The rule's counts $p$ and $n$ (the totals are not used).
 * @param options.p The positives the rule covers.
 * @param options.n The negatives the rule covers.
 * @param classes The number of classes $k$.
 * @returns The estimate, in $(0, 1)$.
 *
 * @example First-class women on the Titanic: 141 survivors, 4 deaths
 * print('Laplace =', laplace({ p: 141, n: 4, P: 711, N: 1490 }))
 * print('142/147 =', 142 / 147)
 */
export function laplace({ p, n }: RuleCounts, classes = 2): number {
  return p + n > 0 ? (p + 1) / (p + n + classes) : -Infinity
}

/**
 * The m-estimate $(p + m\pi)/(p + n + m)$ with prior $\pi = P/(P + N)$: precision shrunk towards the prior;
 * $-\infty$ when nothing is covered.
 *
 * @param options The rule's counts and the totals that give the prior.
 * @param options.p The positives the rule covers.
 * @param options.n The negatives the rule covers.
 * @param options.P The positives in the data.
 * @param options.N The negatives in the data.
 * @param m The pseudo-count $m \ge 0$: 0 is precision, a large $m$ the prior.
 * @returns The estimate.
 *
 * @example Six children, all survivors: the estimate is pulled to the prior as m grows
 * const counts = { p: 6, n: 0, P: 711, N: 1490 }
 * for (const m of [0, 2, 10, 1000]) print('m =', m, ' estimate =', mEstimate(counts, m))
 */
export function mEstimate({ p, n, P, N }: RuleCounts, m = 2): number {
  if (p + n === 0) return -Infinity
  const prior = P + N > 0 ? P / (P + N) : 0
  return (p + m * prior) / (p + n + m)
}

/**
 * Rule accuracy $(p + N - n)/(P + N)$: the accuracy of predicting positive inside the rule and negative outside it;
 * $-\infty$ when nothing is covered.
 *
 * @param options The rule's counts and the totals.
 * @param options.p The positives the rule covers.
 * @param options.n The negatives the rule covers.
 * @param options.P The positives in the data.
 * @param options.N The negatives in the data.
 * @returns The accuracy, in $[0, 1]$.
 *
 * @example Women on the Titanic as a rule for survival
 * print(ruleAccuracy({ p: 344, n: 126, P: 711, N: 1490 }))
 */
export function ruleAccuracy({ p, n, P, N }: RuleCounts): number {
  return p + n > 0 ? (p + N - n) / (P + N) : -Infinity
}

/**
 * CN2's likelihood-ratio statistic $2\,[p \ln(p/e_p) + n \ln(n/e_n)]$ of the covered class counts against the counts
 * expected from the prior; asymptotically $\chi^2$ with one degree of freedom (two classes) when the rule's coverage
 * is independent of the class. A term with a zero count is 0, and a rule covering nothing scores 0.
 *
 * @param options The rule's counts and the totals.
 * @param options.p The positives the rule covers.
 * @param options.n The negatives the rule covers.
 * @param options.P The positives in the data.
 * @param options.N The negatives in the data.
 * @returns The statistic, at least 0.
 *
 * @example Zero for the universal rule, large for a pure one
 * print('universal:', likelihoodRatio({ p: 711, n: 1490, P: 711, N: 1490 }))
 * print('women:', likelihoodRatio({ p: 344, n: 126, P: 711, N: 1490 }))
 */
export function likelihoodRatio({ p, n, P, N }: RuleCounts): number {
  const c = p + n
  if (c === 0 || P + N === 0) return 0
  const ep = (c * P) / (P + N)
  const en = (c * N) / (P + N)
  const term = (o: number, e: number) => (o > 0 && e > 0 ? o * Math.log(o / e) : 0)
  return 2 * (term(p, ep) + term(n, en))
}

/**
 * RIPPER's pruning value $(p - n)/(p + n)$, computed on the pruning set; $-\infty$ when nothing is covered.
 *
 * @param options The rule's counts (the totals are not used).
 * @param options.p The positives the rule covers.
 * @param options.n The negatives the rule covers.
 * @returns The value, in $[-1, 1]$.
 *
 * @example 23 positives and 1 negative on the pruning set
 * print(ripperPruneValue({ p: 23, n: 1, P: 0, N: 0 }))
 */
export function ripperPruneValue({ p, n }: RuleCounts): number {
  return p + n > 0 ? (p - n) / (p + n) : -Infinity
}

/**
 * FOIL gain from counts, on `foilGain` of `aifn-compute/logic/induction` with $t = p$. A rule that keeps no positive
 * but covers a negative gains 0 bits ($p = 0$ positives times a finite or infinite loss, read as 0), and a rule
 * covering nothing, or a parent without positives, gains $-\infty$.
 *
 * @param options The rule's counts and the totals.
 * @param options.p The positives the rule covers.
 * @param options.n The negatives the rule covers.
 * @param options.P The positives in the data.
 * @param options.N The negatives in the data.
 * @param parent The parent rule's counts $(p_0, n_0)$; left out, the empty rule $(P, N)$.
 * @returns The gain in bits.
 */
function foilGainOfCounts({ p, n, P, N }: RuleCounts, parent: { p: number; n: number } = { p: P, n: N }): number {
  if (p === 0) return n > 0 ? 0 : -Infinity
  return foilGain(parent.p, parent.n, p, n, p)
}

/**
 * The quality of a rule under a named measure (see the file comment for the formulas). Coverage is
 * $(p + n)/(P + N)$, precision $p/(p + n)$, WRAcc $(pN - nP)/(P + N)^2$; FOIL gain is $-\infty$ for an empty rule and
 * 0 for one that covers negatives only.
 *
 * @param measure The measure.
 * @param counts The covered positives and negatives and the totals.
 * @param options The m-estimate's $m$, the Laplace class count and FOIL's parent rule.
 * @returns The quality; larger is better for every measure.
 *
 * @example Every measure on one rule (women on the Titanic)
 * const counts = { p: 344, n: 126, P: 711, N: 1490 }
 * for (const m of RULE_MEASURES) print(m.name, ruleQuality(m.key, counts))
 *
 * @example Precision is constant along a line through the origin of coverage space
 * const T = { P: 711, N: 1490 }
 * print('p = 10, n = 5:', ruleQuality('precision', { p: 10, n: 5, ...T }))
 * print('p = 20, n = 10:', ruleQuality('precision', { p: 20, n: 10, ...T }))
 */
export function ruleQuality(measure: RuleMeasure, counts: RuleCounts, options: RuleMeasureOptions = {}): number {
  const { p, n, P, N } = counts
  switch (measure) {
    case 'coverage':
      return P + N > 0 ? (p + n) / (P + N) : 0
    case 'precision':
      return p + n > 0 ? p / (p + n) : -Infinity
    case 'laplace':
      return laplace(counts, options.classes ?? 2)
    case 'm-estimate':
      return mEstimate(counts, options.m ?? 2)
    case 'accuracy':
      return ruleAccuracy(counts)
    case 'wracc':
      return p + n > 0 ? (p * N - n * P) / ((P + N) * (P + N)) : -Infinity
    case 'foil-gain':
      return foilGainOfCounts(counts, options.parent)
    case 'likelihood-ratio':
      return likelihoodRatio(counts)
    case 'ripper-prune':
      return ripperPruneValue(counts)
  }
}

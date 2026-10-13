/**
 * The schedules of successive halving (Jamieson and Talwalkar, 2016) and Hyperband (Li et al., 2018): how many
 * configurations each rung evaluates, at what budget, and how many it promotes.
 *
 * A rung $k$ of successive halving with $n$ configurations, minimum budget $r$, maximum $R$ and reduction factor
 * $\eta$ evaluates $n_k$ configurations at budget $r_k = r\eta^{k + s}$ ($s$ the early-stopping rate, Li et al.,
 * 2020, Algorithm 1) and keeps the best $n_{k+1} = \lfloor n_k / \eta \rfloor$; there are rungs while
 * $r_k \le R$ and $n_k \ge 1$. Hyperband (Li et al., 2018, Algorithm 1) hedges over how aggressively to halve: with
 * $s_{\max} = \lfloor \log_\eta (R / r) \rfloor$, bracket $s = s_{\max}, \dots, 0$ starts
 *
 * $$ n = \left\lceil \frac{s_{\max} + 1}{s + 1} \eta^s \right\rceil $$
 *
 * configurations at budget $R\eta^{-s}$ and runs successive halving for $s + 1$ rungs, so every bracket spends about
 * $(s_{\max} + 1)R$. A schedule is plain data; `multiFidelitySteps` runs one over given learning curves.
 */

import { DomainError } from 'aifn-compute/foundation/errors'

/** One rung of a bracket. */
export interface Rung {
  /** The rung's index $k$ within its bracket, from 0. */
  rung: number
  /** The number of configurations $n_k$ evaluated at the rung. */
  configurations: number
  /** The budget $r_k$ each is evaluated at (epochs, samples: the unit of the learning curves). */
  budget: number
  /** How many of them are promoted to the next rung, $\lfloor n_k / \eta \rfloor$ (0 at the last rung). */
  keep: number
}

/** One bracket: a run of successive halving. */
export interface Bracket {
  /** The bracket's index in the schedule, from 0. */
  bracket: number
  /** Hyperband's $s$: the bracket has $s + 1$ rungs when it is not cut short (0 for plain successive halving). */
  s: number
  /** Its rungs, in order. */
  rungs: Rung[]
  /** The budget it spends when every evaluation starts from scratch, $\sum_k n_k r_k$. */
  budget: number
  /** The budget it spends when a promoted configuration resumes its training, $\sum_k n_k (r_k - r_{k-1})$. */
  resumedBudget: number
}

/** A schedule: one bracket for successive halving, $s_{\max} + 1$ for Hyperband. */
export interface MultiFidelitySchedule {
  /** Which method made it. */
  kind: 'successive-halving' | 'hyperband'
  /** The reduction factor $\eta$. */
  eta: number
  /** The minimum budget $r$. */
  minBudget: number
  /** The maximum budget $R$. */
  maxBudget: number
  /** The brackets, in the order they are run. */
  brackets: Bracket[]
  /** The rungs of the first bracket (the only one for successive halving). */
  rungs: Rung[]
  /** The number of configurations started, over every bracket. */
  configurations: number
  /** The total budget from scratch, over every bracket. */
  budget: number
  /** The total budget when promoted configurations resume, over every bracket. */
  resumedBudget: number
}

/**
 * The largest $s$ with $\eta^s \le \rho$, without the rounding of $\log \rho / \log \eta$.
 *
 * @param rho The ratio $\rho = R / r$ of the budgets, at least 1.
 * @param eta The reduction factor $\eta > 1$.
 * @returns $s = \lfloor \log_\eta \rho \rfloor$.
 */
function floorLog(rho: number, eta: number): number {
  let s = 0
  while (eta ** (s + 1) <= rho * (1 + 1e-12)) s++
  return s
}

/**
 * Checks the shared arguments of the schedules; throws `DomainError` unless $\eta > 1$ and $0 < r \le R$.
 *
 * @param where The caller's name, for error messages.
 * @param minBudget The minimum budget $r$.
 * @param maxBudget The maximum budget $R$.
 * @param eta The reduction factor $\eta$.
 */
function check(where: string, minBudget: number, maxBudget: number, eta: number): void {
  if (!(eta > 1)) throw new DomainError(where, `${where}: eta must exceed 1`)
  if (!(minBudget > 0)) throw new DomainError(where, `${where}: minBudget must be positive`)
  if (!(maxBudget >= minBudget)) throw new DomainError(where, `${where}: maxBudget must be at least minBudget`)
}

/**
 * The rungs of one bracket: $n_k$ halved by $\eta$ (rounding down), at the given budgets, while $k \le K$ and
 * $n_k \ge 1$.
 *
 * @param index The bracket's index in its schedule.
 * @param s The bracket's Hyperband $s$ (0 for plain successive halving).
 * @param n The configurations of the first rung.
 * @param budgetAt The budget $r_k$ of rung $k$.
 * @param last The index $K$ of the last possible rung.
 * @param eta The reduction factor.
 * @returns The bracket.
 */
function bracketOf(
  index: number,
  s: number,
  n: number,
  budgetAt: (k: number) => number,
  last: number,
  eta: number,
): Bracket {
  const rungs: Rung[] = []
  let count = n
  for (let k = 0; k <= last && count >= 1; k++) {
    const keep = k < last ? Math.floor(count / eta) : 0
    rungs.push({ rung: k, configurations: count, budget: budgetAt(k), keep })
    count = keep
  }
  // A rung that keeps nobody is the last one.
  if (rungs.length > 0) rungs[rungs.length - 1].keep = 0
  let budget = 0
  let resumedBudget = 0
  rungs.forEach((g, k) => {
    budget += g.configurations * g.budget
    resumedBudget += g.configurations * (g.budget - (k > 0 ? rungs[k - 1].budget : 0))
  })
  return { bracket: index, s, rungs, budget, resumedBudget }
}

/**
 * The schedule of successive halving (Jamieson and Talwalkar, 2016) in the form of Li et al. (2020), Algorithm 1:
 * $n$ configurations at budget $r\eta^s$, the best $\lfloor n_k / \eta \rfloor$ promoted to $\eta$ times the budget,
 * until the budget would pass $R$ or one configuration is left. Throws `DomainError` unless $n \ge 1$ is an integer,
 * $\eta > 1$, $0 < r \le R$ and $s$ is a non-negative integer with $r\eta^s \le R$.
 *
 * @param options The run.
 * @param options.n The number of configurations started, $n$.
 * @param options.minBudget The minimum budget $r$ (default 1).
 * @param options.maxBudget The maximum budget $R$.
 * @param options.eta The reduction factor $\eta$ (default 3).
 * @param options.earlyStoppingRate The early-stopping rate $s$ (default 0): the first rung's budget is $r\eta^s$.
 * @returns The schedule, with one bracket.
 *
 * @example 27 configurations from budget 1 to 27, thirds kept
 * const sh = successiveHalvingSchedule({ n: 27, minBudget: 1, maxBudget: 27, eta: 3 })
 * for (const g of sh.rungs) print('rung', g.rung, ':', g.configurations, 'configurations at budget', g.budget)
 * print('total budget:', sh.budget, ' (resuming:', sh.resumedBudget, ')')
 */
export function successiveHalvingSchedule({
  n,
  minBudget = 1,
  maxBudget,
  eta = 3,
  earlyStoppingRate = 0,
}: {
  n: number
  minBudget?: number
  maxBudget: number
  eta?: number
  earlyStoppingRate?: number
}): MultiFidelitySchedule {
  const where = 'successiveHalvingSchedule'
  check(where, minBudget, maxBudget, eta)
  if (!(Number.isInteger(n) && n >= 1)) throw new DomainError(where, `${where}: n must be a positive integer`)
  const sMax = floorLog(maxBudget / minBudget, eta)
  if (!(Number.isInteger(earlyStoppingRate) && earlyStoppingRate >= 0 && earlyStoppingRate <= sMax))
    throw new DomainError(where, `${where}: earlyStoppingRate must be an integer from 0 to ${sMax}`)
  const b = bracketOf(0, 0, n, (k) => minBudget * eta ** (k + earlyStoppingRate), sMax - earlyStoppingRate, eta)
  return {
    kind: 'successive-halving',
    eta,
    minBudget,
    maxBudget,
    brackets: [b],
    rungs: b.rungs,
    configurations: n,
    budget: b.budget,
    resumedBudget: b.resumedBudget,
  }
}

/**
 * The brackets of Hyperband (Li et al., 2018, Algorithm 1): for $s = s_{\max}, \dots, 0$, successive halving from
 * $\lceil (s_{\max} + 1)\eta^s / (s + 1) \rceil$ configurations at budget $R\eta^{-s}$, with
 * $s_{\max} = \lfloor \log_\eta (R / r) \rfloor$, the formula of Algorithm 1 (which BOHB uses too). With $R = 81$,
 * $r = 1$ and $\eta = 3$ the brackets start 81, 34, 15, 8 and 5 configurations; the table Li et al. (2018) print in
 * Section 3.3 lists 81, 27, 9, 6 and 5, which Algorithm 1 does not give, and this function follows Algorithm 1.
 * Throws `DomainError` unless $\eta > 1$ and $0 < r \le R$.
 *
 * @param options The run.
 * @param options.maxBudget The maximum budget $R$ any configuration gets.
 * @param options.minBudget The smallest budget $r$ worth evaluating (default 1).
 * @param options.eta The reduction factor $\eta$ (default 3).
 * @returns The schedule, most aggressive bracket first.
 *
 * @example The brackets for R = 81 and eta = 3, by Algorithm 1: 81, 34, 15, 8 and 5 configurations
 * const hb = hyperbandSchedule({ maxBudget: 81, eta: 3 })
 * for (const b of hb.brackets)
 *   print('s =', b.s, ' n_i =', b.rungs.map((g) => g.configurations), ' r_i =', b.rungs.map((g) => g.budget))
 * print('configurations:', hb.configurations, ' total budget:', hb.budget)
 */
export function hyperbandSchedule({
  maxBudget,
  minBudget = 1,
  eta = 3,
}: {
  maxBudget: number
  minBudget?: number
  eta?: number
}): MultiFidelitySchedule {
  check('hyperbandSchedule', minBudget, maxBudget, eta)
  const sMax = floorLog(maxBudget / minBudget, eta)
  const brackets: Bracket[] = []
  for (let s = sMax; s >= 0; s--) {
    const n = Math.ceil(((sMax + 1) / (s + 1)) * eta ** s - 1e-9)
    // Budgets counted down from R, so that the last rung is R exactly.
    brackets.push(bracketOf(sMax - s, s, n, (k) => maxBudget / eta ** (s - k), s, eta))
  }
  return {
    kind: 'hyperband',
    eta,
    minBudget,
    maxBudget,
    brackets,
    rungs: brackets[0].rungs,
    configurations: brackets.reduce((a, b) => a + b.rungs[0].configurations, 0),
    budget: brackets.reduce((a, b) => a + b.budget, 0),
    resumedBudget: brackets.reduce((a, b) => a + b.resumedBudget, 0),
  }
}

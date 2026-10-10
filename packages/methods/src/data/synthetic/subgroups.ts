/**
 * Tables with planted patterns, for subgroup discovery and exceptional model mining. Every attribute is drawn
 * independently; the targets depend on the attributes only through the planted descriptions, so the planted subgroups
 * are the truth a search should find (up to the cut points its discretisation can express).
 *
 * - `plantedSubgroups`: a health-style population where `age ≥ 50 ∧ smoker = yes` has an elevated outcome rate and a
 *   higher cost, and `region = north ∧ exercise = none` a milder elevation of the rate;
 * - `plantedModelFlip`: two numeric targets whose correlation (and regression slope) flips sign inside
 *   `group = b ∧ level ≥ 5`, and a label whose logistic dependence on x, and a binary pair whose association, flip
 *   inside `flag = yes ∧ region = east`.
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import type { Description } from 'aifn-compute/learning/subgroups'
import type { TableData } from '../types'

/**
 * One of `levels`, chosen by a uniform draw with the given weights (inverse CDF).
 *
 * @param u A uniform draw in $[0, 1)$.
 * @param levels The values to choose from.
 * @param weights One non-negative weight per level, not necessarily summing to 1 (default all equal).
 * @returns The level whose share of the cumulative weight contains `u`.
 */
const pick = <T>(u: number, levels: readonly T[], weights?: readonly number[]): T => {
  const w = weights ?? levels.map(() => 1)
  let v = u * w.reduce((a, b) => a + b, 0)
  for (let j = 0; j < levels.length; j++) {
    v -= w[j]
    if (v < 0) return levels[j]
  }
  return levels[levels.length - 1]
}
/**
 * A number rounded to one decimal place.
 *
 * @param v The number.
 * @returns `v` to the nearest tenth.
 */
const round1 = (v: number) => Math.round(v * 10) / 10
/**
 * Throws `DomainError` unless the table size is an integer of at least 20.
 *
 * @param n The number of rows asked for.
 * @param where The caller's name, for the error message.
 */
const check = (n: number, where: string) => {
  if (!(Number.isInteger(n) && n >= 20)) throw new DomainError(where, `${where}: n must be an integer ≥ 20`)
}

/** Options of `plantedSubgroups`. */
export interface PlantedSubgroupsOptions {
  /** Rows (default 800). */
  n?: number
  /** Outcome rate outside the planted subgroups (default 0.2). */
  baseRate?: number
  /** Outcome rate inside `age ≥ 50 ∧ smoker = yes` (default 0.7). */
  rate?: number
  /** Outcome rate inside `region = north ∧ exercise = none` (default 0.45). */
  secondaryRate?: number
  /**
   * Cost increase inside `age ≥ 50 ∧ smoker = yes`, in population standard deviations of the base cost (default 1.5).
   */
  costShift?: number
}

/** The main subgroup planted by `plantedSubgroups`, `age ≥ 50 ∧ smoker = yes`: a higher outcome rate and cost. */
export const PLANTED_MAIN: Description = [
  { attribute: 'age', op: '≥', value: 50 },
  { attribute: 'smoker', op: '=', value: 'yes' },
]
/** The secondary subgroup planted by `plantedSubgroups`, `region = north ∧ exercise = none`: a higher outcome rate. */
export const PLANTED_SECONDARY: Description = [
  { attribute: 'region', op: '=', value: 'north' },
  { attribute: 'exercise', op: '=', value: 'none' },
]

/**
 * A population of $n$ people: age (18–85), smoker, sex, region, exercise, BMI and income, with a binary `outcome`
 * and a numeric `cost`. The outcome rate is `rate` inside `age ≥ 50 ∧ smoker = yes`, `secondaryRate` inside
 * `region = north ∧ exercise = none` (the larger where both hold) and `baseRate` elsewhere; the cost is higher by
 * `costShift` base standard deviations inside the first. The attributes are independent: age uniform over whole years,
 * 30% smokers, sexes and regions equally likely, exercise `none`, `weekly` and `daily` in shares 0.3, 0.4 and 0.3, BMI
 * $\Gauss(27, 4^2)$ and income skewed over 15 to 100; the base cost is $\Gauss(1000, 250^2)$, rounded. Throws
 * `DomainError` when $n$ is not an integer of at least 20.
 *
 * @param s The stream the attributes (child `'attributes'`), the BMI and cost noise (child `'noise'`) and the outcomes
 *   (child `'outcome'`) are drawn from.
 * @param options The size, the three outcome rates and the cost shift.
 * @returns A table with the nine columns, the targets `outcome` and `cost`, and the two planted patterns.
 *
 * @example The planted subgroup's outcome rate and cost stand out
 * const d = plantedSubgroups(stream(1), { n: 2000 })
 * const { age, smoker, outcome, cost } = d.table
 * print('columns:', Object.keys(d.table), ' targets:', d.targets)
 * print('first row:', Object.keys(d.table).map((c) => d.table[c][0]))
 * const inside = age.map((a, i) => a >= 50 && smoker[i] === 'yes')
 * const mean = (v, keep) => v.filter((_, i) => keep(i)).reduce((a, x, _, all) => a + x / all.length, 0)
 * print('outcome rate inside:', mean(outcome, (i) => inside[i]), ' outside:', mean(outcome, (i) => !inside[i]))
 * print('mean cost inside:', mean(cost, (i) => inside[i]), ' outside:', mean(cost, (i) => !inside[i]))
 */
export function plantedSubgroups(s: Stream, options: PlantedSubgroupsOptions = {}): TableData {
  const { n = 800, baseRate = 0.2, rate = 0.7, secondaryRate = 0.45, costShift = 1.5 } = options
  check(n, 'plantedSubgroups')
  const u = units(child(s, 'attributes'), 7 * n)
  const z = standardNormals(child(s, 'noise'), 2 * n)
  const v = units(child(s, 'outcome'), n)
  const age = Array.from({ length: n }, (_, i) => 18 + Math.floor(68 * u[i]))
  const smoker = Array.from({ length: n }, (_, i) => (u[n + i] < 0.3 ? 'yes' : 'no'))
  const sex = Array.from({ length: n }, (_, i) => (u[2 * n + i] < 0.5 ? 'female' : 'male'))
  const region = Array.from({ length: n }, (_, i) => pick(u[3 * n + i], ['north', 'south', 'east', 'west']))
  const exercise = Array.from({ length: n }, (_, i) => pick(u[4 * n + i], ['none', 'weekly', 'daily'], [0.3, 0.4, 0.3]))
  const bmi = Array.from({ length: n }, (_, i) => round1(27 + 4 * z[i]))
  const income = Array.from({ length: n }, (_, i) => round1(15 + 85 * u[5 * n + i] ** 1.5))
  const main = age.map((a, i) => a >= 50 && smoker[i] === 'yes')
  const second = region.map((r, i) => r === 'north' && exercise[i] === 'none')
  const p = main.map((m, i) => Math.max(m ? rate : baseRate, second[i] ? secondaryRate : baseRate))
  const outcome = p.map((pi, i) => (v[i] < pi ? 1 : 0))
  const cost = main.map((m, i) => Math.round(1000 + 250 * z[n + i] + (m ? 250 * costShift : 0)))
  return {
    kind: 'table',
    table: { age, smoker, sex, region, exercise, bmi, income, outcome, cost },
    targets: ['outcome', 'cost'],
    planted: [
      {
        description: PLANTED_MAIN,
        targets: ['outcome', 'cost'],
        effect: `outcome rate ${rate} (elsewhere ${baseRate}); cost higher by ${costShift} sd`,
      },
      {
        description: PLANTED_SECONDARY,
        targets: ['outcome'],
        effect: `outcome rate ${secondaryRate} (elsewhere ${baseRate})`,
      },
    ],
    meta: {
      name: 'planted subgroups',
      description: `${n} people with seven independent attributes; the outcome rate is ${rate} when age ≥ 50 and smoker = yes, ${secondaryRate} when region = north and exercise = none, and ${baseRate} otherwise; cost rises in the first group.`,
    },
  }
}

definer<DatasetInfo>('dataset', 'data/synthetic')(
  {
    key: 'plantedSubgroups',
    name: 'Planted subgroups',
    summary:
      'A population where age ≥ 50 ∧ smoker = yes has an elevated outcome rate and cost, and region = north ∧ exercise = none a milder one.',
    task: 'classification',
    output: 'table',
    knobs: space({
      n: int(20, 20000, { default: 800 }),
      baseRate: real(0, 1, { default: 0.2 }),
      rate: real(0, 1, { default: 0.7 }),
      secondaryRate: real(0, 1, { default: 0.45 }),
      costShift: real(0, 5, { default: 1.5 }),
    }),
    truth: true,
    random: true,
    notes: ['intrinsically-interpretable-models'],
  },
  plantedSubgroups,
)

/** Options of `plantedModelFlip`. */
export interface PlantedModelFlipOptions {
  /** Rows (default 600). */
  n?: number
  /**
   * $\rho$, the size of the correlation of `x` and `y`, in $[0, 1)$: positive outside the planted subgroup and negative
   * inside (default 0.7).
   */
  rho?: number
  /** Logistic slope of the label on x, positive outside and negative inside the second subgroup (default 2.5). */
  slope?: number
}

/** The subgroup of `plantedModelFlip` where the correlation of `x` and `y` flips, `group = b ∧ level ≥ 5`. */
export const PLANTED_CORRELATION: Description = [
  { attribute: 'group', op: '=', value: 'b' },
  { attribute: 'level', op: '≥', value: 5 },
]
/**
 * The subgroup of `plantedModelFlip` where the label's slope on `x` and the association of `a` and `b` flip,
 * `flag = yes ∧ region = east`.
 */
export const PLANTED_CLASSIFIER: Description = [
  { attribute: 'flag', op: '=', value: 'yes' },
  { attribute: 'region', op: '=', value: 'east' },
]

/**
 * Four attributes (group, level 0–10, flag, region) and five targets: `x` and `y` standard normal with correlation
 * $\rho$, flipped to $-\rho$ inside `group = b ∧ level ≥ 5`; a label with $\pr(1 \mid x) = \sigma(\beta x)$, $\beta$
 * the `slope`, negated inside `flag = yes ∧ region = east`; and two binary columns `a` and `b` that agree with
 * probability 0.85, and disagree with that probability inside the same subgroup. The attributes are independent: groups
 * `a` to `d` and regions `north`, `south`, `east` equally likely, level uniform on $[0, 10]$ to one decimal, and 40% of
 * flags `yes`. Throws `DomainError` when $n$ is not an integer of at least 20 or $\rho$ is not in $[0, 1)$.
 *
 * @param s The stream the attributes (child `'attributes'`), `x` and `y` (child `'targets'`) and the binary columns
 *   (child `'labels'`) are drawn from.
 * @param options The size, the correlation and the slope.
 * @returns A table with the nine columns, the five targets, and the two planted patterns.
 *
 * @example The correlation flips inside the planted subgroup
 * const d = plantedModelFlip(stream(1), { n: 2000, rho: 0.7 })
 * const { group, level, x, y } = d.table
 * print('columns:', Object.keys(d.table), ' first row:', Object.keys(d.table).map((c) => d.table[c][0]))
 * const corr = (keep) => {
 *   const i = x.map((_, k) => k).filter(keep)
 *   const m = (v) => i.reduce((a, k) => a + v[k], 0) / i.length
 *   const [mx, my] = [m(x), m(y)]
 *   const c = (u, mu, v, mv) => i.reduce((a, k) => a + (u[k] - mu) * (v[k] - mv), 0)
 *   return c(x, mx, y, my) / Math.sqrt(c(x, mx, x, mx) * c(y, my, y, my))
 * }
 * const inside = (k) => group[k] === 'b' && level[k] >= 5
 * print('correlation inside:', corr(inside), ' outside:', corr((k) => !inside(k)))
 */
export function plantedModelFlip(s: Stream, options: PlantedModelFlipOptions = {}): TableData {
  const { n = 600, rho = 0.7, slope = 2.5 } = options
  check(n, 'plantedModelFlip')
  if (!(rho >= 0 && rho < 1)) throw new DomainError('plantedModelFlip', 'plantedModelFlip: rho must be in [0, 1)')
  const u = units(child(s, 'attributes'), 4 * n)
  const z = standardNormals(child(s, 'targets'), 2 * n)
  const w = units(child(s, 'labels'), 3 * n)
  const group = Array.from({ length: n }, (_, i) => pick(u[i], ['a', 'b', 'c', 'd']))
  const level = Array.from({ length: n }, (_, i) => round1(10 * u[n + i]))
  const flag = Array.from({ length: n }, (_, i) => (u[2 * n + i] < 0.4 ? 'yes' : 'no'))
  const region = Array.from({ length: n }, (_, i) => pick(u[3 * n + i], ['north', 'south', 'east']))
  const first = group.map((g, i) => g === 'b' && level[i] >= 5)
  const second = flag.map((f, i) => f === 'yes' && region[i] === 'east')
  const x = Array.from({ length: n }, (_, i) => z[i])
  const y = x.map((xi, i) => (first[i] ? -rho : rho) * xi + Math.sqrt(1 - rho * rho) * z[n + i])
  const label = x.map((xi, i) => (w[i] < 1 / (1 + Math.exp(-(second[i] ? -slope : slope) * xi)) ? 1 : 0))
  const a = Array.from({ length: n }, (_, i) => (w[n + i] < 0.5 ? 1 : 0))
  const b = a.map((ai, i) => (w[2 * n + i] < 0.85 !== second[i] ? ai : 1 - ai))
  return {
    kind: 'table',
    table: { group, level, flag, region, x, y, label, a, b },
    targets: ['x', 'y', 'label', 'a', 'b'],
    planted: [
      {
        description: PLANTED_CORRELATION,
        targets: ['x', 'y'],
        effect: `correlation of x and y is −${rho} inside, +${rho} outside`,
      },
      {
        description: PLANTED_CLASSIFIER,
        targets: ['x', 'label', 'a', 'b'],
        effect: `logistic slope of label on x is −${slope} inside, +${slope} outside; a and b disagree inside`,
      },
    ],
    meta: {
      name: 'planted model flip',
      description: `${n} rows; x and y correlate at +${rho} except inside group = b ∧ level ≥ 5 (−${rho}); the label's dependence on x and the association of a and b flip inside flag = yes ∧ region = east.`,
    },
  }
}

definer<DatasetInfo>('dataset', 'data/synthetic')(
  {
    key: 'plantedModelFlip',
    name: 'Planted model flip',
    summary:
      'Two numeric targets whose correlation flips sign inside one subgroup, and a classifier and an association that flip inside another.',
    task: 'regression',
    output: 'table',
    knobs: space({
      n: int(20, 20000, { default: 600 }),
      rho: real(0, 0.99, { default: 0.7 }),
      slope: real(0, 10, { default: 2.5 }),
    }),
    truth: true,
    random: true,
    notes: ['simpsons-paradox', 'pearson-correlation'],
  },
  plantedModelFlip,
)

/**
 * Seeded families of learning curves (validation loss against budget) whose final values are known, for
 * multi-fidelity search and learning-curve extrapolation: each curve is a power law $y(t) = a + b\,t^{-c}$ or an
 * exponential $y(t) = a + b\,e^{-ct}$ (the simplest of the families of Domhan, Springenberg and Hutter, 2015), seen
 * with Gaussian noise.
 *
 * Each curve $i$ has an asymptote $a_i$ in $[0.05, 0.3]$ and a loss $s_i$ in $[0.6, 1]$ at the first budget, set
 * through a Gaussian copula: $a_i = 0.05 + 0.25\,\Phi(z_i)$ and $s_i = 0.6 + 0.4\,\Phi(\rho z_i +
 * \sqrt{1 - \rho^2}\,z'_i)$ with $\rho = 1 - 2\kappa$ for the crossing $\kappa \in [0, 1]$. At $\kappa = 0$ the curves
 * that start well end well; at $\kappa = 1$ the best curves start worst, so early stopping drops them. The rate $c$
 * is drawn per curve, and $b$ is set so that the curve passes through $s_i$ at the first budget.
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { normalCdf } from 'aifn-compute/numerics/special'
import { checkCount } from '../types'

/** The parametric form of one learning curve. */
export interface LearningCurveForm {
  /** `'power'`: $y(t) = a + b\,t^{-c}$; `'exponential'`: $y(t) = a + b\,e^{-ct}$. */
  shape: 'power' | 'exponential'
  /** The asymptote $a$, the loss as the budget grows without bound. */
  a: number
  /** The scale $b > 0$. */
  b: number
  /** The rate $c > 0$ (per unit of budget for the exponential). */
  c: number
}

/** Options of `learningCurves`. */
export interface LearningCurvesOptions {
  /** The number of curves (configurations), default 27. */
  n?: number
  /**
   * The budgets $t_1 < \dots < t_B$, all positive (default the integers 1 to 81, so that the schedules with
   * $\eta = 3$ land on them).
   */
  budgets?: Tensor | ArrayLike<number>
  /** The family: `'power'` (default), `'exponential'`, or `'mixed'` (each curve one or the other, with even odds). */
  shape?: 'power' | 'exponential' | 'mixed'
  /** The standard deviation of the Gaussian noise on each observed loss (default 0.01; 0 for clean curves). */
  noise?: number
  /** The crossing $\kappa \in [0, 1]$ (default 0.5): how strongly a good start foretells a bad end. */
  crossing?: number
}

/** A family of learning curves from `learningCurves`. */
export interface LearningCurves {
  /** The budgets, $B$ values. */
  budgets: Tensor
  /** The observed losses, $n \times B$: the clean curves plus noise. */
  values: Tensor
  /** The noise-free losses, $n \times B$. */
  clean: Tensor
  /** Each curve's noise-free loss at the largest budget, $n$ values. */
  final: Tensor
  /** Each curve's asymptote $a_i$, $n$ values. */
  asymptote: Tensor
  /** Each curve's parametric form. */
  forms: LearningCurveForm[]
  /** The curve with the lowest final loss. */
  best: number
  /**
   * The fraction of pairs of curves whose order at the first budget is reversed at the last (noise-free): 0 when no
   * two cross, 1 when every pair does.
   */
  crossings: number
}

/**
 * The loss of a learning curve at a budget.
 *
 * @param form The curve's parametric form.
 * @param t The budget, positive.
 * @returns $a + b\,t^{-c}$ or $a + b\,e^{-ct}$.
 *
 * @example A power law and an exponential through the same start
 * const p = { shape: 'power', a: 0.1, b: 0.8, c: 0.5 }
 * print('power at 1, 10, 100:', [1, 10, 100].map((t) => learningCurveAt(p, t)))
 * const e = { shape: 'exponential', a: 0.1, b: 0.8 * Math.exp(0.05), c: 0.05 }
 * print('exponential at 1, 10, 100:', [1, 10, 100].map((t) => learningCurveAt(e, t)))
 */
export function learningCurveAt(form: LearningCurveForm, t: number): number {
  return form.shape === 'power' ? form.a + form.b * t ** -form.c : form.a + form.b * Math.exp(-form.c * t)
}

/**
 * A seeded family of learning curves with known parametric forms and final values (see the file comment). Power-law
 * rates are uniform on $[0.5, 1.2]$; an exponential's rate is $c = r / (t_B - t_1)$ with $r$ uniform on $[2, 6]$, so
 * it has nearly converged by the last budget. Throws `DomainError` for a count that is not a non-negative integer,
 * budgets that are not positive and increasing, negative noise or a crossing outside $[0, 1]$.
 *
 * @param s The stream: the copula draws (child `'order'`), the shapes (`'shape'`), the rates (`'rate'`) and the noise
 *   (`'noise'`) come from its children, so changing the noise leaves the clean curves as they were.
 * @param options The count, budgets, family, noise and crossing, as `LearningCurvesOptions`.
 * @returns The curves, as `LearningCurves`: a table `multiFidelitySteps` (`aifn-methods/optimisation/multi-fidelity`)
 *   reads directly.
 *
 * @example Crossing curves: the best at the end starts among the worst
 * const c = learningCurves(stream(0), { n: 8, budgets: [1, 3, 9, 27, 81], crossing: 1, noise: 0 })
 * const first = toArray(c.clean).map((row) => row[0])
 * print('best at the end:', c.best, ' its rank at budget 1:', first.filter((v) => v < first[c.best]).length)
 * print('pairs that cross:', c.crossings.toFixed(2))
 *
 * @example No crossing, and the forms behind the curves
 * const c = learningCurves(stream(0), { n: 4, budgets: [1, 9, 81], crossing: 0 })
 * print('final:', c.final, ' asymptote:', c.asymptote)
 * print('form of curve 0:', c.forms[0])
 */
export function learningCurves(s: Stream, options: LearningCurvesOptions = {}): LearningCurves {
  const where = 'learningCurves'
  const { n = 27, shape = 'power', noise = 0.01, crossing = 0.5 } = options
  checkCount(n, where)
  const budgets =
    options.budgets === undefined
      ? Float64Array.from({ length: 81 }, (_, k) => k + 1)
      : Float64Array.from((isTensor(options.budgets) ? toFlat(options.budgets) : options.budgets) as ArrayLike<number>)
  const B = budgets.length
  if (B === 0 || !budgets.every((t, k) => t > 0 && (k === 0 || t > budgets[k - 1])))
    throw new DomainError(where, `${where}: the budgets must be positive and increasing`)
  if (!(noise >= 0)) throw new DomainError(where, `${where}: the noise must be non-negative`)
  if (!(crossing >= 0 && crossing <= 1)) throw new DomainError(where, `${where}: crossing must lie in [0, 1]`)
  if (!['power', 'exponential', 'mixed'].includes(shape))
    throw new DomainError(where, `${where}: unknown shape "${shape}"`)

  const rho = 1 - 2 * crossing
  const z = standardNormals(child(s, 'order'), 2 * n)
  const kinds = units(child(s, 'shape'), n)
  const rates = units(child(s, 'rate'), n)
  const [t1, tB] = [budgets[0], budgets[B - 1]]
  const forms: LearningCurveForm[] = Array.from({ length: n }, (_, i) => {
    const a = 0.05 + 0.25 * (normalCdf(z[2 * i]) as number)
    const start = 0.6 + 0.4 * (normalCdf(rho * z[2 * i] + Math.sqrt(1 - rho * rho) * z[2 * i + 1]) as number)
    const kind = shape === 'mixed' ? (kinds[i] < 0.5 ? 'power' : 'exponential') : shape
    if (kind === 'power') {
      const c = 0.5 + 0.7 * rates[i]
      return { shape: 'power', a, b: (start - a) * t1 ** c, c }
    }
    const c = (2 + 4 * rates[i]) / (tB > t1 ? tB - t1 : 1)
    return { shape: 'exponential', a, b: (start - a) * Math.exp(c * t1), c }
  })
  const clean = new Float64Array(n * B)
  forms.forEach((f, i) => budgets.forEach((t, k) => (clean[i * B + k] = learningCurveAt(f, t))))
  const eps = noise > 0 ? standardNormals(child(s, 'noise'), n * B) : new Float64Array(n * B)
  const values = clean.map((v, k) => v + noise * eps[k])
  const final = Float64Array.from(forms, (_, i) => clean[i * B + B - 1])
  let best = 0
  final.forEach((v, i) => {
    if (v < final[best]) best = i
  })
  let crossed = 0
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) if ((clean[i * B] - clean[j * B]) * (final[i] - final[j]) < 0) crossed++
  return {
    budgets: fromData(budgets, [B]),
    values: fromData(values, [n, B]),
    clean: fromData(clean, [n, B]),
    final: fromData(final, [n]),
    asymptote: fromData(
      Float64Array.from(forms, (f) => f.a),
      [n],
    ),
    forms,
    best,
    crossings: n > 1 ? crossed / ((n * (n - 1)) / 2) : 0,
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const fn = definer<FunctionInfo>('function', 'data/synthetic')

fn(
  {
    key: 'learningCurves',
    name: 'Synthetic learning curves',
    summary:
      'Power-law or exponential validation-loss curves with known forms and final values, and a dial for how often good starters finish badly.',
    role: 'simulation',
    random: true,
    cite: ['domhan2015'],
  },
  learningCurves,
)

fn(
  {
    key: 'learningCurveAt',
    name: 'Learning curve at a budget',
    summary: 'Evaluate a power-law or exponential learning curve a + b t^(−c) or a + b exp(−c t).',
    role: 'property',
  },
  learningCurveAt,
)

/**
 * A seeded family of learning curves indexed by the points of a search space, with the final loss a smooth known
 * function of the configuration: a test bed on which random search, Hyperband and BOHB can be compared with the true
 * optimum known.
 *
 * A configuration is read through its unit-cube coordinates $\uvec \in [0, 1]^m$ (`encode` in
 * `aifn-compute/foundation/space`, so a log-scale dimension counts in $\log \lambda$). Its loss at the maximum budget
 * $R$ is
 *
 * $$ F(\uvec) = 0.05 + 0.25\,q(\uvec), \qquad q(\uvec) = \frac{\sum_j w_j \sin^2(\pi (u_j - c_j) / 2)}{\sum_j w_j}, $$
 *
 * a bowl with its minimum $0.05$ at a drawn centre $\cvec$ (snapped to a point the space can hold) and drawn weights
 * $w_j \in [0.25, 1]$, so some dimensions matter more than others. Its loss at the minimum budget $r$ is
 * $s(\uvec) = 0.6 + 0.4\,m(\uvec)$, where, with $\rho = 1 - 2\kappa$ for the crossing $\kappa \in [0, 1]$,
 * $m = \rho q + (1 - \rho) p$ when $\rho \ge 0$ and $m = \lvert\rho\rvert (1 - q) + (1 - \lvert\rho\rvert) p$
 * otherwise, $p$ being a second bowl of the same kind with its own centre and weights. At $\kappa = 0$ the
 * configurations that end well start well; at $\kappa = 1$ the best ones start worst, so early stopping drops them.
 * Between the two budgets each curve is a power law $A + B t^{-c}$ or an exponential $A + B e^{-ct}$ (the forms of
 * `learningCurves`) through $(r, s)$ and $(R, F)$, with a rate $c$ that is a third smooth function of $\uvec$.
 * Observed losses add Gaussian noise of standard deviation `noise`.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { definer, type FunctionInfo } from 'aifn-compute/foundation/registry'
import { decode, encode, type Space, type SpaceValues, type ValuesOf } from 'aifn-compute/foundation/space'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { learningCurveAt, type LearningCurveForm } from './learning-curves'

/** Options of `learningCurveSurface`. */
export interface LearningCurveSurfaceOptions<S extends Space> {
  /** The search space: real (linear or log scale), integer, choice and boolean dimensions, none conditional. */
  space: S
  /** The crossing $\kappa \in [0, 1]$ (default 0.5): how strongly a good start foretells a bad end. */
  crossing?: number
  /** The standard deviation of the Gaussian noise on each observed loss (default 0.01; 0 for clean curves). */
  noise?: number
  /** The family of every curve: `'power'` (default) or `'exponential'`. */
  shape?: 'power' | 'exponential'
  /** The minimum budget $r > 0$ (default 1), where the curves start. */
  minBudget?: number
  /** The maximum budget $R > r$ (default 81), where the loss is $F(\uvec)$. */
  maxBudget?: number
}

/** A family of learning curves over a space, from `learningCurveSurface`. */
export interface LearningCurveSurface<S extends Space> {
  /** The search space. */
  space: S
  /** The minimum budget $r$. */
  minBudget: number
  /** The maximum budget $R$. */
  maxBudget: number
  /** The noise's standard deviation. */
  noise: number
  /** The crossing $\kappa$. */
  crossing: number
  /**
   * The observed loss of a configuration after a budget: the clean loss plus `noise` times one standard normal drawn
   * from `stream`. The stream is needed only when there is noise.
   */
  at: (point: ValuesOf<S>, budget: number, stream?: Stream) => number
  /** The noise-free loss of a configuration after a budget. */
  clean: (point: ValuesOf<S>, budget: number) => number
  /** The noise-free loss at the maximum budget, $F(\uvec)$. */
  final: (point: ValuesOf<S>) => number
  /** The parametric form of a configuration's curve (its `a` is the asymptote, which may lie below 0). */
  form: (point: ValuesOf<S>) => LearningCurveForm
  /**
   * The curves of a list of configurations, as `multiFidelitySteps` (`aifn-methods/optimisation/multi-fidelity`)
   * reads them: configuration $i$ after budget $t$ is observed with the noise of `child(stream, i, t)`.
   */
  curves: (
    points: readonly ValuesOf<S>[],
    stream: Stream,
  ) => {
    count: number
    at: (config: number, budget: number) => number
    clean: (config: number, budget: number) => number
  }
  /** The best configuration and its final loss, $0.05$. */
  optimum: { point: ValuesOf<S>; loss: number }
  /** The centre $\cvec$ of the final-loss bowl, in unit coordinates, $m$ values. */
  centre: Tensor
  /** The weights $w_j$ of the final-loss bowl, $m$ values. */
  weights: Tensor
}

/**
 * A weighted bowl $\sum_j w_j \sin^2(\pi (u_j - c_j) / 2) / \sum_j w_j$ on the unit cube, in $[0, 1]$.
 *
 * @param u The unit coordinates.
 * @param c The centre.
 * @param w The weights, positive.
 * @returns The bowl's value, 0 at the centre.
 */
function bowl(u: ArrayLike<number>, c: ArrayLike<number>, w: ArrayLike<number>): number {
  let num = 0
  let den = 0
  for (let j = 0; j < u.length; j++) {
    num += w[j] * Math.sin((Math.PI * (u[j] - c[j])) / 2) ** 2
    den += w[j]
  }
  return num / den
}

/**
 * A seeded family of learning curves over the points of a space, whose final loss is a smooth function of the
 * configuration with a known minimum (see the file comment). Noise is drawn per call from the stream it is given, so
 * a search that passes each evaluation its own stream sees reproducible noise. Throws `DomainError` for a space with a
 * conditional, nested or variants dimension or with no dimensions, a crossing outside $[0, 1]$, negative noise, or
 * budgets other than $0 < r < R$.
 *
 * @param s The stream: the centres and weights of the bowls and the rates come from its children, so changing the
 *   noise leaves the clean curves as they were.
 * @param options The space, crossing, noise, family and budgets, as `LearningCurveSurfaceOptions`.
 * @returns The surface, as `LearningCurveSurface`: the observed and clean losses, the final loss, the forms, and the
 *   optimum.
 *
 * @example The optimum, and one configuration's curve
 * import { real, space } from 'aifn-compute/foundation/space'
 * const sp = space({ lr: real(1e-4, 1, { scale: 'log' }), dropout: real(0, 0.8) })
 * const surface = learningCurveSurface(stream(0), { space: sp, noise: 0 })
 * print('optimum:', surface.optimum.point, ' final loss:', surface.optimum.loss)
 * const p = { lr: 0.01, dropout: 0.5 }
 * print('loss at budgets 1, 9, 81:', [1, 9, 81].map((t) => surface.clean(p, t).toFixed(4)))
 *
 * @example Full crossing: the best configurations start worst
 * import { real, space } from 'aifn-compute/foundation/space'
 * const surface = learningCurveSurface(stream(1), { space: space({ x: real(0, 1) }), crossing: 1, noise: 0 })
 * const xs = [0, 0.25, 0.5, 0.75, 1].map((x) => ({ x }))
 * print('at budget 1:', xs.map((p) => surface.clean(p, 1).toFixed(3)))
 * print('at budget 81:', xs.map((p) => surface.final(p).toFixed(3)))
 */
export function learningCurveSurface<const S extends Space>(
  s: Stream,
  options: LearningCurveSurfaceOptions<S>,
): LearningCurveSurface<S> {
  const where = 'learningCurveSurface'
  const { space, crossing = 0.5, noise = 0.01, shape = 'power', minBudget = 1, maxBudget = 81 } = options
  const dims = Object.entries(space.dims)
  if (dims.length === 0) throw new DomainError(where, `${where}: the space has no dimensions`)
  for (const [key, dim] of dims) {
    if (dim.when !== undefined) throw new DomainError(where, `${where}: '${key}' is conditional; not supported`)
    if (!['real', 'int', 'choice', 'bool'].includes(dim.type))
      throw new DomainError(where, `${where}: '${key}' is a ${dim.type} dimension; not supported`)
  }
  if (!(crossing >= 0 && crossing <= 1)) throw new DomainError(where, `${where}: crossing must lie in [0, 1]`)
  if (!(noise >= 0)) throw new DomainError(where, `${where}: the noise must be non-negative`)
  if (!(minBudget > 0 && maxBudget > minBudget))
    throw new DomainError(where, `${where}: the budgets must satisfy 0 < minBudget < maxBudget`)
  if (shape !== 'power' && shape !== 'exponential') throw new DomainError(where, `${where}: unknown shape "${shape}"`)

  const m = dims.length
  const unit = (p: SpaceValues) => encode(space, p)
  // The centre of the final-loss bowl, snapped to a point the space holds, so that the optimum is attained.
  const centre = unit(decode(space, units(child(s, 'centre'), m)) as SpaceValues)
  const weights = Float64Array.from(units(child(s, 'weights'), m), (v) => 0.25 + 0.75 * v)
  const startCentre = units(child(s, 'start-centre'), m)
  const startWeights = Float64Array.from(units(child(s, 'start-weights'), m), (v) => 0.25 + 0.75 * v)
  const rateCentre = units(child(s, 'rate-centre'), m)
  const rateWeights = Float64Array.from(units(child(s, 'rate-weights'), m), (v) => 0.25 + 0.75 * v)
  const rho = 1 - 2 * crossing
  const [r, R] = [minBudget, maxBudget]

  const formAt = (u: ArrayLike<number>): LearningCurveForm => {
    const q = bowl(u, centre, weights)
    const p = bowl(u, startCentre, startWeights)
    const mix = rho >= 0 ? rho * q + (1 - rho) * p : -rho * (1 - q) + (1 + rho) * p
    const end = 0.05 + 0.25 * q
    const start = 0.6 + 0.4 * mix
    const v = bowl(u, rateCentre, rateWeights)
    if (shape === 'power') {
      const c = 0.5 + 0.7 * v
      const b = (start - end) / (r ** -c - R ** -c)
      return { shape, a: end - b * R ** -c, b, c }
    }
    const c = (2 + 4 * v) / (R - r)
    const b = (start - end) / (Math.exp(-c * r) - Math.exp(-c * R))
    return { shape, a: end - b * Math.exp(-c * R), b, c }
  }
  const form = (point: ValuesOf<S>) => formAt(unit(point as SpaceValues))
  const clean = (point: ValuesOf<S>, budget: number) => learningCurveAt(form(point), budget)
  const final = (point: ValuesOf<S>) => 0.05 + 0.25 * bowl(unit(point as SpaceValues), centre, weights)
  const at = (point: ValuesOf<S>, budget: number, stream?: Stream) => {
    const v = clean(point, budget)
    if (noise === 0) return v
    if (!stream) throw new DomainError(where, `${where}: a noisy loss needs a stream`)
    return v + noise * standardNormals(stream, 1)[0]
  }
  const best = decode(space, centre) as ValuesOf<S>
  return {
    space,
    minBudget,
    maxBudget,
    noise,
    crossing,
    at,
    clean,
    final,
    form,
    curves: (points, stream) => ({
      count: points.length,
      at: (config, budget) => at(points[config], budget, child(stream, config, budget)),
      clean: (config, budget) => clean(points[config], budget),
    }),
    optimum: { point: best, loss: final(best) },
    centre: fromData(Float64Array.from(centre), [m]),
    weights: fromData(weights, [m]),
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const fn = definer<FunctionInfo>('function', 'data/synthetic')

fn(
  {
    key: 'learningCurveSurface',
    name: 'Learning curves over a search space',
    summary:
      'Learning curves indexed by configurations, with a smooth known final loss and a dial for how much they cross.',
    role: 'simulation',
    random: true,
    cite: ['domhan2015', 'falkner2018bohb'],
  },
  learningCurveSurface,
)

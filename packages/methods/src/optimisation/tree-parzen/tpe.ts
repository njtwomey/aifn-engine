/**
 * The tree-structured Parzen estimator (Bergstra, Bardenet, Bengio and Kégl, 2011) as a step-through algorithm,
 * with the defaults of Watanabe's (2023) tutorial: it splits the observations at the $\gamma$-quantile of the loss
 * into a good group and a bad group, models each with a Parzen (kernel) density, $\ell(\lambda)$ and $g(\lambda)$,
 * draws candidates from $\ell$ and evaluates the one with the largest ratio
 *
 * $$ \lambda^\star = \arg\max_{\lambda} \frac{\ell(\lambda)}{g(\lambda)}, $$
 *
 * which maximises expected improvement under the model (Bergstra et al., 2011, Section 4.1).
 *
 * The densities live on the unit-cube coordinates of the search `Space` (`encode` in
 * `aifn-compute/foundation/space`), so a log-scale dimension is modelled in $\log \lambda$ and densities are per unit
 * coordinate. A real or integer coordinate gets a Gaussian kernel truncated to $[0, 1]$ at each observation, plus a
 * prior component $\Gauss(\tfrac12, 1)$; a choice or boolean gets the Aitchison–Aitken kernel, which keeps mass
 * $\kappa = (1 + 1/N)/(1 + C/N)$ on the observed category of $C$ (Watanabe, 2023, Appendix C.3.3), with a uniform
 * prior. The prior weighs `priorWeight` and each observation 1. The kernels combine as a product per observation
 * (the multivariate TPE that Watanabe recommends) or, with `multivariate: false`, as a product of per-dimension
 * mixtures (the original TPE). A bandwidth is clipped below at $\max(\Delta, N^{-2})$ in unit coordinates
 * (Watanabe, 2023, Section 4.2.2, $\Delta = 0.03$, $\alpha = 2$), $N$ the group size plus one, and above at 1.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, units, type Stream } from 'aifn-compute/foundation/random'
import { decode, encode, sample, type Space, type SpaceValues, type ValuesOf } from 'aifn-compute/foundation/space'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Status } from 'aifn-compute/foundation/trace'
import { normalCdf, normalQuantile } from 'aifn-compute/numerics/special'

/** Options of `treeParzenSteps`. */
export interface TreeParzenOptions {
  /**
   * The fraction $\gamma$ of observations in the good group: the best $\lceil \gamma n \rceil$ of $n$ (default 0.15,
   * as Bergstra et al. and Watanabe's final recommendation).
   */
  gamma?: number
  /** The number of candidates $N_s$ drawn from $\ell$ per step (default 24, as hyperopt and Optuna). */
  candidates?: number
  /** The number of random evaluations before the model takes over, made by `init` (default 10). */
  init?: number
  /**
   * The bandwidth of the continuous kernels, in unit coordinates: `'scott'` (default),
   * $1.059 \min(\hat\sigma, \mathrm{IQR}/1.34) N^{-1/5}$ per dimension and group; `'hyperopt'`, the larger distance
   * to an observation's neighbours in that dimension (the smaller-side one at either end); or a fixed number.
   */
  bandwidth?: 'scott' | 'hyperopt' | number
  /** The floor $\Delta$ of the bandwidth clip (default 0.03; with $N^{-2}$, Watanabe's magic clipping). */
  minBandwidth?: number
  /** The weight of the prior component against 1 per observation (default 1). */
  priorWeight?: number
  /** Product kernels per observation (default true) or a product of per-dimension mixtures (false). */
  multivariate?: boolean
  /** The number of grid points of each continuous marginal in the state (default 101). */
  grid?: number
  /** The points per side of the joint grid of a two-dimensional continuous space (default 41). */
  jointGrid?: number
}

/** The densities of one dimension on a grid. */
export interface TreeParzenMarginal {
  /** The dimension's key in the space. */
  key: string
  /** `'continuous'` for a real or integer dimension, `'categorical'` for a choice or boolean. */
  kind: 'continuous' | 'categorical'
  /** The grid: unit coordinates from 0 to 1 (continuous), or the category indices (categorical). */
  at: Tensor
  /** The dimension's values at the grid points: numbers on its own scale, or the options. */
  values: (number | string | boolean)[]
  /** The good density $\ell$ at the grid points (its marginal in this dimension). */
  good: Tensor
  /** The bad density $g$ at the grid points. */
  bad: Tensor
  /** The ratio $\ell / g$ at the grid points. */
  ratio: Tensor
}

/** One state of `treeParzenSteps`. */
export interface TreeParzenState extends Status {
  /** Evaluations made after the initial ones (the step count). */
  t: number
  /** Every point evaluated, in order. */
  observations: SpaceValues[]
  /** Their unit-cube coordinates, $n \times m$ (a category at the centre of its bin). */
  u: Tensor
  /** Their losses, $n$ values. */
  losses: Tensor
  /** The indices of the good group: the best $\lceil \gamma n \rceil$ observations. */
  good: number[]
  /** The split $y^\star$: the largest loss in the good group. */
  threshold: number
  /** The densities fitted to the observations, per dimension, on a grid. */
  marginals: TreeParzenMarginal[]
  /**
   * The joint densities on a `jointGrid` $\times$ `jointGrid` grid (the first dimension along the rows) when the
   * space has two continuous dimensions; `null` otherwise.
   */
  joint: { good: Tensor; bad: Tensor; ratio: Tensor } | null
  /** The candidates drawn from $\ell$, $N_s \times m$ unit coordinates. */
  candidates: Tensor
  /** Their scores $\log \ell - \log g$, $N_s$ values. */
  scores: Tensor
  /** The chosen candidate, the next point to evaluate. */
  next: SpaceValues
  /** Its unit coordinates, $m$ values. */
  nextU: Tensor
  /** The best point evaluated so far. */
  incumbent: SpaceValues
  /** Its loss, the lowest so far. */
  best: number
}

/** A dimension as the estimator sees it. */
type Coordinate =
  | { key: string; kind: 'continuous'; dim: Space['dims'][string] }
  | { key: string; kind: 'categorical'; options: readonly (string | number | boolean)[] }

/** A Parzen estimator of one group: per component, its weight, and per dimension its centre and bandwidth. */
interface Parzen {
  /** The mixture weights, prior first, summing to 1. */
  w: Float64Array
  /** The centres per component and dimension (unit coordinates, or category indices); the prior's are unused. */
  mu: number[][]
  /** The bandwidths of continuous dimensions per component, or the kept mass $\kappa$ of categorical ones. */
  b: number[][]
}

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)

/**
 * The log density of a Gaussian $\Gauss(\mu, b^2)$ truncated to $[0, 1]$, at $x$.
 *
 * @param x The point, in $[0, 1]$.
 * @param mu The centre.
 * @param b The bandwidth.
 * @returns The log density.
 */
function logTruncatedNormal(x: number, mu: number, b: number): number {
  const z = (x - mu) / b
  const mass = (normalCdf((1 - mu) / b) as number) - (normalCdf(-mu / b) as number)
  return -0.5 * z * z - LOG_SQRT_2PI - Math.log(b) - Math.log(Math.max(mass, 1e-300))
}

/**
 * The bandwidths of one continuous dimension for a group's observations, before clipping.
 *
 * @param x The group's coordinates in the dimension.
 * @param rule The rule, or a fixed bandwidth.
 * @returns One bandwidth per observation.
 */
function bandwidths(x: readonly number[], rule: 'scott' | 'hyperopt' | number): number[] {
  const n = x.length
  if (typeof rule === 'number') return x.map(() => rule)
  if (rule === 'scott') {
    if (n < 2) return x.map(() => 0)
    const mean = x.reduce((a, v) => a + v, 0) / n
    const sd = Math.sqrt(x.reduce((a, v) => a + (v - mean) ** 2, 0) / (n - 1))
    const sorted = [...x].sort((a, b) => a - b)
    const q = (p: number) => {
      const h = (n - 1) * p
      const lo = Math.floor(h)
      return sorted[lo] + (h - lo) * ((sorted[Math.min(lo + 1, n - 1)] ?? sorted[lo]) - sorted[lo])
    }
    const iqr = q(0.75) - q(0.25)
    const spread = iqr > 0 ? Math.min(sd, iqr / 1.34) : sd
    return x.map(() => 1.059 * spread * n ** -0.2)
  }
  // hyperopt: the larger gap to a neighbour in sorted order (the only one at either end).
  const order = x.map((_, i) => i).sort((a, b) => x[a] - x[b] || a - b)
  const out = new Array<number>(n).fill(1)
  order.forEach((i, r) => {
    const left = r > 0 ? x[i] - x[order[r - 1]] : -Infinity
    const right = r < n - 1 ? x[order[r + 1]] - x[i] : -Infinity
    out[i] = n === 1 ? 1 : Math.max(left, right)
  })
  return out
}

/**
 * Tree-structured Parzen estimation as a step-through algorithm (see the file comment), minimising `objective` over
 * `space`. `init` evaluates the start's points and then random draws (log-uniform on log scales) up to `init`
 * evaluations, and every step evaluates the point proposed by the state before it, refits $\ell$ and $g$ and proposes
 * the next. Each state therefore holds the observations, the split, the densities fitted to those observations on a
 * grid (every dimension's marginal, and the joint when there are two continuous dimensions), the candidates drawn from
 * $\ell$ with their scores, and the chosen next point. Losses that are not finite rank last. The draws come from the
 * streams the runner passes (`child(s, 'init', k)` for the initial points). Throws `DomainError` for a space with a
 * conditional, nested or variants dimension, $\gamma$ outside $(0, 1]$, a non-positive or non-integer count, or a
 * non-positive prior weight or bandwidth.
 *
 * @param objective The loss of a point of the space; lower is better.
 * @param space The search space: real (linear or log scale), integer, choice and boolean dimensions.
 * @param options The split, the candidates, the initial design and the kernels, as `TreeParzenOptions`.
 * @returns The algorithm, to run with `run` or `trace`; its start is `{ points? }`, points to evaluate first.
 *
 * @example Minimise a one-dimensional function: the incumbent closes in on the minimum at 0.3
 * import { real, space } from 'aifn-compute/foundation/space'
 * const objective = ({ x }) => (x - 0.3) ** 2
 * const s = run(treeParzenSteps(objective, space({ x: real(0, 1) })), {}, 30, { stream: stream(0) })
 * print('best x =', s.incumbent.x.toFixed(4), ' loss =', s.best.toExponential(2))
 * print('good group:', s.good.length, 'of', s.observations.length, ' split at', s.threshold.toExponential(2))
 *
 * @example The densities a step chose from
 * import { real, space } from 'aifn-compute/foundation/space'
 * const objective = ({ x }) => Math.sin(6 * x) + x
 * const s = run(treeParzenSteps(objective, space({ x: real(0, 1) }), { grid: 6 }), {}, 5, { stream: stream(1) })
 * const m = s.marginals[0]
 * print('x:', m.values.map((v) => v.toFixed(2)))
 * print('l(x):', m.good, ' g(x):', m.bad)
 * print('next x =', s.next.x.toFixed(4))
 */
export function treeParzenSteps<const S extends Space>(
  objective: (point: ValuesOf<S>) => number,
  space: S,
  options: TreeParzenOptions = {},
): Algorithm<{ points?: readonly ValuesOf<S>[] }, TreeParzenState> {
  const where = 'treeParzenSteps'
  const {
    gamma = 0.15,
    candidates = 24,
    init = 10,
    bandwidth = 'scott',
    minBandwidth = 0.03,
    priorWeight = 1,
    multivariate = true,
    grid = 101,
    jointGrid = 41,
  } = options
  if (!(gamma > 0 && gamma <= 1)) throw new DomainError(where, `${where}: gamma must lie in (0, 1]`)
  for (const [name, v] of [
    ['candidates', candidates],
    ['init', init],
    ['grid', grid],
    ['jointGrid', jointGrid],
  ] as const)
    if (!(Number.isInteger(v) && v >= (name === 'grid' || name === 'jointGrid' ? 2 : 1)))
      throw new DomainError(where, `${where}: ${name} must be an integer of at least ${name.endsWith('rid') ? 2 : 1}`)
  if (!(priorWeight > 0)) throw new DomainError(where, `${where}: priorWeight must be positive`)
  if (typeof bandwidth === 'number' && !(bandwidth > 0))
    throw new DomainError(where, `${where}: a fixed bandwidth must be positive`)
  const coords: Coordinate[] = Object.entries(space.dims).map(([key, dim]) => {
    if (dim.when !== undefined) throw new DomainError(where, `${where}: '${key}' is conditional; not supported`)
    if (dim.type === 'real' || dim.type === 'int') return { key, kind: 'continuous', dim }
    if (dim.type === 'choice') return { key, kind: 'categorical', options: dim.options }
    if (dim.type === 'bool') return { key, kind: 'categorical', options: [false, true] }
    throw new DomainError(where, `${where}: '${key}' is a ${dim.type} dimension; not supported`)
  })
  const m = coords.length
  if (m === 0) throw new DomainError(where, `${where}: the space has no dimensions`)

  // Unit coordinates of a point: a category's index is kept as the centre of its bin, (c + 1/2)/C.
  const toUnit = (p: SpaceValues) => Array.from(encode(space, p))
  const category = (u: number, C: number) => Math.min(C - 1, Math.max(0, Math.floor(u * C)))
  const fromUnit = (u: readonly number[]) => decode(space, u) as SpaceValues

  /** The Parzen estimator of a group of observations (rows of unit coordinates). */
  const fit = (rows: readonly number[][]): Parzen => {
    const N = rows.length
    const w = new Float64Array(N + 1)
    w[0] = priorWeight / (priorWeight + N)
    for (let i = 1; i <= N; i++) w[i] = 1 / (priorWeight + N)
    const mu: number[][] = [coords.map((c) => (c.kind === 'continuous' ? 0.5 : -1))]
    const b: number[][] = [coords.map(() => 1)]
    const perDim = coords.map((c, j) => {
      if (c.kind === 'categorical') {
        const C = c.options.length
        const kappa = C === 1 ? 1 : Math.max(1 / C, (1 + 1 / N) / (1 + C / N))
        return rows.map(() => kappa)
      }
      const floor = Math.max(minBandwidth, (N + 1) ** -2)
      return bandwidths(
        rows.map((r) => r[j]),
        bandwidth,
      ).map((v) => Math.min(1, Math.max(floor, v)))
    })
    rows.forEach((r, i) => {
      mu.push(coords.map((c, j) => (c.kind === 'continuous' ? r[j] : category(r[j], c.options.length))))
      b.push(coords.map((_, j) => perDim[j][i]))
    })
    return { w, mu, b }
  }

  /** The log kernel of component k of an estimator in dimension j at x (a unit coordinate or category index). */
  const logKernel = (p: Parzen, k: number, j: number, x: number): number => {
    const c = coords[j]
    if (c.kind === 'continuous') return logTruncatedNormal(x, p.mu[k][j], p.b[k][j])
    const C = c.options.length
    if (k === 0) return -Math.log(C)
    const kappa = p.b[k][j]
    return Math.log(x === p.mu[k][j] ? kappa : (1 - kappa) / Math.max(1, C - 1))
  }
  const logSumExp = (v: number[]) => {
    const top = Math.max(...v)
    return top === -Infinity ? -Infinity : top + Math.log(v.reduce((a, x) => a + Math.exp(x - top), 0))
  }
  /** The log density of an estimator at a point (per dimension: unit coordinate or category index). */
  const logDensity = (p: Parzen, x: readonly number[]): number => {
    if (multivariate)
      return logSumExp(Array.from(p.w, (w, k) => Math.log(w) + x.reduce((a, xj, j) => a + logKernel(p, k, j, xj), 0)))
    return x.reduce((a, xj, j) => a + logMarginal(p, j, xj), 0)
  }
  /** The log of an estimator's marginal density in dimension j. */
  const logMarginal = (p: Parzen, j: number, x: number): number =>
    logSumExp(Array.from(p.w, (w, k) => Math.log(w) + logKernel(p, k, j, x)))

  /** One draw from an estimator's kernel k in dimension j. */
  const drawKernel = (p: Parzen, k: number, j: number, s: Stream): number => {
    const c = coords[j]
    const [u] = units(s, 1)
    if (c.kind === 'categorical') {
      const C = c.options.length
      if (k === 0) return category(u, C)
      const kappa = p.b[k][j]
      const keep = p.mu[k][j]
      if (u < kappa || C === 1) return keep
      const other = Math.min(C - 2, Math.floor(((u - kappa) / (1 - kappa)) * (C - 1)))
      return other >= keep ? other + 1 : other
    }
    const [mu, b] = [p.mu[k][j], p.b[k][j]]
    const lo = normalCdf(-mu / b) as number
    const hi = normalCdf((1 - mu) / b) as number
    return Math.min(1, Math.max(0, mu + b * (normalQuantile(lo + u * (hi - lo)) as number)))
  }
  const pickComponent = (p: Parzen, s: Stream): number => {
    const [u] = units(s, 1)
    let acc = 0
    for (let k = 0; k < p.w.length; k++) {
      acc += p.w[k]
      if (u < acc) return k
    }
    return p.w.length - 1
  }
  /** A candidate from an estimator, per dimension a unit coordinate or category index. */
  const drawCandidate = (p: Parzen, s: Stream): number[] => {
    if (multivariate) {
      const k = pickComponent(p, s)
      return coords.map((_, j) => drawKernel(p, k, j, s))
    }
    return coords.map((_, j) => drawKernel(p, pickComponent(p, s), j, s))
  }
  // Model coordinates (category index) to unit coordinates (bin centre) and back.
  const toModel = (u: readonly number[]) =>
    u.map((v, j) => {
      const c = coords[j]
      return c.kind === 'continuous' ? v : category(v, c.options.length)
    })
  const toCube = (x: readonly number[]) =>
    x.map((v, j) => {
      const c = coords[j]
      return c.kind === 'continuous' ? v : (v + 0.5) / c.options.length
    })

  /** The value of a continuous dimension at a unit coordinate. */
  const valueAt = (c: Coordinate & { kind: 'continuous' }, u: number): number => {
    const dim = c.dim as { type: 'real' | 'int'; min: number; max: number; scale?: string }
    if (dim.type === 'int') return dim.min + category(Math.min(u, 1 - 1e-12), dim.max - dim.min + 1)
    return dim.scale === 'log' ? dim.min * (dim.max / dim.min) ** u : dim.min + u * (dim.max - dim.min)
  }

  /** Refit to the observations and propose the next point. */
  const propose = (
    t: number,
    observations: SpaceValues[],
    U: number[][],
    losses: number[],
    s: Stream,
  ): TreeParzenState => {
    const n = observations.length
    const rank = (v: number) => (Number.isNaN(v) ? Infinity : v)
    const order = losses.map((_, i) => i).sort((a, b) => rank(losses[a]) - rank(losses[b]) || a - b)
    const nGood = Math.max(1, Math.ceil(gamma * n - 1e-12))
    const goodIdx = order.slice(0, nGood)
    const badIdx = order.slice(nGood)
    const model = U.map(toModel)
    const l = fit(goodIdx.map((i) => model[i]))
    const g = fit(badIdx.map((i) => model[i]))

    const marginals: TreeParzenMarginal[] = coords.map((c, j) => {
      const at =
        c.kind === 'continuous' ? Array.from({ length: grid }, (_, k) => k / (grid - 1)) : c.options.map((_, k) => k)
      const lv = at.map((x) => Math.exp(logMarginal(l, j, x)))
      const gv = at.map((x) => Math.exp(logMarginal(g, j, x)))
      return {
        key: c.key,
        kind: c.kind,
        at: fromData(Float64Array.from(at), [at.length]),
        values: c.kind === 'continuous' ? at.map((u) => valueAt(c, u)) : [...c.options],
        good: fromData(Float64Array.from(lv), [at.length]),
        bad: fromData(Float64Array.from(gv), [at.length]),
        ratio: fromData(
          Float64Array.from(lv, (v, k) => v / gv[k]),
          [at.length],
        ),
      }
    })
    let joint: TreeParzenState['joint'] = null
    if (m === 2 && coords.every((c) => c.kind === 'continuous')) {
      const G = jointGrid
      const [lj, gj] = [new Float64Array(G * G), new Float64Array(G * G)]
      for (let a = 0; a < G; a++)
        for (let b = 0; b < G; b++) {
          const x = [a / (G - 1), b / (G - 1)]
          lj[a * G + b] = Math.exp(logDensity(l, x))
          gj[a * G + b] = Math.exp(logDensity(g, x))
        }
      joint = {
        good: fromData(lj, [G, G]),
        bad: fromData(gj, [G, G]),
        ratio: fromData(
          lj.map((v, k) => v / gj[k]),
          [G, G],
        ),
      }
    }

    const drawn = Array.from({ length: candidates }, () => drawCandidate(l, s))
    const scores = drawn.map((x) => logDensity(l, x) - logDensity(g, x))
    let chosen = 0
    scores.forEach((v, k) => {
      if (v > scores[chosen]) chosen = k
    })
    const cube = drawn.map(toCube)
    const next = fromUnit(cube[chosen])
    return {
      t,
      observations,
      u: fromData(Float64Array.from(U.flat()), [n, m]),
      losses: fromData(Float64Array.from(losses), [n]),
      good: goodIdx,
      threshold: losses[goodIdx[goodIdx.length - 1]],
      marginals,
      joint,
      candidates: fromData(Float64Array.from(cube.flat()), [candidates, m]),
      scores: fromData(Float64Array.from(scores), [candidates]),
      next,
      // The coordinates of the decoded point, so that an integer or a category sits where it is evaluated.
      nextU: fromData(Float64Array.from(toUnit(next)), [m]),
      incumbent: observations[order[0]],
      best: losses[order[0]],
    }
  }

  const evaluate = (p: SpaceValues) => objective(p as ValuesOf<S>)
  return {
    name: 'tree-parzen',
    init: (start, s) => {
      const given = [...(start?.points ?? [])] as SpaceValues[]
      const observations = [...given]
      for (let k = given.length; k < Math.max(init, given.length); k++)
        observations.push(sample(child(s, 'init', k), space) as SpaceValues)
      const U = observations.map(toUnit)
      return propose(0, observations, U, observations.map(evaluate), child(s, 'propose'))
    },
    step: (state, ctx) => {
      const observations = [...state.observations, state.next]
      const n = state.observations.length
      const flat = Array.from(toFlat(state.u) as ArrayLike<number>)
      const prev = Array.from({ length: n }, (_, i) => flat.slice(i * m, i * m + m))
      const losses = [...Array.from(toFlat(state.losses) as ArrayLike<number>), evaluate(state.next)]
      return propose(state.t + 1, observations, [...prev, toUnit(state.next)], losses, ctx.stream)
    },
  }
}

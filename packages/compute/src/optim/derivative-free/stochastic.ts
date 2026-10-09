/**
 * Stochastic derivative-free search: simulated annealing (a Metropolis random walk with a falling temperature) and
 * the covariance matrix adaptation evolution strategy (CMA-ES). Both minimise $f : \reals^n \to \reals$ from its
 * values alone. Step $t$ draws only from the runner's step stream (`ctx.stream`), so a step is a pure function of the
 * state and the root key, and a run is reproduced by passing the same root `stream` to `run` or `trace`.
 */

import { eigh } from 'aifn-compute/numerics/linalg'
import { child, normals, uniform } from 'aifn-compute/foundation/random'
import type { Matrix, Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { StartOptions } from '../options'
import type { IterateState, Schedule, StoppingOptions, ValueFunction } from 'aifn-compute/foundation/contracts'
import { DEFAULT_DIVERGE, divergedAt, evaluateValue } from '../options'
import { dense } from 'aifn-compute/foundation/tensor'

const { data, mat, norm, toF64, vec } = dense
type F64 = dense.F64

/** The state of `simulatedAnnealing`. `x` is the current point of the walk (not necessarily the best). */
export type SimulatedAnnealingState = IterateState & {
  /** The point $\xvec'$ proposed on the last step ($\xvec_0$ at `init`). */
  proposal: Vector
  /** $f(\xvec')$ at the proposal. */
  proposalValue: number
  /**
   * The Metropolis acceptance probability $\min(1, \exp(-\Delta f / T))$ of the last proposal, with
   * $\Delta f = f(\xvec') - f(\xvec)$ (NaN at `init`; 0 for a proposal whose value is NaN).
   */
  acceptance: number
  /** Whether the last proposal was accepted, so that `x` moved to it. */
  accepted: boolean
  /** The temperature used on the last step (at `init`, the temperature of the first step, $T_0$). */
  temperature: number
  /** The best point the walk has accepted (or started from). */
  best: Vector
  /** $f$ at `best`. */
  bestValue: number
  /** Accepted proposals so far. */
  acceptedCount: number
}

/** Options for `simulatedAnnealing`. */
export type SimulatedAnnealingOptions = Pick<StoppingOptions, 'divergeAbove'> & {
  /**
   * The temperature $T_t$ on step $t = 0, 1, \dots$ as a schedule, or a starting temperature $T_0$ cooled
   * geometrically, $T_t = T_0 c^t$ with $c$ = `cooling`. Default $T_0 = 1$.
   */
  temperature?: number | Schedule
  /** Geometric cooling factor when `temperature` is a number. Default 0.995. */
  cooling?: number
  /** Standard deviation of the Gaussian random-walk proposal (a number, or one per coordinate). Default 0.5. */
  proposalScale?: number | readonly number[]
}

/**
 * Simulated annealing (Kirkpatrick, Gelatt and Vecchi, 1983): propose
 * $\xvec' = \xvec + \sigmavec \odot \epsilonvec$, $\epsilonvec \sim \Gauss(\zeros, \Imat)$, and accept it with
 * the Metropolis probability $\min(1, \exp(-(f(\xvec') - f(\xvec))/T_t))$; the temperature falls so the walk
 * settles into low regions. It keeps running until the step limit (`converged` stays false), or until the value or
 * point is not finite or $\lvert f \rvert$ exceeds `divergeAbove`. A proposal whose value is NaN is never accepted,
 * and at $T_t \le 0$ only proposals that do not raise $f$ are. `x` is where the walk is; `best` is the best point it
 * has accepted. `init` takes `{ x0 }`; step $t$ draws from the runner's step stream, one evaluation of $f$ per step.
 *
 * @param f The objective: takes a point (a vector of length $n$) and returns $f(\xvec)$ as a number, or an object with
 *   a `value` field.
 * @param options The temperature (a schedule, or a starting temperature with its geometric `cooling` factor) and the
 *   standard deviations $\sigmavec$ of the proposal.
 * @returns The algorithm, to step with `run` or `trace` from `{ x0 }`.
 *
 * @example Find the deeper of two wells
 * // A tilted double well: its minima are near x = -1 (the deeper) and x = 1, and the walk starts at x = 3.
 * const f = (x) => {
 *   const v = toFlat(x)[0]
 *   return (v * v - 1) ** 2 + 0.3 * v
 * }
 * const alg = simulatedAnnealing(f, { temperature: 1, cooling: 0.99, proposalScale: 0.3 })
 * const s = run(alg, { x0: [3] }, 1000, { stream: stream(1) })
 * print('best =', s.best)
 * print('f(best) =', s.bestValue)
 * print('accepted =', s.acceptedCount, 'of', s.t)
 * print('final temperature =', s.temperature)
 */
export function simulatedAnnealing(
  f: ValueFunction,
  options: SimulatedAnnealingOptions = {},
): Algorithm<StartOptions, SimulatedAnnealingState> {
  const { temperature = 1, cooling = 0.995, proposalScale = 0.5, divergeAbove = DEFAULT_DIVERGE } = options
  const temperatureAt = (t: number) => (typeof temperature === 'function' ? temperature(t) : temperature * cooling ** t)
  const name = 'simulated-annealing'
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const value = evaluateValue(f, x)
      return {
        t: 0,
        x: vec(x),
        value,
        proposal: vec(x),
        proposalValue: value,
        acceptance: NaN,
        accepted: false,
        temperature: temperatureAt(0),
        best: vec(x),
        bestValue: value,
        acceptedCount: 0,
        evaluations: 1,
        converged: false,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (s, ctx) => {
      const x = data(s.x)
      const n = x.length
      const eps = data(normals(child(ctx.stream, 'proposal'), [n]))
      const proposal = x.map(
        (xi, i) => xi + (typeof proposalScale === 'number' ? proposalScale : proposalScale[i]) * eps[i],
      )
      const proposalValue = evaluateValue(f, proposal)
      const T = temperatureAt(s.t)
      const delta = proposalValue - s.value
      // A NaN proposal value is never accepted.
      const acceptance = delta <= 0 ? 1 : T > 0 ? Math.exp(-delta / T) : 0
      const accepted = uniform(child(ctx.stream, 'accept')) < acceptance
      const next = accepted ? proposal : x
      const value = accepted ? proposalValue : s.value
      const improved = accepted && value < s.bestValue
      return {
        ...s,
        t: s.t + 1,
        x: vec(next),
        value,
        proposal: vec(proposal),
        proposalValue,
        acceptance: Number.isNaN(acceptance) ? 0 : acceptance,
        accepted,
        temperature: T,
        best: improved ? vec(proposal) : s.best,
        bestValue: improved ? value : s.bestValue,
        acceptedCount: s.acceptedCount + (accepted ? 1 : 0),
        evaluations: s.evaluations + 1,
        diverged: divergedAt(value, next, divergeAbove),
      }
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// CMA-ES.

/** The state of `cmaEs`. `x` is the distribution mean $\mvec$ and `value` is $f(\mvec)$. */
export type CmaEsState = IterateState & {
  /** Generation count (equal to `t`). */
  generation: number
  /** The global step size $\sigma$, for the next generation. */
  sigma: number
  /** The covariance matrix $\Cmat$ ($n \times n$); samples are $\xvec \sim \Gauss(\mvec, \sigma^2\Cmat)$. */
  covariance: Matrix
  /** The conjugate evolution path $\pvec_\sigma$, which adapts $\sigma$ (zero at `init`). */
  pathSigma: Vector
  /** The evolution path $\pvec_c$ of the rank-one update of $\Cmat$ (zero at `init`). */
  pathCovariance: Vector
  /** The last generation's $\lambda$ samples as the rows of a $\lambda \times n$ matrix, in sampling order. */
  population: Matrix
  /** $f$ at each sample of `population`, in the same order. */
  populationValues: Vector
  /** The best point evaluated over all generations: a sample or a mean. */
  best: Vector
  /** $f$ at `best`. */
  bestValue: number
  /** The square roots of the eigenvalues of $\Cmat$ (axis lengths of the search ellipsoid), descending. */
  axisLengths: Vector
}

/** Options for `cmaEs`. */
export type CmaEsOptions = Pick<StoppingOptions, 'divergeAbove'> & {
  /** Initial step size $\sigma_0$. Default 0.5. */
  sigma?: number
  /**
   * Population size $\lambda$ (at least 2, or the run is flagged diverged at `init`). Default
   * $4 + \lfloor 3 \ln n \rfloor$.
   */
  populationSize?: number
  /** Stop when $\sigma$ times the largest axis length falls below this. Default 1e-10. */
  xTolerance?: number
  /** Stop when the population's value range falls below this. Default 1e-12. */
  fTolerance?: number
}

/**
 * The $(\mu/\mu_w, \lambda)$ CMA-ES with default parameters from Hansen (2016), "The CMA Evolution Strategy: A
 * Tutorial", arXiv:1604.00772, Table 1 and Appendix A: sample $\lambda$ points from $\Gauss(\mvec, \sigma^2\Cmat)$,
 * move $\mvec$ to the weighted mean of the best $\mu = \lfloor \lambda/2 \rfloor$ (positive weights only), update
 * the evolution paths, adapt $\Cmat$ by rank-one and rank-$\mu$ updates, and adapt $\sigma$ by cumulative step-size
 * adaptation. `init` takes `{ x0 }` (the initial mean, with $\Cmat = \Imat$); step $t$ draws from the runner's step
 * stream. Each generation evaluates $f$ at the $\lambda$ samples and at the new mean. It converges when $\sigma$
 * times the longest axis falls below `xTolerance` or the generation's values span less than `fTolerance`, and flags
 * divergence when the mean, its value or $\sigma$ is not finite, or $\lvert f \rvert$ exceeds `divergeAbove`.
 * Samples whose value is NaN rank last.
 *
 * @param f The objective: takes a point (a vector of length $n$) and returns $f(\xvec)$ as a number, or an object with
 *   a `value` field.
 * @param options The initial step size, the population size and the tolerances of the stopping test.
 * @returns The algorithm, to step with `run` or `trace` from `{ x0 }`.
 *
 * @example Minimise a quadratic
 * // The minimum is at (1, -0.5), where f = 0.
 * const f = (x) => {
 *   const [a, b] = toFlat(x)
 *   return (a - 1) ** 2 + 2 * (b + 0.5) ** 2
 * }
 * const s = run(cmaEs(f, { sigma: 0.5 }), { x0: [0, 0] }, 500, { stream: stream(0) })
 * print('mean =', s.x)
 * print('f(mean) =', s.value)
 * print('generations =', s.generation)
 * print('evaluations =', s.evaluations)
 *
 * @example The covariance learns the shape of a valley
 * // Rosenbrock's function: near its minimum (1, 1) the valley runs along the direction (1, 2).
 * const rosen = (x) => {
 *   const [a, b] = toFlat(x)
 *   return (1 - a) ** 2 + 100 * (b - a * a) ** 2
 * }
 * const s = run(cmaEs(rosen), { x0: [-1.2, 1] }, 2000, { stream: stream(0) })
 * print('mean =', s.x)
 * print('generations =', s.generation)
 * print('C =', s.covariance)
 * print('axis lengths =', s.axisLengths)
 */
export function cmaEs(f: ValueFunction, options: CmaEsOptions = {}): Algorithm<StartOptions, CmaEsState> {
  const { sigma: sigma0 = 0.5, xTolerance = 1e-10, fTolerance = 1e-12, divergeAbove = DEFAULT_DIVERGE } = options
  const name = 'cma-es'

  /** Strategy parameters for dimension n (Hansen, 2016, eq. 48–58). */
  const parameters = (n: number) => {
    const lambda = options.populationSize ?? 4 + Math.floor(3 * Math.log(n))
    const mu = Math.floor(lambda / 2)
    const raw = Array.from({ length: mu }, (_, i) => Math.log((lambda + 1) / 2) - Math.log(i + 1))
    const total = raw.reduce((a, b) => a + b, 0)
    const weights = raw.map((w) => w / total)
    const muEff = 1 / weights.reduce((a, w) => a + w * w, 0)
    const cc = (4 + muEff / n) / (n + 4 + (2 * muEff) / n)
    const cs = (muEff + 2) / (n + muEff + 5)
    const c1 = 2 / ((n + 1.3) ** 2 + muEff)
    const cmu = Math.min(1 - c1, (2 * (muEff - 2 + 1 / muEff)) / ((n + 2) ** 2 + muEff))
    const damps = 1 + 2 * Math.max(0, Math.sqrt((muEff - 1) / (n + 1)) - 1) + cs
    const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    return { lambda, mu, weights, muEff, cc, cs, c1, cmu, damps, chiN }
  }

  /** C = B diag(d²) Bᵀ: the eigenvectors (columns of B) and axis lengths d, with negative rounding clipped to 0. */
  const decompose = (C: F64, n: number) => {
    const { values, vectors } = eigh(mat(C, n, n))
    const d = data(values).map((v) => Math.sqrt(Math.max(v, 0)))
    return { B: data(vectors), d }
  }

  return {
    name,
    init: ({ x0 }) => {
      const m = toF64(x0, name)
      const n = m.length
      const C = new Float64Array(n * n)
      for (let i = 0; i < n; i++) C[i * n + i] = 1
      const value = evaluateValue(f, m)
      const { lambda } = parameters(n)
      return {
        t: 0,
        generation: 0,
        x: vec(m),
        value,
        sigma: sigma0,
        covariance: mat(C, n, n),
        pathSigma: vec(new Float64Array(n)),
        pathCovariance: vec(new Float64Array(n)),
        population: mat(new Float64Array(0), 0, n),
        populationValues: vec(new Float64Array(0)),
        best: vec(m),
        bestValue: value,
        axisLengths: vec(new Float64Array(n).fill(1)),
        evaluations: 1,
        converged: false,
        diverged: divergedAt(value, m, divergeAbove) || lambda < 2,
      }
    },
    step: (s, ctx) => {
      const n = s.x.shape[0]
      const { lambda, mu, weights, muEff, cc, cs, c1, cmu, damps, chiN } = parameters(n)
      const m = data(s.x)
      const C = data(s.covariance)
      const { B, d } = decompose(C, n)
      const sigma = s.sigma

      // Sample y_k = B D z_k and x_k = m + σ y_k.
      const z = data(normals(ctx.stream, [lambda, n]))
      const ys: F64[] = []
      const xs: F64[] = []
      const fs: number[] = []
      for (let k = 0; k < lambda; k++) {
        const y = new Float64Array(n)
        for (let i = 0; i < n; i++) {
          let v = 0
          for (let j = 0; j < n; j++) v += B[i * n + j] * d[j] * z[k * n + j]
          y[i] = v
        }
        const x = m.map((mi, i) => mi + sigma * y[i])
        ys.push(y)
        xs.push(x)
        fs.push(evaluateValue(f, x))
      }
      // Rank by value; NaN values rank last.
      const order = fs
        .map((_, k) => k)
        .sort((a, b) => (Number.isNaN(fs[a]) ? 1 : Number.isNaN(fs[b]) ? -1 : fs[a] - fs[b]))

      // Recombination: m' = m + σ Σ w_i y_{i:λ}.
      const yw = new Float64Array(n)
      for (let i = 0; i < mu; i++) for (let j = 0; j < n; j++) yw[j] += weights[i] * ys[order[i]][j]
      const mNext = m.map((mi, j) => mi + sigma * yw[j])

      // C^{-1/2} y_w = B D⁻¹ Bᵀ y_w (zero axis lengths contribute nothing).
      const Bt = new Float64Array(n)
      for (let j = 0; j < n; j++) {
        let v = 0
        for (let i = 0; i < n; i++) v += B[i * n + j] * yw[i]
        Bt[j] = d[j] > 0 ? v / d[j] : 0
      }
      const invSqrtYw = new Float64Array(n)
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) invSqrtYw[i] += B[i * n + j] * Bt[j]

      const ps0 = data(s.pathSigma)
      const pc0 = data(s.pathCovariance)
      const csFactor = Math.sqrt(cs * (2 - cs) * muEff)
      const ps = ps0.map((p, i) => (1 - cs) * p + csFactor * invSqrtYw[i])
      const psNorm = norm(ps)
      const generation = s.generation + 1
      // Stall the rank-one update while ‖p_σ‖ is large (Hansen, 2016, eq. 45, h_σ).
      const hSigma = psNorm / Math.sqrt(1 - (1 - cs) ** (2 * generation)) / chiN < 1.4 + 2 / (n + 1) ? 1 : 0
      const ccFactor = hSigma * Math.sqrt(cc * (2 - cc) * muEff)
      const pc = pc0.map((p, i) => (1 - cc) * p + ccFactor * yw[i])

      const deltaH = (1 - hSigma) * cc * (2 - cc)
      const CNext = new Float64Array(n * n)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          let rankMu = 0
          for (let k = 0; k < mu; k++) rankMu += weights[k] * ys[order[k]][i] * ys[order[k]][j]
          CNext[i * n + j] = (1 - c1 - cmu) * C[i * n + j] + c1 * (pc[i] * pc[j] + deltaH * C[i * n + j]) + cmu * rankMu
        }
      // Keep C exactly symmetric against rounding.
      for (let i = 0; i < n; i++)
        for (let j = 0; j < i; j++) {
          const v = 0.5 * (CNext[i * n + j] + CNext[j * n + i])
          CNext[i * n + j] = v
          CNext[j * n + i] = v
        }
      const sigmaNext = sigma * Math.exp((cs / damps) * (psNorm / chiN - 1))

      const value = evaluateValue(f, mNext)
      const population = new Float64Array(lambda * n)
      xs.forEach((x, k) => population.set(x, k * n))
      const bestK = order[0]
      let best = s.best
      let bestValue = s.bestValue
      if (fs[bestK] < bestValue) {
        best = vec(xs[bestK])
        bestValue = fs[bestK]
      }
      if (value < bestValue) {
        best = vec(mNext)
        bestValue = value
      }
      const axes = decompose(CNext, n).d
      const range = fs[order[lambda - 1]] - fs[order[0]]
      return {
        ...s,
        t: s.t + 1,
        generation,
        x: vec(mNext),
        value,
        sigma: sigmaNext,
        covariance: mat(CNext, n, n),
        pathSigma: vec(ps),
        pathCovariance: vec(pc),
        population: mat(population, lambda, n),
        populationValues: vec(Float64Array.from(fs)),
        best,
        bestValue,
        axisLengths: vec(axes),
        evaluations: s.evaluations + lambda + 1,
        converged: sigmaNext * Math.max(...axes) < xTolerance || range < fTolerance,
        diverged: divergedAt(value, mNext, divergeAbove) || !Number.isFinite(sigmaNext),
      }
    },
  }
}

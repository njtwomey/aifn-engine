/**
 * Hamiltonian Monte Carlo (Duane et al., 1987; Neal, 2011) with the leapfrog integrator, and the No-U-Turn Sampler
 * (Hoffman & Gelman, 2014), with Stan's multinomial sampling (Betancourt, 2017) by default and the original slice
 * sampling (Hoffman & Gelman's Algorithm 3) as an option. Both expose the trajectory, the momenta and the energy along
 * it, and both can tune their step size during warmup by dual averaging (`adapt`; see `./adaptation`).
 *
 * Conventions: the potential is $U(\thetavec) = -\log \pi(\thetavec)$, the kinetic energy
 * $K(\pvec) = \frac{1}{2} \sum_i p_i^2/m_i$ for a diagonal mass matrix $\Mmat = \diag(\mvec)$, momenta are drawn
 * $\pvec \sim \Gauss(\zeros, \Mmat)$, and the Hamiltonian $H = U + K$ is the energy.
 */

import { child, uniform } from 'aifn-compute/foundation/random'
import { logAddExp } from 'aifn-compute/numerics/special'
import type { Matrix, Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import {
  dualAveragingStart,
  dualAveragingUpdate,
  resolveDualAveraging,
  type DualAveragingOptions,
  type DualAveragingState,
} from './adaptation'
import { badLogDensity } from './metropolis'
import type { AcceptRejectState, ChainStart, LogDensity } from './types'
import { allFinite, data, logDensityAndGrad, mat, perCoordinate, standardNormals, toF64, vec, type F64 } from './util'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A leapfrog trajectory: positions and momenta at each of its $L + 1$ points (fewer when it stopped early), and $H$ at
 * each.
 */
export type Leapfrog = {
  /** The positions $\thetavec$, one row per point: $(L + 1) \times d$, the start first. */
  positions: Matrix
  /** The momenta $\pvec$ at the same points, $(L + 1) \times d$. */
  momenta: Matrix
  /** $H(\thetavec, \pvec)$ at each point. */
  energies: Vector
  /** $\log \pi$ at each point. */
  logDensities: Vector
  /** The gradient $\nabla \log \pi$ at the last point. */
  finalGrad: Vector
  /** False when a position, momentum or energy became non-finite (the trajectory stops there). */
  finite: boolean
}

/**
 * Kinetic energy $\frac{1}{2} \sum_i p_i^2/m_i$.
 *
 * @param p The momentum $\pvec$.
 * @param mass The diagonal $\mvec$ of the mass matrix, one value per coordinate of `p`.
 * @returns $K(\pvec)$.
 */
function kinetic(p: F64, mass: F64): number {
  let k = 0
  for (let i = 0; i < p.length; i++) k += (p[i] * p[i]) / mass[i]
  return 0.5 * k
}

/**
 * One leapfrog step of size $\varepsilon$ from $(\thetavec, \pvec)$ with gradient
 * $\gvec = \nabla \log \pi(\thetavec)$: half momentum, full position, half momentum. One gradient evaluation.
 *
 * @param target The target, whose log-density and gradient are evaluated at the new position.
 * @param theta The position $\thetavec$ ($d$ values); not modified.
 * @param p The momentum $\pvec$ ($d$ values); not modified.
 * @param g The gradient $\nabla \log \pi(\thetavec)$ at `theta`, from the previous step.
 * @param eps The step size $\varepsilon$; negative to integrate backwards in time.
 * @param mass The diagonal $\mvec$ of the mass matrix ($d$ values).
 * @returns The new position `theta` and momentum `p`, with $\log \pi$ (`logDensity`) and its gradient (`grad`) there.
 */
function leapfrogStep(target: LogDensity, theta: F64, p: F64, g: F64, eps: number, mass: F64) {
  const d = theta.length
  const pHalf = new Float64Array(d)
  const next = new Float64Array(d)
  for (let i = 0; i < d; i++) {
    pHalf[i] = p[i] + 0.5 * eps * g[i]
    next[i] = theta[i] + (eps * pHalf[i]) / mass[i]
  }
  const { value, grad } = logDensityAndGrad(target, next)
  const pNext = new Float64Array(d)
  for (let i = 0; i < d; i++) pNext[i] = pHalf[i] + 0.5 * eps * grad[i]
  return { theta: next, p: pNext, logDensity: value, grad }
}

/**
 * Integrate Hamilton's equations with `steps` leapfrog steps of size `stepSize` from $(\thetavec_0, \pvec_0)$ (Neal,
 * 2011, eqs. 5.18 to 5.20). The leapfrog map is volume-preserving and reversible, and its energy error stays bounded
 * for a stable step size ($\varepsilon$ below about $2/\sqrt{\lambda_\text{max}}$, $\lambda_\text{max}$ the largest
 * eigenvalue of the precision, for a Gaussian with unit mass), which is why HMC accepts most proposals. The trajectory
 * stops early at the first point whose energy or position is not finite.
 *
 * @param target The target, through `logDensity`, `grad` (or autodiff) and `dim`.
 * @param theta0 The start position $\thetavec_0$ ($d$ values).
 * @param momentum0 The start momentum $\pvec_0$ ($d$ values).
 * @param options `stepSize` $\varepsilon$ (negative integrates backwards), the number of leapfrog `steps` $L$, and the
 *   `mass` diagonal $\mvec$ (one number or one per coordinate; default 1).
 * @returns The trajectory: its positions, momenta, energies and log-densities at every point.
 *
 * @example A standard normal is a harmonic oscillator
 * // H = (theta^2 + p^2) / 2: one period is 2 pi, about 63 steps of 0.1.
 * const target = {
 *   kind: 'log-density', dim: 1, normalised: false,
 *   logDensity: (x) => mul(-0.5, sum(mul(x, x))),
 *   grad: (x) => neg(x),
 * }
 * const path = leapfrog(target, [1], [0], { stepSize: 0.1, steps: 63 })
 * print('end point =', path.positions.data[63], 'momentum', path.momenta.data[63])
 * print('energy range =', min(path.energies), max(path.energies))
 *
 * @example An unstable step size
 * // For the unit Gaussian the leapfrog map is stable only for a step below 2.
 * const target = {
 *   kind: 'log-density', dim: 1, normalised: false,
 *   logDensity: (x) => mul(-0.5, sum(mul(x, x))),
 *   grad: (x) => neg(x),
 * }
 * for (const stepSize of [1.9, 2.1]) {
 *   const path = leapfrog(target, [1], [0], { stepSize, steps: 20 })
 *   print(`step ${stepSize}: final energy =`, get(path.energies, 20))
 * }
 */
export function leapfrog(
  target: LogDensity,
  theta0: Vector | ArrayLike<number>,
  momentum0: Vector | ArrayLike<number>,
  options: { stepSize: number; steps: number; mass?: number | ArrayLike<number> },
): Leapfrog {
  const d = target.dim
  const mass = perCoordinate(options.mass ?? 1, d, 'leapfrog')
  let theta = toF64(theta0, 'leapfrog')
  let p = toF64(momentum0, 'leapfrog')
  let { value, grad } = logDensityAndGrad(target, theta)
  const positions: F64[] = [theta]
  const momenta: F64[] = [p]
  const energies: number[] = [-value + kinetic(p, mass)]
  const logs: number[] = [value]
  let finite = Number.isFinite(energies[0])
  for (let l = 0; l < options.steps && finite; l++) {
    const next = leapfrogStep(target, theta, p, grad, options.stepSize, mass)
    ;({ theta, p, grad } = next)
    value = next.logDensity
    const h = -value + kinetic(p, mass)
    positions.push(theta)
    momenta.push(p)
    energies.push(h)
    logs.push(value)
    finite = Number.isFinite(h) && allFinite(theta)
  }
  const flat = (rows: F64[]) => {
    const out = new Float64Array(rows.length * d)
    rows.forEach((r, i) => out.set(r, i * d))
    return mat(out, rows.length, d)
  }
  return {
    positions: flat(positions),
    momenta: flat(momenta),
    energies: vec(Float64Array.from(energies)),
    logDensities: vec(Float64Array.from(logs)),
    finalGrad: vec(grad),
    finite,
  }
}

/** The state of `hmc`. `proposal` is the trajectory's end point. */
export type HmcState = AcceptRejectState &
  DualAveragingState & {
    /** $\nabla \log \pi(\xvec)$. */
    grad: Vector
    /** The momentum drawn on the last step (zeros at $t = 0$). */
    momentum: Vector
    /** The last trajectory, $(L + 1) \times d$, starting at the previous $\xvec$. */
    trajectory: Matrix
    /** The momenta along the last trajectory, $(L + 1) \times d$. */
    trajectoryMomenta: Matrix
    /** $H$ along the last trajectory (at $t = 0$, $-\log \pi(\xvec_0)$ alone). */
    energies: Vector
    /**
     * $\Delta H = H(\text{end}) - H(\text{start})$ of the last trajectory ($\infty$ when it became non-finite); the log
     * acceptance ratio is its negative.
     */
    energyError: number
    /** True when the last trajectory's energy error exceeded `divergenceThreshold` or became non-finite. */
    divergent: boolean
    /** Divergent trajectories so far (each is also rejected, or accepted only by chance). */
    divergentCount: number
    /** Proposals rejected so far whose trajectory did not diverge: ordinary Metropolis rejections. */
    rejectedCount: number
    /** The energy error above which a trajectory counts as divergent (the resolved `divergenceThreshold`). */
    divergenceLimit: number
    /** Gradient evaluations so far. */
    gradientEvaluations: number
    /** The step size used on the last step (after jitter). */
    stepSize: number
    /**
     * The acceptance statistic of the last step, $\min(1, \exp(-\Delta H))$, which adaptation steers towards its
     * target.
     */
    acceptStat: number
  }

/** Options for `hmc`. */
export type HmcOptions = {
  /** Leapfrog step size $\varepsilon$. Default 0.1. */
  stepSize?: number
  /** Leapfrog steps $L$ per proposal. Default 20. */
  steps?: number
  /** Diagonal $\mvec$ of the mass matrix $\Mmat$ (one number or one per coordinate). Default 1. */
  mass?: number | ArrayLike<number>
  /**
   * Draw the step uniformly in $[\varepsilon(1 - j), \varepsilon(1 + j)]$ each step, to avoid periodic trajectories
   * (Neal, 2011, §5.4.2). Default 0.
   */
  stepJitter?: number
  /** When a trajectory counts as divergent (see `DivergenceThreshold`). Default 1000, Stan's absolute threshold. */
  divergenceThreshold?: DivergenceThreshold
  /**
   * Tune $\varepsilon$ by dual averaging during the first `warmup` steps, starting from `stepSize` (Algorithm 5).
   * Default off.
   */
  adapt?: DualAveragingOptions
}

/**
 * When a trajectory counts as divergent: its energy error $\lvert \Delta H \rvert$ exceeds an absolute number
 * (Stan's 1000), or `{ relative: r }`, $rd$ for a target of dimension $d$. The kinetic energy of $d$ coordinates has
 * variance $d/2$, so a relative threshold scales with the energy's own spread: it flags the divergences of a small
 * model (a 2-D funnel's errors of 50) that an absolute 1000 misses, and stays loose for a large one.
 */
export type DivergenceThreshold = number | { readonly relative: number }

/**
 * The absolute energy-error limit of a threshold for dimension $d$ (checked positive: `DomainError` otherwise).
 *
 * @param threshold An absolute limit, or `{ relative: r }` for $rd$.
 * @param d The dimension of the target.
 * @param name The caller's name for error messages.
 * @returns The limit on $\lvert \Delta H \rvert$.
 *
 * @example Absolute and relative thresholds
 * print('absolute =', divergenceLimit(1000, 2, 'hmc'))
 * print('relative 10, d = 2 =', divergenceLimit({ relative: 10 }, 2, 'hmc'))
 * print('relative 10, d = 500 =', divergenceLimit({ relative: 10 }, 500, 'hmc'))
 */
export function divergenceLimit(threshold: DivergenceThreshold, d: number, name: string): number {
  const limit = typeof threshold === 'number' ? threshold : threshold.relative * d
  if (!(limit > 0)) throw new DomainError(name, `${name}: the divergence threshold must be positive, got ${limit}`)
  return limit
}

/**
 * The log-density, gradient and divergence flag at the start point of `hmc` and `nuts`. Throws `ShapeError` when
 * $\xvec_0$ does not have `target.dim` values.
 *
 * @param target The target, evaluated once at $\xvec_0$.
 * @param x0 The start point $\xvec_0$ as a working array; not modified.
 * @param name The sampler's name, for error messages.
 * @returns `value` $\log \pi(\xvec_0)$, its `grad`, and `diverged`, set when the value is NaN or $+\infty$ or
 *   $\xvec_0$ is not finite.
 */
function hmcStart(target: LogDensity, x0: F64, name: string) {
  if (x0.length !== target.dim)
    throw new ShapeError(name, `${name}: x0 has ${x0.length} values for dimension ${target.dim}`)
  const { value, grad } = logDensityAndGrad(target, x0)
  return { value, grad, diverged: badLogDensity(value) || !allFinite(x0) }
}

/**
 * Hamiltonian Monte Carlo (Duane et al., 1987; Neal, 2011, §5.3.2): draw $\pvec \sim \Gauss(\zeros, \Mmat)$, run $L$
 * leapfrog steps of size $\varepsilon$ from $(\xvec, \pvec)$, and accept the end point with probability
 * $\min(1, \exp(-\Delta H))$. Step $t$ draws the momentum from `child(ctx.stream, 'momentum')`, the jittered step
 * from `child(ctx.stream, 'jitter')` and the uniform from the step stream `ctx.stream` itself. A trajectory that turns
 * non-finite is rejected and counted as divergent. Throws `DomainError` for a step size or option out of range.
 *
 * @param target The target, through `logDensity`, `grad` (or autodiff of `logDensity`) and `dim`.
 * @param options The step size, steps, mass, jitter, divergence threshold and adaptation.
 * @returns The sampler as an algorithm: start it from `{ x0 }` and run it with `run`, `trace` or `sampleChains`.
 *
 * @example Moments of a correlated Gaussian
 * const target = {
 *   kind: 'log-density', dim: 2, normalised: false,
 *   logDensity: (x) => {
 *     const [a, b] = x.data
 *     return -(a * a - 1.6 * a * b + b * b) / 0.72
 *   },
 *   grad: (x) => {
 *     const [a, b] = x.data
 *     return [-(a - 0.8 * b) / 0.36, -(b - 0.8 * a) / 0.36]
 *   },
 * }
 * const { draws } = sampleChains(hmc(target, { stepSize: 0.25, steps: 10 }), { x0: [0, 0] }, {
 *   chains: 2, steps: 200, stream: stream(1),
 * })
 * const x = reshape(draws, [-1, 2])
 * print('mean =', mean(x, 0))
 * print('second moments =', div(matmul(transpose(x), x), x.shape[0]))
 * print('ESS =', effectiveSampleSize(draws))
 *
 * @example Tuning an unstable step size during warmup
 * // Leapfrog is stable here only below 2 sqrt(1 - 0.8): the early steps of 1 diverge.
 * const target = {
 *   kind: 'log-density', dim: 2, normalised: false,
 *   logDensity: (x) => {
 *     const [a, b] = x.data
 *     return -(a * a - 1.6 * a * b + b * b) / 0.72
 *   },
 *   grad: (x) => {
 *     const [a, b] = x.data
 *     return [-(a - 0.8 * b) / 0.36, -(b - 0.8 * a) / 0.36]
 *   },
 * }
 * const s = run(hmc(target, { stepSize: 1, steps: 10, adapt: { warmup: 100 } }), { x0: [0, 0] }, 200)
 * print('adapted step size =', s.stepSize)
 * print('acceptance rate over all 200 steps =', s.acceptanceRate)
 * print('divergent trajectories =', s.divergentCount)
 */
export function hmc(target: LogDensity, options: HmcOptions = {}): Algorithm<ChainStart, HmcState> {
  const name = 'hmc'
  const { stepSize = 0.1, steps = 20, stepJitter = 0 } = options
  const d = target.dim
  const limit = divergenceLimit(options.divergenceThreshold ?? 1000, d, name)
  const adapt = resolveDualAveraging(options.adapt, stepSize, name)
  const mass = perCoordinate(options.mass ?? 1, d, name)
  const sqrtMass = mass.map(Math.sqrt)
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const start = hmcStart(target, x, name)
      const X = vec(x)
      return {
        t: 0,
        x: X,
        logDensity: start.value,
        grad: vec(start.grad),
        proposal: X,
        proposalLogDensity: start.value,
        logAcceptanceRatio: NaN,
        acceptance: NaN,
        accepted: false,
        acceptedCount: 0,
        acceptanceRate: NaN,
        momentum: vec(new Float64Array(d)),
        trajectory: mat(Float64Array.from(x), 1, d),
        trajectoryMomenta: mat(new Float64Array(d), 1, d),
        energies: vec(Float64Array.of(-start.value)),
        energyError: 0,
        divergent: false,
        divergentCount: 0,
        rejectedCount: 0,
        divergenceLimit: limit,
        gradientEvaluations: 1,
        stepSize,
        acceptStat: NaN,
        ...dualAveragingStart(stepSize, adapt),
        diverged: start.diverged,
      }
    },
    step: (s, ctx) => {
      const draws = ctx.stream
      const z = standardNormals(child(draws, 'momentum'), d)
      const p = z.map((zi, i) => zi * sqrtMass[i])
      const base = s.nextStepSize
      const eps = stepJitter > 0 ? base * (1 + stepJitter * (2 * uniform(child(draws, 'jitter')) - 1)) : base
      const path = leapfrog(target, s.x, p, { stepSize: eps, steps, mass })
      const H = data(path.energies)
      const last = H.length - 1
      const energyError = path.finite ? H[last] - H[0] : Infinity
      const divergent = !path.finite || Math.abs(energyError) > limit
      const logRatio = path.finite ? -energyError : -Infinity
      const acceptance = Math.min(1, Math.exp(logRatio))
      const accepted = uniform(draws) < acceptance
      const P = data(path.positions)
      const end = P.slice(last * d, (last + 1) * d)
      const endLog = data(path.logDensities)[last]
      const acceptedCount = s.acceptedCount + (accepted ? 1 : 0)
      return {
        ...s,
        t: s.t + 1,
        x: accepted ? vec(end) : s.x,
        logDensity: accepted ? endLog : s.logDensity,
        grad: accepted ? path.finalGrad : s.grad,
        proposal: vec(end),
        proposalLogDensity: endLog,
        logAcceptanceRatio: logRatio,
        acceptance,
        accepted,
        acceptedCount,
        acceptanceRate: acceptedCount / (s.t + 1),
        momentum: vec(p),
        trajectory: path.positions,
        trajectoryMomenta: path.momenta,
        energies: path.energies,
        energyError,
        divergent,
        divergentCount: s.divergentCount + (divergent ? 1 : 0),
        rejectedCount: s.rejectedCount + (!accepted && !divergent ? 1 : 0),
        gradientEvaluations: s.gradientEvaluations + last,
        stepSize: eps,
        acceptStat: acceptance,
        ...dualAveragingUpdate(s, s.t + 1, acceptance, adapt),
      }
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// NUTS.

/** The state of `nuts`. */
export type NutsState = AcceptRejectState &
  DualAveragingState & {
    /** $\nabla \log \pi(\xvec)$. */
    grad: Vector
    /** The momentum drawn on the last step (zeros at $t = 0$). */
    momentum: Vector
    /** Every point the last tree visited, ordered by integration time (leftmost first), one row each. */
    trajectory: Matrix
    /** The integration time of each trajectory point, in leapfrog steps from the start (negative backwards). */
    trajectoryTimes: Vector
    /** $H$ at each trajectory point. */
    energies: Vector
    /**
     * The number of doublings of the last tree: a full tree of depth $j$ has $2^j$ points. Under `'multinomial'` a last
     * doubling that diverged or turned is not counted.
     */
    treeDepth: number
    /** The leapfrog steps of the last tree, including those of a last doubling that was abandoned. */
    leapfrogSteps: number
    /**
     * True when the last tree hit an energy error above `divergenceThreshold` (Hoffman & Gelman's
     * $\Delta_\text{max}$), or a non-finite energy.
     */
    divergent: boolean
    /** Steps whose tree diverged so far. */
    divergentCount: number
    /** Steps that stayed at x without a divergence (no acceptable point other than x was drawn). */
    rejectedCount: number
    /** The resolved $\Delta_\text{max}$. */
    divergenceLimit: number
    /** Gradient evaluations so far. */
    gradientEvaluations: number
    /** True when the last tree stopped at `maxDepth` rather than by a U-turn. */
    hitMaxDepth: boolean
    /** The step size used on the last step. */
    stepSize: number
    /**
     * The acceptance statistic of the last step: the mean of $\min(1, \exp(H_0 - H))$ over every leapfrog point of the
     * tree (Stan's `accept_stat`; Hoffman & Gelman's Algorithm 6 averages over the last doubling only).
     */
    acceptStat: number
  }

/** Options for `nuts`. */
export type NutsOptions = {
  /** Leapfrog step size $\varepsilon$. Default 0.1. */
  stepSize?: number
  /** Largest tree depth (at most $2^{\text{maxDepth}} - 1$ leapfrog steps). Default 10. */
  maxDepth?: number
  /** Diagonal $\mvec$ of the mass matrix $\Mmat$ (one number or one per coordinate). Default 1. */
  mass?: number | ArrayLike<number>
  /**
   * $\Delta_\text{max}$: stop building when the energy error passes this (see `DivergenceThreshold`). Default 1000.
   */
  divergenceThreshold?: DivergenceThreshold
  /**
   * Tune $\varepsilon$ by dual averaging during the first `warmup` steps, starting from `stepSize` (Algorithm 6).
   * Default off.
   */
  adapt?: DualAveragingOptions
  /**
   * How the next point is drawn from the trajectory. `'multinomial'` (default; Stan since 2.10, Betancourt 2017,
   * appendix A.3): each point is weighted by $\exp(-H)$, subtrees are merged by progressive sampling in proportion to
   * their weights, a new doubling's sample replaces the current one with probability
   * $\min(1, w_\text{new}/w_\text{old})$ (biased progressive sampling, favouring points far from the start), and the
   * U-turn criterion is the generalised one on the summed momentum $\rhovec$, with Stan's extra checks across merged
   * subtrees. `'slice'`: Hoffman and Gelman's Algorithm 3, a slice variable $u \sim \Unif(0, e^{-H})$ and a uniform
   * draw from the points above it, with the end-point U-turn criterion.
   */
  variant?: 'multinomial' | 'slice'
}

/**
 * A point of a NUTS trajectory: position `theta`, momentum `p`, and the gradient `grad` and `logDensity` at `theta`.
 */
type Point = { theta: F64; p: F64; grad: F64; logDensity: number }
/** A subtree of the slice variant (Hoffman & Gelman's `BuildTree`). */
type Tree = {
  /** The leftmost point (earliest in integration time). */
  minus: Point
  /** The rightmost point (latest in integration time). */
  plus: Point
  /** The point drawn uniformly from the subtree's points inside the slice. */
  candidate: Point
  /** How many of its points lie inside the slice. */
  n: number
  /** False once the subtree diverged or made a U-turn. */
  keepGoing: boolean
  /** True when its last-built leaf diverged. */
  divergent: boolean
}

/**
 * The No-U-Turn Sampler (Hoffman & Gelman, 2014): the trajectory doubles forwards or backwards at random until it
 * starts to turn back, and the next point is drawn from the whole trajectory. By default the draw is multinomial
 * (Stan's sampler; Betancourt, 2017): points are weighted by $\exp(H_0 - H)$, and the trajectory stops when the summed
 * momentum $\rhovec$ satisfies $\rhovec^\top \Mmat^{-1}\pvec^- \le 0$ or $\rhovec^\top \Mmat^{-1}\pvec^+ \le 0$ on
 * any subtree. `variant: 'slice'` gives Hoffman and Gelman's Algorithm 3 ("efficient NUTS"): a slice variable
 * $u \sim \Unif(0, e^{-H})$ picks the acceptable points, and the end-point criterion
 * $(\thetavec^+ - \thetavec^-)^\top \Mmat^{-1}\pvec < 0$ stops the doubling. The step size is fixed, or tuned during
 * warmup by dual averaging with `adapt` (Algorithm 6). Step $t$ draws the momentum from `child(ctx.stream, 'momentum')`
 * and every other draw from its step stream `ctx.stream`. Throws `DomainError` for a step size or option out of
 * range.
 *
 * @param target The target, through `logDensity`, `grad` (or autodiff of `logDensity`) and `dim`.
 * @param options The step size, largest depth, mass, divergence threshold, adaptation and `variant`.
 * @returns The sampler as an algorithm: start it from `{ x0 }` and run it with `run`, `trace` or `sampleChains`.
 *
 * @example Moments of a correlated Gaussian
 * const target = {
 *   kind: 'log-density', dim: 2, normalised: false,
 *   logDensity: (x) => {
 *     const [a, b] = x.data
 *     return -(a * a - 1.6 * a * b + b * b) / 0.72
 *   },
 *   grad: (x) => {
 *     const [a, b] = x.data
 *     return [-(a - 0.8 * b) / 0.36, -(b - 0.8 * a) / 0.36]
 *   },
 * }
 * const { draws, traces } = sampleChains(nuts(target, { stepSize: 0.3 }), { x0: [0, 0] }, {
 *   chains: 2, steps: 200, stream: stream(2),
 * })
 * const x = reshape(draws, [-1, 2])
 * print('mean =', mean(x, 0))
 * print('second moments =', div(matmul(transpose(x), x), x.shape[0]))
 * print('ESS =', effectiveSampleSize(draws))
 * print('gradient evaluations per draw =', traces[0].final.gradientEvaluations / 200)
 *
 * @example One step's tree
 * const target = {
 *   kind: 'log-density', dim: 2, normalised: false,
 *   logDensity: (x) => {
 *     const [a, b] = x.data
 *     return -(a * a - 1.6 * a * b + b * b) / 0.72
 *   },
 *   grad: (x) => {
 *     const [a, b] = x.data
 *     return [-(a - 0.8 * b) / 0.36, -(b - 0.8 * a) / 0.36]
 *   },
 * }
 * for (const variant of ['multinomial', 'slice']) {
 *   const s = run(nuts(target, { stepSize: 0.3, variant }), { x0: [1, 1] }, 1)
 *   print(`${variant}: depth`, s.treeDepth, 'leapfrog steps', s.leapfrogSteps, 'accept stat', s.acceptStat)
 *   print('  integration times =', s.trajectoryTimes)
 * }
 */
export function nuts(target: LogDensity, options: NutsOptions = {}): Algorithm<ChainStart, NutsState> {
  const name = 'nuts'
  const { stepSize: stepSize0 = 0.1, maxDepth = 10, variant = 'multinomial' } = options
  const d = target.dim
  const divergenceThreshold = divergenceLimit(options.divergenceThreshold ?? 1000, d, name)
  const adapt = resolveDualAveraging(options.adapt, stepSize0, name)
  const mass = perCoordinate(options.mass ?? 1, d, name)
  const sqrtMass = mass.map(Math.sqrt)

  const noUTurn = (minus: F64, plus: F64, p: F64) => {
    let s = 0
    for (let i = 0; i < d; i++) s += ((plus[i] - minus[i]) * p[i]) / mass[i]
    return s >= 0
  }

  /**
   * The generalised criterion (Betancourt, 2017): the trajectory has not turned while $\rhovec^\top \Mmat^{-1}\pvec > 0$
   * at both ends.
   */
  const sharp = (p: F64) => p.map((pi, i) => pi / mass[i])
  const dot = (a: F64, b: F64) => {
    let s = 0
    for (let i = 0; i < d; i++) s += a[i] * b[i]
    return s
  }
  const addVec = (a: F64, b: F64) => a.map((ai, i) => ai + b[i])
  const criterion = (sharpMinus: F64, sharpPlus: F64, rho: F64) => dot(sharpPlus, rho) > 0 && dot(sharpMinus, rho) > 0

  return {
    name: variant === 'slice' ? 'nuts-slice' : name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const start = hmcStart(target, x, name)
      const X = vec(x)
      return {
        t: 0,
        x: X,
        logDensity: start.value,
        grad: vec(start.grad),
        proposal: X,
        proposalLogDensity: start.value,
        logAcceptanceRatio: NaN,
        acceptance: NaN,
        accepted: false,
        acceptedCount: 0,
        acceptanceRate: NaN,
        momentum: vec(new Float64Array(d)),
        trajectory: mat(Float64Array.from(x), 1, d),
        trajectoryTimes: vec(Float64Array.of(0)),
        energies: vec(Float64Array.of(-start.value)),
        treeDepth: 0,
        leapfrogSteps: 0,
        divergent: false,
        divergentCount: 0,
        rejectedCount: 0,
        divergenceLimit: divergenceThreshold,
        gradientEvaluations: 1,
        hitMaxDepth: false,
        stepSize: stepSize0,
        acceptStat: NaN,
        ...dualAveragingStart(stepSize0, adapt),
        diverged: start.diverged,
      }
    },
    step: (s, ctx) => {
      const draws = ctx.stream
      const z = standardNormals(child(draws, 'momentum'), d)
      const p0 = z.map((zi, i) => zi * sqrtMass[i])
      const x0 = data(s.x)
      const joint0 = s.logDensity - kinetic(p0, mass)
      // Slice variant: log u with u ~ U(0, exp(joint0)).
      const logU = variant === 'slice' ? joint0 + Math.log(uniform(draws)) : NaN
      const visited: { time: number; theta: F64; energy: number }[] = [{ time: 0, theta: x0, energy: -joint0 }]
      const stepSize = s.nextStepSize
      let steps = 0
      // Σ min(1, exp(H₀ − H)) over the leapfrog points, for the acceptance statistic.
      let acceptSum = 0

      const buildTree = (from: Point, direction: 1 | -1, depth: number, time: number): Tree & { time: number } => {
        if (depth === 0) {
          const next = leapfrogStep(target, from.theta, from.p, from.grad, direction * stepSize, mass)
          steps++
          const joint = next.logDensity - kinetic(next.p, mass)
          const nextTime = time + direction
          visited.push({ time: nextTime, theta: next.theta, energy: -joint })
          const point: Point = { theta: next.theta, p: next.p, grad: next.grad, logDensity: next.logDensity }
          const ok = Number.isFinite(joint)
          acceptSum += ok ? Math.min(1, Math.exp(joint - joint0)) : 0
          const divergent = !ok || logU - joint >= divergenceThreshold
          return {
            minus: point,
            plus: point,
            candidate: point,
            n: ok && logU <= joint ? 1 : 0,
            keepGoing: !divergent,
            divergent,
            time: nextTime,
          }
        }
        const first = buildTree(from, direction, depth - 1, time)
        if (!first.keepGoing) return first
        const edge = direction === -1 ? first.minus : first.plus
        const second = buildTree(edge, direction, depth - 1, first.time)
        const n = first.n + second.n
        const candidate = n > 0 && uniform(draws) < second.n / n ? second.candidate : first.candidate
        const minus = direction === -1 ? second.minus : first.minus
        const plus = direction === -1 ? first.plus : second.plus
        return {
          minus,
          plus,
          candidate,
          n,
          keepGoing:
            second.keepGoing && noUTurn(minus.theta, plus.theta, minus.p) && noUTurn(minus.theta, plus.theta, plus.p),
          divergent: second.divergent,
          time: second.time,
        }
      }

      const here: Point = { theta: x0, p: p0, grad: data(s.grad), logDensity: s.logDensity }

      /** Hoffman and Gelman's Algorithm 3 from `here`. */
      const sliceTransition = () => {
        let minus = here
        let plus = here
        let timeMinus = 0
        let timePlus = 0
        let chosen = here
        let n = 1
        let depth = 0
        let keepGoing = true
        let divergent = false
        while (keepGoing && depth < maxDepth) {
          const direction: 1 | -1 = uniform(draws) < 0.5 ? -1 : 1
          const tree = buildTree(
            direction === -1 ? minus : plus,
            direction,
            depth,
            direction === -1 ? timeMinus : timePlus,
          )
          if (direction === -1) {
            minus = tree.minus
            timeMinus = tree.time
          } else {
            plus = tree.plus
            timePlus = tree.time
          }
          divergent ||= tree.divergent
          if (tree.keepGoing && uniform(draws) < Math.min(1, tree.n / n)) chosen = tree.candidate
          n += tree.n
          keepGoing =
            tree.keepGoing && noUTurn(minus.theta, plus.theta, minus.p) && noUTurn(minus.theta, plus.theta, plus.p)
          depth++
        }
        return { chosen, depth, divergent, keepGoing }
      }

      /**
       * Stan's transition (base_nuts.hpp): a subtree of $2^\text{depth}$ points from `from` in `direction`,
       * returning its multinomial sample, $\log \sum \exp(H_0 - H)$, summed momentum, the momenta at both ends (first
       * and last integrated) and whether it is valid (no divergence, no U-turn on it or on any of its subtrees).
       */
      type Subtree = {
        edge: Point
        sample: Point
        logWeight: number
        rho: F64
        pBegin: F64
        pEnd: F64
        valid: boolean
        divergent: boolean
        time: number
      }
      const H0 = -joint0
      const multinomialTree = (from: Point, direction: 1 | -1, depth: number, time: number): Subtree => {
        if (depth === 0) {
          const next = leapfrogStep(target, from.theta, from.p, from.grad, direction * stepSize, mass)
          steps++
          let H = -next.logDensity + kinetic(next.p, mass)
          if (Number.isNaN(H)) H = Infinity
          const nextTime = time + direction
          visited.push({ time: nextTime, theta: next.theta, energy: H })
          const point: Point = { theta: next.theta, p: next.p, grad: next.grad, logDensity: next.logDensity }
          acceptSum += H0 - H > 0 ? 1 : Math.exp(H0 - H)
          const divergent = !Number.isFinite(H) || !allFinite(next.theta) || H - H0 > divergenceThreshold
          return {
            edge: point,
            sample: point,
            logWeight: H0 - H,
            rho: next.p,
            pBegin: next.p,
            pEnd: next.p,
            valid: !divergent,
            divergent,
            time: nextTime,
          }
        }
        const first = multinomialTree(from, direction, depth - 1, time)
        if (!first.valid) return first
        const second = multinomialTree(first.edge, direction, depth - 1, first.time)
        if (!second.valid) return second
        const logWeight = logAddExp(first.logWeight, second.logWeight)
        // Progressive sampling within a subtree: the second half's sample in proportion to its weight.
        const sample =
          second.logWeight > logWeight || uniform(draws) < Math.exp(second.logWeight - logWeight)
            ? second.sample
            : first.sample
        const rho = addVec(first.rho, second.rho)
        // The criterion on the whole subtree, and on each half extended by the neighbouring point of the other half.
        const valid =
          criterion(sharp(first.pBegin), sharp(second.pEnd), rho) &&
          criterion(sharp(first.pBegin), sharp(second.pBegin), addVec(first.rho, second.pBegin)) &&
          criterion(sharp(first.pEnd), sharp(second.pEnd), addVec(second.rho, first.pEnd))
        return {
          edge: second.edge,
          sample,
          logWeight,
          rho,
          pBegin: first.pBegin,
          pEnd: second.pEnd,
          valid,
          divergent: false,
          time: second.time,
        }
      }

      /** Stan's top-level loop: biased progressive sampling between the current trajectory and each new doubling. */
      const multinomialTransition = () => {
        // Each end of the trajectory: its edge point (to integrate from) and the time of that point.
        let back = { edge: here, time: 0 }
        let fwd = { edge: here, time: 0 }
        let rho = p0
        let chosen = here
        let logWeight = 0
        let depth = 0
        let divergent = false
        let keepGoing = true
        while (depth < maxDepth) {
          const direction: 1 | -1 = uniform(draws) < 0.5 ? -1 : 1
          const side = direction === 1 ? fwd : back
          const other = direction === 1 ? back : fwd
          const tree = multinomialTree(side.edge, direction, depth, side.time)
          divergent ||= tree.divergent
          if (!tree.valid) {
            keepGoing = false
            break
          }
          depth++
          if (tree.logWeight > logWeight || uniform(draws) < Math.exp(tree.logWeight - logWeight)) chosen = tree.sample
          logWeight = logAddExp(logWeight, tree.logWeight)
          // The criterion on the whole trajectory; on the old trajectory extended by the new subtree's first point; and
          // on the new subtree extended by the old trajectory's point next to it (as build_tree checks merged halves).
          const rhoOld = rho
          rho = addVec(rhoOld, tree.rho)
          const adjacent = side.edge.p
          const persist =
            criterion(sharp(other.edge.p), sharp(tree.pEnd), rho) &&
            criterion(sharp(other.edge.p), sharp(tree.pBegin), addVec(rhoOld, tree.pBegin)) &&
            criterion(sharp(adjacent), sharp(tree.pEnd), addVec(tree.rho, adjacent))
          if (direction === 1) fwd = { edge: tree.edge, time: tree.time }
          else back = { edge: tree.edge, time: tree.time }
          if (!persist) {
            keepGoing = false
            break
          }
        }
        return { chosen, depth, divergent, keepGoing }
      }

      const { chosen, depth, divergent, keepGoing } = variant === 'slice' ? sliceTransition() : multinomialTransition()
      visited.sort((a, b) => a.time - b.time)
      const trajectory = new Float64Array(visited.length * d)
      visited.forEach((v, i) => trajectory.set(v.theta, i * d))
      const moved = chosen !== here
      const acceptedCount = s.acceptedCount + (moved ? 1 : 0)
      const acceptStat = steps > 0 ? acceptSum / steps : 0
      return {
        ...s,
        t: s.t + 1,
        x: vec(chosen.theta),
        logDensity: chosen.logDensity,
        grad: vec(chosen.grad),
        proposal: vec(chosen.theta),
        proposalLogDensity: chosen.logDensity,
        logAcceptanceRatio: NaN,
        acceptance: NaN,
        accepted: moved,
        acceptedCount,
        acceptanceRate: acceptedCount / (s.t + 1),
        momentum: vec(p0),
        trajectory: mat(trajectory, visited.length, d),
        trajectoryTimes: vec(Float64Array.from(visited, (v) => v.time)),
        energies: vec(Float64Array.from(visited, (v) => v.energy)),
        treeDepth: depth,
        leapfrogSteps: steps,
        divergent,
        divergentCount: s.divergentCount + (divergent ? 1 : 0),
        rejectedCount: s.rejectedCount + (!moved && !divergent ? 1 : 0),
        gradientEvaluations: s.gradientEvaluations + steps,
        hitMaxDepth: keepGoing && depth >= maxDepth,
        stepSize,
        acceptStat,
        ...dualAveragingUpdate(s, s.t + 1, acceptStat, adapt),
      }
    },
  }
}

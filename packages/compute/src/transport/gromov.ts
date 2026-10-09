/**
 * Entropic Gromov–Wasserstein transport between two metric-measure spaces (Peyré, Cuturi and Solomon, 2016,
 * "Gromov–Wasserstein averaging of kernel and distance matrices", ICML, Algorithm 1 with the square loss): points are
 * matched by how they relate to the other points of their own space, not by a cross-space cost. Each step linearises
 * the quadratic objective at the current plan and solves the entropic problem with Sinkhorn.
 *
 * With intra-space matrices $\Cmat^x$ ($n \times n$) and $\Cmat^y$ ($m \times m$) and weights $\avec$, $\bvec$, the
 * objective is $\sum_{ijkl} (C^x_{ik} - C^y_{jl})^2 T_{ij} T_{kl}$ over couplings $\Tmat$ of $\avec$ and $\bvec$. For
 * such a coupling it equals $\inner{\Lmat(\Tmat)}{\Tmat}$ with the linearised cost
 * $\Lmat(\Tmat) = \cvec_x\ones^\top + \ones\cvec_y^\top - 2\Cmat^x\Tmat(\Cmat^y)^\top$, where
 * $(\cvec_x)_i = \sum_k (C^x_{ik})^2 a_k$ and $(\cvec_y)_j = \sum_l (C^y_{jl})^2 b_l$ (their Proposition 1). The
 * problem is not convex, so the plan found depends on the start, here the product coupling $\avec\bvec^\top$.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Scalar, Size, Status } from 'aifn-compute/foundation/contracts'
import { readCost, readVector, sinkhorn, type CostInput, type WeightsInput } from './discrete'

/** The two metric-measure spaces of `gromovWassersteinSteps`. */
export interface GromovProblem {
  /** The intra-space distance (or similarity) matrix $\Cmat^x$ of the first space, $n \times n$. */
  cx: CostInput
  /** The intra-space distance (or similarity) matrix $\Cmat^y$ of the second space, $m \times m$. */
  cy: CostInput
  /** The weights $\avec$ of the first space's $n$ points. */
  a: WeightsInput
  /** The weights $\bvec$ of the second space's $m$ points, with the same total as $\avec$. */
  b: WeightsInput
}

/** Options for `gromovWassersteinSteps` and `gromovWasserstein`. */
export interface GromovOptions {
  /** Entropic regularisation $\varepsilon > 0$ of each inner Sinkhorn solve, in units of the linearised cost. */
  epsilon: Scalar
  /** Sinkhorn steps per outer step. Default 200. */
  innerSteps?: Size
  /** Stop when no entry of the plan changes by as much as this in a step. Default 1e-7. */
  tolerance?: Scalar
}

/** One state of entropic Gromov–Wasserstein. */
export interface GromovState extends Status {
  /** The coupling $\Tmat$, $n \times m$. */
  plan: Tensor
  /**
   * The GW objective $\sum_{ijkl} (C^x_{ik} - C^y_{jl})^2 T_{ij} T_{kl}$ at the plan, computed as
   * $\inner{\Lmat(\Tmat)}{\Tmat}$ with the marginals taken to be $\avec$ and $\bvec$ (exact for the column marginal,
   * and to Sinkhorn's tolerance for the row one).
   */
  loss: Scalar
  /** The linearised cost $\Lmat(\Tmat)$ at the plan, $n \times m$: the cost of the next Sinkhorn solve. */
  linearCost: Tensor
  /** Largest change of a plan entry in the last step (infinite at the start). */
  change: Scalar
  /** The last change is below `tolerance`. */
  converged: boolean
  /** The loss or the plan is not finite. */
  diverged: boolean
}

/**
 * The linearised square-loss cost $\Lmat(\Tmat) = \cvec_x\ones^\top + \ones\cvec_y^\top - 2\Cmat^x\Tmat(\Cmat^y)^\top$
 * at a plan, with $(\cvec_x)_i = \sum_k (C^x_{ik})^2 a_k$ and $(\cvec_y)_j = \sum_l (C^y_{jl})^2 b_l$, and the GW
 * objective $\inner{\Lmat(\Tmat)}{\Tmat}$ there.
 *
 * @param Cx The first space's matrix $\Cmat^x$, row-major, $n \times n$.
 * @param Cy The second space's matrix $\Cmat^y$, row-major, $m \times m$.
 * @param a The weights $\avec$ of the first space ($n$), standing in for the plan's row sums.
 * @param b The weights $\bvec$ of the second space ($m$), standing in for the plan's column sums.
 * @param T The plan $\Tmat$, row-major, $n \times m$; not modified.
 * @returns `L`, the linearised cost (row-major, $n \times m$), and `loss`, the objective at $\Tmat$.
 */
function linearise(Cx: Float64Array, Cy: Float64Array, a: Float64Array, b: Float64Array, T: Float64Array) {
  const n = a.length
  const m = b.length
  // Square loss: L(T) = const − 2 Cx T Cyᵀ with const_ij = Σ_k Cx_ik² a_k + Σ_l Cy_jl² b_l.
  const rowTerm = new Float64Array(n)
  const colTerm = new Float64Array(m)
  for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) rowTerm[i] += Cx[i * n + k] ** 2 * a[k]
  for (let j = 0; j < m; j++) for (let l = 0; l < m; l++) colTerm[j] += Cy[j * m + l] ** 2 * b[l]
  const CxT = new Float64Array(n * m)
  for (let i = 0; i < n; i++)
    for (let k = 0; k < n; k++) {
      const c = Cx[i * n + k]
      if (c === 0) continue
      for (let l = 0; l < m; l++) CxT[i * m + l] += c * T[k * m + l]
    }
  const L = new Float64Array(n * m)
  let loss = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let cross = 0
      for (let l = 0; l < m; l++) cross += CxT[i * m + l] * Cy[j * m + l]
      L[i * m + j] = rowTerm[i] + colTerm[j] - 2 * cross
      loss += L[i * m + j] * T[i * m + j]
    }
  return { L, loss }
}

/**
 * Entropic Gromov–Wasserstein as a traceable algorithm, started from the product coupling $\avec\bvec^\top$ (Peyré,
 * Cuturi and Solomon, 2016, Algorithm 1). Each step solves the entropic transport problem with the linearised cost
 * $\Lmat(\Tmat)$ of the current plan by `sinkhorn` (at most `innerSteps` steps, from zero potentials) and takes its
 * plan as the next. `init` takes no start (`undefined`).
 *
 * @param problem The two spaces: their intra-space matrices and their weights.
 * @param options The regularisation and the stopping rules.
 * @param options.epsilon The entropic regularisation $\varepsilon > 0$ of each Sinkhorn solve.
 * @param options.innerSteps The most Sinkhorn steps per outer step.
 * @param options.tolerance The largest change of a plan entry at which a step counts as `converged`.
 * @returns The algorithm: its state holds the plan, the objective and the linearised cost.
 *
 * @example The objective falls from the product coupling
 * // Points 0, 1, 3 on one line and 12, 13, 10 on another: the same distances, listed in a different order.
 * const dist = (p) => p.map((x) => p.map((y) => Math.abs(x - y)))
 * const problem = { cx: dist([0, 1, 3]), cy: dist([12, 13, 10]), a: uniformWeights(3), b: uniformWeights(3) }
 * const alg = gromovWassersteinSteps(problem, { epsilon: 0.1 })
 * for (const steps of [0, 1, 5]) print(`after ${steps} steps: loss =`, run(alg, undefined, steps).loss)
 */
export function gromovWassersteinSteps(
  problem: GromovProblem,
  { epsilon, innerSteps = 200, tolerance = 1e-7 }: GromovOptions,
): Algorithm<undefined, GromovState> {
  const a = readVector(problem.a, 'gromovWasserstein a')
  const b = readVector(problem.b, 'gromovWasserstein b')
  const n = a.length
  const m = b.length
  const Cx = readCost(problem.cx, n, n, 'gromovWasserstein cx')
  const Cy = readCost(problem.cy, m, m, 'gromovWasserstein cy')
  return {
    name: 'gromov-wasserstein',
    init: () => {
      const T = new Float64Array(n * m)
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[i * m + j] = a[i] * b[j]
      const { L, loss } = linearise(Cx, Cy, a, b, T)
      return {
        t: 0,
        plan: fromData(T, [n, m]),
        loss,
        linearCost: fromData(L, [n, m]),
        change: Infinity,
        converged: false,
        diverged: !Number.isFinite(loss),
      }
    },
    step: (s) => {
      const solved = sinkhorn(a, b, s.linearCost, { epsilon, maxSteps: innerSteps })
      const T = dense.data(solved.plan)
      const prev = dense.data(s.plan)
      let change = 0
      for (let k = 0; k < T.length; k++) change = Math.max(change, Math.abs(T[k] - prev[k]))
      const { L, loss } = linearise(Cx, Cy, a, b, T)
      return {
        t: s.t + 1,
        plan: solved.plan,
        loss,
        linearCost: fromData(L, [n, m]),
        change,
        converged: change < tolerance,
        diverged: !Number.isFinite(loss) || !Number.isFinite(change),
      }
    },
  }
}

/**
 * Entropic Gromov–Wasserstein, `gromovWassersteinSteps` run to convergence, divergence or `maxSteps` (default 50) outer
 * steps. Check `converged` on the result.
 *
 * @param problem The two spaces: their intra-space matrices and their weights.
 * @param options The regularisation, the inner steps and tolerance of `GromovOptions`, and `maxSteps`, the most outer
 *   steps.
 * @returns The last state: the plan, the objective and whether it converged.
 *
 * @example Isometric spaces are matched point for point
 * // 0, 1, 3 and 12, 13, 10 have the same distances: 0 matches 13, 1 matches 12 and 3 matches 10.
 * const dist = (p) => p.map((x) => p.map((y) => Math.abs(x - y)))
 * const problem = { cx: dist([0, 1, 3]), cy: dist([12, 13, 10]), a: uniformWeights(3), b: uniformWeights(3) }
 * const r = gromovWasserstein(problem, { epsilon: 0.1 })
 * print('plan =', r.plan)
 * print('loss =', r.loss, ' converged:', r.converged, ' steps:', r.t)
 */
export function gromovWasserstein(problem: GromovProblem, options: GromovOptions & { maxSteps?: Size }): GromovState {
  return run(gromovWassersteinSteps(problem, options), undefined, options.maxSteps ?? 50)
}

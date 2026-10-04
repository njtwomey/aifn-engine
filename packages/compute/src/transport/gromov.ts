/**
 * Entropic Gromov–Wasserstein transport between two metric-measure spaces (Peyré, Cuturi and Solomon, 2016, "Gromov–
 * Wasserstein averaging of kernel and distance matrices", ICML, Algorithm 1 with the square loss): points are matched by
 * how they relate to the other points of their own space, not by a cross-space cost. Each step linearises the quadratic
 * objective at the current plan and solves the entropic problem with Sinkhorn.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import type { Scalar, Size, Status } from 'aifn-compute/foundation/contracts'
import { readCost, readVector, sinkhorn, type CostInput, type WeightsInput } from './discrete'

/** The two metric-measure spaces of `gromovWassersteinSteps`. */
export interface GromovProblem {
  /** Intra-space distance (or similarity) matrices, n × n and m × m. */
  cx: CostInput
  cy: CostInput
  /** Weights of the points of each space. */
  a: WeightsInput
  b: WeightsInput
}

/** Options for `gromovWassersteinSteps` and `gromovWasserstein`. */
export interface GromovOptions {
  /** Entropic regularisation ε > 0 of each inner Sinkhorn solve. */
  epsilon: Scalar
  /** Sinkhorn steps per outer step. Default 200. */
  innerSteps?: Size
  /** Stop when the plan changes by less than this (max abs). Default 1e-7. */
  tolerance?: Scalar
}

/** One state of entropic Gromov–Wasserstein. */
export interface GromovState extends Status {
  /** The coupling, n × m. */
  plan: Tensor
  /** The GW objective Σ_{ijkl} (Cx_ik − Cy_jl)² T_ij T_kl at the plan. */
  loss: Scalar
  /** The linearised cost the last Sinkhorn solve used, n × m. */
  linearCost: Tensor
  /** Largest change of a plan entry in the last step. */
  change: Scalar
  converged: boolean
  /** The loss or the plan is not finite. */
  diverged: boolean
}

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
 * Entropic Gromov–Wasserstein as a traceable algorithm, started from the product coupling a bᵀ (Peyré, Cuturi and
 * Solomon, 2016, Algorithm 1). `init` takes no start (`undefined`).
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

/** Entropic Gromov–Wasserstein run to convergence or `maxSteps` (default 50) outer steps. */
export function gromovWasserstein(problem: GromovProblem, options: GromovOptions & { maxSteps?: Size }): GromovState {
  return run(gromovWassersteinSteps(problem, options), undefined, options.maxSteps ?? 50)
}

/**
 * Non-negative matrix factorisation X ≈ WH with W, H ≥ 0 (Paatero and Tapper, 1994; Lee and Seung, 1999) as a
 * step-through algorithm, by one of two solvers:
 *
 * - `multiplicative`: the updates of Lee and Seung (2001), for the squared Frobenius error and the generalised
 *   Kullback–Leibler divergence, as scikit-learn's `NMF(solver='mu')`: W first, then H, each a gradient step whose step
 *   size makes it a ratio of the gradient's negative and positive parts, so entries stay non-negative and the
 *   objective never increases.
 * - `hals`: hierarchical alternating least squares (Cichocki and Phan, 2009) for the Frobenius error: each column of W,
 *   then each row of H, is the exact non-negative minimiser with the others fixed, as scikit-learn's
 *   `NMF(solver='cd')` without shuffling. Each update is exact, so the objective never increases either.
 */

import type { MatrixLike, Size, Status } from 'aifn-compute/foundation/contracts'
import { normal, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { run } from 'aifn-compute/foundation/trace'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The objective: the squared Frobenius error ½‖X − WH‖² or the generalised KL divergence Σ X log(X/WH) − X + WH. */
export type NmfLoss = 'frobenius' | 'kullback-leibler'

/** The update rule: Lee–Seung multiplicative updates, or HALS coordinate descent (Frobenius error only). */
export type NmfSolver = 'multiplicative' | 'hals'

/** Options of `nmfSteps` and `nmf`. */
export type NmfOptions = {
  /** The inner dimension k (the number of parts or topics). */
  rank: Size
  /** The objective (default `frobenius`). */
  loss?: NmfLoss
  /** The solver (default `multiplicative`); `hals` requires the Frobenius loss. */
  solver?: NmfSolver
  /** Initial factors W [m, k] and H [k, n]; default random, |N(0, 1)|·√(mean X / k) as scikit-learn's `random` init. */
  init?: { W: MatrixLike; H: MatrixLike }
  /** Stop when the relative decrease of the objective in one step falls below this (default 1e-4). */
  tolerance?: number
}

/** The state of `nmfSteps`. */
export interface NmfState extends Status {
  t: Size
  /** W [m, k] and H [k, n]. */
  W: Tensor
  H: Tensor
  /** The objective at (W, H). */
  objective: number
  converged: boolean
}

/** The smallest denominator used (scikit-learn's float32 epsilon), so a zero column does not divide by zero. */
const EPSILON = 1.1920929e-7

function product(W: Float64Array, H: Float64Array, m: Size, k: Size, n: Size): Float64Array {
  return dense.matMul(W, H, m, k, n)
}

function objectiveOf(X: Float64Array, WH: Float64Array, loss: NmfLoss): number {
  let total = 0
  if (loss === 'frobenius') {
    for (let i = 0; i < X.length; i++) total += (X[i] - WH[i]) ** 2
    return total / 2
  }
  for (let i = 0; i < X.length; i++) {
    const x = X[i]
    const y = Math.max(WH[i], EPSILON)
    total += (x > 0 ? x * Math.log(x / y) : 0) - x + y
  }
  return total
}

/** One multiplicative update of W (m × k) given H (k × n): W ← W ∘ (negative part)/(positive part) of the gradient. */
function updateW(X: Float64Array, W: Float64Array, H: Float64Array, m: Size, k: Size, n: Size, loss: NmfLoss) {
  const out = new Float64Array(m * k)
  if (loss === 'frobenius') {
    // W ← W ∘ (X Hᵀ) / (W H Hᵀ)
    const HHt = new Float64Array(k * k)
    for (let a = 0; a < k; a++)
      for (let b = 0; b < k; b++) {
        let s = 0
        for (let j = 0; j < n; j++) s += H[a * n + j] * H[b * n + j]
        HHt[a * k + b] = s
      }
    for (let i = 0; i < m; i++)
      for (let a = 0; a < k; a++) {
        let num = 0
        for (let j = 0; j < n; j++) num += X[i * n + j] * H[a * n + j]
        let den = 0
        for (let b = 0; b < k; b++) den += W[i * k + b] * HHt[b * k + a]
        out[i * k + a] = (W[i * k + a] * num) / (den === 0 ? EPSILON : den)
      }
    return out
  }
  // KL: W ← W ∘ ((X / WH) Hᵀ) / (1 Hᵀ)
  const WH = product(W, H, m, k, n)
  const rowSums = new Float64Array(k)
  for (let a = 0; a < k; a++) for (let j = 0; j < n; j++) rowSums[a] += H[a * n + j]
  for (let i = 0; i < m; i++)
    for (let a = 0; a < k; a++) {
      let num = 0
      for (let j = 0; j < n; j++) num += (X[i * n + j] / Math.max(WH[i * n + j], EPSILON)) * H[a * n + j]
      out[i * k + a] = (W[i * k + a] * num) / (rowSums[a] === 0 ? EPSILON : rowSums[a])
    }
  return out
}

/**
 * One HALS sweep over the columns of W (m × k) given H (k × n): column a is set to max(0, w_a − ∇_a / (HHᵀ)_aa), its
 * exact minimiser with the other columns fixed, where ∇_a = W (HHᵀ)_{:,a} − (XHᵀ)_{:,a} uses the columns already updated.
 */
function halsW(X: Float64Array, W0: Float64Array, H: Float64Array, m: Size, k: Size, n: Size) {
  const W = Float64Array.from(W0)
  const HHt = new Float64Array(k * k)
  for (let a = 0; a < k; a++)
    for (let b = 0; b < k; b++) {
      let s = 0
      for (let j = 0; j < n; j++) s += H[a * n + j] * H[b * n + j]
      HHt[a * k + b] = s
    }
  const XHt = new Float64Array(m * k)
  for (let i = 0; i < m; i++)
    for (let a = 0; a < k; a++) {
      let s = 0
      for (let j = 0; j < n; j++) s += X[i * n + j] * H[a * n + j]
      XHt[i * k + a] = s
    }
  for (let a = 0; a < k; a++) {
    const hess = HHt[a * k + a]
    if (hess === 0) continue
    for (let i = 0; i < m; i++) {
      let g = -XHt[i * k + a]
      for (let b = 0; b < k; b++) g += W[i * k + b] * HHt[b * k + a]
      W[i * k + a] = Math.max(W[i * k + a] - g / hess, 0)
    }
  }
  return W
}

const transposed = (a: Float64Array, r: Size, c: Size) => {
  const o = new Float64Array(r * c)
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) o[j * r + i] = a[i * c + j]
  return o
}

/** The update of H given W, by the same rule applied to Xᵀ ≈ HᵀWᵀ: Hᵀ (n × k) is updated as the "W" of Xᵀ (n × m). */
function updateH(
  X: Float64Array,
  W: Float64Array,
  H: Float64Array,
  m: Size,
  k: Size,
  n: Size,
  loss: NmfLoss,
  solver: NmfSolver,
) {
  const Xt = transposed(X, m, n)
  const Ht = transposed(H, k, n)
  const Wt = transposed(W, m, k)
  const next = solver === 'hals' ? halsW(Xt, Ht, Wt, n, k, m) : updateW(Xt, Ht, Wt, n, k, m, loss)
  return transposed(next, n, k)
}

function readMatrix(a: MatrixLike, where: string): { data: Float64Array; m: Size; n: Size } {
  const r = dense.toMatrixF64(a, where)
  return { data: Float64Array.from(r.data), m: r.m, n: r.n }
}

/**
 * Non-negative matrix factorisation by multiplicative updates or HALS, one sweep (W then H) per step. `init` takes a stream
 * for the random initial factors (unused when `options.init` gives them). The objective is non-increasing (Lee and
 * Seung, 2001); `converged` is set when its relative decrease falls below the tolerance.
 */
export function nmfSteps(X: MatrixLike, options: NmfOptions): Algorithm<void, NmfState> {
  const { rank: k, loss = 'frobenius', solver = 'multiplicative', tolerance = 1e-4 } = options
  const x = readMatrix(X, 'nmfSteps')
  const { m, n } = x
  for (const v of x.data) if (!(v >= 0)) throw new DomainError('nmfSteps', 'nmfSteps: X must be non-negative')
  if (!(Number.isInteger(k) && k >= 1))
    throw new DomainError('nmfSteps', `nmfSteps: rank must be a positive integer, got ${k}`)
  if (solver === 'hals' && loss !== 'frobenius')
    throw new DomainError('nmfSteps', 'nmfSteps: HALS minimises the Frobenius loss only')
  return {
    name: solver === 'hals' ? 'nmf-hals' : 'nmf-multiplicative',
    init: (_start, s: Stream) => {
      let W: Float64Array
      let H: Float64Array
      if (options.init) {
        W = readMatrix(options.init.W, 'nmfSteps: init W').data
        H = readMatrix(options.init.H, 'nmfSteps: init H').data
        if (W.length !== m * k || H.length !== k * n)
          throw new ShapeError('nmfSteps', 'nmfSteps: init W must be [m, k] and H [k, n]')
      } else {
        let mean = 0
        for (const v of x.data) mean += v / x.data.length
        const scale = Math.sqrt(mean / k)
        H = Float64Array.from(toFlat(normal(s, 0, 1, { shape: [k * n] })), (v) => Math.abs(v) * scale)
        W = Float64Array.from(toFlat(normal(s, 0, 1, { shape: [m * k] })), (v) => Math.abs(v) * scale)
      }
      return {
        t: 0,
        W: fromData(W, [m, k]),
        H: fromData(H, [k, n]),
        objective: objectiveOf(x.data, product(W, H, m, k, n), loss),
        converged: false,
      }
    },
    step: (state) => {
      const W0 = Float64Array.from(state.W.data)
      const H0 = Float64Array.from(state.H.data)
      const W = solver === 'hals' ? halsW(x.data, W0, H0, m, k, n) : updateW(x.data, W0, H0, m, k, n, loss)
      const H = updateH(x.data, W, H0, m, k, n, loss, solver)
      const objective = objectiveOf(x.data, product(W, H, m, k, n), loss)
      const decrease = (state.objective - objective) / Math.max(state.objective, Number.MIN_VALUE)
      return {
        t: state.t + 1,
        W: fromData(W, [m, k]),
        H: fromData(H, [k, n]),
        objective,
        converged: state.t > 0 && decrease >= 0 && decrease < tolerance,
        diverged: !Number.isFinite(objective),
      }
    },
  }
}

/** Run `nmfSteps` to convergence or `maxSteps` sweeps (default 200) and return W, H and the objective. */
export function nmf(
  X: MatrixLike,
  options: NmfOptions & { maxSteps?: Size; stream?: Stream },
): { W: Tensor; H: Tensor; objective: number; steps: Size } {
  const s = run(nmfSteps(X, options), undefined, options.maxSteps ?? 200, { stream: options.stream })
  return { W: s.W, H: s.H, objective: s.objective, steps: s.t }
}

/**
 * System identification: discrete-time models fitted to an input sequence u and an output sequence y.
 *
 * - `arx`: the ARX model A(q)y = B(q)u + e by linear least squares, with its loss, FPE and AIC for order selection.
 * - `predictionErrorMethod`: the prediction-error method (PEM) for the polynomial family A(q)y = B(q)/F(q)·u + C(q)e
 *   (ARX, ARMAX, output error and their mixtures) as a step algorithm: damped Gauss–Newton (Levenberg–Marquardt) on the
 *   mean squared one-step prediction error, its gradient obtained by filtering, as in Ljung (1999), "System
 *   Identification: Theory for the User", 2nd ed., §10.2. `polynomialModel` runs it; `armax` and `outputError` name the
 *   usual structures.
 * - `n4sid`: subspace identification of a state-space model (Van Overschee & De Moor, 1994, "N4SID: subspace
 *   algorithms for the identification of combined deterministic–stochastic systems", Automatica 30(1)): the oblique
 *   projection of future outputs onto past data along future inputs, its SVD (whose singular values choose the order),
 *   A and C from the shift invariance of the extended observability matrix, then B, D and x₀ by least squares.
 *
 * Polynomials are ascending in the delay operator q⁻¹: A = [1, a₁, …, a_na]; B's first nk coefficients are zero.
 */

import { eigh, lstsq } from 'aifn-compute/numerics/linalg'
import { roots } from 'aifn-compute/numerics/polynomial'
import { linearFilter } from 'aifn-compute/foundation/convolution'
import {
  complexAbs,
  dense,
  fromData,
  toFlat,
  type Matrix,
  type Tensor,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { run } from 'aifn-compute/foundation/trace'
import type { MatrixLike, Scalar, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { stateSpace, transferFunction, type LtiOf, type StateSpaceForm, type TransferFunctionForm } from './system'

type F64 = dense.F64

const vec = (a: ArrayLike<number>): Vector => fromData(Float64Array.from(a), [a.length])

/** y filtered by b/a (ascending q⁻¹ coefficients), zero initial conditions. */
function filt(b: ArrayLike<number>, a: ArrayLike<number>, x: F64): F64 {
  return Float64Array.from(
    toFlat(linearFilter(Float64Array.from(b), Float64Array.from(a), fromData(x, [x.length])) as Tensor),
  )
}

/** x delayed by d samples (zeros shifted in). */
function delay(x: F64, d: number): F64 {
  const out = new Float64Array(x.length)
  for (let t = d; t < x.length; t++) out[t] = x[t - d]
  return out
}

/** The product of two ascending polynomials. */
function polyMulAsc(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(a.length + b.length - 1).fill(0)
  a.forEach((x, i) => b.forEach((y, j) => (out[i + j] += x * y)))
  return out
}

/** True when every root of the monic ascending polynomial [1, c₁, …] (in q⁻¹) lies strictly inside the unit circle. */
function isStable(c: readonly number[]): boolean {
  if (c.length < 2) return true
  // In powers of z: zⁿ + c₁zⁿ⁻¹ + … + cₙ has the same coefficients in descending order.
  return toFlat(complexAbs(roots(Float64Array.from(c)))).every((m) => m < 1 - 1e-9)
}

function readSeries(y: VectorLike, u: VectorLike, where: string): { y: F64; u: F64 } {
  const ys = dense.toF64(y, `${where} y`)
  const us = dense.toF64(u, `${where} u`)
  if (ys.length !== us.length) throw new ShapeError(where, `${where}: y has ${ys.length} samples, u ${us.length}`)
  return { y: ys, u: us }
}

// ── ARX by least squares ─────────────────────────────────────────────────────────────────────────────────────────────

/** The orders of a polynomial model: na, nb, nc, nf coefficients and an input delay of nk samples (nk ≥ 0). */
export type PolynomialOrders = { na?: Size; nb: Size; nc?: Size; nf?: Size; nk?: Size }

/** A fitted polynomial model A(q)y = B(q)/F(q)·u + C(q)e. */
export type PolynomialModel = {
  /** Ascending q⁻¹ coefficients, monic A, C, F; B with nk leading zeros. */
  A: number[]
  B: number[]
  C: number[]
  F: number[]
  /** The input-to-output model G = B/(AF), discrete with the given dt. */
  system: LtiOf<TransferFunctionForm>
  /** The noise model H = C/A. */
  noise: LtiOf<TransferFunctionForm>
  /** The one-step prediction errors ε(t). */
  residuals: Vector
  /** Mean squared prediction error V = (1/N)Σε² over the fitted samples (the estimate of the noise variance λ). */
  loss: Scalar
  /** Akaike's final prediction error, V(1 + d/N)/(1 − d/N), with d parameters. */
  fpe: Scalar
  /** Akaike's information criterion, N log V + 2d. */
  aic: Scalar
  /** Number of estimated parameters d. */
  parameters: Size
}

function finishModel(
  A: number[],
  B: number[],
  C: number[],
  F: number[],
  eps: F64,
  from: number,
  dt: Scalar,
): PolynomialModel {
  let s = 0
  for (let t = from; t < eps.length; t++) s += eps[t] * eps[t]
  const N = eps.length - from
  const loss = s / N
  const params = A.length - 1 + (B.length - leadingZeros(B)) + C.length - 1 + F.length - 1
  const AF = polyMulAsc(A, F)
  return {
    A,
    B,
    C,
    F,
    system: transferFunction(B, AF, { dt }),
    noise: transferFunction(C, A, { dt }),
    residuals: vec(eps),
    loss,
    fpe: (loss * (1 + params / N)) / (1 - params / N),
    aic: N * Math.log(loss) + 2 * params,
    parameters: params,
  }
}

function leadingZeros(b: readonly number[]): number {
  let k = 0
  while (k < b.length && b[k] === 0) k++
  return Math.min(k, b.length)
}

/**
 * The ARX model y(t) + a₁y(t−1) + … + a_na y(t−na) = b₁u(t−nk) + … + b_nb u(t−nk−nb+1) + e(t), fitted by linear least
 * squares on the samples t ≥ max(na, nk + nb − 1) (Ljung, 1999, §7.3). The prediction error is linear in the
 * parameters, so the fit is one solve. `dt` (default 1) labels the returned systems.
 */
export function arx(
  y: VectorLike,
  u: VectorLike,
  orders: PolynomialOrders,
  { dt = 1 }: { dt?: Scalar } = {},
): PolynomialModel {
  const { y: ys, u: us } = readSeries(y, u, 'arx')
  const na = orders.na ?? 0
  const nb = orders.nb
  const nk = orders.nk ?? 1
  if (na < 0 || nb < 0 || nk < 0) throw new DomainError('arx', 'arx: orders must be non-negative')
  const from = Math.max(na, nk + nb - 1, 0)
  const N = ys.length
  const rows = N - from
  const d = na + nb
  if (rows <= d) throw new DomainError('arx', `arx: ${N} samples are too few for ${d} parameters`)
  const phi = new Float64Array(rows * d)
  const target = new Float64Array(rows)
  for (let t = from; t < N; t++) {
    const r = t - from
    for (let i = 0; i < na; i++) phi[r * d + i] = -ys[t - 1 - i]
    for (let i = 0; i < nb; i++) phi[r * d + na + i] = us[t - nk - i]
    target[r] = ys[t]
  }
  const theta = toFlat(lstsq(fromData(phi, [rows, d]), fromData(target, [rows])).x)
  const A = [1, ...theta.slice(0, na)]
  const B = [...new Array<number>(nk).fill(0), ...theta.slice(na)]
  if (B.length === 0) B.push(0)
  const eps = new Float64Array(N)
  // ε = A y − B u: the equation error, which for ARX is the one-step prediction error.
  const Ay = filt(A, [1], ys)
  const Bu = filt(B, [1], us)
  for (let t = 0; t < N; t++) eps[t] = Ay[t] - Bu[t]
  return finishModel(A, B, [1], [1], eps, from, dt)
}

// ── The prediction-error method ──────────────────────────────────────────────────────────────────────────────────────

/** The state of `predictionErrorMethod`. */
export interface PredictionErrorState extends Status {
  /** Steps taken. */
  t: Size
  /** The parameters θ = (a, b, c, f). */
  theta: Vector
  /** The loss V(θ) = (1/N) Σ ε(t, θ)². */
  loss: Scalar
  /** The gradient norm ‖∂V/∂θ‖. */
  gradNorm: Scalar
  /** The Levenberg–Marquardt damping μ. */
  damping: Scalar
  /** Rejected trial steps in this step (a step increased the loss or made C or F unstable). */
  rejected: Size
  converged: boolean
  stalled: boolean
}

/** Options for `predictionErrorMethod`. */
export type PredictionErrorOptions = {
  /** Stop when the relative decrease of V falls below this. Default 1e-10. */
  tolerance?: Scalar
  /** The initial damping μ₀. Default 1e-3 (relative to the mean diagonal of JᵀJ). */
  damping?: Scalar
}

/** The orders, resolved. */
type Orders = { na: number; nb: number; nc: number; nf: number; nk: number }

function resolve(orders: PolynomialOrders): Orders {
  const o = { na: orders.na ?? 0, nb: orders.nb, nc: orders.nc ?? 0, nf: orders.nf ?? 0, nk: orders.nk ?? 1 }
  if (Object.values(o).some((v) => !(Number.isInteger(v) && v >= 0)))
    throw new DomainError('predictionErrorMethod', 'predictionErrorMethod: orders must be non-negative integers')
  if (o.nb < 1) throw new DomainError('predictionErrorMethod', 'predictionErrorMethod: nb must be at least 1')
  return o
}

function unpack(theta: ArrayLike<number>, o: Orders) {
  let k = 0
  const take = (n: number) => Array.from({ length: n }, () => theta[k++])
  const a = take(o.na)
  const b = take(o.nb)
  const c = take(o.nc)
  const f = take(o.nf)
  return {
    A: [1, ...a],
    B: [...new Array<number>(o.nk).fill(0), ...b],
    C: [1, ...c],
    F: [1, ...f],
  }
}

/** The prediction errors and their Jacobian ∂ε/∂θ (N × d, row-major) at θ. */
function predictionErrors(theta: ArrayLike<number>, o: Orders, y: F64, u: F64, jacobian: boolean) {
  const { A, B, C, F } = unpack(theta, o)
  const N = y.length
  const w = filt(B, F, u)
  const Ay = filt(A, [1], y)
  const v = new Float64Array(N)
  for (let t = 0; t < N; t++) v[t] = Ay[t] - w[t]
  const eps = filt([1], C, v)
  if (!jacobian) return { eps, J: null }
  const d = o.na + o.nb + o.nc + o.nf
  const J = new Float64Array(N * d)
  let col = 0
  const put = (s: F64) => {
    for (let t = 0; t < N; t++) J[t * d + col] = s[t]
    col++
  }
  const yC = filt([1], C, y)
  for (let i = 1; i <= o.na; i++) put(delay(yC, i))
  const CF = polyMulAsc(C, F)
  const uCF = filt([1], CF, u)
  for (let i = 0; i < o.nb; i++) put(delay(uCF, o.nk + i).map((s) => -s))
  const eC = filt([1], C, eps)
  for (let i = 1; i <= o.nc; i++) put(delay(eC, i).map((s) => -s))
  const wCF = filt([1], CF, w)
  for (let i = 1; i <= o.nf; i++) put(delay(wCF, i))
  return { eps, J }
}

const meanSquare = (e: F64) => e.reduce((s, v) => s + v * v, 0) / e.length

/**
 * The prediction-error method for A(q)y = B(q)/F(q)·u + C(q)e as a traceable algorithm (Ljung, 1999, §10.2). The
 * one-step predictor's error is ε = (A y − (B/F) u)/C; each step solves the damped Gauss–Newton system
 * (JᵀJ + μ·diag(JᵀJ)) δ = −Jᵀε, with J = ∂ε/∂θ obtained by filtering (∂ε/∂aᵢ = y(t−i)/C, ∂ε/∂bᵢ = −u(t−nk−i)/(CF),
 * ∂ε/∂cᵢ = −ε(t−i)/C, ∂ε/∂fᵢ = w(t−i)/(CF) with w = (B/F)u). A step that raises the loss or leaves C or F with a root
 * on or outside the unit circle is rejected and μ grows; an accepted one shrinks μ. `init` takes `{ theta0 }`, default
 * an ARX fit of the same na and nb (zeros for c and f).
 */
export function predictionErrorMethod(
  y: VectorLike,
  u: VectorLike,
  orders: PolynomialOrders,
  options: PredictionErrorOptions = {},
): Algorithm<{ theta0?: VectorLike }, PredictionErrorState> {
  const { y: ys, u: us } = readSeries(y, u, 'predictionErrorMethod')
  const o = resolve(orders)
  const d = o.na + o.nb + o.nc + o.nf
  const tol = options.tolerance ?? 1e-10
  const state = (t: number, theta: F64, mu: number, rejected: number, prevLoss: number): PredictionErrorState => {
    const { eps, J } = predictionErrors(theta, o, ys, us, true)
    const loss = meanSquare(eps)
    const g = dense.scale(2 / ys.length, dense.matTVec(J!, eps, ys.length, d))
    const rel = Number.isFinite(prevLoss) ? (prevLoss - loss) / Math.max(prevLoss, 1e-300) : Infinity
    return {
      t,
      theta: vec(theta),
      loss,
      gradNorm: dense.norm(g),
      damping: mu,
      rejected,
      converged: t > 0 && rel >= 0 && rel < tol,
      stalled: rejected >= 20,
    }
  }
  return {
    name: 'prediction-error-method',
    init: ({ theta0 } = {}) => {
      let theta: F64
      if (theta0 !== undefined) {
        theta = dense.toF64(theta0, 'predictionErrorMethod theta0')
        if (theta.length !== d)
          throw new ShapeError('predictionErrorMethod', `predictionErrorMethod: θ₀ needs ${d} values`)
      } else {
        const start = arx(ys, us, { na: o.na, nb: o.nb, nk: o.nk })
        theta = new Float64Array(d)
        start.A.slice(1).forEach((v, i) => (theta[i] = v))
        start.B.slice(o.nk).forEach((v, i) => (theta[o.na + i] = v))
      }
      return state(0, theta, options.damping ?? 1e-3, 0, Infinity)
    },
    step: (s) => {
      const theta = dense.data(s.theta)
      const { eps, J } = predictionErrors(theta, o, ys, us, true)
      const JtJ = dense.gram(J!, ys.length, d)
      const Jte = dense.matTVec(J!, eps, ys.length, d)
      let mu = s.damping
      let rejected = 0
      while (rejected < 20) {
        const M = Float64Array.from(JtJ)
        for (let i = 0; i < d; i++) M[i * d + i] += mu * Math.max(JtJ[i * d + i], 1e-12)
        const delta = toFlat(
          lstsq(
            fromData(M, [d, d]),
            fromData(
              Jte.map((v) => -v),
              [d],
            ),
          ).x,
        )
        const trial = Float64Array.from(theta, (v, i) => v + delta[i])
        const { C, F } = unpack(trial, o)
        const ok = isStable(C) && isStable(F)
        const trialLoss = ok ? meanSquare(predictionErrors(trial, o, ys, us, false).eps) : Infinity
        if (Number.isFinite(trialLoss) && trialLoss <= s.loss) return state(s.t + 1, trial, mu / 3, rejected, s.loss)
        mu *= 4
        rejected++
      }
      return { ...state(s.t + 1, Float64Array.from(theta), mu, rejected, s.loss), converged: false, stalled: true }
    },
  }
}

/**
 * A polynomial model A(q)y = B(q)/F(q)·u + C(q)e fitted by the prediction-error method (`predictionErrorMethod`, at
 * most `maxSteps` steps, default 100).
 */
export function polynomialModel(
  y: VectorLike,
  u: VectorLike,
  orders: PolynomialOrders,
  options: PredictionErrorOptions & { dt?: Scalar; maxSteps?: Size; theta0?: VectorLike } = {},
): PolynomialModel & { steps: Size; converged: boolean } {
  const { y: ys, u: us } = readSeries(y, u, 'polynomialModel')
  const o = resolve(orders)
  const s = run(predictionErrorMethod(ys, us, orders, options), { theta0: options.theta0 }, options.maxSteps ?? 100)
  const { A, B, C, F } = unpack(dense.data(s.theta), o)
  const { eps } = predictionErrors(dense.data(s.theta), o, ys, us, false)
  return { ...finishModel(A, B, C, F, eps, 0, options.dt ?? 1), steps: s.t, converged: s.converged }
}

/** The ARMAX model A(q)y = B(q)u + C(q)e by the prediction-error method. */
export function armax(
  y: VectorLike,
  u: VectorLike,
  orders: { na: Size; nb: Size; nc: Size; nk?: Size },
  options: PredictionErrorOptions & { dt?: Scalar; maxSteps?: Size } = {},
): PolynomialModel & { steps: Size; converged: boolean } {
  return polynomialModel(y, u, orders, options)
}

/** The output-error model y = B(q)/F(q)·u + e by the prediction-error method. */
export function outputError(
  y: VectorLike,
  u: VectorLike,
  orders: { nb: Size; nf: Size; nk?: Size },
  options: PredictionErrorOptions & { dt?: Scalar; maxSteps?: Size } = {},
): PolynomialModel & { steps: Size; converged: boolean } {
  return polynomialModel(y, u, orders, options)
}

// ── Subspace identification ──────────────────────────────────────────────────────────────────────────────────────────

/** Options for `n4sid`. */
export type N4sidOptions = {
  /** The model order n; default: the largest gap in the logarithms of the singular values. */
  order?: Size
  /** Block rows i of the Hankel matrices (past and future horizon); default 10. Needs i > n. */
  horizon?: Size
  /** Sampling interval of the returned system. Default 1. */
  dt?: Scalar
  /** The largest order the automatic choice considers. Default horizon − 1. */
  maxOrder?: Size
}

/** A state-space model identified by `n4sid`. */
export type SubspaceModel = {
  system: LtiOf<StateSpaceForm>
  /** The singular values of the oblique projection, descending: the order is read from their drop. */
  singularValues: Vector
  order: Size
  /** The initial state of the fitted simulation. */
  x0: Vector
  /** The simulated output of the model from x0 (N × p) and its mean squared error against y. */
  simulated: Matrix
  loss: Scalar
}

/** A sequence as an N × k matrix (a vector is one channel). */
function channels(x: MatrixLike | VectorLike, where: string): { d: F64; N: number; k: number } {
  const shape = (x as { shape?: readonly number[] }).shape
  const nested = !shape && typeof (x as ArrayLike<unknown>)[0] === 'object'
  if ((shape && shape.length === 2) || nested) {
    const { data, m, n } = dense.toMatrixF64(x as MatrixLike, where)
    return { d: data, N: m, k: n }
  }
  const d = dense.toF64(x as VectorLike, where)
  return { d, N: d.length, k: 1 }
}

/** The block Hankel matrix with `blocks` block rows of the k-channel sequence s, starting at block `from`: (blocks·k) × j. */
function hankel(s: F64, k: number, from: number, blocks: number, j: number): F64 {
  const out = new Float64Array(blocks * k * j)
  for (let b = 0; b < blocks; b++)
    for (let c = 0; c < k; c++)
      for (let col = 0; col < j; col++) out[(b * k + c) * j + col] = s[(from + b + col) * k + c]
  return out
}

/** X minus its projection onto the row space of U: X − X Uᵀ(UUᵀ)⁻¹U (rows × j). */
function projectOut(X: F64, rx: number, U: F64, ru: number, j: number): F64 {
  if (ru === 0) return Float64Array.from(X)
  const UUt = dense.matMul(U, dense.transpose(U, ru, j), ru, j, ru)
  const XUt = dense.matMul(X, dense.transpose(U, ru, j), rx, j, ru)
  // Coefficients W (rx × ru) with W UUᵀ = XUᵀ, i.e. UUᵀ Wᵀ = U Xᵀ.
  const Wt = toFlat(lstsq(fromData(UUt, [ru, ru]), fromData(dense.transpose(XUt, rx, ru), [ru, rx])).x)
  const W = dense.transpose(Float64Array.from(Wt), ru, rx)
  return dense.sub(X, dense.matMul(W, U, rx, ru, j))
}

/**
 * Subspace identification of x_{k+1} = Ax_k + Bu_k, y_k = Cx_k + Du_k from inputs u (N × m, or a vector) and outputs
 * y (N × p) by N4SID (Van Overschee & De Moor, 1994). With block Hankel matrices of i past and i future rows, the
 * oblique projection O = Y_f /_{U_f} W_p of the future outputs onto the past data W_p = [U_p; Y_p] along the future
 * inputs equals Γᵢ X̂_f, the extended observability matrix times a state sequence. Its SVD O = USVᵀ gives the order (the
 * number of singular values clearly above the rest) and Γᵢ = U₁S₁^{1/2}; C is Γᵢ's first block row and A solves the
 * shift equation Γ↑A = Γ↓ in least squares. With A and C fixed the output is linear in (B, D, x₀), which are fitted by
 * least squares on the whole record.
 */
export function n4sid(
  y: MatrixLike | VectorLike,
  u: MatrixLike | VectorLike,
  options: N4sidOptions = {},
): SubspaceModel {
  const Y = channels(y, 'n4sid y')
  const U = channels(u, 'n4sid u')
  if (Y.N !== U.N) throw new ShapeError('n4sid', `n4sid: y has ${Y.N} samples, u ${U.N}`)
  const N = Y.N
  const p = Y.k
  const m = U.k
  const i = options.horizon ?? 10
  const j = N - 2 * i + 1
  if (j < 2 * i * (m + p)) throw new DomainError('n4sid', `n4sid: ${N} samples are too few for horizon ${i}`)
  const Up = hankel(U.d, m, 0, i, j)
  const Uf = hankel(U.d, m, i, i, j)
  const Yp = hankel(Y.d, p, 0, i, j)
  const Yf = hankel(Y.d, p, i, i, j)
  const rw = i * (m + p)
  const Wp = new Float64Array(rw * j)
  Wp.set(Up, 0)
  Wp.set(Yp, i * m * j)
  const ry = i * p
  const ru = i * m
  const YfPerp = projectOut(Yf, ry, Uf, ru, j)
  const WpPerp = projectOut(Wp, rw, Uf, ru, j)
  // O = Y_f⊥ W_p⊥ᵀ (W_p⊥ W_p⊥ᵀ)† W_p.
  const G = dense.matMul(WpPerp, dense.transpose(WpPerp, rw, j), rw, j, rw)
  const YW = dense.matMul(YfPerp, dense.transpose(WpPerp, rw, j), ry, j, rw)
  const Kt = toFlat(lstsq(fromData(G, [rw, rw]), fromData(dense.transpose(YW, ry, rw), [rw, ry])).x)
  const K = dense.transpose(Float64Array.from(Kt), rw, ry)
  const O = dense.matMul(K, Wp, ry, rw, j)
  // The left singular vectors and values of O from the eigen-decomposition of OOᵀ (ry × ry).
  const OOt = dense.symmetrise(dense.matMul(O, dense.transpose(O, ry, j), ry, j, ry), ry)
  const e = eigh(fromData(OOt, [ry, ry]))
  const sv = toFlat(e.values).map((v) => Math.sqrt(Math.max(v, 0)))
  const Uo = dense.data(e.vectors)
  let n = options.order ?? 0
  if (!options.order) {
    const top = Math.min(options.maxOrder ?? i - 1, ry - 1)
    let best = -Infinity
    // Values at rounding level are floored, so exact zeros past the order do not make the largest gap.
    const floor = 1e-12 * sv[0]
    for (let k = 1; k <= top; k++) {
      const gap = Math.log(Math.max(sv[k - 1], floor)) - Math.log(Math.max(sv[k], floor))
      if (gap > best) {
        best = gap
        n = k
      }
    }
  }
  if (n < 1 || n >= i) throw new DomainError('n4sid', `n4sid: the order ${n} must be in 1 … ${i - 1}`)
  // Γ = U₁ S₁^{1/2}, ry × n.
  const Gamma = new Float64Array(ry * n)
  for (let r = 0; r < ry; r++) for (let c = 0; c < n; c++) Gamma[r * n + c] = Uo[r * ry + c] * Math.sqrt(sv[c])
  const C = Gamma.slice(0, p * n)
  const up = Gamma.slice(0, (ry - p) * n)
  const down = Gamma.slice(p * n)
  const A = Float64Array.from(toFlat(lstsq(fromData(up, [ry - p, n]), fromData(down, [ry - p, n])).x))
  // (B, D, x₀) by least squares: y(t) = C Aᵗ x₀ + Σ_{k<t} C A^{t−k−1} B u(k) + D u(t).
  const d = n * m + p * m + n
  const Phi = new Float64Array(N * p * d)
  let col = 0
  const fill = (resp: F64) => {
    for (let r = 0; r < N * p; r++) Phi[r * d + col] = resp[r]
    col++
  }
  const simulateUnit = (Bu: F64 | null, Du: F64 | null, x0: F64 | null): F64 => {
    const out = new Float64Array(N * p)
    let x = x0 ? Float64Array.from(x0) : new Float64Array(n)
    for (let t = 0; t < N; t++) {
      const ut = U.d.subarray(t * m, t * m + m)
      const yt = dense.matVec(C, x, p, n)
      if (Du) {
        const add = dense.matVec(Du, ut, p, m)
        for (let r = 0; r < p; r++) yt[r] += add[r]
      }
      out.set(yt, t * p)
      const next = dense.matVec(A, x, n, n)
      if (Bu) {
        const add = dense.matVec(Bu, ut, n, m)
        for (let r = 0; r < n; r++) next[r] += add[r]
      }
      x = next
    }
    return out
  }
  for (let r = 0; r < n; r++)
    for (let c = 0; c < m; c++) {
      const Bu = new Float64Array(n * m)
      Bu[r * m + c] = 1
      fill(simulateUnit(Bu, null, null))
    }
  for (let r = 0; r < p; r++)
    for (let c = 0; c < m; c++) {
      const Du = new Float64Array(p * m)
      Du[r * m + c] = 1
      fill(simulateUnit(null, Du, null))
    }
  for (let r = 0; r < n; r++) {
    const x0 = new Float64Array(n)
    x0[r] = 1
    fill(simulateUnit(null, null, x0))
  }
  const theta = toFlat(lstsq(fromData(Phi, [N * p, d]), fromData(Float64Array.from(Y.d), [N * p])).x)
  const B = Float64Array.from(theta.slice(0, n * m))
  const D = Float64Array.from(theta.slice(n * m, n * m + p * m))
  const x0 = Float64Array.from(theta.slice(n * m + p * m))
  const sim = dense.matVec(Phi, Float64Array.from(theta), N * p, d)
  let loss = 0
  for (let r = 0; r < N * p; r++) loss += (sim[r] - Y.d[r]) ** 2
  const system = stateSpace({
    A: fromData(A, [n, n]),
    B: fromData(B, [n, m]),
    C: fromData(C, [p, n]),
    D: fromData(D, [p, m]),
    dt: options.dt ?? 1,
  })
  return {
    system,
    singularValues: vec(sv),
    order: n,
    x0: vec(x0),
    simulated: fromData(sim, [N, p]),
    loss: loss / (N * p),
  }
}

/** The order-selection curve of ARX models: loss, FPE and AIC for na = nb = 1 … maxOrder (nk fixed). */
export function arxOrderSelection(
  y: VectorLike,
  u: VectorLike,
  { maxOrder = 6, nk = 1 }: { maxOrder?: Size; nk?: Size } = {},
): { order: Size; loss: Scalar; fpe: Scalar; aic: Scalar }[] {
  return Array.from({ length: maxOrder }, (_, k) => {
    const fit = arx(y, u, { na: k + 1, nb: k + 1, nk })
    return { order: k + 1, loss: fit.loss, fpe: fit.fpe, aic: fit.aic }
  })
}

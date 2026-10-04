/**
 * Covariance kernels k(x, x′) as plain objects whose hyperparameters are a tree of values, so that a Gram matrix is
 * differentiable in them (and in the inputs) with `aifn-compute/foundation/autodiff`. Every stationary kernel is parameterised by its
 * lengthscale ℓ (a number, or a vector [d] for automatic relevance determination) and its variance σ² = k(x, x).
 *
 * The forms follow Rasmussen and Williams (2006), "Gaussian Processes for Machine Learning", ch. 4 (eqs. 4.9, 4.14,
 * 4.17, 4.19, 4.31) and scikit-learn's `sklearn.gaussian_process.kernels`, whose conventions the tests check against.
 */

import {
  add,
  elementwise,
  greater,
  div,
  exp,
  fromData,
  log,
  matmul,
  mul,
  neg,
  pow,
  reshape,
  shapeOfValue,
  sin,
  square,
  sub,
  sum,
  transpose,
  unwrap,
  where,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { KernelParams, Kernel } from 'aifn-compute/foundation/contracts'
import { treeMap } from 'aifn-compute/foundation/pytree'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { KernelParams, Kernel } from 'aifn-compute/foundation/contracts'

// ── Inputs and distances ─────────────────────────────────────────────────────────────────────────────────────────

/** Inputs as rows: a vector [n] is n one-dimensional inputs [n, 1]. */
export function asRows(x: Value): Value {
  const shape = shapeOfValue(x)
  if (shape.length === 1) return reshape(x, [shape[0], 1])
  if (shape.length === 2) return x
  throw new ShapeError('kernels', `kernels: inputs must be [n] or [n, d], got shape [${shape.join(', ')}]`)
}

function rows(x: Value): number {
  return shapeOfValue(x)[0]
}

/**
 * √u with derivative 0 at u = 0. A distance r = √(r²) has an infinite derivative in r² at 0, but every r² in a kernel
 * is a sum of squares whose own derivative vanishes there, so the chain-rule product is 0; declaring it keeps
 * gradients of Matérn and periodic kernels finite on the diagonal (Rasmussen and Williams, 2006, §4.2.1).
 */
const distanceFromSquared = elementwise({
  id: 'learning/kernels/distanceFromSquared',
  f: (u) => Math.sqrt(u),
  // 1/(2r) where u > 0 and 0 at u = 0; the guard keeps the unselected 1/(2·0) out of the result.
  derivative: [(u, r) => where(greater(u, 0), div(0.5, where(greater(u, 0), r, 1)), 0)],
  doc: { summary: '√u with derivative 0 at u = 0, for distances inside kernels.' },
  test: { domain: { lo: 0.1, hi: 3 } },
})

/** Squared distances ‖(xᵢ − yⱼ)/ℓ‖² [n, m]; ℓ is a number or a vector [d] (one lengthscale per input dimension). */
export function scaledSquaredDistances(x: Value, y: Value | null, lengthscale: Value): Value {
  const a = div(asRows(x), lengthscale)
  const b = y === null ? a : div(asRows(y), lengthscale)
  const [n, d] = shapeOfValue(a)
  const m = rows(b)
  const diff = sub(reshape(a, [n, 1, d]), reshape(b, [1, m, d]))
  return sum(square(diff), 2)
}

/** Distances ‖(xᵢ − yⱼ)/ℓ‖ [n, m]. */
export function scaledDistances(x: Value, y: Value | null, lengthscale: Value): Value {
  return distanceFromSquared(scaledSquaredDistances(x, y, lengthscale))
}

/** A vector [n] filled with the (possibly traced) scalar v. */
function filled(v: Value, n: number): Value {
  return mul(v, fromData(new Float64Array(n).fill(1), [n]))
}

function checkLengthscale(name: string, lengthscale: Value) {
  const v = unwrap(lengthscale)
  const values = typeof v === 'number' ? [v] : Array.from(v.data)
  if (!values.every((l) => l > 0)) throw new DomainError(name, `${name}: lengthscales must be positive`)
}

/** A stationary kernel σ² g(r²) with r² the scaled squared distance. */
function stationary<P extends { lengthscale: Value; variance: Value }>(
  name: string,
  params: P,
  profile: (r2: Value, params: P) => Value,
  make: (params: P) => Kernel<P>,
): Kernel<P> {
  checkLengthscale(name, params.lengthscale)
  return {
    kind: 'kernel',
    name,
    params,
    stationary: true,
    evaluate: (x, y) => mul(params.variance, profile(scaledSquaredDistances(x, y, params.lengthscale), params)),
    diagonal: (x) => filled(params.variance, rows(asRows(x))),
    withParams: make,
  }
}

// ── Stationary kernels ───────────────────────────────────────────────────────────────────────────────────────────

/** Hyperparameters of a stationary kernel: lengthscale ℓ (number, or [d] for ARD) and variance σ². */
export type StationaryParams = { lengthscale: Value; variance: Value }

/**
 * The squared-exponential (RBF) kernel σ² exp(−r²/2), r = ‖x − x′‖/ℓ (Rasmussen and Williams, 2006, eq. 4.9).
 * Defaults ℓ = 1, σ² = 1. A vector ℓ [d] gives automatic relevance determination.
 */
export function rbf({ lengthscale = 1, variance = 1 }: Partial<StationaryParams> = {}): Kernel<StationaryParams> {
  return stationary('rbf', { lengthscale, variance }, (r2) => exp(mul(-0.5, r2)), rbf)
}

/** The Matérn smoothness ν: ½, 3⁄2 or 5⁄2 (the half-integer cases with closed forms). */
export type MaternNu = 0.5 | 1.5 | 2.5

/**
 * The Matérn kernel with ν ∈ {½, 3⁄2, 5⁄2} (Rasmussen and Williams, 2006, eqs. 4.14 and 4.17): with r = ‖x − x′‖/ℓ,
 * σ² e^{−r}, σ² (1 + √3 r) e^{−√3 r} and σ² (1 + √5 r + 5r²/3) e^{−√5 r}. Sample paths are ⌈ν⌉ − 1 times
 * differentiable. ν = ½ is the exponential (Ornstein–Uhlenbeck) kernel.
 */
export function matern(nu: MaternNu, { lengthscale = 1, variance = 1 }: Partial<StationaryParams> = {}) {
  const make = (p: StationaryParams): Kernel<StationaryParams> =>
    stationary(
      `matern${nu * 2}/2`,
      p,
      (r2) => {
        const r = distanceFromSquared(r2)
        if (nu === 0.5) return exp(neg(r))
        if (nu === 1.5) {
          const s = mul(Math.sqrt(3), r)
          return mul(add(1, s), exp(neg(s)))
        }
        if (nu === 2.5) {
          const s = mul(Math.sqrt(5), r)
          return mul(add(add(1, s), mul(5 / 3, r2)), exp(neg(s)))
        }
        throw new DomainError('matern', `matern: ν must be 0.5, 1.5 or 2.5, got ${nu}`)
      },
      make,
    )
  return make({ lengthscale, variance })
}

/** Matérn ½ (exponential) kernel σ² e^{−r}. */
export const matern12 = (params: Partial<StationaryParams> = {}) => matern(0.5, params)
/** Matérn 3⁄2 kernel σ² (1 + √3 r) e^{−√3 r}. */
export const matern32 = (params: Partial<StationaryParams> = {}) => matern(1.5, params)
/** Matérn 5⁄2 kernel σ² (1 + √5 r + 5r²/3) e^{−√5 r}. */
export const matern52 = (params: Partial<StationaryParams> = {}) => matern(2.5, params)

/** Hyperparameters of the rational quadratic kernel. */
export type RationalQuadraticParams = StationaryParams & { alpha: Value }

/**
 * The rational quadratic kernel σ² (1 + r²/(2α))^{−α} (Rasmussen and Williams, 2006, eq. 4.19): a scale mixture of
 * RBF kernels, tending to the RBF as α → ∞. Defaults ℓ = 1, α = 1, σ² = 1.
 */
export function rationalQuadratic({
  lengthscale = 1,
  alpha = 1,
  variance = 1,
}: Partial<RationalQuadraticParams> = {}): Kernel<RationalQuadraticParams> {
  return stationary(
    'rational-quadratic',
    { lengthscale, alpha, variance },
    (r2, p) => pow(add(1, div(r2, mul(2, p.alpha))), neg(p.alpha)),
    rationalQuadratic,
  )
}

/** Hyperparameters of the periodic kernel. */
export type PeriodicParams = StationaryParams & { period: Value }

/**
 * The periodic (exp-sine-squared) kernel σ² exp(−2 sin²(π‖x − x′‖/p)/ℓ²) (MacKay, 1998; Rasmussen and Williams,
 * 2006, eq. 4.31; scikit-learn's `ExpSineSquared`). Defaults ℓ = 1, p = 1, σ² = 1. ℓ must be a number here.
 */
export function periodic({
  lengthscale = 1,
  period = 1,
  variance = 1,
}: Partial<PeriodicParams> = {}): Kernel<PeriodicParams> {
  checkLengthscale('periodic', lengthscale)
  const params = { lengthscale, period, variance }
  return {
    kind: 'kernel',
    name: 'periodic',
    params,
    stationary: true,
    evaluate: (x, y) => {
      const r = scaledDistances(x, y, 1)
      const s = sin(div(mul(Math.PI, r), period))
      return mul(variance, exp(div(mul(-2, square(s)), square(lengthscale))))
    },
    diagonal: (x) => filled(variance, rows(asRows(x))),
    withParams: periodic,
  }
}

/** Hyperparameters of the white-noise and constant kernels. */
export type VarianceParams = { variance: Value }

/**
 * White noise σ² δ(x, x′): σ²I on a set of inputs against itself (`y === null`), 0 between different sets (as
 * scikit-learn's `WhiteKernel`). Added to a kernel it models independent observation noise.
 */
export function white({ variance = 1 }: Partial<VarianceParams> = {}): Kernel<VarianceParams> {
  const params = { variance }
  return {
    kind: 'kernel',
    name: 'white',
    params,
    stationary: true,
    evaluate: (x, y) => {
      const n = rows(asRows(x))
      if (y !== null) return zeros([n, rows(asRows(y))])
      const eye = new Float64Array(n * n)
      for (let i = 0; i < n; i++) eye[i * n + i] = 1
      return mul(variance, fromData(eye, [n, n]))
    },
    diagonal: (x) => filled(variance, rows(asRows(x))),
    withParams: white,
  }
}

/** The constant kernel k(x, x′) = σ²: a random offset shared by every input. */
export function constant({ variance = 1 }: Partial<VarianceParams> = {}): Kernel<VarianceParams> {
  const params = { variance }
  return {
    kind: 'kernel',
    name: 'constant',
    params,
    stationary: true,
    evaluate: (x, y) => {
      const n = rows(asRows(x))
      const m = y === null ? n : rows(asRows(y))
      return mul(variance, fromData(new Float64Array(n * m).fill(1), [n, m]))
    },
    diagonal: (x) => filled(variance, rows(asRows(x))),
    withParams: constant,
  }
}

// ── Dot-product kernels ──────────────────────────────────────────────────────────────────────────────────────────

/** Hyperparameters of the linear and polynomial kernels. */
export type DotProductParams = { variance: Value; bias: Value }

/**
 * Hyperparameters of `linearKernel`: the weight variance σ² and, for an inhomogeneous kernel, the offset variance σ_b².
 * A homogeneous kernel (σ_b² = 0) has no `bias` leaf, so every leaf is positive and has a finite log.
 */
export type LinearParams = { variance: Value } | { variance: Value; bias: Value }

/** σ_b² + σ² xᵀx′ as a Gram matrix [n, m] (σ_b² = 0 without a bias). */
function dotProducts(x: Value, y: Value | null, params: LinearParams): Value {
  const a = asRows(x)
  const b = y === null ? a : asRows(y)
  const scaled = mul(params.variance, matmul(a, transpose(b)))
  return 'bias' in params ? add(params.bias, scaled) : scaled
}

/** Row-wise σ_b² + σ² ‖xᵢ‖². */
function dotDiagonal(x: Value, params: LinearParams): Value {
  const scaled = mul(params.variance, sum(square(asRows(x)), 1))
  return 'bias' in params ? add(params.bias, scaled) : scaled
}

/**
 * The linear kernel σ_b² + σ² xᵀx′ (Rasmussen and Williams, 2006, §4.2.2): a GP with this kernel is Bayesian linear
 * regression with weight variance σ² and offset variance σ_b². Defaults σ² = 1, σ_b² = 0. A bias of exactly 0 (the
 * default) makes the homogeneous kernel σ² xᵀx′, whose only hyperparameter is σ², so the bias stays fixed at 0 when
 * the hyperparameters are fitted in log space; any other bias is a hyperparameter like σ².
 */
export function linearKernel(params: Partial<DotProductParams> = {}): Kernel<LinearParams> {
  const { variance = 1, bias = 0 } = params
  return makeLinear(bias === 0 ? { variance } : { variance, bias })
}

function makeLinear(params: LinearParams): Kernel<LinearParams> {
  return {
    kind: 'kernel',
    name: 'linear',
    params,
    stationary: false,
    evaluate: (x, y) => dotProducts(x, y, params),
    diagonal: (x) => dotDiagonal(x, params),
    withParams: makeLinear,
  }
}

/**
 * The polynomial kernel (σ_b² + σ² xᵀx′)^degree for an integer degree ≥ 1: the feature space of all monomials up to
 * `degree` (Rasmussen and Williams, 2006, eq. 4.22). Defaults σ² = 1, σ_b² = 1.
 */
export function polynomial(
  degree: number,
  { variance = 1, bias = 1 }: Partial<DotProductParams> = {},
): Kernel<DotProductParams> {
  if (!(Number.isInteger(degree) && degree >= 1))
    throw new DomainError('polynomial', 'polynomial: degree must be an integer ≥ 1')
  const make = (params: DotProductParams): Kernel<DotProductParams> => ({
    kind: 'kernel',
    name: `polynomial${degree}`,
    params,
    stationary: false,
    evaluate: (x, y) => pow(dotProducts(x, y, params), degree),
    diagonal: (x) => pow(dotDiagonal(x, params), degree),
    withParams: make,
  })
  return make({ variance, bias })
}

// ── Combinations ────────────────────────────────────────────────────────────────────────────────────────────────

/** Hyperparameters of a sum or product: one tree per term. */
export type CombinedParams = { terms: readonly KernelParams[] }

function combine(op: 'sum' | 'product', kernels: readonly Kernel[]): Kernel<CombinedParams> {
  if (kernels.length === 0) throw new DomainError(`${op}Kernel`, `${op}Kernel: needs at least one kernel`)
  const combineOp = op === 'sum' ? add : mul
  const fold = (f: (k: Kernel) => Value) => kernels.map(f).reduce((a, b) => combineOp(a, b))
  return {
    kind: 'kernel',
    name: `${op}(${kernels.map((k) => k.name).join(', ')})`,
    params: { terms: kernels.map((k) => k.params) },
    stationary: kernels.every((k) => k.stationary),
    evaluate: (x, y) => fold((k) => k.evaluate(x, y)),
    diagonal: (x) => fold((k) => k.diagonal(x)),
    withParams: (p) => {
      if (p.terms.length !== kernels.length)
        throw new ShapeError(`${op}Kernel`, `${op}Kernel: expected ${kernels.length} term trees`)
      return combine(
        op,
        kernels.map((k, i) => k.withParams(p.terms[i])),
      )
    },
  }
}

/** k₁ + k₂ + …: a sum of independent processes (e.g. a trend plus a periodic part plus noise). */
export function sumKernel(...kernels: Kernel[]): Kernel<CombinedParams> {
  return combine('sum', kernels)
}

/** k₁ · k₂ · …: e.g. a periodic kernel times an RBF gives a locally periodic process. */
export function productKernel(...kernels: Kernel[]): Kernel<CombinedParams> {
  return combine('product', kernels)
}

// ── Gram matrices ────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Gram (covariance) matrix k(X, Y) [n, m] for inputs X [n, d] (or [n]) and Y [m, d]; with Y omitted, k(X, X)
 * [n, n], including any white-noise term. Differentiable in the inputs and in the kernel's hyperparameters.
 */
export function gram(k: Kernel, x: Tensor, y?: Tensor): Tensor
export function gram(k: Kernel, x: Value, y?: Value): Value
export function gram(k: Kernel, x: Value, y?: Value): Value {
  return k.evaluate(x, y ?? null)
}

/** k(xᵢ, xᵢ) for each input, [n] (the diagonal of `gram(k, x)` without forming it). */
export function kernelDiagonal(k: Kernel, x: Tensor): Tensor
export function kernelDiagonal(k: Kernel, x: Value): Value
export function kernelDiagonal(k: Kernel, x: Value): Value {
  return k.diagonal(x)
}

/** The kernel as a function of the lag τ = x − x′ in one dimension, k(τ, 0), on a grid of lags [n] → [n]. */
export function kernelProfile(k: Kernel, lags: Tensor): Tensor {
  const origin = fromData(Float64Array.of(0), [1, 1])
  const n = lags.shape[0]
  const values = k.evaluate(reshape(lags, [n, 1]), origin)
  return reshape(values as Tensor, [n])
}

// ── Hyperparameters in log space ─────────────────────────────────────────────────────────────────────────────────

/**
 * The kernel with hyperparameters exp(θ) for a tree θ of log-hyperparameters shaped like `k.params` (numbers, tensors
 * or traced values at the leaves), so a fit can optimise unconstrained values and differentiate through the map. A
 * fit over one flat vector pairs it with `ravel(logParams(k))` from `aifn-compute/foundation/pytree`.
 */
export function kernelFromLog<P extends KernelParams>(k: Kernel<P>, logParams: P): Kernel<P> {
  return k.withParams(treeMap<P, Value>(logParams, (v) => exp(v)))
}

/** A kernel's hyperparameters in log space: the tree of log θ for every leaf of `k.params` (all are positive). */
export function logParams<P extends KernelParams>(k: Kernel<P>): P {
  return treeMap<P, Value>(k.params, (v) => log(v))
}

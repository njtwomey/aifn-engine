/**
 * Covariance kernels $k(\xvec, \xvec')$ as plain objects whose hyperparameters are a tree of values, so that a Gram
 * matrix is differentiable in them (and in the inputs) with `aifn-compute/foundation/autodiff`. Every stationary
 * kernel but `white` and `constant` is parameterised by its lengthscale $\ell$ (a number, or a vector of $d$ for
 * automatic relevance determination) and its variance $\sigma^2 = k(\xvec, \xvec)$, and is a function of the scaled
 * distance $r = \norm{(\xvec - \xvec') / \ell}$.
 *
 * Inputs are rows: an `[n, d]` matrix of $n$ points, or an `[n]` vector of $n$ one-dimensional points. The forms follow
 * Rasmussen and Williams (2006), "Gaussian Processes for Machine Learning", ch. 4 (eqs. 4.9, 4.14, 4.17, 4.19, 4.31)
 * and scikit-learn's `sklearn.gaussian_process.kernels`, whose conventions the tests check against.
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

/**
 * Inputs as rows: a vector `[n]` is $n$ one-dimensional inputs `[n, 1]`, and a matrix `[n, d]` is returned as it is.
 * Any other rank throws `ShapeError`.
 *
 * @param x The inputs, `[n]` or `[n, d]`.
 * @returns The inputs as an `[n, d]` matrix.
 *
 * @example A vector becomes a column
 * print(asRows(tensor([1, 2, 3])))
 */
export function asRows(x: Value): Value {
  const shape = shapeOfValue(x)
  if (shape.length === 1) return reshape(x, [shape[0], 1])
  if (shape.length === 2) return x
  throw new ShapeError('kernels', `kernels: inputs must be [n] or [n, d], got shape [${shape.join(', ')}]`)
}

/**
 * The number of rows (inputs) of a value: the length of its first axis.
 *
 * @param x The inputs, `[n]` or `[n, d]`.
 * @returns $n$.
 */
function rows(x: Value): number {
  return shapeOfValue(x)[0]
}

/**
 * $\sqrt{u}$ with derivative 0 at $u = 0$. A distance $r = \sqrt{r^2}$ has an infinite derivative in $r^2$ at 0, but
 * every $r^2$ in a kernel is a sum of squares whose own derivative vanishes there, so the chain-rule product is 0;
 * declaring it keeps gradients of Matérn and periodic kernels finite on the diagonal (Rasmussen and Williams, 2006,
 * §4.2.1).
 */
const distanceFromSquared = elementwise({
  id: 'learning/kernels/distanceFromSquared',
  f: (u) => Math.sqrt(u),
  // 1/(2r) where u > 0 and 0 at u = 0; the guard keeps the unselected 1/(2·0) out of the result.
  derivative: [(u, r) => where(greater(u, 0), div(0.5, where(greater(u, 0), r, 1)), 0)],
  doc: { summary: '√u with derivative 0 at u = 0, for distances inside kernels.' },
  test: { domain: { lo: 0.1, hi: 3 } },
})

/**
 * Squared scaled distances $\norm{(\xvec_i - \yvec_j) / \ell}^2$ between every pair of rows, `[n, m]`;
 * differentiable in the inputs and in $\ell$.
 *
 * @param x The first inputs $\xvec_i$, `[n]` or `[n, d]`.
 * @param y The second inputs $\yvec_j$, `[m]` or `[m, d]`, or `null` for `x` against itself.
 * @param lengthscale $\ell$: a number, or a vector of $d$ (one lengthscale per input dimension).
 * @returns The `[n, m]` matrix of squared distances.
 *
 * @example Two points in the plane, unscaled and with a lengthscale per axis
 * const x = tensor([[0, 0], [3, 4]])
 * print('l = 1:', scaledSquaredDistances(x, null, 1))
 * print('l = (3, 4):', scaledSquaredDistances(x, null, tensor([3, 4])))
 */
export function scaledSquaredDistances(x: Value, y: Value | null, lengthscale: Value): Value {
  const a = div(asRows(x), lengthscale)
  const b = y === null ? a : div(asRows(y), lengthscale)
  const [n, d] = shapeOfValue(a)
  const m = rows(b)
  const diff = sub(reshape(a, [n, 1, d]), reshape(b, [1, m, d]))
  return sum(square(diff), 2)
}

/**
 * Scaled distances $\norm{(\xvec_i - \yvec_j) / \ell}$ between every pair of rows, `[n, m]`: the square root of
 * `scaledSquaredDistances`, with derivative 0 (rather than infinite) where a distance is 0.
 *
 * @param x The first inputs $\xvec_i$, `[n]` or `[n, d]`.
 * @param y The second inputs $\yvec_j$, `[m]` or `[m, d]`, or `null` for `x` against itself.
 * @param lengthscale $\ell$: a number, or a vector of $d$.
 * @returns The `[n, m]` matrix of distances.
 *
 * @example A 3-4-5 triangle
 * print(scaledDistances(tensor([[0, 0], [3, 4]]), tensor([[0, 0]]), 1))
 */
export function scaledDistances(x: Value, y: Value | null, lengthscale: Value): Value {
  return distanceFromSquared(scaledSquaredDistances(x, y, lengthscale))
}

/**
 * A vector `[n]` filled with the (possibly traced) scalar $v$, differentiable in $v$.
 *
 * @param v The value of every entry.
 * @param n The length.
 * @returns The vector.
 */
function filled(v: Value, n: number): Value {
  return mul(v, fromData(new Float64Array(n).fill(1), [n]))
}

/**
 * Throw `DomainError` unless every lengthscale is positive (a traced lengthscale is checked at its current value).
 *
 * @param name The kernel's name, for the error message.
 * @param lengthscale $\ell$: a number or a vector.
 */
function checkLengthscale(name: string, lengthscale: Value) {
  const v = unwrap(lengthscale)
  const values = typeof v === 'number' ? [v] : Array.from(v.data)
  if (!values.every((l) => l > 0)) throw new DomainError(name, `${name}: lengthscales must be positive`)
}

/**
 * A stationary kernel $\sigma^2 g(r^2)$ with $r^2$ the scaled squared distance; its diagonal is $\sigma^2$. The
 * lengthscale is checked here, once, when the kernel is made.
 *
 * @param name The kernel's name.
 * @param params The hyperparameters: `lengthscale` $\ell$, `variance` $\sigma^2$ and any of the profile's own.
 * @param profile The profile $g$: from the `[n, m]` squared scaled distances (and the parameters) to the correlations.
 * @param make The kernel's factory, called by `withParams` with new hyperparameters.
 * @returns The kernel.
 */
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

/**
 * Hyperparameters of a stationary kernel: `lengthscale` $\ell$ (a number, or a vector of $d$ for ARD) and `variance`
 * $\sigma^2$.
 */
export type StationaryParams = { lengthscale: Value; variance: Value }

/**
 * The squared-exponential (RBF) kernel $k(\xvec, \xvec') = \sigma^2 \exp(-r^2/2)$, $r = \norm{\xvec - \xvec'} / \ell$
 * (Rasmussen and Williams, 2006, eq. 4.9). Defaults $\ell = 1$, $\sigma^2 = 1$. A vector $\ell$ of $d$ gives automatic
 * relevance determination, $r^2 = \sum_j (x_j - x'_j)^2 / \ell_j^2$. With $\sigma^2 = 1$ it is scikit-learn's
 * `RBF(length_scale)`. A lengthscale that is not positive throws `DomainError`.
 *
 * @param options The hyperparameters.
 * @param options.lengthscale $\ell > 0$: a number, or a vector of $d$ (default 1).
 * @param options.variance $\sigma^2 > 0$, the value on the diagonal (default 1).
 * @returns The kernel.
 *
 * @example The RBF kernel at distance 0 and 1
 * print(gram(rbf(), tensor([0, 1])))
 * print('exp(-1/2) =', Math.exp(-0.5))
 *
 * @example A lengthscale per input dimension
 * // The second coordinate barely matters with a lengthscale of 100.
 * print(gram(rbf({ lengthscale: tensor([1, 100]) }), tensor([[0, 0], [1, 10]])))
 */
export function rbf({ lengthscale = 1, variance = 1 }: Partial<StationaryParams> = {}): Kernel<StationaryParams> {
  return stationary('rbf', { lengthscale, variance }, (r2) => exp(mul(-0.5, r2)), rbf)
}

/** The Matérn smoothness $\nu$: $\tfrac{1}{2}$, $\tfrac{3}{2}$ or $\tfrac{5}{2}$, the cases with closed forms. */
export type MaternNu = 0.5 | 1.5 | 2.5

/**
 * The Matérn kernel with $\nu \in \{\tfrac{1}{2}, \tfrac{3}{2}, \tfrac{5}{2}\}$ (Rasmussen and Williams, 2006, eqs.
 * 4.14 and 4.17): with $r = \norm{\xvec - \xvec'} / \ell$, $\sigma^2 e^{-r}$,
 * $\sigma^2 (1 + \sqrt{3} r) e^{-\sqrt{3} r}$ and $\sigma^2 (1 + \sqrt{5} r + 5r^2/3) e^{-\sqrt{5} r}$. Sample
 * paths are $\lceil \nu \rceil - 1$ times differentiable. $\nu = \tfrac{1}{2}$ is the exponential (Ornstein–Uhlenbeck)
 * kernel. As scikit-learn's `Matern(length_scale, nu)` for $\sigma^2 = 1$. Another $\nu$ throws `DomainError`, but
 * only when the kernel is evaluated.
 *
 * @param nu The smoothness $\nu$: 0.5, 1.5 or 2.5.
 * @param options The hyperparameters.
 * @param options.lengthscale $\ell > 0$: a number, or a vector of $d$ (default 1).
 * @param options.variance $\sigma^2 > 0$ (default 1).
 * @returns The kernel.
 *
 * @example At distance 1, smoother kernels stay more correlated
 * for (const nu of [0.5, 1.5, 2.5]) print(`nu = ${nu}:`, kernelProfile(matern(nu), tensor([0, 1, 2])))
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

/**
 * The Matérn $\tfrac{1}{2}$ (exponential) kernel $\sigma^2 e^{-r}$: `matern(0.5, params)`.
 *
 * @param params The hyperparameters `lengthscale` and `variance` (default 1 each).
 * @returns The kernel.
 *
 * @example At distance 1
 * print('k =', kernelProfile(matern12(), tensor([1])), ' e^-1 =', Math.exp(-1))
 */
export const matern12 = (params: Partial<StationaryParams> = {}) => matern(0.5, params)
/**
 * The Matérn $\tfrac{3}{2}$ kernel $\sigma^2 (1 + \sqrt{3} r) e^{-\sqrt{3} r}$: `matern(1.5, params)`.
 *
 * @param params The hyperparameters `lengthscale` and `variance` (default 1 each).
 * @returns The kernel.
 *
 * @example At distance 1
 * print('k =', kernelProfile(matern32(), tensor([1])), ' closed form =', (1 + Math.sqrt(3)) * Math.exp(-Math.sqrt(3)))
 */
export const matern32 = (params: Partial<StationaryParams> = {}) => matern(1.5, params)
/**
 * The Matérn $\tfrac{5}{2}$ kernel $\sigma^2 (1 + \sqrt{5} r + 5r^2/3) e^{-\sqrt{5} r}$: `matern(2.5, params)`.
 *
 * @param params The hyperparameters `lengthscale` and `variance` (default 1 each).
 * @returns The kernel.
 *
 * @example At distance 1
 * const closed = (1 + Math.sqrt(5) + 5 / 3) * Math.exp(-Math.sqrt(5))
 * print('k =', kernelProfile(matern52(), tensor([1])), ' closed form =', closed)
 */
export const matern52 = (params: Partial<StationaryParams> = {}) => matern(2.5, params)

/** Hyperparameters of the rational quadratic kernel: those of a stationary kernel and the shape `alpha` $\alpha$. */
export type RationalQuadraticParams = StationaryParams & { alpha: Value }

/**
 * The rational quadratic kernel $\sigma^2 (1 + r^2 / (2\alpha))^{-\alpha}$ (Rasmussen and Williams, 2006, eq. 4.19):
 * a scale mixture of RBF kernels, tending to the RBF as $\alpha \to \infty$. Defaults $\ell = 1$, $\alpha = 1$,
 * $\sigma^2 = 1$. As scikit-learn's `RationalQuadratic(length_scale, alpha)` for $\sigma^2 = 1$.
 *
 * @param options The hyperparameters.
 * @param options.lengthscale $\ell > 0$: a number, or a vector of $d$ (default 1).
 * @param options.alpha The shape $\alpha > 0$: small for a heavy mixture of lengthscales, large for an RBF (default 1).
 * @param options.variance $\sigma^2 > 0$ (default 1).
 * @returns The kernel.
 *
 * @example At distance 1: 2/3 for alpha = 1, and the RBF's exp(-1/2) for large alpha
 * print('alpha = 1:', kernelProfile(rationalQuadratic(), tensor([1])))
 * print('alpha = 1e6:', kernelProfile(rationalQuadratic({ alpha: 1e6 }), tensor([1])), ' exp(-1/2) =', Math.exp(-0.5))
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

/** Hyperparameters of the periodic kernel: those of a stationary kernel and the `period` $p > 0$. */
export type PeriodicParams = StationaryParams & { period: Value }

/**
 * The periodic (exp-sine-squared) kernel $\sigma^2 \exp(-2 \sin^2(\pi \norm{\xvec - \xvec'} / p) / \ell^2)$ (MacKay,
 * 1998; Rasmussen and Williams, 2006, eq. 4.31; scikit-learn's `ExpSineSquared`). Defaults $\ell = 1$, $p = 1$,
 * $\sigma^2 = 1$. $\ell$ must be a number here: the distance is unscaled Euclidean, so there is no ARD.
 *
 * @param options The hyperparameters.
 * @param options.lengthscale $\ell > 0$, a number (default 1).
 * @param options.period The period $p > 0$, in the inputs' units (default 1).
 * @param options.variance $\sigma^2 > 0$ (default 1).
 * @returns The kernel.
 *
 * @example A Gram matrix of three points: points a period apart are perfectly correlated
 * print(gram(periodic(), tensor([0, 0.5, 1])))
 * print('half a period: exp(-2) =', Math.exp(-2))
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

/** Hyperparameters of the white-noise and constant kernels: the `variance` $\sigma^2$. */
export type VarianceParams = { variance: Value }

/**
 * White noise $\sigma^2 \delta(\xvec, \xvec')$: $\sigma^2 \Imat$ on a set of inputs against itself (`y === null`), 0
 * between different sets, even where two points coincide (as scikit-learn's `WhiteKernel`). Added to a kernel it
 * models independent observation noise.
 *
 * @param options The hyperparameters.
 * @param options.variance The noise variance $\sigma^2 > 0$ (default 1).
 * @returns The kernel.
 *
 * @example Noise on the diagonal of a set against itself, none between two sets
 * const k = white({ variance: 0.1 })
 * print('gram(k, x) =', gram(k, tensor([0, 1])))
 * print('gram(k, x, y) =', gram(k, tensor([0, 1]), tensor([0, 1])))
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

/**
 * The constant kernel $k(\xvec, \xvec') = \sigma^2$: a random offset shared by every input. Exported under its own
 * name, `constant`.
 *
 * @param options The hyperparameters.
 * @param options.variance The offset's variance $\sigma^2 > 0$ (default 1).
 * @returns The kernel.
 *
 * @example Every entry is the variance
 * print(gram(constant({ variance: 2 }), tensor([0, 1, 5])))
 */
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

/** Hyperparameters of the linear and polynomial kernels: weight `variance` $\sigma^2$, offset `bias` $\sigma_b^2$. */
export type DotProductParams = { variance: Value; bias: Value }

/**
 * Hyperparameters of `linearKernel`: the weight `variance` $\sigma^2$ and, for an inhomogeneous kernel, the offset
 * variance `bias` $\sigma_b^2$. A homogeneous kernel ($\sigma_b^2 = 0$) has no `bias` leaf, so every leaf is positive
 * and has a finite log.
 */
export type LinearParams = { variance: Value } | { variance: Value; bias: Value }

/**
 * $\sigma_b^2 + \sigma^2 \xvec^\top \xvec'$ as a Gram matrix `[n, m]` ($\sigma_b^2 = 0$ without a bias).
 *
 * @param x The first inputs, `[n]` or `[n, d]`.
 * @param y The second inputs, `[m]` or `[m, d]`, or `null` for `x` against itself.
 * @param params $\sigma^2$ and, if present, $\sigma_b^2$.
 * @returns The `[n, m]` matrix.
 */
function dotProducts(x: Value, y: Value | null, params: LinearParams): Value {
  const a = asRows(x)
  const b = y === null ? a : asRows(y)
  const scaled = mul(params.variance, matmul(a, transpose(b)))
  return 'bias' in params ? add(params.bias, scaled) : scaled
}

/**
 * Row-wise $\sigma_b^2 + \sigma^2 \norm{\xvec_i}^2$, the diagonal of `dotProducts(x, null, params)`.
 *
 * @param x The inputs, `[n]` or `[n, d]`.
 * @param params $\sigma^2$ and, if present, $\sigma_b^2$.
 * @returns The vector `[n]`.
 */
function dotDiagonal(x: Value, params: LinearParams): Value {
  const scaled = mul(params.variance, sum(square(asRows(x)), 1))
  return 'bias' in params ? add(params.bias, scaled) : scaled
}

/**
 * The linear kernel $\sigma_b^2 + \sigma^2 \xvec^\top \xvec'$ (Rasmussen and Williams, 2006, §4.2.2): a GP with this
 * kernel is Bayesian linear regression with weight variance $\sigma^2$ and offset variance $\sigma_b^2$. Defaults
 * $\sigma^2 = 1$, $\sigma_b^2 = 0$. A bias of exactly 0 (the default) makes the homogeneous kernel
 * $\sigma^2 \xvec^\top \xvec'$, whose only hyperparameter is $\sigma^2$, so the bias stays fixed at 0 when the
 * hyperparameters are fitted in log space; any other bias is a hyperparameter like $\sigma^2$. Not stationary.
 *
 * @param params The weight variance `variance` $\sigma^2$ (default 1) and the offset variance `bias` $\sigma_b^2$
 *   (default 0, for the homogeneous kernel).
 * @returns The kernel.
 *
 * @example Inner products, and with an offset
 * const x = tensor([[1, 2], [3, 4]])
 * print('homogeneous:', gram(linearKernel(), x))
 * print('bias 1:', gram(linearKernel({ bias: 1 }), x))
 * print('hyperparameters:', Object.keys(linearKernel().params), Object.keys(linearKernel({ bias: 1 }).params))
 */
export function linearKernel(params: Partial<DotProductParams> = {}): Kernel<LinearParams> {
  const { variance = 1, bias = 0 } = params
  return makeLinear(bias === 0 ? { variance } : { variance, bias })
}

/**
 * The linear kernel for given hyperparameters; also its `withParams`, so a fitted tree keeps its shape (with or
 * without `bias`).
 *
 * @param params $\sigma^2$ and, for an inhomogeneous kernel, $\sigma_b^2$.
 * @returns The kernel.
 */
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
 * The polynomial kernel $(\sigma_b^2 + \sigma^2 \xvec^\top \xvec')^p$ for an integer degree $p \ge 1$: the feature
 * space of all monomials up to degree $p$ (Rasmussen and Williams, 2006, eq. 4.22). Defaults $\sigma^2 = 1$,
 * $\sigma_b^2 = 1$; scikit-learn's `polynomial_kernel(X, degree=p, gamma=1, coef0=1)` is the default. A degree that is
 * not a positive integer throws `DomainError`. Not stationary.
 *
 * @param degree The degree $p$, an integer of at least 1; fixed, not a hyperparameter.
 * @param options The hyperparameters.
 * @param options.variance The weight variance $\sigma^2$ (default 1).
 * @param options.bias The offset $\sigma_b^2$ (default 1).
 * @returns The kernel.
 *
 * @example The quadratic kernel on two points: (1 + x x')^2
 * print(gram(polynomial(2), tensor([1, 2])))
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

/** Hyperparameters of a sum or product: `terms`, one hyperparameter tree per term, in order. */
export type CombinedParams = { terms: readonly KernelParams[] }

/**
 * A kernel that adds or multiplies its terms' Gram matrices and diagonals. It is stationary when every term is. No
 * term throws `DomainError`; a `withParams` tree with the wrong number of terms throws `ShapeError`.
 *
 * @param op Whether the terms are added (`sum`) or multiplied (`product`).
 * @param kernels The terms, at least one.
 * @returns The combined kernel, named e.g. `sum(rbf, white)`.
 */
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

/**
 * The sum $k_1 + k_2 + \dots$: the covariance of a sum of independent processes (e.g. a trend plus a periodic part
 * plus noise). Its hyperparameters are `{ terms }`, one tree per kernel.
 *
 * @param kernels The terms, at least one (none throws `DomainError`).
 * @returns The sum kernel.
 *
 * @example An RBF plus white noise on two points
 * print(gram(sumKernel(rbf(), white({ variance: 0.01 })), tensor([0, 1])))
 */
export function sumKernel(...kernels: Kernel[]): Kernel<CombinedParams> {
  return combine('sum', kernels)
}

/**
 * The product $k_1 k_2 \cdots$: e.g. a periodic kernel times an RBF gives a locally periodic process. Its
 * hyperparameters are `{ terms }`, one tree per kernel.
 *
 * @param kernels The factors, at least one (none throws `DomainError`).
 * @returns The product kernel.
 *
 * @example Locally periodic: one period apart, the RBF factor sets the correlation
 * const k = productKernel(periodic(), rbf({ lengthscale: 2 }))
 * print(gram(k, tensor([0, 1])))
 * print('exp(-1/8) =', Math.exp(-1 / 8))
 */
export function productKernel(...kernels: Kernel[]): Kernel<CombinedParams> {
  return combine('product', kernels)
}

// ── Gram matrices ────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The Gram (covariance) matrix $K_{ij} = k(\xvec_i, \yvec_j)$, `[n, m]`, for inputs $\Xmat$ (`[n, d]` or `[n]`) and
 * $\Ymat$ (`[m, d]` or `[m]`); with $\Ymat$ omitted, $k(\Xmat, \Xmat)$, `[n, n]`, including any white-noise term.
 * Differentiable in the inputs and in the kernel's hyperparameters.
 *
 * @param k The kernel.
 * @param x The inputs $\Xmat$, one per row.
 * @param y The second inputs $\Ymat$; left out, `x` against itself (where `white` noise contributes).
 * @returns The `[n, m]` (or `[n, n]`) matrix.
 *
 * @example A Gram matrix of three points, and a cross-covariance
 * const x = tensor([0, 1, 2])
 * print('K =', gram(rbf(), x))
 * print('K(x, y) =', gram(rbf(), x, tensor([0.5])))
 *
 * @example Gradients in the hyperparameters
 * const x = tensor([0, 1])
 * print(grad((p) => sum(gram(rbf(p), x)))({ lengthscale: 1, variance: 1 }))
 */
export function gram(k: Kernel, x: Tensor, y?: Tensor): Tensor
export function gram(k: Kernel, x: Value, y?: Value): Value
export function gram(k: Kernel, x: Value, y?: Value): Value {
  return k.evaluate(x, y ?? null)
}

/**
 * $k(\xvec_i, \xvec_i)$ for each input, `[n]`: the diagonal of `gram(k, x)` without forming it, including any
 * white-noise term.
 *
 * @param k The kernel.
 * @param x The inputs, `[n, d]` or `[n]`.
 * @returns The `[n]` diagonal.
 *
 * @example The variances, and the squared norms under the linear kernel
 * print('rbf + white:', kernelDiagonal(sumKernel(rbf(), white({ variance: 0.1 })), tensor([0, 1, 2])))
 * print('linear:', kernelDiagonal(linearKernel(), tensor([[1, 2], [3, 4]])))
 */
export function kernelDiagonal(k: Kernel, x: Tensor): Tensor
export function kernelDiagonal(k: Kernel, x: Value): Value
export function kernelDiagonal(k: Kernel, x: Value): Value {
  return k.diagonal(x)
}

/**
 * The kernel as a function of the lag $\tau = x - x'$ in one dimension, $k(\tau, 0)$, on a grid of lags: for plotting
 * a stationary kernel's profile.
 *
 * @param k The kernel; its inputs are one-dimensional.
 * @param lags The lags $\tau$, a tensor `[n]`.
 * @returns $k(\tau, 0)$ for each lag, `[n]`.
 *
 * @example The RBF with lengthscale 2
 * print(kernelProfile(rbf({ lengthscale: 2 }), tensor([0, 1, 2])))
 * print('exp(-1/8), exp(-1/2) =', Math.exp(-1 / 8), Math.exp(-0.5))
 */
export function kernelProfile(k: Kernel, lags: Tensor): Tensor {
  const origin = fromData(Float64Array.of(0), [1, 1])
  const n = lags.shape[0]
  const values = k.evaluate(reshape(lags, [n, 1]), origin)
  return reshape(values as Tensor, [n])
}

// ── Hyperparameters in log space ─────────────────────────────────────────────────────────────────────────────────

/**
 * The kernel with hyperparameters $\exp(\thetavec)$ for a tree $\thetavec$ of log-hyperparameters shaped like
 * `k.params` (numbers, tensors or traced values at the leaves), so a fit can optimise unconstrained values and
 * differentiate through the map. A fit over one flat vector pairs it with `ravel(logParams(k))` from
 * `aifn-compute/foundation/pytree`.
 *
 * @param k The kernel whose structure is kept (its kind, $\nu$ or degree, and terms).
 * @param logParams The log-hyperparameters $\thetavec$, a tree of the same shape as `k.params`.
 * @returns The kernel with hyperparameters $\exp(\thetavec)$.
 *
 * @example From log space back to the kernel
 * const k = kernelFromLog(rbf(), { lengthscale: Math.log(2), variance: 0 })
 * print(k.params)
 */
export function kernelFromLog<P extends KernelParams>(k: Kernel<P>, logParams: P): Kernel<P> {
  return k.withParams(treeMap<P, Value>(logParams, (v) => exp(v)))
}

/**
 * A kernel's hyperparameters in log space: the tree of $\log \theta$ for every leaf $\theta$ of `k.params` (all are
 * positive), the starting point of a fit with `kernelFromLog`.
 *
 * @param k The kernel.
 * @returns The tree of log-hyperparameters, shaped like `k.params`.
 *
 * @example A round trip
 * const k = sumKernel(rbf({ lengthscale: 2, variance: 3 }), white({ variance: 0.1 }))
 * print('log params:', logParams(k))
 * print('back:', kernelFromLog(k, logParams(k)).params)
 */
export function logParams<P extends KernelParams>(k: Kernel<P>): P {
  return treeMap<P, Value>(k.params, (v) => log(v))
}

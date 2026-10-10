/**
 * Feature expansions: polynomial features, B-spline features and random Fourier features, as scikit-learn's
 * `PolynomialFeatures`, `SplineTransformer` and `RBFSampler`.
 *
 * Each maps an $n \times d$ matrix to a wider one of fixed basis functions of its columns, so that a linear model on
 * the output is non-linear in the input (Hastie, Tibshirani and Friedman, 2009, "The Elements of Statistical Learning",
 * ch. 5). `fit` learns only what the basis needs (the number of columns, the knots, or the random draws).
 */

import type { FitOptions } from 'aifn-compute/learning/estimators'
import { normals, uniform, child } from 'aifn-compute/foundation/random'
import { bsplineBasis } from 'aifn-compute/numerics/interpolate'
import { quantile } from 'aifn-compute/probability/stats'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { checkColumns, column, matrix, values, type FittedTransform, type Transformer } from './transformer'
import { defineModel } from 'aifn-compute/learning/estimators'
import { bool, int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

// ── Polynomial features ──────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted polynomial expansion. */
export interface PolynomialFeatures extends FittedTransform {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'polynomial-features'
  /**
   * The exponents of each output column, $m \times d$ for $m$ outputs: output $k$ is $\prod_j x_j^{p_{kj}}$ with
   * $p_{kj}$ = `powers[k, j]`.
   */
  readonly powers: Tensor
  /** The output columns' names, such as "1", "x0", "x0^2", "x0 x1". */
  readonly featureNames: readonly string[]
}

/**
 * Multisets of size `size` from $0, \dots, d - 1$ in lexicographic order (with repetition unless `distinct`).
 *
 * @param d The number of indices to choose from.
 * @param size The number of indices in each multiset.
 * @param distinct Whether each index may appear at most once (sets rather than multisets).
 * @returns The multisets, each a non-decreasing (increasing when `distinct`) list of indices.
 */
function combinations(d: number, size: number, distinct: boolean): number[][] {
  const out: number[][] = []
  const walk = (start: number, prefix: number[]) => {
    if (prefix.length === size) return void out.push(prefix)
    for (let j = start; j < d; j++) walk(distinct ? j + 1 : j, [...prefix, j])
  }
  walk(0, [])
  return out
}

/**
 * All monomials of the $d$ input columns of degree at most `degree`, in scikit-learn's `PolynomialFeatures` order: by
 * degree, then lexicographically in the column indices. `interactionOnly` keeps products of distinct columns;
 * `includeBias` leads with the constant column. Throws `DomainError` unless `degree` is a whole number $\ge 0$.
 *
 * @param options The degree and which monomials to keep.
 * @param options.degree The largest total degree of a monomial.
 * @param options.interactionOnly Keep only products of distinct columns (no $x_j^2$ and higher powers).
 * @param options.includeBias Lead with the constant column 1 (the degree-0 monomial).
 * @returns An estimator whose `fit({ x })` on an $n \times d$ matrix returns the fitted `PolynomialFeatures`.
 *
 * @example Degree-2 monomials of two columns
 * const x = tensor([[2, 3], [1, -1]])
 * const model = polynomialFeatures().fit({ x })
 * print('features:', model.featureNames)
 * print(model.transform(x))
 * print('interactions only:', polynomialFeatures({ interactionOnly: true }).fit({ x }).featureNames)
 */
export function polynomialFeatures({
  degree = 2,
  interactionOnly = false,
  includeBias = true,
}: { degree?: number; interactionOnly?: boolean; includeBias?: boolean } = {}): Transformer<
  Tensor,
  PolynomialFeatures
> {
  if (!(Number.isInteger(degree) && degree >= 0))
    throw new DomainError('polynomialFeatures', 'polynomialFeatures: degree must be a whole number')
  return {
    name: 'polynomial-features',
    params: { degree, interactionOnly, includeBias },
    fit({ x }) {
      const { d } = matrix(x, 'polynomialFeatures')
      const terms: number[][] = []
      for (let k = includeBias ? 0 : 1; k <= degree; k++) terms.push(...combinations(d, k, interactionOnly))
      const powers = new Int32Array(terms.length * d)
      terms.forEach((t, r) => t.forEach((j) => powers[r * d + j]++))
      const featureNames = terms.map((t, r) => {
        if (t.length === 0) return '1'
        const parts: string[] = []
        for (let j = 0; j < d; j++) {
          const p = powers[r * d + j]
          if (p > 0) parts.push(p === 1 ? `x${j}` : `x${j}^${p}`)
        }
        return parts.join(' ')
      })
      return {
        kind: 'model',
        name: 'polynomial-features',
        powers: fromData(powers, [terms.length, d]),
        featureNames,
        transform(input) {
          const { n, d: cols, v } = matrix(input, 'polynomialFeatures.transform')
          checkColumns(cols, d, 'polynomialFeatures')
          const out = new Float64Array(n * terms.length)
          for (let i = 0; i < n; i++) {
            terms.forEach((t, r) => {
              let prod = 1
              for (const j of t) prod *= v[i * d + j]
              out[i * terms.length + r] = prod
            })
          }
          return fromData(out, [n, terms.length])
        },
      }
    },
  }
}

// ── Spline features ──────────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted spline expansion. */
export interface SplineFeatures extends FittedTransform {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'spline-features'
  /**
   * The full knot vector of each column (the base knots extended by `degree` knots at each end), $d \times (K + 2p)$
   * for $K$ base knots of degree $p$.
   */
  readonly knots: Tensor
  /** The degree $p$ of the B-splines. */
  readonly degree: number
  /** The number of output columns per input column: $K + p - 1$ (one fewer without the bias). */
  readonly perColumn: number
}

/**
 * B-spline basis expansion of each column, as scikit-learn's `SplineTransformer`: $K$ = `knots` base knots spread
 * uniformly over the training range (or at its quantiles), extended beyond it by $p$ = `degree` knots with the end
 * spacing, give $K + p - 1$ B-splines of degree $p$ per column (de Boor, 1978, "A Practical Guide to Splines"). The
 * output is $n \times d(K + p - 1)$, column by column. Outside the training range, `extrapolation` holds the boundary
 * values (`'constant'`, the default), continues the end polynomials (`'continue'`) or throws `DomainError`
 * (`'error'`). Without `includeBias` the last spline of each column is dropped, so the basis no longer sums to one (for
 * use with an intercept). Throws `DomainError` for fewer than 2 knots.
 *
 * @param options The knots, the degree, and the behaviour outside the training range.
 * @param options.knots The number $K \ge 2$ of base knots, including the two ends of the training range.
 * @param options.degree The degree $p$ of the B-splines (3 for cubic).
 * @param options.knotPlacement `'uniform'` (evenly spaced from the column's minimum to its maximum) or `'quantile'`
 *   (at equally spaced quantiles of the training column).
 * @param options.extrapolation `'constant'` (clamp to the training range), `'continue'` (extend the end polynomials) or
 *   `'error'` (throw).
 * @param options.includeBias Keep every spline; when false the last of each column is dropped.
 * @returns An estimator whose `fit({ x })` on an $n \times d$ matrix returns the fitted `SplineFeatures`.
 *
 * @example Quadratic B-splines on three knots sum to one
 * const x = tensor([[0], [0.25], [0.5], [1]])
 * const model = splineFeatures({ knots: 3, degree: 2 }).fit({ x })
 * print('splines per column:', model.perColumn)
 * const B = model.transform(x)
 * print('B =', B)
 * print('row sums =', sum(B, 1))
 */
export function splineFeatures({
  knots = 5,
  degree = 3,
  knotPlacement = 'uniform',
  extrapolation = 'constant',
  includeBias = true,
}: {
  knots?: number
  degree?: number
  knotPlacement?: 'uniform' | 'quantile'
  extrapolation?: 'constant' | 'continue' | 'error'
  includeBias?: boolean
} = {}): Transformer<Tensor, SplineFeatures> {
  if (knots < 2) throw new DomainError('splineFeatures', 'splineFeatures: needs at least 2 knots')
  return {
    name: 'spline-features',
    params: { knots, degree, knotPlacement, extrapolation, includeBias },
    fit({ x }) {
      const { n, d, v } = matrix(x, 'splineFeatures')
      const m = knots + 2 * degree
      const nBasis = knots + degree - 1
      const perColumn = includeBias ? nBasis : nBasis - 1
      const all = new Float64Array(d * m)
      for (let j = 0; j < d; j++) {
        const c = column(v, n, d, j)
        const lo = Math.min(...c)
        const hi = Math.max(...c)
        const base =
          knotPlacement === 'uniform'
            ? Float64Array.from({ length: knots }, (_, k) => lo + ((hi - lo) * k) / (knots - 1))
            : dense.data(
                quantile(
                  c,
                  Array.from({ length: knots }, (_, k) => k / (knots - 1)),
                ),
              )
        const first = base[1] - base[0]
        const last = base[knots - 1] - base[knots - 2]
        for (let k = 0; k < degree; k++) {
          all[j * m + k] = base[0] - (degree - k) * first
          all[j * m + degree + knots + k] = base[knots - 1] + (k + 1) * last
        }
        for (let k = 0; k < knots; k++) all[j * m + degree + k] = base[k]
      }
      const knotRows = Array.from({ length: d }, (_, j) => all.subarray(j * m, (j + 1) * m))
      return {
        kind: 'model',
        name: 'spline-features',
        knots: fromData(all, [d, m]),
        degree,
        perColumn,
        transform(input) {
          const { n: rows, d: cols, v: z } = matrix(input, 'splineFeatures.transform')
          checkColumns(cols, d, 'splineFeatures')
          const width = d * perColumn
          const out = new Float64Array(rows * width)
          for (let j = 0; j < d; j++) {
            const t = knotRows[j]
            const lo = t[degree]
            const hi = t[nBasis]
            const xs = new Float64Array(rows)
            for (let i = 0; i < rows; i++) {
              let xi = z[i * d + j]
              if (xi < lo || xi > hi) {
                if (extrapolation === 'error')
                  throw new DomainError('splineFeatures', `splineFeatures: ${xi} outside [${lo}, ${hi}]`)
                if (extrapolation === 'constant') xi = xi < lo ? lo : hi
              }
              xs[i] = xi
            }
            // One definition of the B-spline basis: aifn-compute/smooth's Cox–de Boor (end pieces continued outside).
            const B = bsplineBasis(fromData(xs, [rows]), fromData(Float64Array.from(t), [t.length]), degree).data
            for (let i = 0; i < rows; i++)
              for (let b = 0; b < perColumn; b++) out[i * width + j * perColumn + b] = B[i * nBasis + b]
          }
          return fromData(out, [rows, width])
        },
      }
    },
  }
}

// ── Random Fourier features ──────────────────────────────────────────────────────────────────────────────────────────

/** Fitted random Fourier features. */
export interface RandomFourierFeatures extends FittedTransform {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'random-fourier-features'
  /**
   * The frequencies $\omegavec_k$ as columns, $d \times D$, drawn from $\Gauss(\zeros, \ell^{-2} \Imat)$.
   */
  readonly frequencies: Tensor
  /** The phases $b_k$, $D$ values, drawn from $\Unif(0, 2\pi)$. */
  readonly phases: Tensor
  /** The kernel's lengthscale $\ell$. */
  readonly lengthscale: number
}

/**
 * Random Fourier features for the squared-exponential kernel
 * $k(\xvec, \xvec') = \exp(-\lVert \xvec - \xvec' \rVert^2 / 2\ell^2)$ (Rahimi and Recht, 2007, "Random features for
 * large-scale kernel machines", NeurIPS): $z_k(\xvec) = \sqrt{2/D} \cos(\omegavec_k^\top \xvec + b_k)$ for
 * $k = 1, \dots, D$, with $\omegavec_k \sim \Gauss(\zeros, \ell^{-2} \Imat)$ and $b_k \sim \Unif(0, 2\pi)$, so that
 * $\zvec(\xvec)^\top \zvec(\xvec')$ is an unbiased estimate of $k(\xvec, \xvec')$ with variance $O(1/D)$. The draws
 * come from the fit's stream (required: `fit` throws `DomainError` without one); scikit-learn's `RBFSampler(gamma)` is
 * $\ell = 1/\sqrt{2\gamma}$.
 *
 * @param options The number of features and the kernel's lengthscale.
 * @param options.components The number $D$ of random features (output columns).
 * @param options.lengthscale The lengthscale $\ell > 0$ of the approximated kernel.
 * @returns An estimator whose `fit({ x }, { stream })` draws the frequencies and phases and returns the fitted
 *   `RandomFourierFeatures`; `transform` gives an $n \times D$ matrix.
 *
 * @example Inner products of the features approximate the kernel
 * const x = tensor([[0, 0], [1, 0], [0, 2]])
 * const model = randomFourierFeatures({ components: 2000 }).fit({ x }, { stream: stream(1) })
 * const z = model.transform(x)
 * print('z zᵀ =', matmul(z, transpose(z)))
 * print('k(x0, x1) = exp(-1/2) =', Math.exp(-0.5), ' k(x0, x2) = exp(-2) =', Math.exp(-2))
 */
export function randomFourierFeatures({
  components = 100,
  lengthscale = 1,
}: { components?: number; lengthscale?: number } = {}): Transformer<Tensor, RandomFourierFeatures> {
  return {
    name: 'random-fourier-features',
    params: { components, lengthscale },
    fit({ x }, options: FitOptions = {}) {
      if (!options.stream)
        throw new DomainError('randomFourierFeatures', 'randomFourierFeatures: fit needs a stream ({ stream })')
      const { d } = matrix(x, 'randomFourierFeatures')
      const D = components
      const omega = values(normals(child(options.stream, 'frequencies'), [d, D], 0, 1 / lengthscale))
      const b = values(uniform(child(options.stream, 'phases'), 0, 2 * Math.PI, { shape: [D] }))
      const scale = Math.sqrt(2 / D)
      return {
        kind: 'model',
        name: 'random-fourier-features',
        frequencies: fromData(omega, [d, D]),
        phases: fromData(b, [D]),
        lengthscale,
        transform(input) {
          const { n, d: cols, v } = matrix(input, 'randomFourierFeatures.transform')
          checkColumns(cols, d, 'randomFourierFeatures')
          const out = new Float64Array(n * D)
          for (let i = 0; i < n; i++) {
            for (let k = 0; k < D; k++) {
              let s = b[k]
              for (let j = 0; j < d; j++) s += v[i * d + j] * omega[j * D + k]
              out[i * D + k] = scale * Math.cos(s)
            }
          }
          return fromData(out, [n, D])
        },
      }
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'polynomialFeatures',
    module: 'learning/preprocessing',
    name: 'Polynomial features',
    summary: 'All monomials of the features up to a degree.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({ degree: int(1, 6, { default: 2 }), interactionOnly: bool(), includeBias: bool({ default: true }) }),
    notes: ['basis-expansions'],
    cite: ['hastie2009'],
  },
  polynomialFeatures,
)

defineModel(
  {
    key: 'splineFeatures',
    module: 'learning/preprocessing',
    name: 'Spline features',
    summary: 'A B-spline basis per feature.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({
      knots: int(2, 50, { default: 5 }),
      degree: int(0, 5, { default: 3 }),
      knotPlacement: oneOf(['uniform', 'quantile']),
      extrapolation: oneOf(['constant', 'continue', 'error']),
      includeBias: bool({ default: true }),
    }),
    notes: ['regression-splines', 'b-splines'],
    cite: ['hastie2009'],
  },
  splineFeatures,
)

defineModel(
  {
    key: 'randomFourierFeatures',
    module: 'learning/preprocessing',
    name: 'Random Fourier features',
    summary: 'Random cosine features whose inner products approximate an RBF kernel.',
    task: 'preprocessing',
    capabilities: ['transform'],
    hyper: space({
      components: int(1, 2000, { default: 100 }),
      lengthscale: real(1e-2, 1e2, { default: 1, scale: 'log' }),
    }),
    notes: ['random-fourier-features'],
    cite: ['rahimi2007'],
  },
  randomFourierFeatures,
)

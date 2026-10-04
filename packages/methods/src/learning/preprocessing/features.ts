/**
 * Feature expansions: polynomial features, B-spline features and random Fourier features.
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
  /** Exponents of each output column, [outputs, d]: output k is Πⱼ xⱼ^powers[k, j]. */
  readonly powers: Tensor
  /** Names such as "1", "x0", "x0^2", "x0 x1". */
  readonly featureNames: readonly string[]
}

/** Multisets of size `size` from 0 … d−1 in lexicographic order (with repetition unless `distinct`). */
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
 * All monomials of the d input columns of degree at most `degree`, in scikit-learn's `PolynomialFeatures` order: by
 * degree, then lexicographically in the column indices. `interactionOnly` keeps products of distinct columns;
 * `includeBias` (default true) leads with the constant column.
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
  /** The full knot vector of each column (the base knots extended by `degree` knots at each end), [d, knots + 2·degree]. */
  readonly knots: Tensor
  readonly degree: number
  /** Output columns per input column: knots + degree − 1 (one fewer without the bias). */
  readonly perColumn: number
}

/**
 * B-spline basis expansion of each column, as scikit-learn's `SplineTransformer`: `knots` base knots spread uniformly
 * over the training range (or at its quantiles), extended beyond it by `degree` knots with the end spacing, give
 * knots + degree − 1 B-splines of degree `degree` per column (de Boor, 1978, "A Practical Guide to Splines"). Outside
 * the training range, `extrapolation` holds the boundary values (`'constant'`, the default), continues the end
 * polynomials (`'continue'`) or throws (`'error'`). Without `includeBias` the last spline of each column is dropped, so
 * the basis no longer sums to one (for use with an intercept).
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
  /** Frequencies ω, [d, D], drawn from N(0, ℓ⁻² I). */
  readonly frequencies: Tensor
  /** Phases b, [D], drawn from U(0, 2π). */
  readonly phases: Tensor
  readonly lengthscale: number
}

/**
 * Random Fourier features for the squared-exponential kernel k(x, x′) = exp(−‖x − x′‖² / 2ℓ²) (Rahimi and Recht,
 * 2007, "Random features for large-scale kernel machines", NeurIPS): z(x) = √(2/D) cos(ωᵀx + b) with ω ~ N(0, ℓ⁻² I)
 * and b ~ U(0, 2π), so that z(x)·z(x′) is an unbiased estimate of k(x, x′) with variance O(1/D). The draws come from
 * the fit's stream (required); scikit-learn's `RBFSampler(gamma)` is ℓ = 1/√(2γ).
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

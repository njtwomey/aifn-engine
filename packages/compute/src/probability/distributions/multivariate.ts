/**
 * Multivariate families: the multivariate normal (with conditioning and marginals), Dirichlet, multinomial and
 * Wishart. Log-densities are compositions of primitives (`aifn-compute/numerics/linalg`'s Cholesky factor and triangular solves are
 * differentiable), so they are differentiable in the value and in the parameters.
 */

import { AifnError, DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  cholesky,
  choleskyLogDet,
  inverse,
  solve,
  solveTriangular,
  matrixTrace,
  type CholeskyOptions,
} from 'aifn-compute/numerics/linalg'
import {
  chiSquare as chiSquareDraws,
  dirichlet as dirichletDraws,
  multinomial as multinomialDraws,
  multivariateNormal as multivariateNormalDraws,
} from 'aifn-compute/probability/samplers'
import { normal as normalDraws, standardNormals, type Stream } from 'aifn-compute/foundation/random'
import { digamma, logFactorial, logGamma } from 'aifn-compute/numerics/special'
import {
  add,
  broadcastTo,
  diagonal,
  div,
  expandDims,
  exp,
  eye,
  fromData,
  item,
  log,
  matmul,
  mul,
  outer,
  reshape,
  shapeOfValue,
  slice,
  sqrt,
  square,
  stack,
  sub,
  sum,
  toFlat,
  transpose,
  unwrap,
  where,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Multivariate, SampleOptions, Support } from './types'
import { check, drawShape, isInteger, LOG_2PI, raw, scalarOf, sumLast } from './util'
import { xlogy } from 'aifn-compute/numerics/special'

/** A mask over the rows (last axis) of x: 1 where `test(row)` holds; a number for a single row. */
function rowMask(x: Value, test: (row: number[]) => boolean): Raw {
  const r = unwrap(x)
  if (typeof r === 'number') throw new ShapeError('multivariate', 'expected a vector (or a batch of vectors)')
  const k = r.shape[r.shape.length - 1]
  const flat = toFlat(r)
  const rows = flat.length / k
  const out = new Float64Array(rows)
  for (let i = 0; i < rows; i++) out[i] = test(flat.slice(i * k, (i + 1) * k)) ? 1 : 0
  return r.shape.length === 1 ? out[0] : fromData(out, r.shape.slice(0, -1))
}

/** `expr` where the mask holds, `fill` elsewhere (the mask of a single row is a number). */
function maskRows(ok: Raw, expr: Value, fill: number): Value {
  if (ok === 1) return expr
  if (ok === 0 && typeof ok === 'number') return fill
  return where(ok, expr, fill)
}

/** Selection matrix S (rows e_{indices[i]}ᵀ), so that S x picks the entries and S Σ Sᵀ the sub-block. */
function selector(indices: readonly number[], d: number): Tensor {
  const out = new Float64Array(indices.length * d)
  indices.forEach((j, i) => {
    if (!Number.isInteger(j) || j < 0 || j >= d)
      throw new DomainError('multivariate', `index ${j} is outside 0 … ${d - 1}`)
    out[i * d + j] = 1
  })
  return fromData(out, [indices.length, d])
}

/** Build a multivariate object: fills in `prob`, `stddev` and the sample-shape handling. */
function multivariate<P extends Value, Extra extends object>(
  base: {
    name: string
    params: Record<string, Value>
    batchShape: readonly number[]
    eventShape: readonly number[]
    support: Support
    discrete?: boolean
    logProb(x: Value): Value
    mean(): Value
    variance(): Value
    covariance(): Value
    entropy(): Value
    mode(): Value
    sample(s: Stream, shape: number[]): Tensor
    /** A pathwise draw of the full shape, written with primitives (differentiable in the parameters). */
    rsample?(s: Stream, shape: number[]): Value
    expFamily?: Multivariate['expFamily']
  },
  extra: Extra,
): Multivariate<P> & Extra {
  return {
    ...extra,
    kind: 'distribution',
    name: base.name,
    params: base.params,
    batchShape: base.batchShape,
    eventShape: base.eventShape,
    support: base.support,
    discrete: base.discrete ?? false,
    expFamily: base.expFamily,
    logProb: base.logProb,
    prob: (x: Value) => exp(base.logProb(x)),
    mean: base.mean,
    variance: base.variance,
    stddev: () => sqrt(base.variance()),
    covariance: base.covariance,
    entropy: base.entropy,
    mode: base.mode,
    sample: (s: Stream, options?: SampleOptions) => {
      for (const [k, v] of Object.entries(base.params)) raw(v, `${base.name}.sample (parameter ${k})`)
      return base.sample(s, drawShape(base.batchShape, options).shape)
    },
    ...(base.rsample
      ? { rsample: (s: Stream, options?: SampleOptions) => base.rsample!(s, drawShape(base.batchShape, options).shape) }
      : {}),
  } as unknown as Multivariate<P> & Extra
}

// ── Multivariate normal ──────────────────────────────────────────────────────────────────────────────────────────────

/** The spread of a multivariate normal: covariance Σ, precision Λ = Σ⁻¹, or a lower Cholesky factor L of Σ. */
export type MultivariateNormalSpread = { covariance: Value } | { precision: Value } | { scaleTril: Value }

/** A multivariate normal: the shared protocol plus its factor and the Gaussian operations. */
export type MultivariateNormal<P extends Value = Value> = Multivariate<P> & {
  /** The mean, `[...batchShape, d]`. */
  readonly loc: Value
  /** Lower-triangular L with Σ = L Lᵀ (d × d, shared by the batch). */
  readonly scaleTril: Value
  /** The diagonal jitter added to factor Σ (0 unless `options.jitter` allowed it). */
  readonly jitter: number
  /**
   * The conditional distribution of the other coordinates (in increasing index order) given x[indices] = values:
   * mean μ_f + Σ_fo Σ_oo⁻¹ (x_o − μ_o) and covariance Σ_ff − Σ_fo Σ_oo⁻¹ Σ_of (Bishop, 2006, PRML §2.3.1).
   * `values` has shape `[..., indices.length]` and broadcasts with the batch.
   */
  condition(indices: readonly number[], values: Value): MultivariateNormal<Value>
  /** The marginal distribution of x[indices]: mean μ_I and covariance Σ_II. */
  marginal(indices: readonly number[]): MultivariateNormal<Value>
}

/**
 * The multivariate normal N(loc, Σ) in d dimensions. `loc` has shape `[d]` (or `[..., d]` for a batch sharing one
 * covariance). The spread is `{ covariance }`, `{ precision }` or `{ scaleTril }` (a lower Cholesky factor of Σ).
 * A covariance that does not factor is an error unless `options.jitter` allows jitter (the `cholesky` options; the
 * jitter used is reported on the result). Densities, entropy and conditioning are differentiable in loc and in the
 * spread.
 */
export function MultivariateNormal<M extends Value, S extends Value>(
  loc: M,
  spread: { covariance: S } | { precision: S } | { scaleTril: S },
  options: { jitter?: CholeskyOptions['jitter'] } = {},
): MultivariateNormal<M | S> {
  const locShape = shapeOfValue(loc)
  if (locShape.length === 0) throw new ShapeError('MultivariateNormal', 'MultivariateNormal: loc must be a vector')
  const d = locShape[locShape.length - 1]
  const batchShape = locShape.slice(0, -1)
  let L: Value
  let jitter = 0
  if ('scaleTril' in spread) L = spread.scaleTril
  else {
    const covariance = 'covariance' in spread ? spread.covariance : inverse(spread.precision)
    const c = cholesky(covariance, { jitter: options.jitter ?? false })
    if (c.failed)
      throw new DomainError(
        'MultivariateNormal',
        `MultivariateNormal: the covariance is not positive definite (pivot ${c.failedAt}); pass options.jitter`,
      )
    L = c.L
    jitter = c.jitter
  }
  const lShape = shapeOfValue(L)
  if (lShape.length !== 2 || lShape[0] !== d || lShape[1] !== d)
    throw new ShapeError(
      'MultivariateNormal',
      `MultivariateNormal: needs a ${d} × ${d} spread, got shape [${lShape.join(', ')}]`,
    )
  const covariance = () => matmul(L, transpose(L))
  const halfLogDet = mul(0.5, choleskyLogDet(L))
  const atBatch = (v: Value, event: number[]) =>
    batchShape.length === 0 ? v : broadcastTo(v, [...batchShape, ...event])

  const logProb = (x: Value): Value => {
    const diff = sub(x, loc)
    const shape = shapeOfValue(diff)
    const lead = shape.slice(0, -1)
    let maha: Value
    if (lead.length === 0) maha = sum(square(solveTriangular(L, diff)))
    else {
      const m = lead.reduce((a, b) => a * b, 1)
      const z = solveTriangular(L, transpose(reshape(diff, [m, d])))
      maha = reshape(sum(square(z), 0), lead)
    }
    return sub(mul(-0.5, maha), add(halfLogDet, 0.5 * d * LOG_2PI))
  }

  const self = multivariate<M | S, object>(
    {
      name: 'MultivariateNormal',
      params: 'scaleTril' in spread ? { loc, scaleTril: L } : { loc, ...spread },
      batchShape,
      eventShape: [d],
      support: { type: 'real-vector' },
      logProb,
      mean: () => loc,
      variance: () => atBatch(diagonal(covariance()), [d]),
      covariance: () => atBatch(covariance(), [d, d]),
      entropy: () => atBatch(add(0.5 * d * (1 + LOG_2PI), halfLogDet), []),
      mode: () => loc,
      sample: (s, shape) =>
        multivariateNormalDraws(
          s,
          raw(loc, 'MultivariateNormal') as Tensor,
          { choleskyFactor: raw(L, 'sample') as Tensor },
          { shape },
        ),
      // loc + L z with z ~ N(0, I): the same draws as `sample`, differentiable in loc and in the factor (and through
      // the Cholesky factorisation, in a covariance or precision).
      rsample: (s, shape) =>
        add(
          loc,
          matmul(fromData(standardNormals(s, shape.reduce((a, b) => a * b, 1) * d), [...shape, d]), transpose(L)),
        ),
      expFamily: {
        naturalParams: () => {
          const precision = inverse(covariance())
          return [matmul(loc, precision), mul(-0.5, precision)]
        },
        sufficientStats: (x) => [x, outer(x, x)],
        logPartition: () => {
          const precision = inverse(covariance())
          return add(mul(0.5, sum(mul(matmul(loc, precision), loc))), halfLogDet)
        },
        logBaseMeasure: () => -0.5 * d * LOG_2PI,
      },
    },
    {},
  )

  const condition = (indices: readonly number[], values: Value): MultivariateNormal<Value> => {
    const observed = new Set(indices)
    const free = Array.from({ length: d }, (_, j) => j).filter((j) => !observed.has(j))
    if (free.length === 0)
      throw new DomainError('MultivariateNormal', 'MultivariateNormal.condition: every coordinate is observed')
    const So = selector(indices, d)
    const Sf = selector(free, d)
    const sigma = covariance()
    const sigmaOO = matmul(matmul(So, sigma), transpose(So))
    const sigmaOF = matmul(matmul(So, sigma), transpose(Sf))
    const sigmaFF = matmul(matmul(Sf, sigma), transpose(Sf))
    // The gain K = Σ_fo Σ_oo⁻¹, as the transpose of Σ_oo⁻¹ Σ_of (a solve, never an explicit inverse).
    const gainT = solve(sigmaOO, sigmaOF)
    const residual = sub(values, matmul(loc, transpose(So)))
    const mean = add(matmul(loc, transpose(Sf)), matmul(residual, gainT))
    const cov = sub(sigmaFF, matmul(transpose(sigmaOF), gainT))
    // Symmetrise: rounding leaves the Schur complement slightly asymmetric.
    return MultivariateNormal(mean as Value, { covariance: mul(0.5, add(cov, transpose(cov))) as Value }, options)
  }

  const marginal = (indices: readonly number[]): MultivariateNormal<Value> => {
    const S = selector(indices, d)
    const mean: Value = matmul(loc, transpose(S))
    return MultivariateNormal(mean, { covariance: matmul(matmul(S, covariance()), transpose(S)) as Value })
  }

  return Object.assign(self, { loc, scaleTril: L, jitter, condition, marginal }) as MultivariateNormal<M | S>
}

// ── Dirichlet ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Covariance of a vector with probabilities q and scale c: c (diag(q) − q qᵀ), batched over leading axes. */
function simplexCovariance(q: Value, K: number, c: Value): Value {
  const column = expandDims(q, -1)
  const row = expandDims(q, -2)
  return mul(c, sub(mul(column, eye(K)), mul(column, row)))
}

/** Tolerance on Σ x = 1 for a point of the simplex. */
const SIMPLEX_TOLERANCE = 1e-9

/**
 * The Dirichlet distribution on the (K − 1)-simplex with concentrations α > 0 (last axis K; a batch `[..., K]`).
 * Points off the simplex (a negative entry, or a sum further than 1e-9·K from 1) have log-density −∞. The mode is
 * (α − 1)/(α₀ − K) when every αₖ > 1 and NaN otherwise. An exponential family with η = α − 1 and T(x) = log x.
 */
export function Dirichlet<A extends Value>(concentration: A): Multivariate<A> {
  const shape = shapeOfValue(concentration)
  if (shape.length === 0) throw new ShapeError('Dirichlet', 'Dirichlet: concentration must be a vector')
  check('Dirichlet', 'concentration', concentration, (x) => x > 0, 'positive')
  const K = shape[shape.length - 1]
  const alpha = concentration
  const total = sum(alpha, -1, true)
  const totalFlat = sumLast(alpha)
  const logNormaliser = sub(sumLast(logGamma(alpha)), logGamma(totalFlat))
  const mean = div(alpha, total)
  return multivariate<A, object>(
    {
      name: 'Dirichlet',
      params: { concentration },
      batchShape: shape.slice(0, -1),
      eventShape: [K],
      support: { type: 'simplex' },
      logProb: (x) => {
        const ok = rowMask(x, (row) => {
          let s = 0
          for (const v of row) {
            if (!(v >= 0)) return false
            s += v
          }
          return Math.abs(s - 1) <= SIMPLEX_TOLERANCE * K
        })
        return maskRows(ok, sub(sumLast(xlogy(sub(alpha, 1), x)), logNormaliser), -Infinity)
      },
      mean: () => mean,
      variance: () => div(mul(mean, sub(1, mean)), add(total, 1)),
      covariance: () => simplexCovariance(mean, K, div(1, add(expandDims(total, -1), 1))),
      entropy: () =>
        sub(
          add(logNormaliser, mul(sub(totalFlat, K), digamma(totalFlat))),
          sumLast(mul(sub(alpha, 1), digamma(alpha))),
        ),
      mode: () => {
        const ok = rowMask(alpha, (row) => row.every((a) => a > 1))
        const m = div(sub(alpha, 1), sub(total, K))
        return ok === 1 ? m : where(typeof ok === 'number' ? ok : expandDims(ok, -1), m, NaN)
      },
      sample: (s, drawShape_) => dirichletDraws(s, raw(alpha, 'Dirichlet') as Tensor, { shape: drawShape_ }),
      expFamily: {
        naturalParams: () => [sub(alpha, 1)],
        sufficientStats: (x) => [log(x)],
        logPartition: () => logNormaliser,
        logBaseMeasure: () => 0,
      },
    },
    {},
  )
}

// ── Multinomial ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** Visit every vector of K non-negative integers summing to n. */
function compositions(n: number, K: number, visit: (counts: number[]) => void): void {
  const counts = new Array<number>(K).fill(0)
  const go = (j: number, left: number) => {
    if (j === K - 1) {
      counts[j] = left
      visit(counts)
      return
    }
    for (let c = 0; c <= left; c++) {
      counts[j] = c
      go(j + 1, left - c)
    }
  }
  go(0, n)
}

/** The number of compositions of n into K parts, C(n + K − 1, K − 1). */
function compositionCount(n: number, K: number): number {
  let c = 1
  for (let i = 1; i < K; i++) c = (c * (n + i)) / i
  return c
}

/**
 * The multinomial distribution: counts in K categories from n trials (a non-negative integer, a constant) with
 * probabilities p (last axis K, normalised here; a batch `[..., K]`). Count vectors that are not non-negative integers
 * summing to n have log-mass −∞. The entropy is an exact sum over all C(n + K − 1, K − 1) outcomes, allowed up to
 * 10⁶ of them; `mode()` has no closed form and throws.
 */
export function Multinomial<N extends Value, P extends Value>(n: N, p: P): Multivariate<N | P> {
  check('Multinomial', 'n', n, (x) => isInteger(x) && x >= 0, 'a non-negative integer')
  const shape = shapeOfValue(p)
  if (shape.length === 0) throw new ShapeError('Multinomial', 'Multinomial: p must be a vector')
  check('Multinomial', 'p', p, (x) => x >= 0, 'non-negative')
  const K = shape[shape.length - 1]
  const probs = div(p, sum(p, -1, true))
  const trials = typeof unwrap(n) === 'number' ? (unwrap(n) as number) : item(unwrap(n) as Tensor)
  const logProb = (x: Value) => {
    const ok = rowMask(
      x,
      (row) => row.every((v) => isInteger(v) && v >= 0) && row.reduce((a, b) => a + b, 0) === trials,
    )
    return maskRows(ok, add(sub(logFactorial(n), sumLast(logFactorial(x))), sumLast(xlogy(x, probs))), -Infinity)
  }
  return multivariate<N | P, object>(
    {
      name: 'Multinomial',
      params: { n, p },
      batchShape: shape.slice(0, -1),
      eventShape: [K],
      support: { type: 'count-vector', total: n },
      discrete: true,
      logProb,
      mean: () => mul(n, probs),
      variance: () => mul(mul(n, probs), sub(1, probs)),
      covariance: () => simplexCovariance(probs, K, n),
      entropy: () => {
        raw(p, 'Multinomial.entropy')
        if (shape.length !== 1)
          throw new AifnError('Multinomial', 'Multinomial.entropy: only for a single distribution (no batch)')
        if (compositionCount(trials, K) > 1e6)
          throw new AifnError('Multinomial', 'Multinomial.entropy: more than 10⁶ outcomes to sum')
        let h = 0
        compositions(trials, K, (counts) => {
          const lp = unwrap(logProb(fromData(Float64Array.from(counts), [K]))) as number
          if (lp > -Infinity) h -= Math.exp(lp) * lp
        })
        return h
      },
      mode: () => {
        throw new AifnError('Multinomial', 'Multinomial.mode: no closed form')
      },
      sample: (s, drawShape_) =>
        multinomialDraws(s, raw(n, 'Multinomial') as number, raw(probs, 'Multinomial') as Tensor, {
          shape: drawShape_,
        }),
    },
    {},
  )
}

// ── Wishart ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** log Γ_d(a) = d(d − 1)/4 · log π + Σⱼ₌₁ᵈ log Γ(a + (1 − j)/2), the multivariate log-gamma function. */
function logMultivariateGamma(a: Value, d: number): Value {
  let total: Value = (d * (d - 1) * Math.log(Math.PI)) / 4
  for (let j = 1; j <= d; j++) total = add(total, logGamma(add(a, (1 - j) / 2)))
  return total
}

/** ψ_d(a) = Σⱼ₌₁ᵈ ψ(a + (1 − j)/2), the derivative of log Γ_d. */
function multivariateDigamma(a: Value, d: number): Value {
  let total: Value = 0
  for (let j = 1; j <= d; j++) total = add(total, digamma(add(a, (1 - j) / 2)))
  return total
}

/**
 * The Wishart distribution over d × d positive-definite matrices with `df` = ν > d − 1 degrees of freedom and scale
 * matrix V (scipy's wishart(df, scale)); the mean is νV. `logProb` takes one matrix `[d, d]` or a stack `[..., d, d]`;
 * a matrix that is not positive definite has log-density −∞. Draws use Bartlett's decomposition (Bartlett, 1933;
 * Smith and Hocking, 1972): X = (L A)(L A)ᵀ with V = L Lᵀ, A lower triangular, A_ii = √χ²(ν − i) and standard
 * normals below the diagonal. `covariance()` (a 4-tensor) is not provided and throws.
 */
export function Wishart<D extends Value, V extends Value>(df: D, scale: V): Multivariate<D | V> {
  const shape = shapeOfValue(scale)
  if (shape.length !== 2 || shape[0] !== shape[1])
    throw new ShapeError('Wishart', 'Wishart: scale must be a square matrix')
  const d = shape[0]
  check('Wishart', 'df', df, (x) => x > d - 1, `greater than d − 1 = ${d - 1}`)
  const factor = cholesky(scale, { jitter: false })
  if (factor.failed) throw new ShapeError('Wishart', 'Wishart: the scale matrix is not positive definite')
  const LV = factor.L
  const logDetV = choleskyLogDet(LV)
  // log of the normaliser 2^{νd/2} |V|^{ν/2} Γ_d(ν/2).
  const constant = add(
    add(mul(0.5 * d * Math.LN2, df), mul(mul(0.5, df), logDetV)),
    logMultivariateGamma(mul(0.5, df), d),
  )
  const one = (X: Value): Value => {
    const c = cholesky(X, { jitter: false })
    if (c.failed) return -Infinity
    const logDetX = choleskyLogDet(c.L)
    // tr(V⁻¹ X) through the factor of V: V⁻¹X = L⁻ᵀ L⁻¹ X.
    const quadratic = matrixTrace(solveTriangular(LV, solveTriangular(LV, X), { transpose: true }))
    return sub(mul(mul(0.5, sub(df, d + 1)), logDetX), add(mul(0.5, quadratic), constant))
  }
  const logProb = (x: Value): Value => {
    const xs = shapeOfValue(x)
    if (xs.length === 2) return one(x)
    const lead = xs.slice(0, -2)
    const m = lead.reduce((a, b) => a * b, 1)
    const flat = reshape(x, [m, d, d])
    const out: Value[] = []
    for (let i = 0; i < m; i++) out.push(one(slice(flat, i)))
    return reshape(stack(out.map((v) => (typeof v === 'number' ? fromData(new Float64Array([v]), []) : v))), lead)
  }
  const diagV = diagonal(scale)
  return multivariate<D | V, object>(
    {
      name: 'Wishart',
      params: { df, scale },
      batchShape: [],
      eventShape: [d, d],
      support: { type: 'positive-definite' },
      logProb,
      mean: () => mul(df, scale),
      variance: () => mul(df, add(square(scale), outer(diagV, diagV))),
      covariance: () => {
        throw new AifnError(
          'Wishart',
          'Wishart.covariance: the covariance of a matrix is a 4-tensor; use variance() for its entries',
        )
      },
      entropy: () =>
        add(
          sub(
            add(add(mul((d + 1) / 2, logDetV), (d * (d + 1) * Math.LN2) / 2), logMultivariateGamma(mul(0.5, df), d)),
            mul(mul(0.5, sub(df, d + 1)), multivariateDigamma(mul(0.5, df), d)),
          ),
          mul(0.5 * d, df),
        ),
      mode: () => where(scalarOf(unwrap(df)) >= d + 1 ? 1 : 0, mul(sub(df, d + 1), scale), NaN),
      sample: (s, drawShape_) => {
        const nu = raw(df, 'Wishart') as number
        const L = toFlat(raw(LV, 'Wishart') as Tensor)
        const count = drawShape_.reduce((a, b) => a * b, 1)
        const out = new Float64Array(count * d * d)
        const A = new Float64Array(d * d)
        const B = new Float64Array(d * d)
        for (let r = 0; r < count; r++) {
          A.fill(0)
          for (let i = 0; i < d; i++) {
            A[i * d + i] = Math.sqrt(chiSquareDraws(s, nu - i) as number)
            for (let j = 0; j < i; j++) A[i * d + j] = normalDraws(s) as number
          }
          // B = L A (both lower triangular), then X = B Bᵀ.
          for (let i = 0; i < d; i++)
            for (let j = 0; j <= i; j++) {
              let v = 0
              for (let k = j; k <= i; k++) v += L[i * d + k] * A[k * d + j]
              B[i * d + j] = v
            }
          for (let i = 0; i < d; i++)
            for (let j = 0; j < d; j++) {
              let v = 0
              for (let k = 0; k <= Math.min(i, j); k++) v += B[i * d + k] * B[j * d + k]
              out[r * d * d + i * d + j] = v
            }
        }
        return fromData(out, [...drawShape_, d, d])
      },
    },
    {},
  )
}

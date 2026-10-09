/**
 * Multivariate families: the multivariate normal (with conditioning and marginals), Dirichlet, multinomial and
 * Wishart. Log-densities are compositions of primitives (`aifn-compute/numerics/linalg`'s Cholesky factor and
 * triangular solves are differentiable), so they are differentiable in the value and in the parameters.
 *
 * Vector families take their parameters with the event on the last axis, $[\dots, d]$, the leading axes being the
 * batch; values outside the support (off the simplex, counts not summing to $n$, a matrix that is not positive
 * definite) have log-density $-\infty$ rather than throwing.
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

/**
 * A mask over the rows (last axis) of `x`: 1 where `test(row)` holds, else 0. Throws `ShapeError` for a scalar.
 *
 * @param x A vector `[K]` or a batch of them `[..., K]`, read through its raw values.
 * @param test The condition on one row, given as a plain array of its $K$ values.
 * @returns A number for a single vector, else a tensor of the leading shape `[...]`.
 */
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

/**
 * `expr` where the mask holds, `fill` elsewhere (the mask of a single row is a number).
 *
 * @param ok The row mask, as `rowMask` returns it.
 * @param expr The value where the row is valid, with the shape of the mask.
 * @param fill The value where it is not (such as $-\infty$ for a log-density).
 * @returns `expr` itself when a single row is valid, `fill` when it is not, else the elementwise choice.
 */
function maskRows(ok: Raw, expr: Value, fill: number): Value {
  if (ok === 1) return expr
  if (ok === 0 && typeof ok === 'number') return fill
  return where(ok, expr, fill)
}

/**
 * The selection matrix $\Smat$ whose row $i$ is $\evec_{j_i}^\top$ for the $i$-th index $j_i$, so that $\Smat\xvec$
 * picks the entries and $\Smat\Sigmamat\Smat^\top$ the sub-block. Throws `DomainError` for an index that is not an
 * integer in $0, \dots, d - 1$.
 *
 * @param indices The coordinates to pick, in the order they are wanted.
 * @param d The dimension $d$ of the vectors picked from.
 * @returns $\Smat$, of shape `[indices.length, d]`.
 */
function selector(indices: readonly number[], d: number): Tensor {
  const out = new Float64Array(indices.length * d)
  indices.forEach((j, i) => {
    if (!Number.isInteger(j) || j < 0 || j >= d)
      throw new DomainError('multivariate', `index ${j} is outside 0 … ${d - 1}`)
    out[i * d + j] = 1
  })
  return fromData(out, [indices.length, d])
}

/**
 * Build a multivariate object: fills in `prob`, `stddev` and the sample-shape handling (draws have shape
 * `[...shape, ...batchShape, ...eventShape]`, and parameters must be untraced to sample).
 *
 * @param base The family's own parts: its name, parameters, shapes, support, log-density, moments, entropy, mode,
 *   sampler (given the full draw shape `[...shape, ...batchShape]`), and optionally a pathwise `rsample` and its
 *   exponential-family form.
 * @param extra Further members merged into the result (the multivariate normal adds its own after the call).
 * @returns The distribution, with the members of `extra`.
 */
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

/**
 * The spread of a multivariate normal, one of: `covariance`, the covariance $\Sigmamat$ ($d \times d$); `precision`,
 * $\Lambdamat = \Sigmamat^{-1}$; or `scaleTril`, a lower-triangular Cholesky factor $\Lmat$ with
 * $\Lmat\Lmat^\top = \Sigmamat$.
 */
export type MultivariateNormalSpread = { covariance: Value } | { precision: Value } | { scaleTril: Value }

/** A multivariate normal: the shared protocol plus its factor and the Gaussian operations. */
export type MultivariateNormal<P extends Value = Value> = Multivariate<P> & {
  /** The mean, `[...batchShape, d]`. */
  readonly loc: Value
  /** Lower-triangular $\Lmat$ with $\Sigmamat = \Lmat\Lmat^\top$ ($d \times d$, shared by the batch). */
  readonly scaleTril: Value
  /** The diagonal jitter added to factor $\Sigmamat$ (0 unless `options.jitter` allowed it). */
  readonly jitter: number
  /**
   * The conditional distribution of the other coordinates $f$ (in increasing index order) given that the coordinates
   * $o$ at `indices` equal `values`: mean $\muvec_f + \Sigmamat_{fo} \Sigmamat_{oo}^{-1} (\xvec_o - \muvec_o)$ and
   * covariance $\Sigmamat_{ff} - \Sigmamat_{fo} \Sigmamat_{oo}^{-1} \Sigmamat_{of}$ (Bishop, 2006, PRML §2.3.1).
   * `values` has shape `[..., indices.length]` and broadcasts with the batch. Throws `DomainError` when every
   * coordinate is observed.
   */
  condition(indices: readonly number[], values: Value): MultivariateNormal<Value>
  /**
   * The marginal distribution of the coordinates $I$ at `indices` (in that order): mean $\muvec_I$ and covariance
   * $\Sigmamat_{II}$.
   */
  marginal(indices: readonly number[]): MultivariateNormal<Value>
}

/**
 * The multivariate normal $\Gauss(\muvec, \Sigmamat)$ in $d$ dimensions,
 * $p(\xvec) = (2\pi)^{-d/2} \det(\Sigmamat)^{-1/2}
 * \exp\left(-\tfrac{1}{2}(\xvec - \muvec)^\top \Sigmamat^{-1} (\xvec - \muvec)\right)$, evaluated through the
 * Cholesky factor $\Lmat$ of $\Sigmamat$. A covariance that does not factor throws `DomainError` unless
 * `options.jitter` allows jitter (the `cholesky` options; the jitter used is reported on the result). Densities,
 * entropy, pathwise draws (`rsample`) and conditioning are differentiable in $\muvec$ and in the spread. The result
 * also has `condition` and `marginal` (Bishop, 2006, PRML §2.3.1 and §2.3.2).
 *
 * @param loc The mean $\muvec$: shape `[d]`, or `[..., d]` for a batch sharing one covariance.
 * @param spread `{ covariance }` ($\Sigmamat$), `{ precision }` ($\Sigmamat^{-1}$, inverted here) or
 *   `{ scaleTril }` (a lower Cholesky factor $\Lmat$ of $\Sigmamat$, used as given), each $d \times d$.
 * @param options `jitter`: how much diagonal jitter `cholesky` may add to factor the covariance (default `false`,
 *   none). Passed on to the distributions `condition` returns (`marginal` does not pass it on).
 * @returns The distribution, with `loc`, `scaleTril`, `jitter`, `condition` and `marginal`.
 *
 * @example Density at the mean, and the moments
 * const d = MultivariateNormal(tensor([0, 0]), { covariance: tensor([[1, 0], [0, 4]]) })
 * print('p(0) =', d.prob(tensor([0, 0])), ' 1/(2π · 2) =', 1 / (4 * Math.PI))
 * print('variance =', d.variance())
 * print('draws =', d.sample(stream(0), { shape: [3] }))
 *
 * @example Conditioning on an observed coordinate
 * const d = MultivariateNormal(tensor([0, 0]), { covariance: tensor([[1, 0.8], [0.8, 1]]) })
 * const c = d.condition([0], tensor([1]))
 * print('given x0 = 1: mean =', c.mean(), ' variance =', c.variance())
 * print('marginal of x1: mean =', d.marginal([1]).mean(), ' variance =', d.marginal([1]).variance())
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

/**
 * The covariance $c\,(\diag(\qvec) - \qvec\qvec^\top)$ of a vector with probabilities $\qvec$ and scale $c$,
 * batched over leading axes.
 *
 * @param q The probabilities $\qvec$, shape `[..., K]`.
 * @param K The number of categories, the length of the last axis of `q`.
 * @param c The scale $c$: a number, or a tensor `[..., 1, 1]` that broadcasts against the batch.
 * @returns The covariance, shape `[..., K, K]`.
 */
function simplexCovariance(q: Value, K: number, c: Value): Value {
  const column = expandDims(q, -1)
  const row = expandDims(q, -2)
  return mul(c, sub(mul(column, eye(K)), mul(column, row)))
}

/** Tolerance on $\sum_k x_k = 1$ for a point of the simplex, per category. */
const SIMPLEX_TOLERANCE = 1e-9

/**
 * The Dirichlet distribution on the $(K - 1)$-simplex with concentrations $\alphavec > 0$,
 * $p(\xvec) = \frac{\Gamma(\alpha_0)}{\prod_k \Gamma(\alpha_k)} \prod_k x_k^{\alpha_k - 1}$ with
 * $\alpha_0 = \sum_k \alpha_k$. The mean is $\alphavec / \alpha_0$ and the variance of each coordinate
 * $\bar\alpha_k (1 - \bar\alpha_k) / (\alpha_0 + 1)$, with $\bar\alpha_k = \alpha_k / \alpha_0$. Points off the
 * simplex (a negative entry, or a sum further than $10^{-9} K$ from 1) have log-density $-\infty$. The mode is
 * $(\alphavec - 1)/(\alpha_0 - K)$ when every $\alpha_k > 1$ and NaN otherwise. An exponential family with
 * $\etavec = \alphavec - 1$ and $T(\xvec) = \log \xvec$. Throws `ShapeError` for a scalar and `DomainError` for a
 * concentration that is not positive.
 *
 * @param concentration The concentrations $\alphavec$, all positive: shape `[K]`, or `[..., K]` for a batch.
 * @returns The distribution, with event shape `[K]`.
 *
 * @example Density and moments
 * const d = Dirichlet(tensor([1, 2, 1]))
 * print('p(0.25, 0.5, 0.25) =', d.prob(tensor([0.25, 0.5, 0.25])), ' (Γ(4) / Γ(2)) · 0.5 =', 3)
 * print('mean =', d.mean())
 * print('variance =', d.variance())
 *
 * @example The flat Dirichlet is uniform on the simplex
 * const d = Dirichlet(tensor([1, 1, 1]))
 * print('p =', d.prob(tensor([0.2, 0.3, 0.5])), d.prob(tensor([0.6, 0.3, 0.1])))
 * print('off the simplex:', d.prob(tensor([0.5, 0.5, 0.5])))
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

/**
 * Visit every vector of $K$ non-negative integers summing to $n$, in lexicographic order.
 *
 * @param n The total $n$.
 * @param K The number of parts $K$ (at least 1).
 * @param visit Called once per vector; it receives the same array each time, overwritten between calls, so it must
 *   copy what it keeps.
 */
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

/**
 * The number of compositions of $n$ into $K$ non-negative parts, $\binom{n + K - 1}{K - 1}$, as a float.
 *
 * @param n The total $n$.
 * @param K The number of parts $K$.
 */
function compositionCount(n: number, K: number): number {
  let c = 1
  for (let i = 1; i < K; i++) c = (c * (n + i)) / i
  return c
}

/**
 * The multinomial distribution: counts $\xvec$ in $K$ categories from $n$ trials with probabilities $\pvec$,
 * $\pr(\xvec) = \frac{n!}{\prod_k x_k!} \prod_k p_k^{x_k}$. The mean is $n\pvec$ and the covariance
 * $n(\diag(\pvec) - \pvec\pvec^\top)$. Count vectors that are not non-negative integers summing to $n$ have
 * log-mass $-\infty$. The entropy (no batch only) is an exact sum over all $\binom{n + K - 1}{K - 1}$ outcomes,
 * allowed up to $10^6$ of them; `mode()` has no closed form and throws. Throws `DomainError` for an invalid $n$ or a
 * negative weight, and `ShapeError` for scalar weights.
 *
 * @param n The number of trials $n$, a non-negative integer (a constant; one value for the whole batch).
 * @param p The category weights, non-negative along the last axis (length $K$; `[..., K]` for a batch), normalised
 *   here to $\pvec$.
 * @returns The distribution, with event shape `[K]`.
 *
 * @example Four trials over three categories
 * const d = Multinomial(4, tensor([0.5, 0.25, 0.25]))
 * print('P(2, 1, 1) =', d.prob(tensor([2, 1, 1])), ' 12 / 4³ =', 12 / 64)
 * print('mean =', d.mean(), ' variance =', d.variance())
 * print('draws =', d.sample(stream(2), { shape: [3] }))
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

/**
 * The multivariate log-gamma function
 * $\log \Gamma_d(a) = \frac{d(d - 1)}{4} \log \pi + \sum_{j=1}^{d} \log \Gamma\left(a + \frac{1 - j}{2}\right)$
 * (differentiable in $a$).
 *
 * @param a The argument $a$, greater than $(d - 1)/2$.
 * @param d The dimension $d$.
 */
function logMultivariateGamma(a: Value, d: number): Value {
  let total: Value = (d * (d - 1) * Math.log(Math.PI)) / 4
  for (let j = 1; j <= d; j++) total = add(total, logGamma(add(a, (1 - j) / 2)))
  return total
}

/**
 * The multivariate digamma function $\psi_d(a) = \sum_{j=1}^{d} \psi\left(a + \frac{1 - j}{2}\right)$, the
 * derivative of $\log \Gamma_d$.
 *
 * @param a The argument $a$, greater than $(d - 1)/2$.
 * @param d The dimension $d$.
 */
function multivariateDigamma(a: Value, d: number): Value {
  let total: Value = 0
  for (let j = 1; j <= d; j++) total = add(total, digamma(add(a, (1 - j) / 2)))
  return total
}

/**
 * The Wishart distribution over $d \times d$ positive-definite matrices with $\nu > d - 1$ degrees of freedom and
 * scale matrix $\Vmat$ (scipy's wishart(df, scale)):
 * $p(\Xmat) = \frac{\det(\Xmat)^{(\nu - d - 1)/2} \exp\left(-\tfrac{1}{2}\trace(\Vmat^{-1}\Xmat)\right)}
 * {2^{\nu d/2} \det(\Vmat)^{\nu/2} \Gamma_d(\nu/2)}$. The mean is $\nu\Vmat$, the variance of entry $ij$ is
 * $\nu(V_{ij}^2 + V_{ii} V_{jj})$, and the mode is $(\nu - d - 1)\Vmat$ when $\nu \ge d + 1$ (NaN otherwise).
 * `logProb` takes one matrix `[d, d]` or a stack `[..., d, d]`; a matrix that is not positive definite has
 * log-density $-\infty$. Draws use Bartlett's decomposition (Bartlett, 1933; Smith and Hocking, 1972):
 * $\Xmat = (\Lmat\Amat)(\Lmat\Amat)^\top$ with $\Vmat = \Lmat\Lmat^\top$, $\Amat$ lower triangular,
 * $A_{ii} = \sqrt{c_i}$ with $c_i \sim \ChiSq(\nu - i)$ for $i = 0, \dots, d - 1$, and standard normals below the
 * diagonal. `covariance()` (a 4-tensor) is not provided and throws. Throws `ShapeError` for a scale that is not
 * square or not positive definite, and `DomainError` for $\nu \le d - 1$.
 *
 * @param df The degrees of freedom $\nu > d - 1$, a single number (no batch).
 * @param scale The scale matrix $\Vmat$, $d \times d$ symmetric positive definite.
 * @returns The distribution, with event shape `[d, d]` and no batch.
 *
 * @example Density at the identity, and the moments
 * const d = Wishart(3, tensor([[1, 0], [0, 1]]))
 * print('p(I) =', d.prob(tensor([[1, 0], [0, 1]])), ' e^-1 / (4π) =', Math.exp(-1) / (4 * Math.PI))
 * print('mean =', d.mean())
 * print('variance =', d.variance())
 *
 * @example A seeded draw is symmetric positive definite
 * print('X =', Wishart(3, tensor([[1, 0.5], [0.5, 1]])).sample(stream(4)))
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

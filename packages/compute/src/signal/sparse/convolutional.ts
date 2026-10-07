/**
 * Convolutional sparse coding and convolutional dictionary learning: a signal $\yvec \in \reals^N$ written as a sum
 * of $K$ short filters $\dvec_k \in \reals^L$, each convolved with a sparse activation map $\zvec_k \in \reals^M$,
 *
 * $$\yvec \approx \sum_{k=1}^K \zvec_k * \dvec_k, \qquad M = N - L + 1,$$
 *
 * with the full convolution, so that every placed copy of a filter lies inside the signal (Grosse et al., 2007;
 * Zeiler et al., 2010). It is sparse coding over the $N \times KM$ dictionary of every shift of every filter, a banded
 * Toeplitz block per filter, applied by convolution and never formed.
 *
 * - Coding (`convolutionalSparseCode`, `convolutionalSparseCodeSteps`): the convolutional lasso
 *   $\min_{\zvec} \frac{1}{2}\lVert \yvec - \sum_k \zvec_k * \dvec_k \rVert^2 + \lambda \sum_k \lVert \zvec_k \rVert_1$
 *   by FISTA (`proximalGradient` with `proxL1`). The gradient with respect to $\zvec_k$ is $-(\rvec \star \dvec_k)$, the
 *   residual correlated with the filter over the valid lags, which is the adjoint of convolution with $\dvec_k$. The
 *   step is $1/(1.1\,\hat L)$, with $\hat L$ a power-iteration estimate of the largest eigenvalue of the synthesis
 *   operator's Gram operator, the Lipschitz constant of the gradient.
 * - Learning (`convolutionalDictionaryLearning`, `convolutionalDictionaryLearningSteps`): alternate coding, warm-started
 *   from the previous maps, with a filter update for fixed maps, the least-squares filters subject to
 *   $\lVert \dvec_k \rVert \le 1$ (Garcia-Cardona and Wohlberg, 2018). Column $(k, l)$ of the filter update's design
 *   matrix is $\zvec_k$ shifted by $l$, so its normal equations need only correlations of the maps; they are solved by
 *   projected FISTA from the current filters, with step $1/L_G$ for $L_G$ the Gershgorin bound on the Gram matrix. The
 *   constraint fixes the scale ambiguity $(\dvec_k, \zvec_k) \to (c\,\dvec_k, \zvec_k / c)$ that would let the penalty
 *   shrink without limit, and the filter update never increases the objective.
 *
 * Filters are the rows of a $K \times L$ matrix $\Dmat$, and the maps the rows of a $K \times M$ matrix $\Zmat$: unlike
 * the rest of `aifn-compute/signal/sparse`, whose dictionaries hold their atoms as columns, a filter here is a row. Fast
 * solvers work in the Fourier domain with ADMM (Bristow et al., 2013; Wohlberg, 2016); this module keeps the direct
 * convolutions, which suit short filters and signals of a few thousand samples.
 */

import type { MatrixLike, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { convolve, correlate } from 'aifn-compute/foundation/convolution'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, permutation, type Stream } from 'aifn-compute/foundation/random'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { proximalGradient, proxL1, type Prox, type ProximalGradientState } from 'aifn-compute/optim/proximal'

type F64 = dense.F64

/**
 * The rows of a row-major matrix, each as its own array.
 *
 * @param data The matrix, row-major.
 * @param r The number of rows.
 * @param c The number of columns.
 * @returns The $r$ rows, each $c$ values.
 */
const rows = (data: ArrayLike<number>, r: Size, c: Size): F64[] =>
  Array.from({ length: r }, (_, i) => Float64Array.from({ length: c }, (_, j) => data[i * c + j]))

/**
 * Rows joined into one row-major array.
 *
 * @param parts The rows, in order.
 * @returns Their values one after another.
 */
const join = (parts: readonly ArrayLike<number>[]): F64 => {
  const out = new Float64Array(parts.reduce((n, p) => n + p.length, 0))
  let at = 0
  for (const p of parts) {
    out.set(p, at)
    at += p.length
  }
  return out
}

/**
 * A copy of a vector's values.
 *
 * @param x The vector.
 * @returns Its values.
 */
const flat = (x: Tensor | VectorLike): F64 => Float64Array.from(dense.toF64(x, 'convolutional'))

/**
 * Read filters as a $K \times L$ matrix, one per row.
 *
 * @param D The filters.
 * @param where The caller's name, for error messages.
 * @returns The filters as rows, with their number $K$ and length $L$.
 */
function readFilters(D: MatrixLike, where: string): { d: F64[]; K: Size; L: Size } {
  const { data, m, n } = dense.toMatrixF64(D, where)
  if (m === 0 || n === 0) throw new DomainError(where, `${where}: there are no filters`)
  return { d: rows(data, m, n), K: m, L: n }
}

/**
 * The synthesis $\sum_k \zvec_k * \dvec_k$ (full convolution) for filters and maps as rows.
 *
 * @param d The filters, $K$ rows of length $L$.
 * @param z The maps, $K$ rows of length $M$.
 * @returns The signal, $M + L - 1$ values.
 */
function synth(d: readonly F64[], z: readonly F64[]): F64 {
  const N = z[0].length + d[0].length - 1
  const out = new Float64Array(N)
  d.forEach((dk, k) => {
    const part = dense.toF64(convolve(z[k], dk, { method: 'direct' }), 'convolutional')
    for (let n = 0; n < N; n++) out[n] += part[n]
  })
  return out
}

/**
 * The adjoint of the synthesis: $\rvec$ correlated with every filter over the valid lags.
 *
 * @param d The filters, $K$ rows of length $L$.
 * @param r A signal, $N$ values.
 * @returns $K$ maps of $M = N - L + 1$ values, joined row after row.
 */
function adjoint(d: readonly F64[], r: F64): F64 {
  return join(d.map((dk) => dense.toF64(correlate(r, dk, { mode: 'valid', method: 'direct' }), 'convolutional')))
}

/**
 * The largest eigenvalue of $\Amat^\top\Amat$ for the synthesis $\Amat: \Zmat \mapsto \sum_k \zvec_k * \dvec_k$, by
 * power iteration from a fixed start, so it is deterministic. Power iteration approaches it from below; callers add a
 * margin.
 *
 * @param d The filters, $K$ rows of length $L$.
 * @param M The length of each map.
 * @param iterations The power iterations. Default 10.
 * @returns The estimate $\hat L$ of the gradient's Lipschitz constant.
 */
function synthesisLipschitz(d: readonly F64[], M: Size, iterations = 10): number {
  const K = d.length
  let v = Float64Array.from({ length: K * M }, (_, i) => 1 + 0.5 * Math.sin(i))
  let value = 0
  for (let it = 0; it < iterations; it++) {
    const norm = Math.sqrt(dense.dot(v, v)) || 1
    for (let i = 0; i < v.length; i++) v[i] /= norm
    v = adjoint(d, synth(d, rows(v, K, M)))
    value = Math.sqrt(dense.dot(v, v))
  }
  return value
}

/**
 * The synthesis $\sum_k \zvec_k * \dvec_k$ of filters and activation maps: each filter placed at every position its
 * map is non-zero, scaled by the map's value, and the copies added.
 *
 * @param D The filters $\Dmat$, $K \times L$, one per row.
 * @param Z The activation maps $\Zmat$, $K \times M$, one per row.
 * @returns The signal, $M + L - 1$ values.
 *
 * @example Two copies of one filter
 * print(convolutionalSynthesis([[1, 2, 1]], [[0, 1, 0, 0, -0.5]]))
 */
export function convolutionalSynthesis(D: MatrixLike, Z: MatrixLike): Tensor {
  const { d, K } = readFilters(D, 'convolutionalSynthesis')
  const z = dense.toMatrixF64(Z, 'convolutionalSynthesis')
  if (z.m !== K)
    throw new ShapeError('convolutionalSynthesis', `convolutionalSynthesis: ${K} filters but ${z.m} activation maps`)
  return dense.vec(synth(d, rows(z.data, z.m, z.n)))
}

/** Options of `convolutionalSparseCodeSteps` and `convolutionalSparseCode`. */
export type ConvolutionalCodeOptions = {
  /** The weight $\lambda > 0$ of the $\ell_1$ penalty: larger gives fewer, larger activations and a looser fit. */
  lambda: number
  /** The starting maps, $K \times M$ (a warm start); default zero. */
  init?: MatrixLike
  /** FISTA's extrapolation (default true); false runs ISTA. */
  accelerated?: boolean
  /** Stop when the gradient mapping's norm is at most this. Default $10^{-7}$. */
  tolerance?: number
}

/**
 * Convolutional sparse coding as a step-through algorithm: FISTA on
 * $\frac{1}{2}\lVert \yvec - \sum_k \zvec_k * \dvec_k \rVert^2 + \lambda \sum_k \lVert \zvec_k \rVert_1$ over the maps,
 * with step $1/(1.1\,\hat L)$ for $\hat L$ the power-iteration estimate of the gradient's Lipschitz constant.
 *
 * @param D The filters $\Dmat$, $K \times L$, one per row, $L \le N$.
 * @param y The signal $\yvec$, $N$ values.
 * @param options The penalty, the warm start, the acceleration and the tolerance.
 * @returns The algorithm; its state is that of `proximalGradient`, with the maps' rows joined in `x` ($KM$ values,
 *   map $k$ at $[kM, (k + 1)M)$) and the objective in `value`.
 *
 * @example The objective falls with the steps
 * const y = [0, 0, 1, 2, 1, 0, 0, 0, -1, -2, -1, 0]
 * const alg = convolutionalSparseCodeSteps([[1, 2, 1]], y, { lambda: 0.1 })
 * for (const t of [1, 10, 100]) print(`after ${t} steps: objective =`, run(alg, undefined, t).value)
 */
export function convolutionalSparseCodeSteps(
  D: MatrixLike,
  y: VectorLike,
  options: ConvolutionalCodeOptions,
): Algorithm<void, ProximalGradientState> {
  const { lambda, accelerated = true, tolerance = 1e-7 } = options
  const where = 'convolutionalSparseCode'
  if (!(lambda > 0)) throw new DomainError(where, `${where}: lambda must be positive, got ${lambda}`)
  const { d, K, L } = readFilters(D, where)
  const signal = flat(y)
  const N = signal.length
  if (L > N) throw new ShapeError(where, `${where}: filters of length ${L} are longer than the signal (${N})`)
  const M = N - L + 1
  let x0 = new Float64Array(K * M)
  if (options.init) {
    const z0 = dense.toMatrixF64(options.init, where)
    if (z0.m !== K || z0.n !== M)
      throw new ShapeError(where, `${where}: init must be ${K} × ${M}, got ${z0.m} × ${z0.n}`)
    x0 = Float64Array.from(z0.data)
  }
  const f = (x: Tensor | VectorLike) => {
    const fitted = synth(d, rows(flat(x), K, M))
    const r = signal.map((v, n) => v - fitted[n])
    return { value: 0.5 * dense.dot(r, r), grad: adjoint(d, r).map((g) => -g) }
  }
  // With every filter zero the gradient is zero and any step will do.
  const lipschitz = synthesisLipschitz(d, M)
  const alg = proximalGradient(f, proxL1(lambda), {
    stepSize: lipschitz > 1e-12 ? 1 / (1.1 * lipschitz) : 1,
    accelerated,
    tolerance,
  })
  return { ...alg, name: where, init: (_start, s) => alg.init({ x0 }, s) }
}

/** The result of `convolutionalSparseCode`. */
export type ConvolutionalCode = {
  /** The activation maps $\Zmat$, $K \times M$. */
  Z: Tensor
  /** $\yvec - \sum_k \zvec_k * \dvec_k$, $N$ values. */
  residual: Tensor
  /** The residual's Euclidean norm. */
  residualNorm: number
  /** The objective at the maps. */
  objective: number
  /** Non-zero activations in all maps. */
  nonzeros: Size
  /** FISTA steps taken. */
  steps: Size
  /** Whether the tolerance was met. */
  converged: boolean
}

/**
 * Convolutional sparse coding run to the tolerance or for at most `maxSteps` steps: the one-call form of
 * `convolutionalSparseCodeSteps`.
 *
 * @param D The filters $\Dmat$, $K \times L$, one per row.
 * @param y The signal $\yvec$, $N$ values.
 * @param options The options of `convolutionalSparseCodeSteps` and the step budget.
 * @param options.maxSteps The most FISTA steps. Default 200.
 * @returns The maps, the residual and its norm, the objective, the number of non-zeros, the steps and whether the
 *   tolerance was met.
 *
 * @example Find where a pulse occurs
 * const y = [0, 0, 1, 2, 1, 0, 0, 0, -1, -2, -1, 0]
 * const { Z, nonzeros } = convolutionalSparseCode([[1, 2, 1]], y, { lambda: 0.1 })
 * print('Z =', Z, ' non-zeros:', nonzeros)
 */
export function convolutionalSparseCode(
  D: MatrixLike,
  y: VectorLike,
  options: ConvolutionalCodeOptions & { maxSteps?: Size },
): ConvolutionalCode {
  const { d, K, L } = readFilters(D, 'convolutionalSparseCode')
  const s = run(convolutionalSparseCodeSteps(D, y, options), undefined, options.maxSteps ?? 200)
  const signal = flat(y)
  const M = signal.length - L + 1
  const zf = flat(s.x)
  const fitted = synth(d, rows(zf, K, M))
  const residual = signal.map((v, n) => v - fitted[n])
  return {
    Z: dense.mat(zf, K, M),
    residual: dense.vec(residual),
    residualNorm: Math.sqrt(dense.dot(residual, residual)),
    objective: s.value,
    nonzeros: zf.reduce((n, v) => n + (v !== 0 ? 1 : 0), 0),
    steps: s.t,
    converged: s.converged ?? false,
  }
}

/**
 * The least-squares filters for fixed maps subject to $\lVert \dvec_k \rVert \le 1$, by projected FISTA from `d0`.
 * The Gram entry for taps $(a, l)$ and $(b, l')$ is the correlation of $\zvec_a$ and $\zvec_b$ at lag $l - l'$, and
 * the right-hand side is $\yvec$ correlated with each map.
 *
 * @param signal The signal, $N$ values.
 * @param z The maps as rows, $K$ of length $M$.
 * @param d0 The current filters as rows, the start of the projected FISTA.
 * @param steps The projected FISTA steps.
 * @returns The new filters as rows.
 */
function updateFilters(signal: F64, z: readonly F64[], d0: readonly F64[], steps: Size): F64[] {
  const K = z.length
  const L = d0[0].length
  const P = K * L
  const G = new Float64Array(P * P)
  const centre = z[0].length - 1
  for (let a = 0; a < K; a++)
    for (let b = 0; b < K; b++) {
      // full[c] is the correlation of z_a with z_b at lag c − (M − 1).
      const full = dense.toF64(correlate(z[a], z[b], { mode: 'full' }), 'convolutional')
      for (let l = 0; l < L; l++) for (let lp = 0; lp < L; lp++) G[(a * L + l) * P + b * L + lp] = full[centre + lp - l]
    }
  const rhs = join(
    z.map((zk) => dense.toF64(correlate(signal, zk, { mode: 'valid', method: 'direct' }), 'convolutional')),
  )
  let bound = Number.MIN_VALUE
  for (let i = 0; i < P; i++) {
    let s = 0
    for (let j = 0; j < P; j++) s += Math.abs(G[i * P + j])
    bound = Math.max(bound, s)
  }
  const f = (x: Tensor | VectorLike) => {
    const v = flat(x)
    const Gv = dense.matVec(G, v, P, P)
    return { value: 0.5 * dense.dot(v, Gv) - dense.dot(v, rhs), grad: Gv.map((g, i) => g - rhs[i]) }
  }
  // The projection onto the unit ball for every filter: a filter that is too long is scaled back to unit norm.
  const shrink = (dk: F64) => {
    const n = Math.sqrt(dense.dot(dk, dk))
    return n > 1 ? dk.map((x) => x / n) : dk
  }
  const balls: Prox = {
    name: 'unit-balls',
    value: (x) => (rows(flat(x), K, L).every((dk) => Math.sqrt(dense.dot(dk, dk)) <= 1 + 1e-9) ? 0 : Infinity),
    prox: (v) => dense.vec(join(rows(flat(v), K, L).map(shrink))),
  }
  const alg = proximalGradient(f, balls, { stepSize: 1 / bound, accelerated: true, tolerance: 1e-9 })
  return rows(flat(run(alg, { x0: join(d0) }, steps).x), K, L)
}

/** Options of `convolutionalDictionaryLearningSteps` and `convolutionalDictionaryLearning`. */
export type ConvolutionalDictionaryOptions = {
  /** The number of filters $K$. */
  filters: Size
  /** The filter length $L$, at most the signal's length. */
  length: Size
  /** The penalty $\lambda > 0$ of the coding. */
  lambda: number
  /** FISTA steps of coding per round, warm-started from the previous maps. Default 25. */
  codeSteps?: Size
  /** Projected FISTA steps of the filter update per round. Default 60. */
  filterSteps?: Size
  /** The initial filters, $K \times L$ (scaled to unit norm); default $K$ stretches of the signal at random. */
  init?: MatrixLike
  /** Stop when the objective changes by less than this fraction of itself in one round. Default $10^{-6}$. */
  tolerance?: number
}

/** The state of `convolutionalDictionaryLearningSteps`. */
export interface ConvolutionalDictionaryState extends Status {
  /** Rounds taken: filter update then coding. */
  t: Size
  /** The filters $\Dmat$, $K \times L$, each of norm at most 1. */
  D: Tensor
  /** The activation maps $\Zmat$, $K \times M$. */
  Z: Tensor
  /** $\frac{1}{2}\lVert \yvec - \sum_k \zvec_k * \dvec_k \rVert^2 + \lambda \sum_k \lVert \zvec_k \rVert_1$. */
  objective: number
  /** The squared-error part, $\frac{1}{2}\lVert \yvec - \sum_k \zvec_k * \dvec_k \rVert^2$. */
  fit: number
  /** Non-zero activations in all maps. */
  nonzeros: Size
  /** Set when the objective's relative change in a round falls below the tolerance. */
  converged: boolean
}

/**
 * Convolutional dictionary learning as a step-through algorithm. The initial state codes the signal over the initial
 * filters; each step updates the filters for the current maps (least squares with $\lVert \dvec_k \rVert \le 1$) and
 * then recodes, warm-started. Neither stage increases the objective, up to the coding's finite step budget. The
 * objective is not convex in filters and maps together, so the result depends on the start: a filter that starts as a
 * misaligned stretch of a pattern can settle on a piece of it, coded by two neighbouring activations. A filter a few
 * samples longer than the patterns avoids this, and holds each pattern at some shift within its window (a shifted
 * filter with a shifted map fits equally well).
 *
 * @param y The signal $\yvec$, $N$ values.
 * @param options The number and length of the filters, the penalty, the step budgets, the start and the tolerance.
 * @returns The algorithm; `init` takes a stream for the random initial filters (unused when `options.init` gives them).
 *
 * @example The objective falls round by round
 * // Six copies of the pulse (1, 3, 1) at irregular places and scales.
 * const y = new Array(60).fill(0)
 * for (const [t, a] of [[3, 1], [11, -0.5], [19, 2], [30, 1.5], [41, -1], [50, 0.8]])
 *   [1, 3, 1].forEach((v, l) => (y[t + l] += a * v))
 * const alg = convolutionalDictionaryLearningSteps(y, { filters: 1, length: 6, lambda: 0.1 })
 * for (const t of [0, 5, 20, 60]) print(`round ${t}: objective =`, run(alg, undefined, t).objective)
 */
export function convolutionalDictionaryLearningSteps(
  y: VectorLike,
  options: ConvolutionalDictionaryOptions,
): Algorithm<void, ConvolutionalDictionaryState> {
  const { filters: K, length: L, lambda, codeSteps = 25, filterSteps = 60, tolerance = 1e-6 } = options
  const where = 'convolutionalDictionaryLearning'
  const signal = flat(y)
  const N = signal.length
  if (!(Number.isInteger(K) && K >= 1))
    throw new DomainError(where, `${where}: filters must be a positive integer, got ${K}`)
  if (!(Number.isInteger(L) && L >= 1 && L <= N))
    throw new DomainError(where, `${where}: length must be an integer from 1 to ${N}, got ${L}`)
  if (!(lambda > 0)) throw new DomainError(where, `${where}: lambda must be positive, got ${lambda}`)
  const M = N - L + 1
  const code = (d: readonly F64[], z: readonly F64[]) => {
    const alg = convolutionalSparseCodeSteps(dense.mat(join(d), K, L), signal, {
      lambda,
      init: dense.mat(join(z), K, M),
    })
    return rows(flat(run(alg, undefined, codeSteps).x), K, M)
  }
  const measure = (d: readonly F64[], z: readonly F64[]) => {
    const fitted = synth(d, z)
    const r = signal.map((v, n) => v - fitted[n])
    const fit = 0.5 * dense.dot(r, r)
    let l1 = 0
    let nonzeros = 0
    for (const zk of z)
      for (const v of zk) {
        l1 += Math.abs(v)
        if (v !== 0) nonzeros++
      }
    return { D: dense.mat(join(d), K, L), Z: dense.mat(join(z), K, M), objective: fit + lambda * l1, fit, nonzeros }
  }
  const unit = (dk: F64) => {
    const n = Math.sqrt(dense.dot(dk, dk))
    return n > 0 ? dk.map((v) => v / n) : dk
  }
  return {
    name: where,
    init: (_start, s: Stream) => {
      let d: F64[]
      if (options.init) {
        const r = readFilters(options.init, `${where}: init`)
        if (r.K !== K || r.L !== L)
          throw new ShapeError(where, `${where}: init must be ${K} × ${L}, got ${r.K} × ${r.L}`)
        d = r.d
      } else {
        // K stretches of the signal at random positions, less their means, skipping flat ones; a unit spike when the
        // signal has too few stretches that are not flat.
        const order = dense.toF64(permutation(child(s, 'filters'), M), where)
        const centred = (t: number) => {
          const w = signal.slice(t, t + L)
          const mean = w.reduce((a, b) => a + b, 0) / L
          return w.map((v) => v - mean)
        }
        d = []
        for (let i = 0; i < M && d.length < K; i++) {
          const w = centred(order[i])
          if (dense.dot(w, w) > 1e-12) d.push(w)
        }
        while (d.length < K) {
          const spike = new Float64Array(L)
          spike[d.length % L] = 1
          d.push(spike)
        }
      }
      d = d.map(unit)
      const z = code(
        d,
        Array.from({ length: K }, () => new Float64Array(M)),
      )
      return { t: 0, ...measure(d, z), converged: false }
    },
    step: (state) => {
      const z0 = rows(dense.toMatrixF64(state.Z, where).data, K, M)
      const d = updateFilters(signal, z0, rows(dense.toMatrixF64(state.D, where).data, K, L), filterSteps)
      const z = code(d, z0)
      const next = measure(d, z)
      const change = Math.abs(state.objective - next.objective) / Math.max(state.objective, Number.MIN_VALUE)
      return {
        t: state.t + 1,
        ...next,
        converged: change < tolerance,
        diverged: !Number.isFinite(next.objective),
      }
    },
  }
}

/**
 * Convolutional dictionary learning run to the tolerance or for at most `maxSteps` rounds: the one-call form of
 * `convolutionalDictionaryLearningSteps`.
 *
 * @param y The signal $\yvec$, $N$ values.
 * @param options The options of `convolutionalDictionaryLearningSteps`, the round budget and the random stream.
 * @param options.maxSteps The most rounds. Default 20.
 * @param options.stream The stream for the random initial filters.
 * @returns The filters, the maps, the objective and the rounds taken.
 *
 * @example Recover a pulse and where it occurs
 * // Six copies of the pulse (1, 3, 1) at irregular places and scales. The learnt filter is the pulse at unit norm,
 * // (0.30, 0.90, 0.30), at some shift within its six samples, and its map is non-zero once per copy.
 * const y = new Array(60).fill(0)
 * for (const [t, a] of [[3, 1], [11, -0.5], [19, 2], [30, 1.5], [41, -1], [50, 0.8]])
 *   [1, 3, 1].forEach((v, l) => (y[t + l] += a * v))
 * const { D, Z } = convolutionalDictionaryLearning(y, { filters: 1, length: 6, lambda: 0.1, maxSteps: 60 })
 * print('filter =', D)
 * print('activations at', toArray(Z)[0].flatMap((v, t) => (v !== 0 ? [t] : [])))
 */
export function convolutionalDictionaryLearning(
  y: VectorLike,
  options: ConvolutionalDictionaryOptions & { maxSteps?: Size; stream?: Stream },
): { D: Tensor; Z: Tensor; objective: number; steps: Size } {
  const s = run(convolutionalDictionaryLearningSteps(y, options), undefined, options.maxSteps ?? 20, {
    stream: options.stream,
  })
  return { D: s.D, Z: s.Z, objective: s.objective, steps: s.t }
}

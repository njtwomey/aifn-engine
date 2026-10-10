/**
 * Channel capacity $C = \max_p I(X; Y)$ and the rate–distortion function $R(D)$ of discrete distributions, by the
 * Blahut–Arimoto algorithms.
 *
 * The algorithms are those of Blahut (1972), "Computation of channel capacity and rate-distortion functions", IEEE
 * Trans. Inf. Theory 18(4), and Arimoto (1972), in the same issue; Cover and Thomas (2006), §10.8, give both. Each is a
 * traceable `Algorithm` whose state carries the current distributions, and for capacity the bounds
 * $\text{lower} \le C \le \text{upper}$, so a figure can show them converging; `channelCapacity`, `rateDistortion` and
 * `rateDistortionCurve` run them to convergence. Distributions are over symbols $0, \dots, n - 1$, and a channel or a
 * distortion is an array of rows, one per input symbol. Everything is computed in nats; the convenience functions
 * convert rates to another `base` (2 for bits). Malformed input throws `DomainError` or `ShapeError`.
 */

import type { MatrixLike, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { dense, fromData, logsumexp, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { softmax } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A matrix as its dimensions and a fresh row-major copy of its entries.
 *
 * @param m The matrix.
 * @param where The caller's name, for error messages.
 * @returns The number of `rows` and `cols`, and the $\text{rows} \cdot \text{cols}$ `values` row by row.
 */
function matrixOf(m: MatrixLike, where: string): { rows: number; cols: number; values: Float64Array } {
  const { data, m: rows, n: cols } = dense.toMatrixF64(m, where)
  return { rows, cols, values: Float64Array.from(data) }
}

/**
 * A vector as a fresh array of its values.
 *
 * @param v The vector.
 * @param where The caller's name, for error messages.
 * @returns A copy of its values.
 */
const vectorOf = (v: VectorLike, where: string): Float64Array => Float64Array.from(dense.toF64(v, where))

// ── Channel capacity ─────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of the capacity algorithm. */
export type CapacityOptions = {
  /** Converged once $\text{upper} - \text{lower} < \text{tolerance}$, in nats (default 1e-10). */
  tolerance?: number
}

/**
 * Where the capacity algorithm starts: `initial`, an input distribution over the channel's rows (normalised to sum to
 * 1; default uniform).
 */
export type CapacityStart = { initial?: VectorLike }

/** A Blahut–Arimoto capacity state. All information quantities are in nats. */
export type CapacityState = Status & {
  /** Reweightings done. */
  t: number
  /** The current input distribution $p(x)$. */
  input: Tensor
  /** The output distribution $q(y) = \sum_x p(x) W(y \mid x)$. */
  output: Tensor
  /** $D_x = \KL(W(\cdot \mid x) \,\Vert\, q)$ for each input $x$. */
  divergences: Tensor
  /** $I(p; W) = \sum_x p(x) D_x$, the mutual information of the current input. */
  information: number
  /** $\log \sum_x p(x) e^{D_x} \le C$ (Blahut, 1972, Theorem 2). */
  lower: number
  /** $\max_x D_x \ge C$. */
  upper: number
  /** The bounds are within the tolerance. */
  converged: boolean
}

/**
 * The capacity state of an input distribution: its output distribution, the divergences $D_x$, the mutual
 * information and the bounds on $C$. It is flagged `diverged` when the information is not finite.
 *
 * @param W The channel $W(y \mid x)$ as a row-major array of $n_x n_y$ values, row $x$ the distribution of the output
 *   given input $x$; not modified.
 * @param nx The number of inputs $n_x$ (rows of `W`).
 * @param ny The number of outputs $n_y$ (columns of `W`).
 * @param p The input distribution, $n_x$ values summing to 1; kept by the state, not copied.
 * @param tolerance The gap $\text{upper} - \text{lower}$ (nats) below which the state is `converged`.
 * @param t The number of reweightings done, stored as the state's `t`.
 * @returns The state, in nats.
 */
function capacityState(
  W: Float64Array,
  nx: number,
  ny: number,
  p: Float64Array,
  tolerance: number,
  t: number,
): CapacityState {
  const q = new Float64Array(ny)
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) q[y] += p[x] * W[x * ny + y]
  const D = new Float64Array(nx)
  for (let x = 0; x < nx; x++) {
    let s = 0
    for (let y = 0; y < ny; y++) {
      const w = W[x * ny + y]
      if (w > 0) s += w * Math.log(w / q[y])
    }
    D[x] = s
  }
  let information = 0
  let upper = -Infinity
  for (let x = 0; x < nx; x++) {
    information += p[x] * D[x]
    upper = Math.max(upper, D[x])
  }
  // The lower bound log Σₓ p(x) e^{D_x} (inputs with p(x) = 0 contribute log 0 = −∞).
  const lower = logsumexp(logWeighted(p, D)) as number
  return {
    t,
    input: fromData(p, [nx]),
    output: fromData(q, [ny]),
    divergences: fromData(D, [nx]),
    information,
    lower,
    upper,
    converged: upper - lower < tolerance,
    diverged: !Number.isFinite(information),
  }
}

/**
 * Blahut–Arimoto for the capacity $C = \max_p I(X; Y)$ of a discrete memoryless channel $W(y \mid x)$, as a traceable
 * algorithm. Each step reweights the input, $p'(x) \propto p(x) \exp \KL(W(\cdot \mid x) \,\Vert\, q)$, which
 * increases $I(p; W)$ monotonically; the state carries the bounds $\text{lower} \le C \le \text{upper}$ and is
 * converged once they are within the tolerance. `init` takes `{ initial }`, the starting input distribution (default
 * uniform). Throws `DomainError` when an entry of the channel is negative or a row does not sum to 1 (within
 * $10^{-9}$).
 *
 * @param channel The channel $W$: row $x$ is the distribution of the output given input $x$, so rows are inputs and
 *   columns outputs.
 * @param options The stopping rule.
 * @param options.tolerance The gap $\text{upper} - \text{lower}$, in nats, below which the state is converged.
 * @returns The algorithm: `init` from a `CapacityStart`, `step` one reweighting, states in nats.
 *
 * @example The Z channel: bounds close on $C = \log 1.25$ nats
 * // Input 0 always arrives; input 1 is flipped to 0 half the time.
 * const Z = [[1, 0], [0.5, 0.5]]
 * const first = run(blahutArimotoCapacity(Z), {}, 0)
 * const last = run(blahutArimotoCapacity(Z), {}, 1000)
 * print('start: lower, upper =', first.lower, first.upper)
 * print('end: lower, upper =', last.lower, last.upper, 'after', last.t, 'steps')
 * print('log 1.25 =', Math.log(1.25))
 * print('capacity-achieving input =', last.input)
 */
export function blahutArimotoCapacity(
  channel: MatrixLike,
  { tolerance = 1e-10 }: CapacityOptions = {},
): Algorithm<CapacityStart, CapacityState> {
  const { rows, cols, values } = matrixOf(channel, 'blahutArimotoCapacity')
  for (let x = 0; x < rows; x++) {
    let s = 0
    for (let y = 0; y < cols; y++) {
      const w = values[x * cols + y]
      if (!(w >= 0))
        throw new DomainError('blahutArimotoCapacity', 'blahutArimotoCapacity: channel entries must be non-negative')
      s += w
    }
    if (Math.abs(s - 1) > 1e-9)
      throw new DomainError('blahutArimotoCapacity', `blahutArimotoCapacity: row ${x} of the channel sums to ${s}`)
  }
  return {
    name: 'blahutArimotoCapacity',
    init({ initial } = {}) {
      const p = initial ? vectorOf(initial, 'blahutArimotoCapacity') : new Float64Array(rows).fill(1 / rows)
      const total = p.reduce((a, b) => a + b, 0)
      return capacityState(
        values,
        rows,
        cols,
        p.map((v) => v / total),
        tolerance,
        0,
      )
    },
    step: (state) => capacityStep(values, rows, cols, tolerance, state),
  }
}

/**
 * $\log p(x) + D_x$ for each input: the log weights of the reweighting $p'(x) \propto p(x) e^{D_x}$. An input with
 * $p(x) = 0$ gets $-\infty$.
 *
 * @param p The input distribution, $n_x$ values.
 * @param D The divergences $D_x$, $n_x$ values.
 * @returns A tensor of $n_x$ log weights.
 */
function logWeighted(p: ArrayLike<number>, D: ArrayLike<number>): Tensor {
  return fromData(
    Float64Array.from(D, (d, x) => Math.log(p[x]) + d),
    [D.length],
  )
}

/**
 * One Blahut–Arimoto reweighting $p'(x) \propto p(x) e^{D_x}$, and the state of the new input.
 *
 * @param W The channel as a row-major array of $n_x n_y$ values (see `capacityState`).
 * @param nx The number of inputs $n_x$.
 * @param ny The number of outputs $n_y$.
 * @param tolerance The convergence gap, in nats.
 * @param state The current state, whose `input` and `divergences` are reweighted; not modified.
 * @returns The next state, with `t` one more.
 */
function capacityStep(W: Float64Array, nx: number, ny: number, tolerance: number, state: CapacityState): CapacityState {
  const next = Float64Array.from(toFlat(softmax(logWeighted(toFlat(state.input), toFlat(state.divergences)))))
  return capacityState(W, nx, ny, next, tolerance, state.t + 1)
}

/** The result of `channelCapacity`, in the requested base. */
export type ChannelCapacity = {
  /** The capacity: the final lower bound, so it never exceeds $C$. */
  capacity: number
  /** The capacity-achieving input distribution (approximately, at the tolerance). */
  input: Tensor
  /** The output distribution $q(y)$ that input induces. */
  output: Tensor
  /** The final lower bound on $C$. */
  lower: number
  /** The final upper bound on $C$. */
  upper: number
  /** Reweightings done. */
  steps: number
  /** Whether the bounds met within the tolerance before `maxSteps`. */
  converged: boolean
}

/**
 * The capacity of a discrete memoryless channel $W(y \mid x)$ by Blahut–Arimoto (`blahutArimotoCapacity` from the
 * uniform input), in nats or the given `base`. Runs until the bounds meet within the tolerance or `maxSteps` is
 * reached; `converged` says which. The reported capacity is the lower bound. Throws `DomainError` for a channel whose
 * entries are negative or whose rows do not sum to 1.
 *
 * @param channel The channel $W$: row $x$ is the distribution of the output given input $x$.
 * @param options The stopping rule and the unit.
 * @param options.tolerance The gap between the bounds, in nats whatever the `base`, at which to stop (default
 *   1e-10).
 * @param options.maxSteps The largest number of reweightings.
 * @param options.base The base of the logarithm the result is reported in (2 for bits); nats when left out.
 * @returns The capacity, its bounds and the input that achieves it, with the number of steps taken.
 *
 * @example A binary symmetric channel has capacity $1 - H(p)$ bits
 * const p = 0.1
 * const { capacity, input } = channelCapacity([[1 - p, p], [p, 1 - p]], { base: 2 })
 * print('C =', capacity)
 * print('1 − H(p) =', 1 + p * Math.log2(p) + (1 - p) * Math.log2(1 - p))
 * print('input =', input)
 *
 * @example An erasure channel loses a fraction $e$ of its capacity
 * // Outputs 0, erased, 1: each input is erased with probability 0.25, so C = 1 − 0.25 bits.
 * const e = 0.25
 * print('C =', channelCapacity([[1 - e, e, 0], [0, e, 1 - e]], { base: 2 }).capacity)
 */
export function channelCapacity(
  channel: MatrixLike,
  { tolerance, maxSteps = 10_000, base }: { tolerance?: number; maxSteps?: number; base?: number } = {},
): ChannelCapacity {
  const s = run(blahutArimotoCapacity(channel, { tolerance }), {}, maxSteps)
  const unit = base === undefined ? 1 : 1 / Math.log(base)
  return {
    capacity: s.lower * unit,
    input: s.input,
    output: s.output,
    lower: s.lower * unit,
    upper: s.upper * unit,
    steps: s.t,
    converged: s.converged,
  }
}

// ── Rate–distortion ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of the rate–distortion algorithm. */
export type RateDistortionOptions = {
  /**
   * Converged once no entry of the reproduction marginal $q(\hat x)$ changes by this much in a step (default
   * 1e-12).
   */
  tolerance?: number
}

/** A Blahut–Arimoto rate–distortion state (nats). */
export type RateDistortionState = Status & {
  /** Alternations done. */
  t: number
  /** The test channel $Q(\hat x \mid x)$, $\lvert \Xcal \rvert \times \lvert \hat\Xcal \rvert$, rows summing to 1. */
  conditional: Tensor
  /** The reproduction marginal $q(\hat x) = \sum_x p(x) Q(\hat x \mid x)$. */
  marginal: Tensor
  /** $I(X; \hat X)$ under $p(x) Q(\hat x \mid x)$, in nats. */
  rate: number
  /** $\expect[d(X, \hat X)]$, in the units of the distortion matrix. */
  distortion: number
  /** $\max_{\hat x} \lvert q'(\hat x) - q(\hat x) \rvert$ in the last step ($\infty$ in the initial state). */
  change: number
  /** The marginal changed by less than the tolerance. */
  converged: boolean
}

/**
 * The test channel from a reproduction marginal, $Q(\hat x \mid x) \propto q(\hat x) e^{-\beta d(x, \hat x)}$, and
 * its marginal, rate and distortion. A reproduction with $q(\hat x) = 0$ keeps probability 0. The state is flagged
 * `diverged` when the rate is NaN.
 *
 * @param p The source distribution $p(x)$, $n_x$ values summing to 1.
 * @param d The distortion $d(x, \hat x)$ as a row-major array of $n_x n_y$ values, row $x$ the source symbol.
 * @param nx The number of source symbols $n_x$.
 * @param ny The number of reproduction symbols $n_y$.
 * @param beta The slope parameter $\beta \ge 0$.
 * @param q The reproduction marginal the test channel is built from, $n_y$ values.
 * @param previous The marginal of the previous state, to measure the `change` against; null for the initial state,
 *   whose change is $\infty$.
 * @param tolerance The change below which the state is `converged`.
 * @param t The number of alternations done, stored as the state's `t`.
 * @returns The state, in nats.
 */
function rateDistortionState(
  p: Float64Array,
  d: Float64Array,
  nx: number,
  ny: number,
  beta: number,
  q: Float64Array,
  previous: Float64Array | null,
  tolerance: number,
  t: number,
): RateDistortionState {
  const Q = new Float64Array(nx * ny)
  for (let x = 0; x < nx; x++) {
    // Normalise with the largest exponent subtracted, so large β does not underflow every entry.
    let m = -Infinity
    for (let y = 0; y < ny; y++) if (q[y] > 0) m = Math.max(m, Math.log(q[y]) - beta * d[x * ny + y])
    let z = 0
    for (let y = 0; y < ny; y++) {
      const v = q[y] > 0 ? Math.exp(Math.log(q[y]) - beta * d[x * ny + y] - m) : 0
      Q[x * ny + y] = v
      z += v
    }
    for (let y = 0; y < ny; y++) Q[x * ny + y] /= z
  }
  const marginal = new Float64Array(ny)
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) marginal[y] += p[x] * Q[x * ny + y]
  let rate = 0
  let distortion = 0
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++) {
      const v = Q[x * ny + y]
      if (v > 0 && p[x] > 0) rate += p[x] * v * Math.log(v / marginal[y])
      distortion += p[x] * v * d[x * ny + y]
    }
  let change = Infinity
  if (previous) {
    change = 0
    for (let y = 0; y < ny; y++) change = Math.max(change, Math.abs(marginal[y] - previous[y]))
  }
  return {
    t,
    conditional: fromData(Q, [nx, ny]),
    marginal: fromData(marginal, [ny]),
    rate,
    distortion,
    change,
    converged: change < tolerance,
    diverged: Number.isNaN(rate),
  }
}

/**
 * Blahut–Arimoto for one point of the rate–distortion function
 * $R(D) = \min I(X; \hat X)$ subject to $\expect[d(X, \hat X)] \le D$, as a traceable algorithm. For a slope $\beta$
 * it alternates $Q(\hat x \mid x) \propto q(\hat x) e^{-\beta d(x, \hat x)}$ and
 * $q(\hat x) = \sum_x p(x) Q(\hat x \mid x)$ (Blahut, 1972, §IV; Cover and Thomas, 2006, §10.8). The state's
 * (distortion, rate) converges to the point of the curve whose slope is $-\beta$ (rate in nats); larger $\beta$ gives
 * lower distortion. No start: $q(\hat x)$ begins uniform. Throws `DomainError` for a negative $\beta$ and
 * `ShapeError` when the distortion does not have one row per source symbol.
 *
 * @param source The source distribution $p(x)$, normalised to sum to 1.
 * @param distortion The distortion $d(x, \hat x) \ge 0$: one row per source symbol, one column per reproduction
 *   symbol.
 * @param beta The slope parameter $\beta \ge 0$, minus the slope of $R(D)$ at the point found.
 * @param options The stopping rule.
 * @param options.tolerance The largest change of the reproduction marginal at which the state is converged.
 * @returns The algorithm: `init` takes no start, `step` is one alternation, states in nats.
 *
 * @example A binary source under Hamming distortion converges at once
 * // By symmetry the uniform start is already optimal: D = 1 / (1 + e^β).
 * const s = run(blahutArimotoRateDistortion([0.5, 0.5], [[0, 1], [1, 0]], 3), undefined, 100)
 * print('rate (nats) =', s.rate, 'distortion =', s.distortion, 'after', s.t, 'steps')
 * print('1 / (1 + e^3) =', 1 / (1 + Math.exp(3)))
 * print('test channel =', s.conditional)
 *
 * @example A skewed source moves the reproduction marginal
 * const s = run(blahutArimotoRateDistortion([0.7, 0.2, 0.1], [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 2), undefined, 1000)
 * print('rate (nats) =', s.rate, 'distortion =', s.distortion, 'after', s.t, 'steps')
 * print('q(x̂) =', s.marginal)
 */
export function blahutArimotoRateDistortion(
  source: VectorLike,
  distortion: MatrixLike,
  beta: number,
  { tolerance = 1e-12 }: RateDistortionOptions = {},
): Algorithm<void, RateDistortionState> {
  if (!(beta >= 0))
    throw new DomainError('blahutArimotoRateDistortion', 'blahutArimotoRateDistortion: beta must be non-negative')
  const raw = vectorOf(source, 'blahutArimotoRateDistortion')
  const total = raw.reduce((a, b) => a + b, 0)
  const p = raw.map((v) => v / total)
  const { rows, cols, values } = matrixOf(distortion, 'blahutArimotoRateDistortion')
  if (rows !== p.length)
    throw new ShapeError(
      'blahutArimotoRateDistortion',
      'blahutArimotoRateDistortion: distortion needs one row per source symbol',
    )
  return {
    name: 'blahutArimotoRateDistortion',
    init: () =>
      rateDistortionState(p, values, rows, cols, beta, new Float64Array(cols).fill(1 / cols), null, tolerance, 0),
    step(state) {
      const q = Float64Array.from(toFlat(state.marginal))
      return rateDistortionState(p, values, rows, cols, beta, q, q, tolerance, state.t + 1)
    },
  }
}

/** One point of $R(D)$, as `rateDistortion` returns it. */
export type RateDistortionPoint = {
  /** The rate $I(X; \hat X)$, in nats or the requested base. */
  rate: number
  /** The expected distortion $D$, in the units of the distortion matrix. */
  distortion: number
  /** The test channel $Q(\hat x \mid x)$ that achieves the point, one row per source symbol. */
  conditional: Tensor
  /** The reproduction marginal $q(\hat x)$. */
  marginal: Tensor
  /** Alternations done. */
  steps: number
  /** Whether the marginal settled within the tolerance before `maxSteps`. */
  converged: boolean
}

/**
 * The point of the rate–distortion curve whose slope is $-\beta$ (rate in nats), by Blahut–Arimoto
 * (`blahutArimotoRateDistortion`): rate in nats (or `base`), distortion in the units of `distortion`. Runs to the
 * tolerance or `maxSteps`; `converged` says which. Throws as `blahutArimotoRateDistortion` does.
 *
 * @param source The source distribution $p(x)$, normalised to sum to 1.
 * @param distortion The distortion $d(x, \hat x) \ge 0$: one row per source symbol, one column per reproduction
 *   symbol.
 * @param beta The slope parameter $\beta \ge 0$; larger values give lower distortion and higher rate.
 * @param options The stopping rule and the unit.
 * @param options.tolerance The largest change of the reproduction marginal at which to stop (default 1e-12).
 * @param options.maxSteps The largest number of alternations.
 * @param options.base The base of the logarithm the rate is reported in (2 for bits); nats when left out. The
 *   distortion is unaffected.
 * @returns The rate and distortion of the point, with its test channel and reproduction marginal.
 *
 * @example A binary source under Hamming distortion: $R(D) = 1 - H(D)$ bits
 * const { rate, distortion: D } = rateDistortion([0.5, 0.5], [[0, 1], [1, 0]], 3, { base: 2 })
 * print('R =', rate, 'at D =', D)
 * print('1 − H(D) =', 1 + D * Math.log2(D) + (1 - D) * Math.log2(1 - D))
 */
export function rateDistortion(
  source: VectorLike,
  distortion: MatrixLike,
  beta: number,
  { tolerance, maxSteps = 10_000, base }: { tolerance?: number; maxSteps?: number; base?: number } = {},
): RateDistortionPoint {
  const s = run(blahutArimotoRateDistortion(source, distortion, beta, { tolerance }), undefined, maxSteps)
  return {
    rate: base === undefined ? s.rate : s.rate / Math.log(base),
    distortion: s.distortion,
    conditional: s.conditional,
    marginal: s.marginal,
    steps: s.t,
    converged: s.converged,
  }
}

/**
 * $R(D)$ traced out over slopes $\beta$: `rateDistortion` at each, with `rate` and `distortion` tensors aligned with
 * `betas`. `converged` is true only when every point converged.
 *
 * @param source The source distribution $p(x)$, normalised to sum to 1.
 * @param distortion The distortion $d(x, \hat x) \ge 0$: one row per source symbol, one column per reproduction
 *   symbol.
 * @param betas The slope parameters, each $\ge 0$; one point of the curve per value, in the order given.
 * @param options The `tolerance`, `maxSteps` and `base` of `rateDistortion`, used for every point.
 * @returns The rates and distortions, one per $\beta$, and whether all points converged.
 *
 * @example The binary Hamming curve, from high distortion to low
 * const curve = rateDistortionCurve([0.5, 0.5], [[0, 1], [1, 0]], [0.5, 1, 2, 4, 8], { base: 2 })
 * print('D =', curve.distortion)
 * print('R =', curve.rate)
 * print('converged =', curve.converged)
 */
export function rateDistortionCurve(
  source: VectorLike,
  distortion: MatrixLike,
  betas: ArrayLike<number>,
  options: { tolerance?: number; maxSteps?: number; base?: number } = {},
): { rate: Tensor; distortion: Tensor; converged: boolean } {
  const n = betas.length
  const rate = new Float64Array(n)
  const dist = new Float64Array(n)
  let converged = true
  for (let i = 0; i < n; i++) {
    const point = rateDistortion(source, distortion, betas[i], options)
    rate[i] = point.rate
    dist[i] = point.distortion
    converged &&= point.converged
  }
  return { rate: fromData(rate, [n]), distortion: fromData(dist, [n]), converged }
}

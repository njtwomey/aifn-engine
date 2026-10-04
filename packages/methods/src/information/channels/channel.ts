/**
 * Channel capacity and the rate–distortion function by the Blahut–Arimoto algorithms (Blahut, 1972, "Computation of
 * channel capacity and rate-distortion functions", IEEE Trans. Inf. Theory 18(4); Arimoto, 1972, same issue). Both are
 * traceable `Algorithm`s whose states carry the current distributions and bounds, so a figure can show them
 * converging. Internally in nats; the convenience functions take a `base`.
 */

import type { MatrixLike, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { dense, fromData, logsumexp, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { softmax } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

function matrixOf(m: MatrixLike, where: string): { rows: number; cols: number; values: Float64Array } {
  const { data, m: rows, n: cols } = dense.toMatrixF64(m, where)
  return { rows, cols, values: Float64Array.from(data) }
}

const vectorOf = (v: VectorLike, where: string): Float64Array => Float64Array.from(dense.toF64(v, where))

// ── Channel capacity ─────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of the capacity algorithm. */
export type CapacityOptions = {
  /** Stop when upper − lower bound < tolerance (nats; default 1e-10). */
  tolerance?: number
}

/** Where the capacity algorithm starts: an input distribution (default uniform). */
export type CapacityStart = { initial?: VectorLike }

/** A Blahut–Arimoto capacity state. All information quantities are in nats. */
export type CapacityState = Status & {
  /** Reweightings done. */
  t: number
  /** The current input distribution p(x). */
  input: Tensor
  /** The output distribution q(y) = Σₓ p(x) W(y | x). */
  output: Tensor
  /** D(W(·|x) ‖ q) for each input x. */
  divergences: Tensor
  /** I(p; W) = Σₓ p(x) D_x, the mutual information of the current input. */
  information: number
  /** log Σₓ p(x) e^{D_x} ≤ C (Blahut, 1972, Theorem 2). */
  lower: number
  /** maxₓ D_x ≥ C. */
  upper: number
  /** The bounds are within the tolerance. */
  converged: boolean
}

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
 * Blahut–Arimoto for the capacity C = max_p I(X; Y) of a discrete memoryless channel W(y | x) (rows are inputs x and
 * sum to 1, columns are outputs y). Each step reweights the input, p′(x) ∝ p(x) exp D(W(·|x) ‖ q), which increases
 * I(p; W) monotonically; the state carries the bounds lower ≤ C ≤ upper and is converged once they are within the
 * tolerance. `init` takes `{ initial }`, the starting input distribution (default uniform).
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

/** log p(x) + D_x as a tensor [nx]: the log weights of the reweighting p′(x) ∝ p(x) e^{D_x}. */
function logWeighted(p: ArrayLike<number>, D: ArrayLike<number>): Tensor {
  return fromData(
    Float64Array.from(D, (d, x) => Math.log(p[x]) + d),
    [D.length],
  )
}

/** One Blahut–Arimoto reweighting p′(x) ∝ p(x) exp D_x. */
function capacityStep(W: Float64Array, nx: number, ny: number, tolerance: number, state: CapacityState): CapacityState {
  const next = Float64Array.from(toFlat(softmax(logWeighted(toFlat(state.input), toFlat(state.divergences)))))
  return capacityState(W, nx, ny, next, tolerance, state.t + 1)
}

/** The result of `channelCapacity`, in the requested base. */
export type ChannelCapacity = {
  capacity: number
  /** The capacity-achieving input distribution (approximately, at the tolerance). */
  input: Tensor
  output: Tensor
  lower: number
  upper: number
  /** Reweightings done. */
  steps: number
  converged: boolean
}

/**
 * The capacity of a discrete memoryless channel W(y | x) (rows x) by Blahut–Arimoto, in nats or the given `base`
 * (2 for bits). Runs until the bounds meet within the tolerance or `maxSteps` (default 10,000) is reached;
 * `converged` says which. The reported capacity is the lower bound.
 *
 * @example channelCapacity([[0.9, 0.1], [0.1, 0.9]], { base: 2 }).capacity // 1 − H₂(0.1) ≈ 0.531
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
  /** Stop when the reproduction marginal changes by less than this (max abs; default 1e-12). */
  tolerance?: number
}

/** A Blahut–Arimoto rate–distortion state (nats). */
export type RateDistortionState = Status & {
  /** Alternations done. */
  t: number
  /** The test channel Q(x̂ | x), [|X|, |X̂|]. */
  conditional: Tensor
  /** The reproduction marginal q(x̂) = Σₓ p(x) Q(x̂ | x). */
  marginal: Tensor
  /** I(X; X̂) under p(x) Q(x̂ | x). */
  rate: number
  /** E[d(X, X̂)]. */
  distortion: number
  /** max |q′ − q| in the last step. */
  change: number
  /** The marginal changed by less than the tolerance. */
  converged: boolean
}

/** The test channel from a reproduction marginal: Q(x̂|x) ∝ q(x̂) e^{−β d(x, x̂)}, and its rate and distortion. */
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
 * Blahut–Arimoto for one point of the rate–distortion function R(D) = min I(X; X̂) subject to E d(X, X̂) ≤ D. For
 * a slope β it alternates Q(x̂|x) ∝ q(x̂) e^{−β d(x, x̂)} and q(x̂) = Σₓ p(x) Q(x̂|x) (Blahut, 1972, §IV; Cover and
 * Thomas, 2006, §10.8) for the source p(x), the distortion d(x, x̂) ≥ 0 ([|X|, |X̂|]) and the slope parameter β ≥ 0
 * (minus the slope of R(D) at the point found; larger β gives lower distortion). The state's (distortion, rate)
 * converges to the point of the curve with slope −β. No start: q(x̂) begins uniform.
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

/** One point of R(D). */
export type RateDistortionPoint = {
  rate: number
  distortion: number
  conditional: Tensor
  marginal: Tensor
  /** Alternations done. */
  steps: number
  converged: boolean
}

/**
 * The point of the rate–distortion curve with slope −β, by Blahut–Arimoto: rate in nats (or `base`), distortion in
 * the units of `distortion`. Runs to the tolerance or `maxSteps` (default 10,000).
 *
 * @example rateDistortion([0.5, 0.5], [[0, 1], [1, 0]], 3, { base: 2 }) // Hamming distortion: R = 1 − H₂(D)
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

/** R(D) traced out over slopes β: `rate` and `distortion` tensors aligned with `betas`. */
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

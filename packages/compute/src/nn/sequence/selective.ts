/**
 * The selective scan of Mamba (Gu and Dao, 2023, Algorithm 2): a diagonal state-space recurrence whose step Δ_t and
 * input and output maps B_t, C_t depend on the input, so the layer can choose per token what to write into its state
 * and what to forget. With A diagonal (D channels × N states), per channel d and state n:
 *
 *   h_t = exp(Δ_t A) ⊙ h_{t−1} + Δ_t B_t x_t,   y_t = C_t · h_t + D x_t.
 *
 * Input-dependent coefficients rule out the convolutional mode of S4, but the recurrence is still linear in h, so it
 * is an associative scan (`linearRecurrence`), computed for every t in O(log L) rounds.
 */

import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import type { Size } from 'aifn-compute/foundation/contracts'
import { softplus } from 'aifn-compute/numerics/special'
import {
  add,
  associativeScan,
  concat,
  exp,
  expandDims,
  fromData,
  mul,
  neg,
  ones,
  shapeOfValue,
  slice,
  sum,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { silu } from 'aifn-compute/nn/functional'
import { Linear, linear, tap, type Layer, type LinearParams } from 'aifn-compute/nn/layers'

/** Options of `selectiveScan`. */
export type SelectiveScanOptions = {
  /** The skip connection D [D] (default none). */
  D?: Value
  /** A gate z [..., L, D]: the output is multiplied by SiLU(z) (Mamba's gated branch). */
  z?: Value
  /** The state before the first token [..., D, N] (default zeros). */
  initial?: Value
}

/** The outputs of `selectiveScan`: y [..., L, D], every state h [..., L, D, N] and the last one. */
export type SelectiveScanResult = { outputs: Value; states: Value; final: Value }

/**
 * Mamba's selective scan (Gu and Dao, 2023; `selective_scan_ref` of the reference code, time first): inputs x [..., L,
 * D], steps Δ [..., L, D] (positive), A [D, N] (negative for stability), B and C [..., L, N]. Discretises with
 * Ā_t = exp(Δ_t A) and B̄_t x_t = Δ_t B_t x_t, runs h_t = Ā_t ⊙ h_{t−1} + B̄_t x_t by the associative scan, and reads
 * y_t = Σ_n C_t[n] h_t[·, n] (+ D x_t, ⊙ SiLU(z_t)). Differentiable in every argument.
 */
export function selectiveScan(
  x: Value,
  delta: Value,
  A: Value,
  B: Value,
  C: Value,
  options: SelectiveScanOptions = {},
): SelectiveScanResult {
  const rank = shapeOfValue(x).length
  const time = rank - 2
  const deltaA = exp(mul(expandDims(delta, -1), A))
  let input = mul(expandDims(mul(delta, x), -1), expandDims(B, -2))
  if (options.initial !== undefined) {
    const L = shapeOfValue(x)[time]
    const lead = Array.from({ length: time }, () => null)
    const firstA = slice(deltaA, ...lead, [0, 1])
    const firstB = slice(input, ...lead, [0, 1])
    const first = add(mul(firstA, expandDims(options.initial, time)), firstB)
    input = concat([first, slice(input, ...lead, [1, L])], time)
  }
  const [, states] = associativeScan(
    ([a1, b1]: readonly Value[], [a2, b2]: readonly Value[]) => [mul(a1, a2), add(mul(a2, b1), b2)],
    [deltaA, input],
    { axis: time },
  )
  let outputs = sum(mul(states, expandDims(C, -2)), -1)
  if (options.D !== undefined) outputs = add(outputs, mul(x, options.D))
  if (options.z !== undefined) outputs = mul(outputs, silu(options.z))
  const L = shapeOfValue(x)[time]
  const lead = Array.from({ length: time }, () => null)
  return { outputs, states, final: slice(states, ...lead, L - 1) }
}

/** Parameters of `SelectiveSsm`. */
export type SelectiveSsmParams = {
  /** Δ's low-rank projection d → r and back r → d (with Δ's bias). */
  deltaDown: LinearParams
  deltaUp: LinearParams
  /** B_t = x_t W_B and C_t = x_t W_C, d → N. */
  inputMap: LinearParams
  outputMap: LinearParams
  /** log(−A) [d, N], so A = −exp(logA) stays negative. */
  logA: Tensor
  /** The skip D [d]. */
  D: Tensor
}

/** Options of `SelectiveSsm`. */
export type SelectiveSsmOptions = {
  /** States per channel N (default 16). */
  state?: Size
  /** Rank r of Δ's projection (default ⌈d/16⌉). */
  deltaRank?: Size
  /** Range of the initial step Δ (default [0.001, 0.1], log-uniform, as Mamba). */
  deltaRange?: readonly [number, number]
}

/**
 * Mamba's selective state-space layer over x [..., L, d] (the SSM inside a Mamba block, without its convolution and
 * gate; Gu and Dao, 2023, §3.2, Algorithm 2): Δ = softplus(x W_↓ W_↑ + b), B = x W_B, C = x W_C, A = −exp(logA)
 * initialised to −(1, 2, …, N) per channel (S4D-real), D = 1; then `selectiveScan`. The states are tapped at
 * `<path>.states`.
 */
export function SelectiveSsm(d: Size, options: SelectiveSsmOptions = {}): Layer<SelectiveSsmParams> {
  const N = options.state ?? 16
  const r = options.deltaRank ?? Math.ceil(d / 16)
  const [lo, hi] = options.deltaRange ?? [0.001, 0.1]
  const down = Linear(d, r, { bias: false })
  const up = Linear(r, d)
  const map = Linear(d, N, { bias: false })
  return {
    kind: 'SelectiveSsm',
    label: `SelectiveSsm(${d} channels, ${N} states)`,
    init: (s: Stream) => {
      // Δ's bias is softplus⁻¹ of a log-uniform step in [lo, hi], so the initial steps cover several time scales.
      const u = uniform(child(s, 'delta'), 0, 1, { shape: [d] }) as Tensor
      const bias = Float64Array.from(u.data as Float64Array, (v) => {
        const step = Math.exp(Math.log(lo) + v * (Math.log(hi) - Math.log(lo)))
        return step + Math.log(-Math.expm1(-step))
      })
      const logA = new Float64Array(d * N)
      for (let c = 0; c < d; c++) for (let n = 0; n < N; n++) logA[c * N + n] = Math.log(n + 1)
      return {
        deltaDown: down.init(child(s, 'deltaDown')),
        deltaUp: { ...up.init(child(s, 'deltaUp')), bias: fromData(bias, [d]) },
        inputMap: map.init(child(s, 'inputMap')),
        outputMap: map.init(child(s, 'outputMap')),
        logA: fromData(logA, [d, N]),
        D: ones([d]),
      }
    },
    apply: (p, x, ctx) => {
      const delta = softplus(linear(linear(x, p.deltaDown.weight), p.deltaUp.weight, p.deltaUp.bias))
      const B = linear(x, p.inputMap.weight)
      const C = linear(x, p.outputMap.weight)
      const { outputs, states } = selectiveScan(x, delta, neg(exp(p.logA)), B, C, { D: p.D })
      tap(ctx, states, 'states')
      return tap(ctx, outputs)
    },
  }
}

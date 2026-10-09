/**
 * The selective scan of Mamba (Gu and Dao, 2023, Algorithm 2): a diagonal state-space recurrence whose step
 * $\Delta_t$ and input and output maps $\bvec_t$, $\cvec_t$ depend on the input, so the layer can choose per token
 * what to write into its state and what to forget. With $\Amat$ diagonal ($D$ channels $\times$ $N$ states), per
 * channel $d$ and state $n$ the state is $h_{t,dn} = e^{\Delta_{t,d} A_{dn}} h_{t-1,dn} + \Delta_{t,d} b_{t,n} x_{t,d}$
 * and the output $y_{t,d} = \sum_n c_{t,n} h_{t,dn} + D_d x_{t,d}$.
 *
 * Input-dependent coefficients rule out the convolutional mode of S4, but the recurrence is still linear in $h$, so it
 * is an associative scan (`linearRecurrence`), computed for every $t$ in $O(\log L)$ rounds.
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
  /** The skip connection, one weight $D_d$ per channel (default none). */
  D?: Value
  /**
   * A gate $\zvec$ of the shape of `x`, $[\dots, L, D]$: the output is multiplied by $\mathrm{SiLU}(\zvec)$ (Mamba's
   * gated branch). Default none.
   */
  z?: Value
  /** The state before the first token, $[\dots, D, N]$ (default zeros). */
  initial?: Value
}

/**
 * The outputs of `selectiveScan`: `outputs` $\yvec$ $[\dots, L, D]$, every state `states` $[\dots, L, D, N]$ and the
 * last one `final` $[\dots, D, N]$.
 */
export type SelectiveScanResult = { outputs: Value; states: Value; final: Value }

/**
 * Mamba's selective scan (Gu and Dao, 2023; `selective_scan_ref` of the reference code, but time first). Discretises
 * with $\bar{\Amat}_t = \exp(\Delta_t \Amat)$ and $\bar{\Bmat}_t x_t = \Delta_t \bvec_t x_t$, runs
 * $\hvec_t = \bar{\Amat}_t \odot \hvec_{t-1} + \bar{\Bmat}_t x_t$ by the associative scan along the time axis, and
 * reads $y_{t,d} = \sum_n c_{t,n} h_{t,dn}$, plus $D_d x_{t,d}$ when `D` is given, times $\mathrm{SiLU}(z_{t,d})$ when
 * `z` is. Differentiable in every argument.
 *
 * @param x The inputs, $[\dots, L, D]$: $L$ tokens of $D$ channels, after any batch axes.
 * @param delta The steps $\Delta$, $[\dots, L, D]$ like `x` (positive; a large step overwrites the state with the
 *   token, a small one keeps the state).
 * @param A The diagonal state matrices, $D \times N$ ($N$ states per channel; negative entries for a stable scan).
 * @param B The input maps $\bvec_t$, $[\dots, L, N]$, shared by the channels.
 * @param C The output maps $\cvec_t$, $[\dots, L, N]$, shared by the channels.
 * @param options The skip `D`, the gate `z` and the initial state.
 * @returns The outputs, every state and the final state.
 *
 * @example A step chooses what is kept and what is forgotten
 * // One channel, one state, three tokens; only the first token is non-zero.
 * const x = tensor([[1], [0], [0]])
 * const A = tensor([[-1]])
 * const B = tensor([[1], [1], [1]])
 * const C = tensor([[1], [1], [1]])
 * // Small steps after the first token hold it in the state.
 * print('kept:', selectiveScan(x, tensor([[1], [0.01], [0.01]]), A, B, C).outputs)
 * // A large step on the second token (whose input is 0) wipes the state.
 * const forget = selectiveScan(x, tensor([[1], [5], [0.01]]), A, B, C)
 * print('forgotten:', forget.outputs)
 * print('final state:', forget.final)
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
  /** The down projection of $\Delta$'s low-rank map, $d \to r$ (weight $d \times r$, no bias). */
  deltaDown: LinearParams
  /** The up projection of $\Delta$'s low-rank map, $r \to d$, with $\Delta$'s bias. */
  deltaUp: LinearParams
  /** $\Wmat_B$ of $\bvec_t = \xvec_t \Wmat_B$, $d \to N$ (no bias). */
  inputMap: LinearParams
  /** $\Wmat_C$ of $\cvec_t = \xvec_t \Wmat_C$, $d \to N$ (no bias). */
  outputMap: LinearParams
  /** $\log(-\Amat)$, $d \times N$, so $\Amat = -\exp(\log(-\Amat))$ stays negative. */
  logA: Tensor
  /** The skip $D$, one weight per channel ($d$ values). */
  D: Tensor
}

/** Options of `SelectiveSsm`. */
export type SelectiveSsmOptions = {
  /** States per channel $N$ (default 16). */
  state?: Size
  /** Rank $r$ of $\Delta$'s projection (default $\lceil d/16 \rceil$). */
  deltaRank?: Size
  /**
   * The range the initial step $\Delta$ of each channel is drawn from, log-uniformly, as in Mamba (default
   * $[0.001, 0.1]$).
   */
  deltaRange?: readonly [number, number]
}

/**
 * Mamba's selective state-space layer over inputs $[\dots, L, d]$ (the SSM inside a Mamba block, without its
 * convolution and gate; Gu and Dao, 2023, §3.2, Algorithm 2):
 * $\Delta = \mathrm{softplus}(\xvec \Wmat_{\downarrow} \Wmat_{\uparrow} + \bvec)$, $\bvec_t = \xvec_t \Wmat_B$,
 * $\cvec_t = \xvec_t \Wmat_C$, $\Amat = -\exp(\log(-\Amat))$ initialised to $-(1, 2, \dots, N)$ in every channel
 * (S4D-real), $D = 1$; then `selectiveScan`. The bias is initialised to $\mathrm{softplus}^{-1}$ of a step drawn
 * log-uniformly from `deltaRange`, the projections by `Linear`'s default. The states are tapped at `<path>.states`.
 *
 * @param d The number of channels $d$, the width of the input and of the output.
 * @param options The state size $N$, the rank of $\Delta$'s projection and the range of its initial steps.
 * @returns The layer: `init` draws the parameters from a stream, `apply` maps $[\dots, L, d]$ to $[\dots, L, d]$.
 *
 * @example A layer of two channels and three states over four tokens
 * const layer = SelectiveSsm(2, { state: 3 })
 * const p = layer.init(stream(0))
 * print('A =', neg(exp(p.logA)))
 * const y = layer.apply(p, normals(stream(1), [4, 2]))
 * print('y =', y)
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

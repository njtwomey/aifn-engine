/**
 * Recurrent cells and their unrolling: the Elman RNN, the GRU (Cho et al., 2014) and the LSTM (Hochreiter &
 * Schmidhuber, 1997, with the forget gate of Gers, Schmidhuber & Cummins, 2000).
 *
 * A cell maps an input and a state to the next state; `unrollRecurrent` scans it over a sequence. Gate order and
 * equations follow PyTorch's `RNNCell`, `GRUCell` and `LSTMCell`, with weights stored transposed (`[in, G * H]` for
 * $G$ gates of hidden size $H$) and, for the RNN and LSTM, the two biases summed into one. Every parameter is drawn
 * from $\Unif(-1/\sqrt{H}, 1/\sqrt{H})$ by default, as in PyTorch.
 */

import { child, type Stream } from 'aifn-compute/foundation/random'
import { sigmoid } from 'aifn-compute/numerics/special'
import {
  add,
  fromData,
  mul,
  slice,
  shapeOfValue,
  stack,
  sub,
  tanh,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { activationFn, type Activation } from 'aifn-compute/nn/functional'
import { lecunUniform, type Initialiser } from 'aifn-compute/nn/init'
import { linear, tap, type Context } from './layers'

/** A recurrent state: the hidden vector `h` and, for the LSTM, the cell `c`. Shapes `[..., H]`. */
export type RecurrentState = { h: Value; c?: Value }

/**
 * Parameters of a cell with $G$ gates of hidden size $H$: `inputWeight`, `[in, G * H]`, `hiddenWeight`, `[H, G * H]`,
 * and `bias`, `[G * H]`, with the gates side by side along the last axis; `hiddenBias`, `[G * H]`, is the GRU's bias
 * on the recurrent term.
 */
export type CellParams = { inputWeight: Tensor; hiddenWeight: Tensor; bias: Tensor; hiddenBias?: Tensor }

/** A recurrent cell: an initialiser, the zero state and a pure step. */
export interface Cell {
  /** The cell type, e.g. `GruCell`. */
  readonly kind: string
  /** A short description, e.g. `GruCell(2 → 4)`. */
  readonly label: string
  /** The hidden size $H$. */
  readonly hidden: number
  /** Fresh parameters drawn from `s`. */
  init(s: Stream): CellParams
  /** The zero state for a batch of leading shape `batch` (default none: vectors of length H). */
  initialState(batch?: readonly number[]): RecurrentState
  /** The next state from the input `x`, `[..., in]`, and the state; pure. */
  step(params: CellParams, x: Value, state: RecurrentState, ctx?: Context): RecurrentState
}

/**
 * Draw a cell's parameters, each from its own child of the stream.
 *
 * @param s The stream; children `inputWeight`, `hiddenWeight`, `bias` and `hiddenBias` are drawn from.
 * @param init The initialiser, called with fans $H$ and $G H$ (so `lecunUniform` gives PyTorch's bound $1/\sqrt{H}$).
 * @param inF The input width.
 * @param hidden The hidden size $H$.
 * @param gates The number of gates $G$.
 * @param twoBiases Whether to draw a second bias, `hiddenBias`, for the recurrent term.
 * @returns The parameters, as `CellParams`.
 */
function cellParams(s: Stream, init: Initialiser, inF: number, hidden: number, gates: number, twoBiases: boolean) {
  // PyTorch draws every recurrent parameter from U(−1/√H, 1/√H); `lecunUniform` with fanIn = H gives that bound.
  const fans = { fanIn: hidden, fanOut: gates * hidden }
  return {
    inputWeight: init(child(s, 'inputWeight'), [inF, gates * hidden], fans),
    hiddenWeight: init(child(s, 'hiddenWeight'), [hidden, gates * hidden], fans),
    bias: init(child(s, 'bias'), [gates * hidden], fans),
    ...(twoBiases ? { hiddenBias: init(child(s, 'hiddenBias'), [gates * hidden], fans) } : {}),
  }
}

/**
 * Gate $k$ of a pre-activation whose gates lie side by side on the last axis.
 *
 * @param z The pre-activation, `[..., G * H]`.
 * @param k The gate's index, from 0.
 * @param hidden The hidden size $H$, the width of each gate.
 * @returns Entries $kH$ to $(k + 1)H - 1$ of the last axis, `[..., H]`.
 */
function gate(z: Value, k: number, hidden: number): Value {
  const rank = shapeOfValue(z).length
  const specs = Array.from({ length: rank }, (_, a) => (a === rank - 1 ? [k * hidden, (k + 1) * hidden] : null))
  return slice(z, ...(specs as [number, number][]))
}

/**
 * A zero hidden (or cell) state.
 *
 * @param hidden The hidden size $H$.
 * @param batch The leading shape; empty for a single vector.
 * @returns Zeros of shape `[...batch, H]`.
 */
const zeroState = (hidden: number, batch: readonly number[] = []): Tensor => zeros([...batch, hidden])

/** Options of the cells: `init`, the initialiser of every parameter (default `lecunUniform`, PyTorch's scale). */
export type CellOptions = { init?: Initialiser }

/**
 * The Elman RNN cell $\hvec' = \phi(\xvec\Wmat_x + \hvec\Wmat_h + \bvec)$, with $\phi = \tanh$ by default.
 * Differentiable; the new $\hvec$ goes through the context's tap.
 *
 * @param inFeatures The input width.
 * @param hidden The hidden size $H$.
 * @param options The initialiser and the activation.
 * @param options.init The initialiser of every parameter.
 * @param options.activation The activation $\phi$, by name or as a function.
 * @returns The cell, with one gate.
 *
 * @example From the zero state, one step is $\tanh(\xvec\Wmat_x + \bvec)$
 * const cell = RnnCell(2, 3)
 * const p = cell.init(stream(0))
 * const x = tensor([1, -1])
 * print('step:', cell.step(p, x, cell.initialState()).h)
 * print('by hand:', tanh(add(matmul(x, p.inputWeight), p.bias)))
 */
export function RnnCell(
  inFeatures: number,
  hidden: number,
  { init = lecunUniform(), activation = 'tanh' }: CellOptions & { activation?: Activation } = {},
): Cell {
  const act = activationFn(activation)
  return {
    kind: 'RnnCell',
    label: `RnnCell(${inFeatures} → ${hidden})`,
    hidden,
    init: (s) => cellParams(s, init, inFeatures, hidden, 1, false),
    initialState: (batch) => ({ h: zeroState(hidden, batch) }),
    step: (p, x, { h }, ctx) => ({
      h: tap(ctx, act(add(linear(x, p.inputWeight, p.bias), linear(h, p.hiddenWeight)))),
    }),
  }
}

/**
 * The GRU cell (Cho et al., 2014), in PyTorch's form with gates $(\rvec, \zvec, \nvec)$:
 * $\rvec = \sigma(\xvec\Wmat_r + \bvec_r + \hvec\Umat_r + \cvec_r)$,
 * $\zvec = \sigma(\xvec\Wmat_z + \bvec_z + \hvec\Umat_z + \cvec_z)$,
 * $\nvec = \tanh(\xvec\Wmat_n + \bvec_n + \rvec \odot (\hvec\Umat_n + \cvec_n))$,
 * $\hvec' = (1 - \zvec) \odot \nvec + \zvec \odot \hvec$. It keeps both biases because $\cvec_n$ sits inside the reset
 * gate. Differentiable; the new $\hvec$ goes through the context's tap.
 *
 * @param inFeatures The input width.
 * @param hidden The hidden size $H$.
 * @param options The initialiser.
 * @param options.init The initialiser of every parameter.
 * @returns The cell, with three gates and both biases.
 *
 * @example A batch of four inputs from the zero state
 * const cell = GruCell(2, 3)
 * const p = cell.init(stream(0))
 * const h = cell.step(p, tensor([[1, 0], [0, 1], [-1, 0], [0, -1]]), cell.initialState([4])).h
 * print(cell.label, ' weights:', p.inputWeight.shape, p.hiddenWeight.shape, ' state:', h.shape)
 * print('h =', h)
 */
export function GruCell(inFeatures: number, hidden: number, { init = lecunUniform() }: CellOptions = {}): Cell {
  return {
    kind: 'GruCell',
    label: `GruCell(${inFeatures} → ${hidden})`,
    hidden,
    init: (s) => cellParams(s, init, inFeatures, hidden, 3, true),
    initialState: (batch) => ({ h: zeroState(hidden, batch) }),
    step: (p, x, { h }, ctx) => {
      const gx = linear(x, p.inputWeight, p.bias)
      const gh = linear(h, p.hiddenWeight, p.hiddenBias)
      const r = sigmoid(add(gate(gx, 0, hidden), gate(gh, 0, hidden)))
      const z = sigmoid(add(gate(gx, 1, hidden), gate(gh, 1, hidden)))
      const n = tanh(add(gate(gx, 2, hidden), mul(r, gate(gh, 2, hidden))))
      return { h: tap(ctx, add(mul(sub(1, z), n), mul(z, h))) }
    },
  }
}

/**
 * The LSTM cell, in PyTorch's form with gates $(\ivec, \fvec, \gvec, \ovec)$, the slices of
 * $\xvec\Wmat + \hvec\Umat + \bvec$ through $\sigma$, $\sigma$, $\tanh$ and $\sigma$:
 * $\cvec' = \fvec \odot \cvec + \ivec \odot \gvec$, $\hvec' = \ovec \odot \tanh(\cvec')$. `forgetBias` is added to the
 * forget gate's bias at initialisation (0 as in PyTorch; 1 is the common choice of Jozefowicz et al., 2015). A state
 * without `c` starts the cell at zero. Differentiable; the new $\hvec$ goes through the context's tap, and $\cvec$
 * under `cell`.
 *
 * @param inFeatures The input width.
 * @param hidden The hidden size $H$.
 * @param options The initialiser and the forget bias.
 * @param options.init The initialiser of every parameter.
 * @param options.forgetBias Added to entries $H$ to $2H - 1$ of the drawn bias, the forget gate's.
 * @returns The cell, with four gates; its state has `h` and `c`.
 *
 * @example A forget bias of 1 shifts the second gate's block of the bias
 * const p0 = LstmCell(1, 2).init(stream(0))
 * const p1 = LstmCell(1, 2, { forgetBias: 1 }).init(stream(0))
 * print('bias:', p0.bias)
 * print('with forgetBias 1:', p1.bias)
 * const cell = LstmCell(1, 2)
 * print('one step from zero:', cell.step(p0, tensor([1]), cell.initialState()))
 */
export function LstmCell(
  inFeatures: number,
  hidden: number,
  { init = lecunUniform(), forgetBias = 0 }: CellOptions & { forgetBias?: number } = {},
): Cell {
  return {
    kind: 'LstmCell',
    label: `LstmCell(${inFeatures} → ${hidden})`,
    hidden,
    init: (s) => {
      const p = cellParams(s, init, inFeatures, hidden, 4, false)
      if (forgetBias === 0) return p
      const b = Float64Array.from(p.bias.data)
      for (let j = hidden; j < 2 * hidden; j++) b[j] += forgetBias
      return { ...p, bias: fromData(b, [4 * hidden]) }
    },
    initialState: (batch) => ({ h: zeroState(hidden, batch), c: zeroState(hidden, batch) }),
    step: (p, x, { h, c }, ctx) => {
      const z = add(linear(x, p.inputWeight, p.bias), linear(h, p.hiddenWeight))
      const i = sigmoid(gate(z, 0, hidden))
      const f = sigmoid(gate(z, 1, hidden))
      const g = tanh(gate(z, 2, hidden))
      const o = sigmoid(gate(z, 3, hidden))
      const cNext = add(mul(f, c ?? zeroState(hidden)), mul(i, g))
      return { h: tap(ctx, mul(o, tanh(cNext))), c: tap(ctx, cNext, 'cell') }
    },
  }
}

/** The result of `unrollRecurrent`. */
export type Unrolled = {
  /** The hidden states $\hvec_1, \dots, \hvec_T$ stacked, shape `[T, ..., H]`. */
  outputs: Value
  /** Every state, the initial one first (T + 1 entries). */
  states: RecurrentState[]
  /** The last state, after step $T$ (the initial one when $T = 0$). */
  final: RecurrentState
}

/**
 * Run a cell over a sequence `xs` of shape `[T, ..., in]` (time first) from `initial` (default zeros), collecting the
 * hidden states. Differentiating through it is backpropagation through time. A loop of $T$ steps, each with its own
 * path (`<path>.t`) in the context.
 *
 * @param cell The cell to step.
 * @param params The cell's parameters, shared by every step.
 * @param xs The inputs, `[T, ..., in]`: step $t$ reads `xs[t]`, and the axes between time and features are the batch.
 * @param initial The state to start from; left out, the cell's zero state for the batch shape of `xs`.
 * @param ctx The context passed to every step, with the step index appended to its path.
 * @returns The stacked hidden states, every state and the final one.
 *
 * @example Five steps of an RNN over a constant input
 * const cell = RnnCell(1, 2)
 * const p = cell.init(stream(0))
 * const { outputs, states, final } = unrollRecurrent(cell, p, ones([5, 1]))
 * print('outputs:', outputs)
 * print('states:', states.length, ' final h:', final.h)
 */
export function unrollRecurrent(
  cell: Cell,
  params: CellParams,
  xs: Value,
  initial?: RecurrentState,
  ctx?: Context,
): Unrolled {
  const shape = shapeOfValue(xs)
  const T = shape[0]
  let state = initial ?? cell.initialState(shape.slice(1, -1))
  const states = [state]
  for (let t = 0; t < T; t++) {
    state = cell.step(
      params,
      slice(xs, t),
      state,
      ctx && { ...ctx, path: `${ctx.path ?? ''}${ctx.path ? '.' : ''}${t}` },
    )
    states.push(state)
  }
  return { outputs: stack(states.slice(1).map((s) => s.h)), states, final: state }
}

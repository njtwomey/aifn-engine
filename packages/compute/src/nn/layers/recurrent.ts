/**
 * Recurrent cells and their unrolling: the Elman RNN, the GRU (Cho et al., 2014) and the LSTM (Hochreiter &
 * Schmidhuber, 1997, with the forget gate of Gers, Schmidhuber & Cummins, 2000). A cell maps an input and a state to
 * the next state; `unrollRecurrent` scans it over a sequence. Gate order and equations follow PyTorch's `RNNCell`, `GRUCell`
 * and `LSTMCell`, with weights stored transposed ([in, gates·H]) and, for the RNN and LSTM, the two biases summed.
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

/** A recurrent state: the hidden vector h and, for the LSTM, the cell c. Shapes [..., H]. */
export type RecurrentState = { h: Value; c?: Value }

/** Parameters of a cell: input weights [in, G·H], recurrent weights [H, G·H] and biases [G·H] (G gates). */
export type CellParams = { inputWeight: Tensor; hiddenWeight: Tensor; bias: Tensor; hiddenBias?: Tensor }

/** A recurrent cell. */
export interface Cell {
  readonly kind: string
  readonly label: string
  readonly hidden: number
  init(s: Stream): CellParams
  /** The zero state for a batch of leading shape `batch` (default none: vectors of length H). */
  initialState(batch?: readonly number[]): RecurrentState
  /** The next state from input x [..., in] and state; pure. */
  step(params: CellParams, x: Value, state: RecurrentState, ctx?: Context): RecurrentState
}

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

/** Gate k of a [..., G·H] pre-activation. */
function gate(z: Value, k: number, hidden: number): Value {
  const rank = shapeOfValue(z).length
  const specs = Array.from({ length: rank }, (_, a) => (a === rank - 1 ? [k * hidden, (k + 1) * hidden] : null))
  return slice(z, ...(specs as [number, number][]))
}

const zeroState = (hidden: number, batch: readonly number[] = []): Tensor => zeros([...batch, hidden])

/** Options of the cells. */
export type CellOptions = { init?: Initialiser }

/** The Elman RNN cell h′ = act(x·W_x + h·W_h + b), act = tanh by default. */
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
 * The GRU cell (Cho et al., 2014), in PyTorch's form with gates (r, z, n):
 * r = σ(x·W_r + b_r + h·U_r + c_r), z = σ(x·W_z + b_z + h·U_z + c_z), n = tanh(x·W_n + b_n + r ⊙ (h·U_n + c_n)),
 * h′ = (1 − z) ⊙ n + z ⊙ h. It keeps both biases because c_n sits inside the reset gate.
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
 * The LSTM cell, in PyTorch's form with gates (i, f, g, o): i = σ(·), f = σ(·), g = tanh(·), o = σ(·) of x·W + h·U + b,
 * c′ = f ⊙ c + i ⊙ g, h′ = o ⊙ tanh(c′). `forgetBias` is added to the forget gate's bias at initialisation (0 as in
 * PyTorch; 1 is the common choice of Jozefowicz et al., 2015).
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
  /** The hidden states h₁ … h_T stacked, shape [T, ..., H]. */
  outputs: Value
  /** Every state, the initial one first (T + 1 entries). */
  states: RecurrentState[]
  final: RecurrentState
}

/**
 * Run a cell over a sequence xs of shape [T, ..., in] (time first) from `initial` (default zeros), collecting the
 * hidden states. Differentiating through it is backpropagation through time.
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

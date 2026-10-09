/**
 * Backpropagation through time with the gradient recorded at every step (Werbos, 1990; Bengio, Simard and Frasconi,
 * 1994; Pascanu, Mikolov and Bengio, 2013): the norm $\lVert \partial L / \partial \hvec_t \rVert$ of the loss's
 * gradient with respect to each hidden state shows how far back a training signal reaches. When the recurrent
 * Jacobians $\partial \hvec_{t+1} / \partial \hvec_t$ have norms below one the norms shrink geometrically into the
 * past (vanishing gradients); above one they grow (exploding gradients).
 *
 * The gradients come from zero probes $\pvec_t$ added to each hidden state:
 * $\partial L / \partial \pvec_t = \partial L / \partial \hvec_t$, the total derivative through every later step,
 * from one reverse pass. Truncated BPTT (Williams and Peng, 1990) stops the gradient flowing into the state every $k$
 * steps.
 */

import { stopGradient, valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import type { Size } from 'aifn-compute/foundation/contracts'
import {
  add,
  norm,
  shapeOfValue,
  slice,
  stack,
  unwrap,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Cell, CellParams, RecurrentState } from 'aifn-compute/nn/layers'

/** Options of `gradientsThroughTime`. */
export type ThroughTimeOptions = {
  /** The state before the first step (default the cell's zero state for the batch shape of `xs`). */
  initial?: RecurrentState
  /**
   * Truncate backpropagation every $k$ steps: the state passed from step $jk$ to step $jk + 1$ ($j = 1, 2, \dots$) is
   * held constant, so no gradient crosses it. Default none.
   */
  truncate?: Size
}

/** What `gradientsThroughTime` returns. */
export type ThroughTime = {
  /** The loss $L$. */
  loss: number
  /** The hidden states $\hvec_1, \dots, \hvec_T$, shape $[T, \dots, H]$. */
  outputs: Tensor
  /** $\partial L / \partial \hvec_t$ for each step, shape $[T, \dots, H]$. */
  hiddenGrads: Tensor
  /** $\lVert \partial L / \partial \hvec_t \rVert_2$ per step $t = 1, \dots, T$ (over the whole batch at that step). */
  norms: number[]
  /** $\partial L / \partial \thetavec$, with the structure of the cell's parameters. */
  paramGrads: CellParams
}

/**
 * Backpropagation through time: unroll `cell` over `xs`, evaluate `lossOf` on the stacked hidden states, and return
 * the loss, $\partial L / \partial \hvec_t$ and its norm at every step, and the parameter gradient. The sequence is
 * run twice (once under `valueAndGrad`, once for the returned `outputs`). Only the hidden state $\hvec$ is probed, so
 * for an LSTM the gradients are those of $\hvec_t$, not of the cell $\cvec_t$.
 *
 * @param cell The recurrent cell (as `RnnCell`, `GruCell` or `LstmCell` of `aifn-compute/nn/layers` make), stepped
 *   once per time step.
 * @param params The cell's parameters, the point at which the gradients are taken.
 * @param xs The inputs, time first: shape $[T, \dots, d]$, a batch of any leading shape between time and the $d$
 *   input features.
 * @param lossOf The loss as a function of the stacked hidden states $[T, \dots, H]$; it returns a scalar, and is
 *   differentiated in them.
 * @param options The initial state and the truncation length.
 * @returns The loss, the hidden states, $\partial L / \partial \hvec_t$ with its norm per step, and
 *   $\partial L / \partial \thetavec$.
 *
 * @example An RNN over three steps: the gradient reaching back from the last state
 * // An Elman cell h' = tanh(x Wx + h Wh + b), one input and two hidden units, written out by hand.
 * const cell = {
 *   kind: 'rnn',
 *   label: 'rnn',
 *   hidden: 2,
 *   init: (s) => ({ inputWeight: normals(s, [1, 2]), hiddenWeight: normals(s, [2, 2]), bias: zeros([2]) }),
 *   initialState: (batch = []) => ({ h: zeros([...batch, 2]) }),
 *   step: (p, x, { h }) => ({ h: tanh(add(add(matmul(x, p.inputWeight), matmul(h, p.hiddenWeight)), p.bias)) }),
 * }
 * const params = cell.init(stream(0))
 * const xs = tensor([[1], [0], [0]]) // three steps of one input
 * // The loss reads only the last hidden state: earlier steps get only what flows back through the cell.
 * const r = gradientsThroughTime(cell, params, xs, (hs) => sum(slice(hs, 2)))
 * print('loss =', r.loss)
 * print('hidden states =', r.outputs)
 * print('dL/dh_t =', r.hiddenGrads)
 * print('norms =', r.norms)
 * print('dL/dWx =', r.paramGrads.inputWeight)
 *
 * @example Truncation every two steps cuts the gradient off from the first two states
 * const cell = {
 *   kind: 'rnn',
 *   label: 'rnn',
 *   hidden: 2,
 *   init: (s) => ({ inputWeight: normals(s, [1, 2]), hiddenWeight: normals(s, [2, 2]), bias: zeros([2]) }),
 *   initialState: (batch = []) => ({ h: zeros([...batch, 2]) }),
 *   step: (p, x, { h }) => ({ h: tanh(add(add(matmul(x, p.inputWeight), matmul(h, p.hiddenWeight)), p.bias)) }),
 * }
 * const xs = tensor([[1], [0], [0]])
 * const r = gradientsThroughTime(cell, cell.init(stream(0)), xs, (hs) => sum(slice(hs, 2)), { truncate: 2 })
 * print('norms =', r.norms)
 */
export function gradientsThroughTime(
  cell: Cell,
  params: CellParams,
  xs: Value,
  lossOf: (outputs: Value) => Value,
  options: ThroughTimeOptions = {},
): ThroughTime {
  const shape = shapeOfValue(xs)
  const T = shape[0]
  const batch = shape.slice(1, -1)
  const unroll = (p: CellParams, probes: Value[]) => {
    let state = options.initial ?? cell.initialState(batch)
    const hs: Value[] = []
    for (let t = 0; t < T; t++) {
      if (options.truncate && t > 0 && t % options.truncate === 0)
        state = { h: stopGradient(state.h), ...(state.c === undefined ? {} : { c: stopGradient(state.c) }) }
      const next = cell.step(p, slice(xs, t), state)
      state = { ...next, h: add(next.h, probes[t]) }
      hs.push(state.h)
    }
    return stack(hs, 0)
  }
  const probes = Array.from({ length: T }, () => zeros([...batch, cell.hidden]) as Value)
  const f = (p: CellParams, pr: Value[]) => lossOf(unroll(p, pr))
  const { value, grad } = valueAndGrad(f, { argnums: [0, 1] })(params, probes)
  const [paramGrads, hiddenGradList] = grad as [CellParams, Value[]]
  const raw = unwrap(value)
  const hiddenGrads = unwrap(stack(hiddenGradList, 0)) as Tensor
  return {
    loss: typeof raw === 'number' ? raw : (raw as Tensor).data[0],
    outputs: unwrap(unroll(params, probes)) as Tensor,
    hiddenGrads,
    norms: hiddenGradList.map((g) => norm(unwrap(g) as Tensor)),
    paramGrads: paramGrads as CellParams & Params,
  }
}

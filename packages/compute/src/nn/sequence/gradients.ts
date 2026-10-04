/**
 * Backpropagation through time with the gradient recorded at every step (Werbos, 1990; Bengio, Simard and Frasconi,
 * 1994; Pascanu, Mikolov and Bengio, 2013): the norm ‖∂L/∂h_t‖ of the loss's gradient with respect to each hidden state
 * shows how far back a training signal reaches. When the recurrent Jacobians ∂h_{t+1}/∂h_t have norms below one the
 * norms shrink geometrically into the past (vanishing gradients); above one they grow (exploding gradients).
 *
 * The gradients come from zero probes added to each hidden state: ∂L/∂probe_t = ∂L/∂h_t, the total derivative through
 * every later step, from one reverse pass. Truncated BPTT (Williams and Peng, 1990) stops the gradient flowing into
 * the state every k steps.
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
  /** The state before the first step (default the cell's zero state). */
  initial?: RecurrentState
  /** Truncate backpropagation every k steps: the state passed from step t·k to t·k + 1 is a constant. Default none. */
  truncate?: Size
}

/** What `gradientsThroughTime` returns. */
export type ThroughTime = {
  loss: number
  /** The hidden states h₁ … h_T, [T, ..., H]. */
  outputs: Tensor
  /** ∂L/∂h_t for each step, [T, ..., H]. */
  hiddenGrads: Tensor
  /** ‖∂L/∂h_t‖₂ per step t = 1 … T. */
  norms: number[]
  /** ∂L/∂θ, with the structure of the cell's parameters. */
  paramGrads: CellParams
}

/**
 * Unroll `cell` over xs [T, ..., in], evaluate `lossOf(outputs)` on the stacked hidden states [T, ..., H], and return
 * the loss, ∂L/∂h_t and its norm at every step, and the parameter gradient (backpropagation through time).
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

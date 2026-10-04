/**
 * aifn-compute/nn/training: `fullBatchTraining` and `treeObjective`, a parameter tree trained by a vector method through
 * `ravel`: the round trip θ ↔ tree, the raveled gradient against per-leaf gradients (and finite differences), and
 * full-batch L-BFGS against gradient descent at the same number of gradient evaluations.
 */
import { describe, expect, it } from 'vitest'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { ravel, treeLeaves } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { fullBatchTraining, treeObjective } from 'aifn-compute/nn/training'
import { checkProtocol } from '../../protocol'

const data = {
  x: tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0.5, 0.2],
    [0.2, 0.8],
  ]),
  y: tensor([[0], [1], [1], [0], [1], [1]]),
}
const model = Mlp([2, 6, 1], { activation: 'tanh', init: xavierUniform() })
type P = ReturnType<typeof model.init>
const loss = (p: P, b: typeof data) => binaryCrossEntropyWithLogits(model.apply(p, b.x), b.y)
const leafValues = (tree: unknown) =>
  treeLeaves(tree).map(({ path, value }) => [path, typeof value === 'number' ? [value] : toFlat(value as Tensor)])

describe('ravel on network parameters', () => {
  it('round-trips an MLP parameter tree, keeping its structure and empty activation entries', () => {
    const params = model.init(stream('round-trip'))
    const { vector, unravel } = ravel(params)
    expect(vector.length).toBe(2 * 6 + 6 + 6 + 1)
    const back = unravel(vector)
    expect(leafValues(back)).toEqual(leafValues(params))
    expect(back[1]).toEqual({})
    expect(ravel(back).vector).toEqual(vector)
  })

  it('gives the gradient through ravel equal to the per-leaf gradients, and to finite differences', () => {
    const params = model.init(stream('gradient'))
    const { vector, unravel } = ravel(params)
    const f = treeObjective(loss, data, unravel)
    const out = f(fromData(vector))
    const lossAndGrad = valueAndGrad((p: Tensor[]) => loss(p as unknown as P, data))
    expect(out.grad).toEqual(ravel(lossAndGrad(params as unknown as Tensor[]).grad).vector)
    const g = out.grad as Float64Array
    const h = 1e-6
    for (const i of [0, 5, 13, vector.length - 1]) {
      const plus = Float64Array.from(vector)
      const minus = Float64Array.from(vector)
      plus[i] += h
      minus[i] -= h
      const fd = (f(fromData(plus)).value - f(fromData(minus)).value) / (2 * h)
      expect(g[i]).toBeCloseTo(fd, 7)
    }
  })
})

describe('fullBatchTraining', () => {
  it('follows the protocol, keeping params equal to unravel(x)', () => {
    const alg = fullBatchTraining({ loss, data, options: { memory: 5 } })
    checkProtocol(alg, { params: model.init(stream('protocol')) }, { steps: 6 })
    const end = run(alg, { params: model.init(stream('protocol')) }, 6)
    expect(ravel(end.params).vector).toEqual(Float64Array.from(toFlat(end.x)))
  })

  it('with L-BFGS reaches a lower loss than gradient descent given the same gradient evaluations', () => {
    const start = { params: model.init(stream('race')) }
    const lbfgs = run(fullBatchTraining({ loss, data, method: 'lbfgs', options: { tolerance: 0 } }), start, 40)
    const budget = lbfgs.evaluations
    // Gradient descent evaluates once at init and once per step: budget − 1 steps use the same evaluations.
    const best = Math.min(
      ...[0.3, 1, 3].map(
        (stepSize) =>
          run(fullBatchTraining({ loss, data, method: 'gradient-descent', options: { stepSize } }), start, budget - 1)
            .value,
      ),
    )
    expect(lbfgs.value).toBeLessThan(best / 10)
  })
})

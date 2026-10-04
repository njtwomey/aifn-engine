/**
 * aifn-compute/nn/training: `methodTraining`, one state shape over first-order and full-batch L-BFGS training: each method
 * matches the algorithm it wraps step for step, and L-BFGS holds its state once it stops.
 */
import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { tensor } from 'aifn-compute/foundation/tensor'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { fullBatchTraining, methodTraining, trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'

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
const params = model.init(stream('method'))
const root = stream('steps')

function steps<S>(alg: { init: (i: { params: P }, s: typeof root) => S; step: (s: S, c: never) => S }, n: number) {
  let s = alg.init({ params }, child(root, 'init'))
  for (let t = 0; t < n; t++) s = alg.step(s, { t, stream: child(root, 'step', t) } as never)
  return s
}

describe('methodTraining', () => {
  it('first-order: the same parameters and loss as trainingLoop with the same rule', () => {
    const optimizer = adamRule({ stepSize: 0.05 })
    const a = steps(methodTraining(loss, data, { method: 'first-order', optimizer }), 20)
    const b = steps(trainingLoop({ loss, data, optimizer }), 20)
    expect(a.loss).toBe(b.loss)
    expect(a.params).toEqual(b.params)
    expect(a.t).toBe(20)
  })

  it('L-BFGS: the same iterates as fullBatchTraining, the loss on the whole set', () => {
    const a = steps(methodTraining(loss, data, { method: 'lbfgs', memory: 5 }), 15)
    const b = steps(fullBatchTraining({ loss, data, method: 'lbfgs', options: { memory: 5 } }), 15)
    expect(a.loss).toBe(b.value)
    expect(a.params).toEqual(b.params)
    expect(a.evaluations).toBe(b.evaluations)
    expect(a.loss).toBeLessThan(0.3)
  })

  it('L-BFGS: holds the state once it stops', () => {
    const alg = methodTraining(loss, data, { method: 'lbfgs', tolerance: 1e-2 })
    const s = steps(alg, 200)
    expect(s.stopped).toBe(true)
    const next = alg.step(s, { t: 200, stream: child(root, 'step', 200) })
    expect(next.params).toBe(s.params)
    expect(next.loss).toBe(s.loss)
  })
})

describe('methodTraining with plain-data Adam', () => {
  it('equals the first-order method with adamRule of the same options', () => {
    const a = steps(methodTraining(loss, data, { method: 'adam', stepSize: 0.05, batchSize: 3 }), 10)
    const b = steps(
      methodTraining(loss, data, { method: 'first-order', optimizer: adamRule({ stepSize: 0.05 }), batchSize: 3 }),
      10,
    )
    expect(a.params).toEqual(b.params)
  })
})

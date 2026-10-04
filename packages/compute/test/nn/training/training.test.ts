/**
 * aifn-compute/nn/training: `trainingLoop` on the Algorithm protocol with optim's pytree update rules (`adamRule`, `sgdRule`,
 * gradient clipping by `clipByGlobalNorm`), minibatch epochs as plain data, and `inspect`.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { sum, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { countParams, treeLeaves } from 'aifn-compute/foundation/pytree'
import { run, trace } from 'aifn-compute/foundation/trace'
import { binaryCrossEntropyWithLogits, softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { xavierUniform } from 'aifn-compute/nn/init'
import { BatchNorm, Dropout, Linear, linear, Mlp, Sequential } from 'aifn-compute/nn/layers'
import { MultiHeadAttention } from 'aifn-compute/nn/attention'
import { activations, inspect, trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, globalNorm, sgdRule } from 'aifn-compute/optim/first-order'
import { checkProtocol } from '../../protocol'

const flat = (v: unknown) => (typeof v === 'number' ? [v] : Array.from(toFlat(v as Tensor)))

const xor = {
  x: tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ]),
  y: tensor([[0], [1], [1], [0]]),
}
const model = Mlp([2, 8, 1], { activation: 'tanh', init: xavierUniform() })
type P = ReturnType<typeof model.init>
const lossOf = (p: P, b: typeof xor) => binaryCrossEntropyWithLogits(model.apply(p, b.x), b.y)

describe('trainingLoop', () => {
  it('reduces the loss on XOR with Adam and with SGD with momentum', () => {
    for (const optimizer of [adamRule({ stepSize: 0.05 }), sgdRule({ stepSize: 0.5, momentum: 0.9 })]) {
      const alg = trainingLoop({ loss: lossOf, data: xor, optimizer })
      const start = alg.init({ params: model.init(stream('xor')) }, stream(0))
      const end = run(alg, { params: model.init(stream('xor')) }, 400)
      expect(end.loss, alg.name).toBeLessThan(0.05)
      expect(end.loss).toBeLessThan(start.loss)
      expect(end.t).toBe(400)
      expect(end.diverged).toBe(false)
    }
  })

  it('one full-batch SGD step is θ − η∇L', () => {
    const params = model.init(stream('one'))
    const alg = trainingLoop({ loss: lossOf, data: xor, optimizer: sgdRule({ stepSize: 0.1 }) })
    const s0 = alg.init({ params }, stream(0))
    const s1 = run(alg, { params }, 1)
    const before = treeLeaves(params)
    const grads = treeLeaves(s0.grads)
    const after = treeLeaves(s1.params)
    before.forEach((leaf, k) => {
      const expected = flat(leaf.value).map((v, i) => v - 0.1 * flat(grads[k].value)[i])
      flat(after[k].value).forEach((v, i) => expect(v).toBeCloseTo(expected[i], 14))
    })
    expect(s0.gradNorm).toBeCloseTo(globalNorm(s0.grads), 14)
    expect(s0.paramNorm).toBeCloseTo(globalNorm(params), 14)
    expect(Object.keys(s0.gradNorms)).toContain('[0].weight')
    expect(s0.batch).toBeNull()
  })

  it('clipNorm bounds the update but reports the unclipped norm', () => {
    const params = model.init(stream('clip'))
    const sgd = sgdRule({ stepSize: 1 })
    const clipped = trainingLoop({ loss: lossOf, data: xor, optimizer: sgd, clipNorm: 1e-3 })
    const s0 = clipped.init({ params }, stream(0))
    const s1 = run(clipped, { params }, 1)
    const step = treeLeaves(s1.params).flatMap((leaf, k) =>
      flat(leaf.value).map((v, i) => v - flat(treeLeaves(params)[k].value)[i]),
    )
    expect(Math.hypot(...step)).toBeCloseTo(1e-3, 10)
    expect(s0.gradNorm).toBeGreaterThan(1e-3)
  })

  it('cuts shuffled epochs into minibatches, each example once per epoch', () => {
    const data = {
      x: tensor(Array.from({ length: 12 }, (_, i) => [Math.cos(i), Math.sin(i)])),
      y: tensor(Array.from({ length: 12 }, (_, i) => [i % 2])),
    }
    const alg = trainingLoop({
      loss: (p: P, b: typeof data) => binaryCrossEntropyWithLogits(model.apply(p, b.x), b.y),
      data,
      batchSize: 4,
    })
    const tr = trace(alg, { params: model.init(stream('epochs')) }, 8, { stream: stream(1), keep: 'all' })
    expect(tr.steps.map((s) => s.epoch)).toEqual([0, 0, 0, 1, 1, 1, 2, 2, 2])
    for (const e of [0, 1]) {
      const seen = tr.steps.filter((s) => s.epoch === e).flatMap((s) => flat(s.batch))
      expect(seen.slice().sort((a, b) => a - b)).toEqual(Array.from({ length: 12 }, (_, i) => i))
    }
    expect(flat(tr.steps[3].order)).not.toEqual(flat(tr.steps[0].order))
    expect(() =>
      trainingLoop({ loss: lossOf, data: { x: xor.x, y: tensor([1, 2]) } }).init(
        { params: model.init(stream('a')) },
        stream(0),
      ),
    ).toThrow(/rows/)
  })

  it('follows the Algorithm protocol, with minibatches and with dropout', () => {
    const data = {
      x: tensor(Array.from({ length: 12 }, (_, i) => [Math.cos(i), Math.sin(i)])),
      y: tensor(Array.from({ length: 12 }, (_, i) => [i % 2])),
    }
    const minibatch = trainingLoop({
      loss: (p: P, b: typeof data) => binaryCrossEntropyWithLogits(model.apply(p, b.x), b.y),
      data,
      batchSize: 4,
    })
    checkProtocol(minibatch, { params: model.init(stream('protocol')) }, { steps: 10, random: true })
    const net = Sequential(Linear(2, 6), Dropout(0.3), Linear(6, 1))
    type Q = ReturnType<typeof net.init>
    const dropped = trainingLoop({
      loss: (p: Q, b: typeof data, ctx) => binaryCrossEntropyWithLogits(net.apply(p, b.x, ctx), b.y),
      data,
      batchSize: 6,
      optimizer: sgdRule({ stepSize: 0.1, momentum: 0.5 }),
    })
    checkProtocol(dropped, { params: net.init(stream('dropout')) }, { steps: 6, random: true })
  })

  it('carries batch-norm running statistics: buffers after t + 1 loss evaluations on a full batch', () => {
    const data = { x: tensor([[0], [1], [3], [4]]), y: tensor([[0], [0], [1], [1]]) }
    const net = Sequential(Linear(1, 2), BatchNorm(2), Linear(2, 1))
    type Q = ReturnType<typeof net.init>
    const alg = trainingLoop({
      loss: (p: Q, b: typeof data, ctx) => binaryCrossEntropyWithLogits(net.apply(p, b.x, ctx), b.y),
      data,
      // A zero step keeps the parameters, so every evaluation sees the same batch statistics.
      optimizer: sgdRule({ stepSize: 0 }),
    })
    const params = net.init(stream('bn'))
    const end = run(alg, { params }, 4)
    const first = params[0] as { weight: Tensor; bias: Tensor }
    const h = flat(linear(data.x, first.weight, first.bias))
    const mean = [0, 1].map((c) => (h[c] + h[2 + c] + h[4 + c] + h[6 + c]) / 4)
    const variance = [0, 1].map((c) => [0, 1, 2, 3].reduce((a, i) => a + (h[2 * i + c] - mean[c]) ** 2, 0) / 3)
    const decay = 1 - 0.9 ** 5
    const running = end.buffers['1'] as { mean: Tensor; variance: Tensor }
    flat(running.mean).forEach((v, c) => expect(v).toBeCloseTo(decay * mean[c], 12))
    flat(running.variance).forEach((v, c) => expect(v).toBeCloseTo(0.9 ** 5 + decay * variance[c], 12))
    expect(alg.init({ params }, stream(0)).buffers).toHaveProperty('1')
    checkProtocol(alg, { params }, { steps: 4 })
  })

  it('a softmax classifier trains on three classes', () => {
    const data = {
      x: tensor([
        [0, 0],
        [1, 0],
        [0, 1],
        [0.1, 0.1],
        [0.9, 0.1],
        [0.1, 0.9],
      ]),
      y: tensor([0, 1, 2, 0, 1, 2]),
    }
    const net = Linear(2, 3)
    const alg = trainingLoop({
      loss: (p: ReturnType<typeof net.init>, b: typeof data) => softmaxCrossEntropy(net.apply(p, b.x), b.y),
      data,
      optimizer: adamRule({ stepSize: 0.1 }),
    })
    expect(run(alg, { params: net.init(stream('softmax')) }, 300).loss).toBeLessThan(0.2)
  })
})

describe('inspection', () => {
  it('inspect gives activation gradients consistent with parameter gradients', () => {
    const model = Mlp([2, 4, 1], { activation: 'tanh' })
    const params = model.init(stream('inspect'))
    const x = tensor([[0.5, -1]])
    const r = inspect(model, params, x, (y) => sum(y))
    expect(Object.keys(r.activations)).toEqual(['0', '1', '2'])
    // The output's own gradient is 1; the last layer's input gradient is its weight row.
    expect(flat(r.activationGrads['2'] as Tensor)).toEqual([1])
    expect(countParams(params)).toBe(2 * 4 + 4 + 4 + 1)
    const mha = MultiHeadAttention(4, { heads: 2 })
    const mp = mha.init(stream('m'))
    const att = inspect(
      mha,
      mp,
      tensor([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
      ]),
      (y) => sum(y),
    )
    expect(Object.keys(att.activationGrads)).toContain('weights')
  })

  it('activations records every tapped output', () => {
    const net = Mlp([2, 3, 1])
    const a = activations(net, net.init(stream('act')), tensor([[1, 2]]))
    expect(Object.keys(a.activations).length).toBeGreaterThan(0)
  })
})

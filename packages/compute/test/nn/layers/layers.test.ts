/**
 * aifn-compute/nn/layers against torch (float64 goldens, `fixtures/nn.json`): normalisation and recurrent cells with
 * gradients; layer compositions by finite differences; dropout.
 */
import { describe, expect, it } from 'vitest'
import { grad, gradCheck } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import {
  add,
  mul,
  square,
  sum,
  take,
  tensor,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Params } from 'aifn-compute/foundation/pytree'
import {
  ActivationLayer,
  BatchNorm,
  batchNorm,
  Conv2d,
  dropout,
  Embedding,
  Flatten,
  GruCell,
  layerNorm,
  Linear,
  LstmCell,
  RnnCell,
  rmsNorm,
  Sequential,
  unrollRecurrent,
} from 'aifn-compute/nn/layers'
import { fixture } from '../../fixtures'

type N = number[] | number[][] | number[][][] | number[][][][]
type Case = Record<string, N | number | number[]>
const F = fixture<Record<string, Case>>('nn')
const T = (x: unknown) => tensor(x as N)
const flat = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}
const flatN = (x: unknown): number[] => (Array.isArray(x) ? (x as unknown[]).flatMap(flatN) : [x as number])

function close(actual: Value | number[], expected: unknown, tol = 1e-9) {
  const a = Array.isArray(actual) ? actual : flat(actual)
  const e = flatN(expected)
  expect(a.length).toBe(e.length)
  let worst = 0
  a.forEach((v, i) => (worst = Math.max(worst, Math.abs(v - e[i]) / Math.max(1, Math.abs(e[i])))))
  expect(worst).toBeLessThan(tol)
}

/** Σ w·y as a function to differentiate. */
const weighted = (w: unknown) => (y: Value) => sum(mul(y, T(w)))

describe('normalisation', () => {
  it('layer norm matches torch', () => {
    const c = F.layernorm
    close(layerNorm(T(c.x), T(c.gamma), T(c.beta)), c.y)
    const [gx, gg, gb] = grad((x: Value, g: Value, b: Value) => weighted(c.w)(layerNorm(x, g, b)), {
      argnums: [0, 1, 2],
    })(T(c.x), T(c.gamma), T(c.beta))
    close(gx, c.gx, 1e-8)
    close(gg, c.ggamma)
    close(gb, c.gbeta)
  })

  it('RMS norm matches torch', () => {
    const c = F.rmsnorm
    close(rmsNorm(T(c.x), T(c.gamma)), c.y)
    const [gx, gg] = grad((x: Value, g: Value) => weighted(c.w)(rmsNorm(x, g)), { argnums: [0, 1] })(T(c.x), T(c.gamma))
    close(gx, c.gx, 1e-8)
    close(gg, c.ggamma)
  })

  it('batch norm matches torch in training mode', () => {
    const c = F.batchnorm
    close(batchNorm(T(c.x), T(c.gamma), T(c.beta)), c.y)
    const [gx, gg] = grad((x: Value, g: Value) => weighted(c.w)(batchNorm(x, g, T(c.beta))), { argnums: [0, 1] })(
      T(c.x),
      T(c.gamma),
    )
    close(gx, c.gx, 1e-8)
    close(gg, c.ggamma)
  })

  it('BatchNorm keeps running statistics in training and normalises with them in evaluation, as torch', () => {
    const c = F.batchnorm_running as unknown as Record<string, unknown> & { batches: unknown[]; trainY: unknown[] }
    const layer = BatchNorm(3)
    const p = { gamma: T(c.gamma), beta: T(c.beta) }
    // At path '1' inside a Sequential, as a model would hold it; the first layer is unused here.
    let buffers: Readonly<Record<string, Params>> = {}
    c.batches.forEach((xb, k) => {
      const bufferUpdates: Record<string, Params> = {}
      close(layer.apply(p, T(xb), { train: true, path: '1', buffers, bufferUpdates }), c.trainY[k])
      expect(Object.keys(bufferUpdates)).toEqual(['1'])
      buffers = { ...buffers, ...bufferUpdates }
    })
    const running = buffers['1'] as { mean: Tensor; variance: Tensor }
    close(running.mean, c.runningMean, 1e-12)
    close(running.variance, c.runningVariance, 1e-12)
    close(layer.apply(p, T(c.x), { path: '1', buffers }), c.y)
    const [gx, gg] = grad(
      (x: Value, g: Value) =>
        weighted(c.w)(layer.apply({ gamma: g as Tensor, beta: p.beta }, x, { path: '1', buffers })),
      { argnums: [0, 1] },
    )(T(c.x), T(c.gamma))
    close(gx, c.gx, 1e-8)
    close(gg, c.ggamma)
    // Before any training, evaluation uses mean 0 and variance 1; without tracking, the batch's statistics.
    close(
      layer.apply(p, T(c.x)),
      flat(batchNorm(T(c.x), p.gamma, p.beta, { mean: T([0, 0, 0]), variance: T([1, 1, 1]) })),
    )
    close(BatchNorm(3, { trackRunningStats: false }).apply(p, T(c.x)), flat(batchNorm(T(c.x), p.gamma, p.beta)))
    // Containers give the layer its path, and grad of a training-mode loss records concrete tensors.
    const net = Sequential(Linear(1, 3), BatchNorm(3))
    const params = net.init(stream('bn'))
    const bufferUpdates: Record<string, Params> = {}
    const xs = T([[1], [2], [4]])
    const loss = (q: unknown) => sum(square(net.apply(q as Params[], xs, { train: true, bufferUpdates })))
    grad(loss)(params)
    const written = bufferUpdates['1'] as { mean: Tensor; variance: Tensor }
    expect(unwrap(written.mean)).toBe(written.mean)
  })

  it('RMS norm divides by the root mean square', () => {
    const y = flat(rmsNorm(tensor([3, 4]), tensor([1, 2]), 0))
    const rms = Math.sqrt((9 + 16) / 2)
    close(y, [3 / rms, (2 * 4) / rms], 1e-14)
  })
})

describe('recurrent cells', () => {
  it('LSTM, GRU and RNN cells match torch', () => {
    const l = F.lstm
    const lp = { inputWeight: T(l.inputWeight), hiddenWeight: T(l.hiddenWeight), bias: T(l.bias) }
    const lstm = LstmCell(3, 4)
    const next = lstm.step(lp, T(l.x), { h: T(l.h), c: T(l.c) })
    close(next.h, l.h1)
    close(next.c!, l.c1)
    const [gx, gh, gc] = grad(
      (x: Value, h: Value, c: Value) => {
        const s = lstm.step(lp, x, { h, c })
        return weighted(l.w)(add(s.h, mul(0.5, s.c!)))
      },
      { argnums: [0, 1, 2] },
    )(T(l.x), T(l.h), T(l.c))
    close(gx, l.gx)
    close(gh, l.gh)
    close(gc, l.gc)

    const g = F.gru
    const gp = {
      inputWeight: T(g.inputWeight),
      hiddenWeight: T(g.hiddenWeight),
      bias: T(g.bias),
      hiddenBias: T(g.hiddenBias),
    }
    const gru = GruCell(3, 4)
    close(gru.step(gp, T(g.x), { h: T(g.h) }).h, g.h1)
    close(grad((h: Value) => weighted(g.w)(gru.step(gp, T(g.x), { h }).h))(T(g.h)), g.gh)

    const r = F.rnn
    const rp = { inputWeight: T(r.inputWeight), hiddenWeight: T(r.hiddenWeight), bias: T(r.bias) }
    close(RnnCell(3, 4).step(rp, T(r.x), { h: T(r.h) }).h, r.h1)
  })

  it('unroll stacks hidden states over time', () => {
    const cell = GruCell(2, 3)
    const p = cell.init(stream('gru'))
    const xs = tensor([[[1, 0]], [[0, 1]], [[1, 1]]])
    const u = unrollRecurrent(cell, p, xs)
    expect((unwrap(u.outputs) as Tensor).shape).toEqual([3, 1, 3])
    expect(u.states).toHaveLength(4)
  })
})

describe('layers', () => {
  it('layer gradients agree with finite differences', () => {
    const s = stream('gradcheck')
    const model = Sequential(Conv2d(1, 2, 2), ActivationLayer('tanh'), Flatten(), Linear(8, 1))
    const params = model.init(s)
    const x = tensor([
      [
        [
          [0.1, 0.5, -0.3],
          [0.8, -0.2, 0.4],
          [0.0, 0.3, 0.9],
        ],
      ],
    ])
    expect(gradCheck((p: typeof params) => sum(model.apply(p, x)), params).ok).toBe(true)
    const emb = Embedding(5, 3)
    const ep = emb.init(s)
    expect(gradCheck((p: typeof ep) => sum(mul(emb.apply(p, tensor([1, 3, 1])), 1.5)), ep).ok).toBe(true)
  })

  it('take adds gradients over repeated rows', () => {
    const table = tensor([
      [1, 2],
      [3, 4],
    ])
    expect(toFlat(grad((t: Value) => sum(take(t, [1, 1, 0])))(table) as Tensor)).toEqual([1, 1, 2, 2])
  })

  it('dropout keeps the expectation and is reproducible', () => {
    const x = tensor(Array.from({ length: 4000 }, () => 1))
    const y = flat(dropout(stream('d'), x, 0.3))
    expect(y.reduce((a, b) => a + b, 0) / y.length).toBeCloseTo(1, 1)
    expect(flat(dropout(stream('d'), x, 0.3))).toEqual(y)
  })
})

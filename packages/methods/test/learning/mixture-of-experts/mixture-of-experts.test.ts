/**
 * aifn-methods/learning/mixture-of-experts: the mixture likelihood against a hand computation, the hierarchical gate
 * as a product of softmaxes, EM's monotone likelihood and recovery of piecewise-linear regimes, the auxiliary
 * load-balancing loss preventing the collapse of top-1 routing, and the streaming run's snapshots.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, toRows, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { piecewiseLinear, quadrantPlanes } from 'aifn-methods/data/synthetic'
import {
  mixtureOfExpertsRun,
  moeEm,
  moeForward,
  moeLoss,
  moeModel,
  moePredict,
  type MoeParams,
  type MoeSnapshot,
} from 'aifn-methods/learning/mixture-of-experts'

const lastOf = (g: Generator<MoeSnapshot>) => {
  let last: MoeSnapshot | undefined
  for (const s of g) last = s
  return last!
}
const scalar = (v: unknown) => {
  const r = unwrap(v as Tensor)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

describe('the model', () => {
  it('the mixture objective is −mean log Σᵢ gᵢ N(y; wᵢx + bᵢ, σᵢ²), by hand', () => {
    const model = moeModel({ inputs: 1, task: 'regression', experts: 2 })
    const params: MoeParams = {
      moe: {
        router: { weight: tensor([[1.5, -0.5]]), bias: tensor([0.2, 0]) },
        experts: [
          { weight: tensor([[2]]), bias: tensor([1]) },
          { weight: tensor([[-1]]), bias: tensor([0.5]) },
        ],
      },
      logSigma: tensor([Math.log(0.5), Math.log(2)]),
    }
    const xs = [-1, 0.3, 2]
    const ys = [0.1, 1.2, -0.4]
    const loss = scalar(moeLoss(model, params, tensor(xs.map((v) => [v])), tensor(ys)).data)
    const expected =
      -xs.reduce((acc, x, t) => {
        const z = [1.5 * x + 0.2, -0.5 * x]
        const m = Math.max(...z)
        const e = z.map((v) => Math.exp(v - m))
        const g = e.map((v) => v / (e[0] + e[1]))
        const mu = [2 * x + 1, -x + 0.5]
        const s = [0.5, 2]
        const p = g.reduce(
          (a, w, i) => a + (w * Math.exp(-0.5 * ((ys[t] - mu[i]) / s[i]) ** 2)) / (s[i] * Math.sqrt(2 * Math.PI)),
          0,
        )
        return acc + Math.log(p)
      }, 0) / 3
    expect(loss).toBeCloseTo(expected, 12)
  })

  it('the hierarchical gate is the top softmax times the softmax within each group', () => {
    const model = moeModel({ inputs: 2, task: 'regression', experts: 4, gate: 'hierarchical', groups: 2 })
    const params = model.init(stream('h'))
    const x = tensor([
      [0.3, -1.2],
      [1.0, 0.4],
    ])
    const g = toRows(unwrap(moeForward(model, params, x).routing.combine) as Tensor)
    const W = toRows(params.moe.router.weight)
    const b = toFlat(params.moe.router.bias!)
    const U = toRows(params.top!.weight)
    const c = toFlat(params.top!.bias!)
    const soft = (z: number[]) => {
      const e = z.map((v) => Math.exp(v))
      const s = e.reduce((a, v) => a + v, 0)
      return e.map((v) => v / s)
    }
    toRows(x).forEach((row, t) => {
      const z = [0, 1, 2, 3].map((i) => row[0] * W[0][i] + row[1] * W[1][i] + b[i])
      const top = soft([0, 1].map((j) => row[0] * U[0][j] + row[1] * U[1][j] + c[j]))
      const lower = [soft(z.slice(0, 2)), soft(z.slice(2, 4))]
      ;[0, 1, 2, 3].forEach((i) => expect(g[t][i]).toBeCloseTo(top[i >> 1] * lower[i >> 1][i % 2], 12))
      expect(g[t].reduce((a, v) => a + v, 0)).toBeCloseTo(1, 12)
    })
  })

  it('classification predictions are probabilities', () => {
    const model = moeModel({ inputs: 2, task: 'classification', experts: 3, gate: 'top-k', k: 2 })
    const p = moePredict(model, model.init(stream('c')), fromData(Float64Array.from([0, 1, -1, 2, 3, -2]), [3, 2]))
    p.prediction.forEach((v) => expect(v >= 0 && v <= 1).toBe(true))
    p.gate.forEach((row) => expect(row.filter((w) => w > 0).length).toBe(2))
  })
})

describe('EM', () => {
  const d = piecewiseLinear(stream('pw'), {})
  const data = { x: d.x as Tensor, y: d.y as Tensor }

  it('never decreases the likelihood (flat and hierarchical gates)', () => {
    for (const gate of ['softmax', 'hierarchical'] as const) {
      const model = moeModel({ inputs: 1, task: 'regression', experts: 4, gate })
      const alg = moeEm(model, data)
      let s = alg.init({ params: model.init(stream(`em-${gate}`)) }, stream('i'))
      for (let t = 0; t < 15; t++) {
        const next = alg.step(s, { t, stream: stream('s') })
        expect(next.loss).toBeLessThanOrEqual(s.loss + 1e-9)
        s = next
      }
    }
  })

  it('recovers the regimes of piecewise-linear data with as many experts as pieces', () => {
    const last = lastOf(
      mixtureOfExpertsRun({ data: { ...data, regime: d.regime }, task: 'regression', method: 'em', experts: 3 }),
    )
    expect(last.history.agreement.at(-1)).toBeGreaterThan(0.95)
    // The fitted noise is close to the true sd 0.15: the mean NLL is near that of N(0, 0.15²), ≈ −0.48.
    expect(last.history.loss.at(-1)).toBeLessThan(-0.35)
  })

  it('refuses MLP experts, sparse gates and the blend objective', () => {
    expect(() => moeEm(moeModel({ inputs: 1, task: 'regression', expert: 'mlp' }), data)).toThrow(/linear/)
    expect(() => moeEm(moeModel({ inputs: 1, task: 'regression', gate: 'switch' }), data)).toThrow(/dense/)
    expect(() => moeEm(moeModel({ inputs: 1, task: 'regression', objective: 'blend' }), data)).toThrow(/mixture/)
  })
})

describe('gradient training and routing balance', () => {
  const q = quadrantPlanes(stream('q'), {})
  const data = { x: q.x as Tensor, y: q.y as Tensor, regime: q.regime }
  const run = (balance: number) =>
    lastOf(
      mixtureOfExpertsRun({
        data,
        task: 'regression',
        method: 'adam',
        experts: 4,
        expert: 'mlp',
        gate: 'switch',
        objective: 'blend',
        balance,
        steps: 300,
        seed: 'b',
      }),
    )

  it('Switch routing without the balancing loss collapses onto few experts; with it, every expert works', () => {
    const free = run(0)
    const balanced = run(0.5)
    const entropy = (load: number[]) => -load.reduce((a, p) => a + (p > 0 ? p * Math.log(p) : 0), 0)
    expect(free.history.idle.at(-1)).toBeGreaterThanOrEqual(2)
    expect(balanced.history.idle.at(-1)).toBe(0)
    expect(entropy(balanced.history.load.at(-1)!)).toBeGreaterThan(entropy(free.history.load.at(-1)!) + 0.5)
    // The balancing loss is near its floor of 1 when balanced.
    expect(balanced.history.balance.at(-1)).toBeLessThan(1.1)
  })

  it('the run streams snapshots from step 0 to the last, with checkpoints for a player', () => {
    const snaps = [...mixtureOfExpertsRun({ data, task: 'regression', experts: 2, steps: 20, every: 5, seed: 'r' })]
    expect(snaps.map((s) => s.step)).toEqual([5, 10, 15, 20])
    const last = snaps.at(-1)!
    expect(last.done).toBe(true)
    expect(last.checkpoints.map((c) => c.step)).toEqual([0, 5, 10, 15, 20])
    expect(last.history.step).toEqual(Array.from({ length: 21 }, (_, i) => i))
    expect(last.history.load.every((l) => Math.abs(l.reduce((a, b) => a + b, 0) - 1) < 1e-9)).toBe(true)
  })
})

describe('mixtureOfExpertsRun by full-batch L-BFGS', () => {
  it('lowers the loss and records the method', () => {
    const d = piecewiseLinear(stream('lbfgs'), {})
    const snaps = [
      ...mixtureOfExpertsRun({
        data: { x: d.x as Tensor, y: d.y as Tensor },
        task: 'regression',
        experts: 2,
        method: 'lbfgs',
        steps: 40,
        every: 10,
        seed: 'l',
      }),
    ]
    const last = snaps.at(-1)!
    expect(last.method).toBe('lbfgs')
    expect(last.done).toBe(true)
    expect(last.history.loss.at(-1)!).toBeLessThan(last.history.loss[0])
  })
})

/**
 * aifn-compute/nn/experts: routing laws for every gate (how many experts each token keeps, the weights' sums and values,
 * capacity in GShard's priority order, expert choice's per-expert counts, temperature and noise), the auxiliary losses
 * against hand computation, routing statistics, and the layer's output and gradients.
 */
import { describe, expect, it } from 'vitest'
import { gradCheck } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import {
  add,
  mul,
  reshape,
  sum,
  tensor,
  toFlat,
  toRows,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { Linear, type LinearParams } from 'aifn-compute/nn/layers'
import {
  importanceLoss,
  loadBalancingLoss,
  MixtureOfExperts,
  route,
  routerZLoss,
  routingStatistics,
} from 'aifn-compute/nn/experts'

const LOGITS = [
  [2.0, 0.5, -1.0, 0.1],
  [-0.3, 1.2, 0.8, -2.0],
  [0.0, -0.5, 0.4, 1.5],
  [1.1, 1.0, -0.2, 0.3],
  [-1.0, 0.2, 2.2, 0.6],
]
const z = tensor(LOGITS)
const total = (...vs: Value[]) => vs.reduce((a, b) => add(a, b))
const rows = (v: Value) => toRows(unwrap(v) as Tensor)
const softmaxRow = (r: number[], tau = 1) => {
  const m = Math.max(...r)
  const e = r.map((v) => Math.exp((v - m) / tau))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}
const topK = (r: number[], k: number) =>
  r
    .map((v, i) => [v, i])
    .sort((a, b) => b[0] - a[0])
    .slice(0, k)
    .map((p) => p[1])

describe('route: gates', () => {
  it('softmax is dense: the weights are the softmax and every expert sees every token', () => {
    const r = route(z, { gate: 'softmax' })
    rows(r.combine).forEach((row, t) => row.forEach((w, i) => expect(w).toBeCloseTo(softmaxRow(LOGITS[t])[i], 12)))
    expect(toFlat(r.dispatch).every((v) => v === 1)).toBe(true)
  })

  it('top-k keeps the k largest logits per token and renormalises the softmax over them', () => {
    for (const k of [1, 2, 3]) {
      const r = route(z, { gate: 'top-k', k })
      rows(r.combine).forEach((row, t) => {
        const keep = topK(LOGITS[t], k)
        const p = softmaxRow(keep.map((i) => LOGITS[t][i]))
        expect(row.filter((w) => w > 0).length).toBe(k)
        keep.forEach((i, j) => expect(row[i]).toBeCloseTo(p[j], 12))
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      })
    }
  })

  it('switch routes to the argmax with the full softmax probability as its weight', () => {
    const r = route(z, { gate: 'switch', k: 3 })
    expect(r.k).toBe(1)
    rows(r.combine).forEach((row, t) => {
      const i = topK(LOGITS[t], 1)[0]
      expect(row.filter((w) => w > 0).length).toBe(1)
      expect(row[i]).toBeCloseTo(softmaxRow(LOGITS[t])[i], 12)
    })
  })

  it('a small temperature makes the softmax gate nearly hard; a large one nearly uniform', () => {
    const hard = rows(route(z, { gate: 'softmax', temperature: 0.01 }).combine)
    hard.forEach((row, t) => expect(row[topK(LOGITS[t], 1)[0]]).toBeGreaterThan(0.999))
    const soft = rows(route(z, { gate: 'softmax', temperature: 1000 }).combine)
    soft.forEach((row) => row.forEach((w) => expect(w).toBeCloseTo(0.25, 2)))
    rows(route(z, { gate: 'softmax', temperature: 2 }).combine).forEach((row, t) =>
      row.forEach((w, i) => expect(w).toBeCloseTo(softmaxRow(LOGITS[t], 2)[i], 12)),
    )
  })

  it('noisy top-k equals top-k without a stream or with zero noise, and is reproducible from its stream', () => {
    const plain = toFlat(unwrap(route(z, { gate: 'top-k', k: 2 }).combine) as Tensor)
    expect(toFlat(unwrap(route(z, { gate: 'noisy-top-k', k: 2 }).combine) as Tensor)).toEqual(plain)
    const zero = route(z, { gate: 'noisy-top-k', k: 2, noiseScale: 0, stream: stream('n') })
    expect(toFlat(unwrap(zero.combine) as Tensor)).toEqual(plain)
    const a = route(z, { gate: 'noisy-top-k', k: 2, noiseScale: 3, stream: stream('n') })
    const b = route(z, { gate: 'noisy-top-k', k: 2, noiseScale: 3, stream: stream('n') })
    expect(toFlat(unwrap(a.combine) as Tensor)).toEqual(toFlat(unwrap(b.combine) as Tensor))
    expect(toFlat(a.selected)).not.toEqual(toFlat(route(z, { gate: 'top-k', k: 2 }).selected))
  })

  it('expert choice gives every expert exactly C tokens, weighted by the router probability', () => {
    const r = route(z, { gate: 'expert-choice', capacityFactor: 1.6 })
    // C = ⌈1.6 · 5 / 4⌉ = 2
    expect(r.capacity).toBe(2)
    const d = toRows(r.dispatch)
    for (let i = 0; i < 4; i++) {
      const col = d.map((row) => row[i])
      expect(col.reduce((a, b) => a + b, 0)).toBe(2)
      const probs = LOGITS.map((row) => softmaxRow(row)[i])
      const top = topK(probs, 2)
      top.forEach((t) => expect(d[t][i]).toBe(1))
    }
    rows(r.combine).forEach((row, t) =>
      row.forEach((w, i) => expect(w).toBeCloseTo(d[t][i] * softmaxRow(LOGITS[t])[i], 12)),
    )
  })
})

describe('route: capacity and dropping', () => {
  it('drops assignments past ⌈c·T·k/N⌉ in token order', () => {
    const same = tensor(Array.from({ length: 6 }, () => [1, 0]))
    const r = route(same, { gate: 'switch', capacityFactor: 1 })
    expect(r.capacity).toBe(3)
    expect(toRows(r.dispatch)).toEqual([
      [1, 0],
      [1, 0],
      [1, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ])
    rows(r.combine)
      .slice(3)
      .forEach((row) => expect(row).toEqual([0, 0]))
    const s = routingStatistics(r)
    expect(s.dropped).toBeCloseTo(0.5, 12)
    expect(s.unrouted).toBeCloseTo(0.5, 12)
    expect(s.counts).toEqual([3, 0])
    expect(s.idle).toBe(1)
  })

  it("fills buffers with every token's first choice before any second choice (GShard)", () => {
    // Four tokens, three experts, k = 2, c = 0.75 → C = ⌈0.75·4·2/3⌉ = 2. Expert 0 is everyone's first choice.
    const l = tensor([
      [3, 2, 0],
      [3, 0, 2],
      [3, 2, 0],
      [0, 2, 3],
    ])
    const r = route(l, { gate: 'top-k', k: 2, capacityFactor: 0.75 })
    expect(r.capacity).toBe(2)
    // First choices: t0→0, t1→0, t2→0 (full), t3→2. Second choices: t0→1, t1→2, t2→1, t3→1 (full).
    expect(toRows(r.dispatch)).toEqual([
      [1, 1, 0],
      [1, 0, 1],
      [0, 1, 0],
      [0, 0, 1],
    ])
    // The surviving weight is the renormalised top-2 weight, not renormalised again after dropping.
    const w = rows(r.combine)
    expect(w[2][1]).toBeCloseTo(softmaxRow([3, 2])[1], 12)
    expect(w[3][2]).toBeCloseTo(softmaxRow([3, 2])[0], 12)
  })
})

describe('auxiliary losses against hand computation', () => {
  it('load balancing: N Σ fᵢ Pᵢ with f the share of assignments and P the mean probability', () => {
    for (const k of [1, 2]) {
      const r = route(z, { gate: 'top-k', k })
      const N = 4
      const counts = [0, 0, 0, 0]
      LOGITS.forEach((row) => topK(row, k).forEach((i) => counts[i]++))
      const f = counts.map((c) => c / (LOGITS.length * k))
      const P = [0, 1, 2, 3].map((i) => LOGITS.reduce((a, row) => a + softmaxRow(row)[i], 0) / LOGITS.length)
      const expected = N * f.reduce((a, v, i) => a + v * P[i], 0)
      expect(unwrap(loadBalancingLoss(r))).toBeCloseTo(expected, 12)
    }
  })

  it('load balancing is 1 when balanced and approaches N when one expert takes everything', () => {
    const balanced = tensor([
      [5, 0],
      [0, 5],
    ])
    // f = (1/2, 1/2) whatever P is, so the loss is 2 · (P₀ + P₁)/2 = 1.
    expect(unwrap(loadBalancingLoss(route(balanced, { gate: 'switch' })))).toBeCloseTo(1, 12)
    const collapsed = tensor(Array.from({ length: 8 }, () => [40, 0, 0]))
    expect(unwrap(loadBalancingLoss(route(collapsed, { gate: 'switch' })))).toBeCloseTo(3, 10)
  })

  it('importance: CV² of the column sums of the combine weights', () => {
    const r = route(z, { gate: 'softmax' })
    const imp = [0, 1, 2, 3].map((i) => LOGITS.reduce((a, row) => a + softmaxRow(row)[i], 0))
    const m = imp.reduce((a, b) => a + b, 0) / 4
    const v = imp.reduce((a, b) => a + (b - m) ** 2, 0) / 4
    expect(unwrap(importanceLoss(r))).toBeCloseTo(v / (m * m), 12)
  })

  it('router z-loss: mean of the squared log-sum-exp of each row of logits', () => {
    const lse = LOGITS.map((row) => {
      const m = Math.max(...row)
      return m + Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0))
    })
    const expected = lse.reduce((a, v) => a + v * v, 0) / lse.length
    expect(unwrap(routerZLoss(z))).toBeCloseTo(expected, 12)
    // The temperature does not enter: the z-loss reads the raw logits.
    expect(unwrap(routerZLoss(route(z, { gate: 'top-k', temperature: 3 })))).toBeCloseTo(expected, 12)
  })

  it('the losses are differentiable in the logits (finite differences)', () => {
    for (const gate of ['top-k', 'switch', 'softmax', 'expert-choice'] as const) {
      const f = (l: Tensor) => {
        const r = route(l, { gate, k: 2 })
        return total(loadBalancingLoss(r), importanceLoss(r), mul(0.1, routerZLoss(r)), sum(mul(r.combine, l)))
      }
      expect(gradCheck(f, z, { rtol: 1e-5, atol: 1e-7 }).ok).toBe(true)
    }
  })
})

describe('routingStatistics', () => {
  it('entropy is log N for uniform probabilities and the load sums to 1', () => {
    const flat = tensor(Array.from({ length: 6 }, () => [0, 0, 0]))
    const s = routingStatistics(route(flat, { gate: 'softmax' }))
    expect(s.entropy).toBeCloseTo(Math.log(3), 12)
    expect(s.loadEntropy).toBeCloseTo(Math.log(3), 12)
    const t = routingStatistics(route(z, { gate: 'top-k', k: 2 }))
    expect(t.load.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(t.counts.reduce((a, b) => a + b, 0)).toBe(10)
  })
})

describe('MixtureOfExperts', () => {
  const experts = [Linear(3, 2), Linear(3, 2), Linear(3, 2)]
  const x = tensor([
    [0.5, -1.0, 0.3],
    [1.2, 0.4, -0.7],
    [-0.8, 0.9, 0.1],
    [0.0, 0.3, 1.1],
  ])

  it('outputs Σᵢ wₜᵢ Eᵢ(xₜ) with the weights of route', () => {
    const layer = MixtureOfExperts(3, experts, { gate: 'top-k', k: 2 })
    const p = layer.init(stream('moe'))
    const { output, routing, expertOutputs } = layer.forward(p, x)
    const w = rows(routing.combine)
    const ys = expertOutputs.map(rows)
    rows(output).forEach((row, t) =>
      row.forEach((v, j) =>
        expect(v).toBeCloseTo(
          ys.reduce((a, y, i) => a + w[t][i] * y[t][j], 0),
          12,
        ),
      ),
    )
    // Batched inputs [B, S, d] route their flattened tokens.
    const batched = layer.apply(p, reshape(x, [2, 2, 3]))
    expect(toFlat(unwrap(batched) as Tensor)).toEqual(toFlat(unwrap(output) as Tensor))
  })

  it('a zero router chooses every expert equally under the softmax gate', () => {
    const layer = MixtureOfExperts(3, experts, { gate: 'softmax', zeroRouter: true })
    const r = layer.forward(layer.init(stream('z')), x).routing
    rows(r.combine).forEach((row) => row.forEach((w) => expect(w).toBeCloseTo(1 / 3, 12)))
  })

  it('gradients through router and experts match finite differences, for every gate', () => {
    for (const gate of ['softmax', 'top-k', 'switch', 'expert-choice', 'noisy-top-k'] as const) {
      const layer = MixtureOfExperts(3, experts, { gate, k: 2 })
      const p = layer.init(stream(`g-${gate}`))
      const f = (q: typeof p) => {
        const { output, routing } = layer.forward(q, x)
        return total(sum(mul(output, output)), loadBalancingLoss(routing), routerZLoss(routing))
      }
      expect(gradCheck(f, p, { rtol: 1e-4, atol: 1e-6 }).ok).toBe(true)
    }
  })

  it('the noisy gate draws noise only in training mode with a stream', () => {
    const layer = MixtureOfExperts(3, experts, { gate: 'noisy-top-k', k: 1 })
    const p = layer.init(stream('noisy'))
    p.noise = { weight: p.router.weight, bias: (p.router as LinearParams).bias }
    const evalA = toFlat(unwrap(layer.apply(p, x)) as Tensor)
    expect(toFlat(unwrap(layer.apply(p, x, { train: false, stream: stream('s') })) as Tensor)).toEqual(evalA)
    const a = toFlat(unwrap(layer.apply(p, x, { train: true, stream: stream('s') })) as Tensor)
    const b = toFlat(unwrap(layer.apply(p, x, { train: true, stream: stream('s') })) as Tensor)
    expect(a).toEqual(b)
  })
})

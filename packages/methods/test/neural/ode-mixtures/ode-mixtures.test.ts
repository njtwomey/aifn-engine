/**
 * Stochastic vector field mixtures: forward filtering against brute-force enumeration, gradients against finite
 * differences, one deterministic component as a plain neural ODE, the losses against hand-computed values, the FLoss
 * rule, realised paths and the datasets.
 */
import { describe, expect, it } from 'vitest'
import {
  checkLossSettings,
  classMixtureLoss,
  instanceWork,
  interpolatePaths,
  mixtureDensityLoss,
  realisation,
  samplePaths,
  svfm,
  svfmObjective,
  transportLoss,
  varianceLoss,
  flatOf,
  type Propagation,
  type SvfmParams,
} from 'aifn-methods/neural/ode-mixtures'
import { floorplanWalks, FLOORPLAN, odeFailureCase } from 'aifn-methods/data/synthetic'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  add,
  concat,
  fromData,
  matmul,
  mul,
  ones,
  toFlat,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { odeFlow } from 'aifn-compute/dynamics/ode'
import { relu } from 'aifn-compute/nn/functional'

const t = (a: number[], shape: number[]) => fromData(Float64Array.from(a), shape)

/** Set a network to a constant output: zero weights, the last bias given (per stacked copy). */
function constantNet(net: SvfmParams['fields'], bias: number[][]): SvfmParams['fields'] {
  return net.map((layer, l) => {
    const [k, , out] = layer.weight.shape
    const last = l === net.length - 1
    return {
      weight: zeros(layer.weight.shape),
      bias: last ? t(bias.flat(), [k, 1, out]) : zeros([k, 1, out]),
    }
  })
}

describe('forward filtering (eq. 5)', () => {
  // K = 3 constant VFs vₖ in 1-d, constant ψ and Ψ, T = 3: the filter against all K^(T+1) component sequences.
  const K = 3
  const T = 3
  const v = [1, -0.5, 2]
  const pi0 = [0.5, 0.3, 0.2]
  const psi = [0.6, 0.3, 0.1]
  const Psi = [
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.25, 0.25, 0.5],
  ]
  const model = svfm({ dim: 1, components: K, selection: 'forward-filtering', grid: T, stepSize: 0.1 })
  const base = model.init(stream(1))
  const params: SvfmParams = {
    ...base,
    fields: constantNet(
      base.fields,
      v.map((x) => [x]),
    ),
    // The prior's logits are 0.2 × its output: bias = 5 log π₀.
    prior: constantNet(base.prior, [pi0.map((p) => 5 * Math.log(p))]),
    emission: constantNet(base.emission, [psi.map(Math.log)]),
    transition: constantNet(base.transition, [Psi.flat().map(Math.log)]),
  }
  const h0 = 0.3
  const p = model.propagate(params, t([h0], [1, 1]), null)

  // Brute force: s₀ ~ π₀; interval i follows s_{i−1}; at tᵢ weight ψ(s_{i−1}) Ψ(s_{i−1}, sᵢ).
  const dt = 1 / T
  const weight = new Array<number>(K).fill(0)
  const first = new Array<number>(K).fill(0)
  const second = new Array<number>(K).fill(0)
  const visit = (seq: number[]) => {
    if (seq.length === T + 1) {
      let w = pi0[seq[0]]
      let h = h0
      for (let i = 1; i <= T; i++) {
        w *= psi[seq[i - 1]] * Psi[seq[i - 1]][seq[i]]
        h += dt * v[seq[i - 1]]
      }
      const k = seq[T]
      weight[k] += w
      first[k] += w * h
      second[k] += w * h * h
      return
    }
    for (let k = 0; k < K; k++) visit([...seq, k])
  }
  visit([])
  const Z = weight.reduce((a, b) => a + b, 0)

  it('gives π(t_T) of the enumeration', () => {
    const got = Array.from(flatOf(p.logWeights[T]), Math.exp)
    weight.forEach((w, k) => expect(got[k]).toBeCloseTo(w / Z, 10))
  })
  it('gives the component-conditioned means and spreads of the enumeration (exact for constant fields)', () => {
    const m = flatOf(p.states[T])
    const r = flatOf(p.spread[T])
    for (let k = 0; k < K; k++) {
      const mean = first[k] / weight[k]
      expect(m[k]).toBeCloseTo(mean, 10)
      expect(r[k] ** 2).toBeCloseTo(second[k] / weight[k] - mean * mean, 8)
    }
  })
  it('is pick and stick with identity transitions and uniform emissions', () => {
    const sticky = svfm({
      dim: 1,
      components: K,
      selection: 'forward-filtering',
      grid: T,
      transitions: 'fixed',
      stickiness: 1,
      emissions: 'uniform',
    })
    const stick = svfm({ dim: 1, components: K, selection: 'pick-and-stick', grid: T })
    const q: SvfmParams = { ...sticky.init(stream(1)), fields: params.fields, prior: params.prior }
    const a = sticky.propagate(q, t([h0], [1, 1]), null)
    const b = stick.propagate(
      { ...stick.init(stream(1)), fields: params.fields, prior: params.prior },
      t([h0], [1, 1]),
      null,
    )
    expect(Array.from(flatOf(a.logWeights[T]))).toEqual(
      Array.from(flatOf(b.logWeights[T])).map((x) => expect.closeTo(x, 7)),
    )
    expect(Array.from(flatOf(a.states[T]))).toEqual(Array.from(flatOf(b.states[T])).map((x) => expect.closeTo(x, 7)))
  })
})

describe('one deterministic component is a plain neural ODE', () => {
  const model = svfm({ dim: 2, components: 1, grid: 4, stepSize: 0.05, layers: 1, hidden: 8, classes: 2 })
  const params = model.init(stream(3))
  const x = t([0.2, -0.4, 1.0, 0.5, -0.7, 0.1], [3, 2])
  it('carries h(t) as ∇h = f(h, t) with the same network', () => {
    const [l1, l2] = params.fields
    const W1 = fromData(Float64Array.from(toFlat(l1.weight)), [3, 8])
    const b1 = fromData(Float64Array.from(toFlat(l1.bias)), [8])
    const W2 = fromData(Float64Array.from(toFlat(l2.weight)), [8, 2])
    const b2 = fromData(Float64Array.from(toFlat(l2.bias)), [2])
    const f = (tt: number | Value, h: Value) =>
      add(matmul(relu(add(matmul(concat([h, mul(tt, ones([3, 1]))], 1), W1), b1)), W2), b2)
    const times = [0, 0.25, 0.5, 0.75, 1]
    const plain = odeFlow((tt, h) => f(tt, h), times, { method: 'rk4', stepSize: 0.05 })(x, 0)
    const p = model.propagate(params, x, null)
    for (let i = 0; i < times.length; i++) {
      const a = flatOf(p.states[i])
      const b = flatOf(plain[i])
      b.forEach((v, j) => expect(a[j]).toBeCloseTo(v, 10))
    }
    expect(Array.from(flatOf(p.spread[4]))).toEqual([0, 0, 0])
  })
  it('makes MDLoss for labels the cross-entropy of the readout', () => {
    const p = model.propagate(params, x, null)
    const y = fromData(Int32Array.of(0, 1, 1), [3])
    const loss = flatOf(classMixtureLoss(model, params, p, y, 2))[0]
    const z = flatOf(p.states[4])
    const W = toFlat(params.readout.weight!)
    const bb = toFlat(params.readout.bias!)
    let ce = 0
    for (let i = 0; i < 3; i++) {
      const logits = [0, 1].map((c) => z[2 * i] * W[c] + z[2 * i + 1] * W[2 + c] + bb[c])
      const lse = Math.log(Math.exp(logits[0]) + Math.exp(logits[1]))
      ce += (lse - logits[[0, 1, 1][i]]) / 3
    }
    expect(loss).toBeCloseTo(ce, 10)
  })
})

describe('losses against hand-computed values', () => {
  // Two components in 1-d over T = 2: states m_k(tᵢ), spreads, weights and fields set by hand.
  const s = (a: number[]) => t(a, [2, 1, 1])
  const p: Propagation = {
    times: [0, 0.5, 1],
    states: [s([0, 0]), s([1, -1]), s([3, -1])],
    arrived: [s([0, 0]), s([1, -1]), s([3, -1])],
    spread: [t([0, 0], [2, 1]), t([0.3, 0.1], [2, 1]), t([0.4, 0.2], [2, 1])],
    logWeights: [
      t([Math.log(0.25), Math.log(0.75)], [1, 2]),
      t([Math.log(0.25), Math.log(0.75)], [1, 2]),
      t([Math.log(0.5), Math.log(0.5)], [1, 2]),
    ],
    fields: [s([0, 0]), s([2, -2]), s([4, 0])],
  }
  it('TLoss (eq. 8)', () => {
    // i = 1: 0.25·1² + 0.75·1² = 1; i = 2: 0.25·2² + 0.75·0² = 1 → mean 1.
    expect(flatOf(transportLoss(p))[0]).toBeCloseTo(1, 12)
  })
  it('VLoss (eq. 9)', () => {
    // Component 1: fields 2, 4 (mean 3): deviations 1, 1; component 2: −2, 0 (mean −1): 1, 1.
    // i = 1: 0.25·1 + 0.75·1 = 1; i = 2: 0.5·1 + 0.5·1 = 1 → 1.
    expect(flatOf(varianceLoss(p))[0]).toBeCloseTo(1, 12)
  })
  it('MDLoss (eq. 10)', () => {
    const model = svfm({ dim: 1, components: 2, stochastic: true })
    const params = { ...model.init(stream(0)), logNoise: t([Math.log(0.5)], []) }
    const y = 2.5
    const sd = [Math.sqrt(0.4 ** 2 + 0.25), Math.sqrt(0.2 ** 2 + 0.25)]
    const dens = [3, -1].map(
      (m, k) => (0.5 * Math.exp(-0.5 * ((y - m) / sd[k]) ** 2)) / (sd[k] * Math.sqrt(2 * Math.PI)),
    )
    expect(flatOf(mixtureDensityLoss(model, params, p, 2, t([y], [1, 1])))[0]).toBeCloseTo(
      -Math.log(dens[0] + dens[1]),
      10,
    )
  })
  it('FLoss targets: a cubic spline reproduces a cubic path', () => {
    const times = [0, 0.2, 0.5, 0.7, 1]
    const path = times.flatMap((u) => [u ** 3 - u, 2 * u])
    const grid = [0, 0.25, 0.5, 0.75, 1]
    const got = toFlat(interpolatePaths(Float64Array.from(path), times, 2, grid))
    grid.forEach((u, g) => {
      expect(got[2 * g]).toBeCloseTo(u ** 3 - u, 10)
      expect(got[2 * g + 1]).toBeCloseTo(2 * u, 10)
    })
  })
  it('refuses FLoss with TLoss or VLoss (§4.1.3)', () => {
    expect(() => checkLossSettings({ forecast: true, transport: true })).toThrow(/FLoss/)
    expect(() => checkLossSettings({ forecast: true, variance: true })).toThrow(/FLoss/)
    expect(() => checkLossSettings({ forecast: true })).not.toThrow()
    expect(() => checkLossSettings({ transport: true, variance: true })).not.toThrow()
  })
})

describe('gradients against finite differences', () => {
  for (const selection of ['pick-and-stick', 'forward-filtering'] as const)
    it(`of the SVFM objective with TLoss and VLoss (${selection})`, () => {
      const model = svfm({
        dim: 2,
        components: 2,
        stochastic: true,
        selection,
        grid: 3,
        stepSize: 0.1,
        hidden: 6,
        layers: 1,
      })
      const params = model.init(stream(7))
      const batch = { x: t([0.3, -0.2, -0.5, 0.8, 1.1, 0.1], [3, 2]), targets: t([1, 0, -1, 1, 0.5, -0.5], [3, 2]) }
      const objective = svfmObjective(model, { transport: true, variance: true, lambda: 0.3 })
      const f = (q: SvfmParams) => flatOf(objective(q, batch).total)[0]
      const g = valueAndGrad((q: unknown) => objective(q as SvfmParams, batch).total)(params).grad as SvfmParams
      const probes: [keyof SvfmParams, number, number][] = [
        ['fields', 0, 1],
        ['fields', 1, 5],
        ['prior', 0, 2],
        ...(selection === 'forward-filtering'
          ? ([
              ['transition', 0, 3],
              ['emission', 0, 4],
            ] as [keyof SvfmParams, number, number][])
          : []),
      ]
      for (const [net, layer, entry] of probes) {
        const leaf = (q: SvfmParams) => (q[net] as SvfmParams['fields'])[layer].weight
        const at = (delta: number): SvfmParams => {
          const w = Float64Array.from(toFlat(leaf(params)))
          w[entry] += delta
          const copy = (params[net] as SvfmParams['fields']).map((l, i) =>
            i === layer ? { ...l, weight: fromData(w, l.weight.shape) } : l,
          )
          return { ...params, [net]: copy }
        }
        const h = 1e-6
        const fd = (f(at(h)) - f(at(-h))) / (2 * h)
        const ad = toFlat(leaf(g) as Tensor)[entry]
        expect(ad).toBeCloseTo(fd, 5)
      }
    })
})

describe('realised paths and their work', () => {
  const model = svfm({ dim: 2, components: 2, stochastic: true, grid: 4, selection: 'forward-filtering' })
  const params = model.init(stream(2))
  const x = Float64Array.of(0.1, 0.2, -0.3, 0.4, 0.5, -0.6)
  const real = realisation(child(stream(4), 'r'), 3, 2)
  it('end where the per-instance solve ends, at every tolerance', () => {
    const paths = samplePaths(model, params, x, null, real, { rtol: 1e-8, atol: 1e-10, framesPerInterval: 2 })
    const work = instanceWork(model, params, x, null, real, { rtol: 1e-8, atol: 1e-10 })
    const F = paths.times.length - 1
    for (let j = 0; j < 6; j++) expect(paths.paths[F * 6 + j]).toBeCloseTo(work.final[j], 5)
    for (const n of work.perInstance) expect(n).toBeGreaterThan(7)
    // π stays on the simplex along the paths.
    for (let i = 0; i < paths.weights.length; i += 2) expect(paths.weights[i] + paths.weights[i + 1]).toBeCloseTo(1, 10)
  })
})

describe('datasets', () => {
  it('odeFailureCase: crossing maps x to −x, splitting goes to ±1', () => {
    const c = odeFailureCase(stream(1), { kind: 'crossing', n: 50 })
    const xs = toFlat(c.x)
    toFlat(c.y!).forEach((y, i) => expect(y).toBeCloseTo(-xs[i], 12))
    const s = odeFailureCase(stream(1), { kind: 'splitting', n: 200, noise: 0.01 })
    for (const y of toFlat(s.y!)) expect(Math.abs(Math.abs(y) - 1)).toBeLessThan(0.05)
  })
  it('floorplanWalks: walks start at the sofa and end at their target', () => {
    const d = floorplanWalks(stream(2), { n: 40, noise: 0 })
    const [n, cols] = d.x.shape
    const M = (cols - 1) / 2
    const a = toFlat(d.x)
    const y = toFlat(d.y!)
    for (let i = 0; i < n; i++) {
      expect(Math.hypot(a[i * cols] - FLOORPLAN.origin[0], a[i * cols + 1] - FLOORPLAN.origin[1])).toBeLessThan(0.5)
      const goal = FLOORPLAN.targets[y[i]].at
      expect(Math.hypot(a[i * cols + 2 * M - 2] - goal[0], a[i * cols + 2 * M - 1] - goal[1])).toBeLessThan(0.6)
    }
    // By day the four targets come in equal shares.
    expect([0, 1, 2, 3].map((k) => y.filter((v) => v === k).length)).toEqual([10, 10, 10, 10])
  })
})

/**
 * Off-policy estimators: hand cases, identities between them (switch-DR at τ = ∞ is DR and at τ = 0 the direct method;
 * clipping above the largest weight is IPS), and unbiasedness laws by simulation: on fixed contexts, logs are redrawn
 * from a seeded stream and the mean estimate must sit within a few standard errors of the true value.
 */
import { describe, expect, it } from 'vitest'
import { bernoulli, categorical, child, normal, permutation, stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  clippedIps,
  directMethod,
  doublyRobust,
  empiricalPropensities,
  estimatePropensities,
  importanceWeights,
  ips,
  slateIps,
  slatePseudoInverse,
  snips,
  switchDoublyRobust,
  type BanditLog,
} from 'aifn-compute/learning/off-policy'

const flat = (t: Tensor) => Array.from(toFlat(t))
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
const softmax = (z: number[], beta: number) => {
  const e = z.map((v) => Math.exp(beta * v - Math.max(...z.map((u) => beta * u))))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/** A small contextual bandit with known truth: n contexts, K actions, Bernoulli rewards. */
function problem(n: number, K: number, sharpness: number) {
  const s = stream('ope-problem')
  const x = Array.from({ length: n }, () => [normal(s), normal(s)])
  const theta = Array.from({ length: K }, () => [normal(s), normal(s)])
  const phi = Array.from({ length: K }, () => [normal(s), normal(s)])
  const mu = x.map((xi) => theta.map((t) => sigmoid(t[0] * xi[0] + t[1] * xi[1])))
  const logging = x.map((xi) => {
    const p = softmax(
      phi.map((f) => f[0] * xi[0] + f[1] * xi[1]),
      sharpness,
    )
    return p.map((v) => 0.9 * v + 0.1 / K)
  })
  const target = mu.map((m) => softmax(m, 6))
  const truth = target.reduce((acc, p, i) => acc + p.reduce((a, v, k) => a + v * mu[i][k], 0), 0) / n
  // A biased reward model: every logit shifted by +0.8.
  const model = mu.map((m) => m.map((v) => sigmoid(Math.log(v / (1 - v)) + 0.8)))
  return { x, mu, logging, target, truth, model }
}

function draw(P: ReturnType<typeof problem>, seed: number): BanditLog & { logging: number[][] } {
  const s = child(stream('ope-logs'), 'log', seed)
  const actions: number[] = []
  const rewards: number[] = []
  const propensities: number[] = []
  P.logging.forEach((p, i) => {
    const a = categorical(s, p)
    actions.push(a)
    propensities.push(p[a])
    rewards.push(bernoulli(s, P.mu[i][a]))
  })
  return { actions, rewards, propensities, logging: P.logging }
}

describe('hand cases', () => {
  const log: BanditLog = { actions: [0, 1, 1, 0], rewards: [1, 0, 1, 1], propensities: [0.5, 0.25, 0.25, 0.5] }
  const target = [
    [1, 0],
    [0.5, 0.5],
    [0, 1],
    [1, 0],
  ]
  const model = [
    [0.8, 0.2],
    [0.5, 0.5],
    [0.6, 0.6],
    [0.8, 0.1],
  ]

  it('IPS, SNIPS and the weights', () => {
    expect(flat(importanceWeights(log, target))).toEqual([2, 2, 4, 2])
    expect(ips(log, target).value).toBeCloseTo((2 + 0 + 4 + 2) / 4, 12)
    expect(snips(log, target).value).toBeCloseTo((2 + 0 + 4 + 2) / 10, 12)
    expect(ips(log, target).effectiveSampleSize).toBeCloseTo(100 / 28, 12)
    expect(clippedIps(log, target, { clip: 3 }).value).toBeCloseTo((2 + 0 + 3 + 2) / 4, 12)
  })

  it('DM, DR and switch-DR', () => {
    const dm = (0.8 + 0.5 + 0.6 + 0.8) / 4
    expect(directMethod(log, target, model).value).toBeCloseTo(dm, 12)
    const corrections = [2 * (1 - 0.8), 2 * (0 - 0.5), 4 * (1 - 0.6), 2 * (1 - 0.8)]
    expect(doublyRobust(log, target, model).value).toBeCloseTo(dm + corrections.reduce((a, b) => a + b) / 4, 12)
    // τ = 3 drops the round with weight 4.
    const sw = switchDoublyRobust(log, target, model, { tau: 3 })
    expect(sw.value).toBeCloseTo(dm + (corrections[0] + corrections[1] + corrections[3]) / 4, 12)
  })

  it('switch-DR spans the direct method (τ = 0) and DR (τ = ∞); clipping above every weight is IPS', () => {
    const P = problem(200, 4, 2)
    const L = draw(P, 1)
    expect(switchDoublyRobust(L, P.target, P.model, { tau: Infinity }).value).toBeCloseTo(
      doublyRobust(L, P.target, P.model).value,
      12,
    )
    expect(switchDoublyRobust(L, P.target, P.model, { tau: 0 }).value).toBeCloseTo(
      directMethod(L, P.target, P.model).value,
      12,
    )
    const wmax = Math.max(...flat(importanceWeights(L, P.target)))
    expect(clippedIps(L, P.target, { clip: wmax }).value).toBeCloseTo(ips(L, P.target).value, 12)
  })

  it('rejects zero propensities and mismatched shapes', () => {
    expect(() => ips({ ...log, propensities: [0.5, 0, 0.25, 0.5] }, target)).toThrow(/propensity/)
    expect(() => ips(log, target.slice(0, 3))).toThrow(/rows/)
  })
})

describe('unbiasedness by simulation', () => {
  const P = problem(150, 4, 2)
  const R = 300
  const logs = Array.from({ length: R }, (_, r) => draw(P, r))
  const meanAndSe = (values: number[]) => {
    const m = values.reduce((a, b) => a + b, 0) / values.length
    const v = values.reduce((a, b) => a + (b - m) ** 2, 0) / (values.length - 1)
    return { m, se: Math.sqrt(v / values.length), sd: Math.sqrt(v) }
  }

  it('IPS and DR (with a biased model) are unbiased; the direct method keeps the model’s bias', () => {
    const ipsRuns = meanAndSe(logs.map((L) => ips(L, P.target).value))
    const drRuns = meanAndSe(logs.map((L) => doublyRobust(L, P.target, P.model).value))
    expect(Math.abs(ipsRuns.m - P.truth)).toBeLessThan(4 * ipsRuns.se)
    expect(Math.abs(drRuns.m - P.truth)).toBeLessThan(4 * drRuns.se)
    const dm = directMethod(logs[0], P.target, P.model).value
    expect(dm - P.truth).toBeGreaterThan(0.05)
    // The reported standard error estimates the spread across logs.
    const se = meanAndSe(logs.map((L) => ips(L, P.target).standardError)).m
    expect(se / ipsRuns.sd).toBeGreaterThan(0.7)
    expect(se / ipsRuns.sd).toBeLessThan(1.3)
  })

  it('clipping biases IPS downward for non-negative rewards; SNIPS is close', () => {
    const clipped = meanAndSe(logs.map((L) => clippedIps(L, P.target, { clip: 2 }).value))
    expect(P.truth - clipped.m).toBeGreaterThan(4 * clipped.se)
    const sn = meanAndSe(logs.map((L) => snips(L, P.target).value))
    expect(Math.abs(sn.m - P.truth)).toBeLessThan(0.02)
  })
})

describe('slates', () => {
  const m = 5
  const l = 2
  const phi = [
    [0.9, 0.7, 0.5, 0.3, 0.1],
    [0.45, 0.35, 0.25, 0.15, 0.05],
  ]
  const target = [0, 1]
  const truth = phi[0][0] + phi[1][1]
  const simulate = (seed: number, n: number) => {
    const s = child(stream('slates'), 'log', seed)
    const slates: number[][] = []
    const rewards: number[] = []
    for (let i = 0; i < n; i++) {
      const p = flat(permutation(s, m) as Tensor).slice(0, l)
      slates.push(p)
      rewards.push(phi[0][p[0]] + phi[1][p[1]] + 0.1 * normal(s))
    }
    return { slates, rewards, target: slates.map(() => target) }
  }

  it('the PI weights under uniform logging match numpy’s pinv (θᵀΓ⁺1ₛ)', () => {
    const log = {
      slates: [
        [0, 1],
        [0, 2],
        [3, 4],
        [1, 0],
      ],
      rewards: [1, 1, 1, 1],
    }
    const est = slatePseudoInverse(
      log,
      log.slates.map(() => target),
      { kind: 'uniform', items: m },
    )
    flat(est.weights).forEach((w, i) => expect(w).toBeCloseTo([9, 11 / 3, -5 / 3, 1][i], 9))
  })

  it('an item outside the logging policy’s catalogue throws instead of landing in the next slot', () => {
    const log = { slates: [[0, 1]], rewards: [1] }
    // Item m in slot 0 would index slot 1's first pair.
    expect(() => slatePseudoInverse(log, [[m, 1]], { kind: 'uniform', items: m })).toThrow(/not one of/)
    const list = { kind: 'list' as const, slates: [[0, 1]], probabilities: [1] }
    expect(() => slatePseudoInverse({ slates: [[0, 3]], rewards: [1] }, [[0, 1]], list)).toThrow(/not one of/)
  })

  it('an explicit list of every ordered slate equals uniform logging', () => {
    const all: number[][] = []
    for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) if (a !== b) all.push([a, b])
    const log = simulate(1, 50)
    const u = slatePseudoInverse(log, log.target, { kind: 'uniform', items: m })
    const e = slatePseudoInverse(log, log.target, {
      kind: 'list',
      slates: all,
      probabilities: all.map(() => 1 / all.length),
    })
    expect(e.value).toBeCloseTo(u.value, 9)
    expect(
      slateIps(log, log.target, { kind: 'list', slates: all, probabilities: all.map(() => 1 / all.length) }).value,
    ).toBeCloseTo(slateIps(log, log.target, { kind: 'uniform', items: m }).value, 12)
  })

  it('PI and slate IPS are unbiased under an additive reward; PI has the smaller variance', () => {
    const runs = Array.from({ length: 200 }, (_, r) => simulate(r + 10, 200))
    const pi = runs.map((L) => slatePseudoInverse(L, L.target, { kind: 'uniform', items: m }).value)
    const si = runs.map((L) => slateIps(L, L.target, { kind: 'uniform', items: m }).value)
    const stats = (v: number[]) => {
      const mean = v.reduce((a, b) => a + b, 0) / v.length
      const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1))
      return { mean, sd, se: sd / Math.sqrt(v.length) }
    }
    const a = stats(pi)
    const b = stats(si)
    expect(Math.abs(a.mean - truth)).toBeLessThan(4 * a.se)
    expect(Math.abs(b.mean - truth)).toBeLessThan(4 * b.se)
    expect(a.sd).toBeLessThan(b.sd)
  })
})

describe('propensities', () => {
  it('a multinomial logistic model recovers a softmax-linear logging policy', () => {
    const s = stream('propensity')
    const n = 3000
    const W = [
      [1.5, -1, 0],
      [0, 1, -1.2],
    ]
    const x: number[][] = []
    const truth: number[][] = []
    const actions: number[] = []
    for (let i = 0; i < n; i++) {
      const xi = [normal(s), normal(s)]
      const p = softmax(
        [0, 1, 2].map((k) => xi[0] * W[0][k] + xi[1] * W[1][k]),
        1,
      )
      x.push(xi)
      truth.push(p)
      actions.push(categorical(s, p))
    }
    const fit = estimatePropensities(x, actions, { l2: 0 })
    expect(fit.converged).toBe(true)
    const P = flat(fit.probabilities)
    let err = 0
    truth.forEach((p, i) => p.forEach((v, k) => (err += Math.abs(v - P[i * 3 + k]))))
    expect(err / (n * 3)).toBeLessThan(0.03)
    expect(flat(fit.predict([[0, 0]])).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('empirical propensities are smoothed shares within each group', () => {
    const { propensities, table } = empiricalPropensities([0, 0, 0, 1], [0, 0, 1, 1], { smoothing: 1 })
    expect(flat(table)).toEqual([3 / 5, 2 / 5, 1 / 3, 2 / 3])
    expect(flat(propensities)).toEqual([3 / 5, 3 / 5, 2 / 5, 2 / 3])
  })
})

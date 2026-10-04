/**
 * Online learning against its regret bounds (laws over seeded loss sequences, adaptive adversaries included), hand
 * cases, and brute force for the best switching comparator.
 */
import { describe, expect, it } from 'vitest'
import { child, normals, stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import {
  bestSwitchingLoss,
  fixedShare,
  followTheRegularisedLeader,
  hedge,
  hedgeRegretBound,
  hedgeTunedRate,
  ogdRegretBound,
  ogdStepSize,
  onlineAdagrad,
  onlineGradientDescent,
  onlineNewtonStep,
  onlineToBatch,
  projectOnto,
  regretTrace,
  weightedMajority,
  type OnlineLoss,
} from 'aifn-compute/optim/online'

const flat = (t: Tensor) => Array.from(toFlat(t))
/** A [T, N] matrix of uniform losses in [0, 1]. */
const uniformLosses = (seed: number, T: number, N: number): number[][] =>
  toRows(uniform(stream(seed), 0, 1, { shape: [T, N] }) as Tensor) as number[][]
/** A linear loss fₜ(w) = ⟨gₜ, w⟩ from the rows of g. */
const linear =
  (g: number[][]): OnlineLoss =>
  (t, w) => ({ value: flat(w).reduce((s, v, i) => s + v * g[t - 1][i], 0), grad: g[t - 1] })

describe('hedge', () => {
  it('plays exp(−ηL) normalised (hand case)', () => {
    const s = trace(
      hedge(
        [
          [0, 1],
          [1, 0],
          [0, 1],
        ],
        { eta: Math.log(2) },
      ),
      undefined,
      3,
      { keep: 'all' },
    )
    expect(flat(s.steps[0].weights)).toEqual([0.5, 0.5])
    flat(s.steps[1].weights).forEach((v, i) => expect(v).toBeCloseTo([2 / 3, 1 / 3][i], 12))
    expect(s.steps[1].learnerLoss).toBeCloseTo(0.5, 12)
    expect(s.final.learnerLoss).toBeCloseTo(0.5 + (2 / 3) * 1 + 0.5 * 1, 12)
    expect(s.final.regret).toBeCloseTo(s.final.learnerLoss - 1, 12)
    expect(s.final.done).toBe(true)
  })

  it('keeps its regret under ln N/η + ηT/8 for every seed and rate', () => {
    for (const seed of [1, 2, 3]) {
      const L = uniformLosses(seed, 400, 10)
      for (const eta of [0.02, 0.1, 0.5, 2]) {
        const s = run(hedge(L, { eta }), undefined, 1000)
        expect(s.t).toBe(400)
        expect(s.regret).toBeLessThanOrEqual(hedgeRegretBound(400, 10, eta) + 1e-9)
      }
      expect(run(hedge(L), undefined, 1000).regret).toBeLessThanOrEqual(hedgeRegretBound(400, 10))
      expect(run(hedge(L, { eta: 'anytime' }), undefined, 1000).regret).toBeLessThanOrEqual(
        hedgeRegretBound(400, 10, 'anytime'),
      )
    }
  })

  it('beats an adaptive adversary that punishes the leader; follow-the-leader does not', () => {
    // Loss 1 to the expert with most weight (the leader), 0 to the rest: FTL pays every round.
    const adversary = (_: number, p: Tensor) => {
      const w = flat(p)
      const k = w.indexOf(Math.max(...w))
      return w.map((_, i) => (i === k ? 1 : 0))
    }
    const T = 500
    const tuned = run(hedge(adversary, { experts: 2, rounds: T, eta: hedgeTunedRate(T, 2) }), undefined, T)
    expect(tuned.regret).toBeLessThanOrEqual(hedgeRegretBound(T, 2) + 1e-9)
    const ftl = run(hedge(adversary, { experts: 2, rounds: T, eta: 1e6 }), undefined, T)
    // FTL pays ½ on the tied rounds and 1 on the others while the best expert pays ½ per round: regret T/4.
    expect(ftl.regret).toBeGreaterThan(0.24 * T)
  })
})

describe('fixedShare', () => {
  it('is Hedge with α = 0, and keeps every weight at least α/N', () => {
    const L = uniformLosses(4, 100, 5)
    const a = run(hedge(L, { eta: 0.3 }), undefined, 200)
    const b = run(fixedShare(L, { eta: 0.3, alpha: 0 }), undefined, 200)
    flat(b.weights).forEach((v, i) => expect(v).toBeCloseTo(flat(a.weights)[i], 12))
    const tr = trace(fixedShare(L, { eta: 2, alpha: 0.1 }), undefined, 200, { keep: 'all' })
    for (const s of tr.steps) for (const w of flat(s.weights)) expect(w).toBeGreaterThanOrEqual(0.1 / 5 - 1e-15)
  })

  it('tracks a switching best expert better than Hedge', () => {
    // Expert 0 is best for 300 rounds, then expert 1 for 300.
    const T = 600
    const L = Array.from({ length: T }, (_, t) => (t < 300 ? [0.1, 0.9, 0.5] : [0.9, 0.1, 0.5]))
    const switching = flat(bestSwitchingLoss(L, 1))[1]
    const h = run(hedge(L, { eta: 1 }), undefined, T)
    const f = run(fixedShare(L, { eta: 1, alpha: 1 / T }), undefined, T)
    expect(f.learnerLoss - switching).toBeLessThan(0.3 * (h.learnerLoss - switching))
  })
})

describe('weightedMajority', () => {
  it('halving makes at most log₂ N mistakes with a perfect expert', () => {
    const N = 16
    const T = 60
    const bits = toRows(uniform(stream(5), 0, 1, { shape: [T, N] }) as Tensor) as number[][]
    const advice = bits.map((r) => r.map((u) => (u < 0.5 ? 1 : 0)))
    const outcomes = advice.map((r) => r[7])
    const s = run(weightedMajority(advice, outcomes, { beta: 0 }), undefined, T)
    expect(s.mistakes).toBeLessThanOrEqual(Math.log2(N))
    expect(s.bestMistakes).toBe(0)
  })

  it('stays under its mistake bound, deterministic and randomised', () => {
    for (const seed of [6, 7]) {
      const T = 300
      const N = 8
      const u = toRows(uniform(stream(seed), 0, 1, { shape: [T, N + 1] }) as Tensor) as number[][]
      const outcomes = u.map((r) => (r[N] < 0.5 ? 1 : 0))
      // Expert i is right with probability 0.55 + 0.04 i.
      const advice = u.map((r, t) => r.slice(0, N).map((v, i) => (v < 0.55 + 0.04 * i ? outcomes[t] : 1 - outcomes[t])))
      for (const randomised of [false, true])
        for (const beta of [0.3, 0.5, 0.8]) {
          const s = run(weightedMajority(advice, outcomes, { beta, randomised }), undefined, T)
          expect(s.mistakes).toBeLessThanOrEqual(s.bound + 1e-9)
        }
    }
  })
})

describe('online convex optimisation', () => {
  const T = 400
  const d = 3
  const G = toRows(uniform(stream(8), -1, 1, { shape: [T, d] }) as Tensor) as number[][]
  // The best fixed point of Σ⟨gₜ, u⟩ on the ball of radius r is −r S/‖S‖ with S = Σ gₜ.
  const bestOnBall = (r: number) => {
    const S = G.reduce((a, g) => a.map((v, i) => v + g[i]), [0, 0, 0])
    const n = Math.hypot(...S)
    return S.map((v) => (-r * v) / n)
  }
  const Gmax = Math.max(...G.map((g) => Math.hypot(...g)))

  it('OGD with ηₜ = D/(G√t) stays under (3/2)DG√T', () => {
    const r = 2
    const u = bestOnBall(r)
    const s = run(
      onlineGradientDescent(linear(G), {
        dim: d,
        domain: { kind: 'ball', radius: r },
        stepSize: ogdStepSize(2 * r, Gmax),
        comparator: u,
        rounds: T,
      }),
      undefined,
      T,
    )
    expect(s.regret).toBeLessThanOrEqual(ogdRegretBound(T, 2 * r, Gmax))
    expect(Math.hypot(...flat(s.w))).toBeLessThanOrEqual(r + 1e-12)
  })

  it('FTRL stays under ‖u‖²/(2η) + η Σ‖gₜ‖², and its first step is −ηg₁', () => {
    const r = 2
    const u = bestOnBall(r)
    const sumSq = G.reduce((a, g) => a + g.reduce((b, v) => b + v * v, 0), 0)
    const eta = r / Math.sqrt(2 * sumSq)
    const tr = trace(
      followTheRegularisedLeader(linear(G), { dim: d, eta, domain: { kind: 'ball', radius: r }, comparator: u }),
      undefined,
      T,
      { keep: 'all' },
    )
    flat(tr.steps[1].w).forEach((v, i) => expect(v).toBeCloseTo(-eta * G[0][i], 12))
    expect(tr.final.regret).toBeLessThanOrEqual((r * r) / (2 * eta) + eta * sumSq)
  })

  it('FTRL with L1 sets every coordinate with |Σg| ≤ λ₁ exactly to zero', () => {
    const g = Array.from({ length: 50 }, (_, t) => [1, 0.01 * (t % 2 ? 1 : -1), -0.5])
    const s = run(followTheRegularisedLeader(linear(g), { dim: 3, eta: 0.1, l1: 1 }), undefined, 50)
    const w = flat(s.w)
    expect(w[1]).toBe(0)
    expect(w[0]).toBeCloseTo(-0.1 * (50 - 1), 12)
    expect(w[2]).toBeCloseTo(0.1 * (25 - 1), 12)
  })

  it('adaptive FTRL-Proximal and AdaGrad stay under the AdaGrad bound on a box', () => {
    const Dinf = 2
    const sparse = G.map((g, t) => g.map((v, i) => (i === 2 && t % 10 !== 0 ? 0 : v)))
    const S = sparse.reduce((a, g) => a.map((v, i) => v + g[i]), [0, 0, 0])
    const u = S.map((v) => (v > 0 ? -1 : 1))
    const bound = Math.SQRT2 * Dinf * [0, 1, 2].reduce((a, i) => a + Math.hypot(...sparse.map((g) => g[i])), 0)
    const box = { kind: 'box' as const, lower: -1, upper: 1 }
    const ada = run(
      onlineAdagrad(linear(sparse), { dim: d, domain: box, eta: Dinf / Math.SQRT2, comparator: u }),
      undefined,
      T,
    )
    expect(ada.regret).toBeLessThanOrEqual(bound)
    const prox = run(
      followTheRegularisedLeader(linear(sparse), {
        dim: d,
        domain: box,
        adaptive: true,
        eta: 1,
        beta: 0,
        comparator: u,
      }),
      undefined,
      T,
    )
    expect(prox.regret).toBeLessThanOrEqual(bound)
  })

  it('the online Newton step has logarithmic regret on squared loss (exp-concave on a ball)', () => {
    // fₜ(w) = (⟨xₜ, w⟩ − yₜ)² with ‖x‖ ≤ 1, y = ⟨x, w⋆⟩ + noise, w⋆ inside the ball.
    const wStar = [0.3, -0.2, 0.1]
    const make = (n: number) => {
      const x = toRows(uniform(child(stream(9), 'x'), -0.57, 0.57, { shape: [n, 3] }) as Tensor) as number[][]
      const e = flat(normals(child(stream(9), 'e'), n, 0, 0.1) as Tensor)
      const y = x.map((r, t) => r.reduce((s, v, i) => s + v * wStar[i], 0) + e[t])
      const loss: OnlineLoss = (t, w) => {
        const r = x[t - 1].reduce((s, v, i) => s + v * flat(w)[i], 0) - y[t - 1]
        return { value: r * r, grad: x[t - 1].map((v) => 2 * r * v) }
      }
      return loss
    }
    const regretAt = (n: number) =>
      run(
        onlineNewtonStep(make(n), {
          dim: 3,
          domain: { kind: 'ball', radius: 1 },
          gamma: 0.25,
          epsilon: 1,
          comparator: wStar,
        }),
        undefined,
        n,
      ).regret
    const r1 = regretAt(500)
    const r2 = regretAt(5000)
    // √T growth would multiply the regret by 3.2; logarithmic growth by about ln 5000/ln 500 = 1.37.
    expect(r2).toBeLessThan(2 * Math.max(r1, 1))
  })

  it('projects onto balls, boxes and the simplex', () => {
    flat(projectOnto([3, 4], { kind: 'ball', radius: 1 })).forEach((v, i) => expect(v).toBeCloseTo([0.6, 0.8][i], 15))
    expect(flat(projectOnto([3, -4], { kind: 'box', lower: -1, upper: 2 }))).toEqual([2, -1])
    const p = flat(projectOnto([0.5, 0.9, -0.2], { kind: 'simplex' }))
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    p.forEach((v, i) => expect(v).toBeCloseTo([0.3, 0.7, 0][i], 12))
  })
})

describe('regret bookkeeping', () => {
  it('regretTrace against experts uses the best expert so far', () => {
    const r = regretTrace([0.5, 0.5, 0.5], {
      experts: [
        [0, 1],
        [1, 0],
        [1, 0],
      ],
    })
    expect(flat(r.comparator)).toEqual([0, 1, 1])
    expect(flat(r.regret)).toEqual([0.5, 0, 0.5])
    expect(flat(regretTrace([1, 2], { comparator: [0.5, 0.5] }).regret)).toEqual([0.5, 2])
  })

  it('bestSwitchingLoss matches brute force over every expert sequence', () => {
    const L = uniformLosses(10, 7, 3)
    const best = [Infinity, Infinity, Infinity]
    const total = 3 ** 7
    for (let code = 0; code < total; code++) {
      const seq = Array.from({ length: 7 }, (_, t) => Math.floor(code / 3 ** t) % 3)
      const switches = seq.slice(1).filter((e, t) => e !== seq[t]).length
      const loss = seq.reduce((a, e, t) => a + L[t][e], 0)
      for (let m = switches; m < 3; m++) best[m] = Math.min(best[m], loss)
    }
    flat(bestSwitchingLoss(L, 2)).forEach((v, m) => expect(v).toBeCloseTo(best[m], 12))
  })

  it('onlineToBatch averages the iterates after the burn-in', () => {
    expect(
      flat(
        onlineToBatch([
          [0, 0],
          [2, 4],
          [4, 8],
        ]),
      ),
    ).toEqual([2, 4])
    expect(
      flat(
        onlineToBatch(
          [
            [0, 0],
            [2, 4],
            [4, 8],
          ],
          { from: 1 },
        ),
      ),
    ).toEqual([3, 6])
  })
})

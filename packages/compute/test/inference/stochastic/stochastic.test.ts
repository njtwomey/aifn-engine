import {
  bivariateGaussianConditionals,
  effectiveSampleSize,
  gibbs,
  hmc,
  independenceMetropolis,
  integratedAutocorrelationTime,
  leapfrog,
  mala,
  metropolisHastings,
  monteCarloStandardError,
  nuts,
  particleFilter,
  randomWalkMetropolis,
  resample,
  resamplingSchemes,
  sampleChains,
  sgld,
  sliceSampler,
  splitRhat,
  summarise,
  temperedSmc,
  unadjustedLangevin,
  type ChainStart,
  type HmcState,
  type NutsState,
  type LogDensity,
} from 'aifn-compute/inference/stochastic'
import { autocorrelation, importanceEffectiveSampleSize } from 'aifn-compute/probability/stats'
import { child, normal, normals, stream } from 'aifn-compute/foundation/random'
import { banana, funnel, gaussianTarget } from './targets'
import { mul, square, sum, toFlat, type Tensor, type Value, type Vector } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import { run, trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { describe, expect, it } from 'vitest'
import { fixture } from '../../fixtures'
import { checkProtocol } from '../../protocol'

const std2 = gaussianTarget(
  [0, 0],
  [
    [1, 0.5],
    [0.5, 1],
  ],
)

/** Mean and variance of column k of an m×n×d draws tensor, pooled. */
function moments(draws: Tensor, k: number) {
  const [m, n, d] = draws.shape
  const x = toFlat(draws)
  let s = 0
  let s2 = 0
  for (let i = 0; i < m * n; i++) {
    s += x[i * d + k]
    s2 += x[i * d + k] ** 2
  }
  const mean = s / (m * n)
  return { mean, variance: s2 / (m * n) - mean * mean }
}

describe('samplers recover known moments', () => {
  const start: ChainStart = { x0: [0.5, -0.5] }
  it.each([
    ['random-walk', randomWalkMetropolis(std2, { scale: 1.2 }), 3000],
    ['hmc', hmc(std2, { stepSize: 0.3, steps: 8 }), 800],
    ['nuts', nuts(std2, { stepSize: 0.4 }), 800],
    ['nuts (slice)', nuts(std2, { stepSize: 0.4, variant: 'slice' }), 800],
    ['mala', mala(std2, { stepSize: 0.4 }), 3000],
    ['slice', sliceSampler(std2, { width: 2 }), 1500],
    ['gibbs', gibbs(bivariateGaussianConditionals(0.5)), 1500],
  ] as [string, Algorithm<ChainStart, { x: Vector } & Status>, number][])('%s', (_, alg, steps) => {
    const { draws } = sampleChains(alg, start, { chains: 4, steps, stream: stream(11), warmup: 100 })
    for (const k of [0, 1]) {
      const { mean, variance } = moments(draws, k)
      expect(Math.abs(mean)).toBeLessThan(0.12)
      expect(Math.abs(variance - 1)).toBeLessThan(0.15)
    }
    expect(Math.max(...toFlat(splitRhat(draws) as Tensor))).toBeLessThan(1.05)
  })

  it('banana moments under HMC', () => {
    const t = banana({ a: 1, b: 0.5 })
    const { draws } = sampleChains(
      hmc(t, { stepSize: 0.2, steps: 15 }),
      { x0: [0, 0] },
      {
        chains: 4,
        steps: 1500,
        stream: stream(3),
        warmup: 100,
      },
    )
    expect(Math.abs(moments(draws, 0).mean)).toBeLessThan(0.1)
    expect(moments(draws, 1).variance).toBeCloseTo(1.5, 0)
  })

  it('independence sampler with a wide Gaussian proposal', () => {
    const target = gaussianTarget([1], [[0.25]])
    const alg = independenceMetropolis(target, {
      sample: (s) => [normal(s, 0, 2)],
      logDensity: (x) => -0.5 * (toFlat(x)[0] / 2) ** 2,
    })
    const { draws } = sampleChains(alg, { x0: [0] }, { chains: 2, steps: 4000, stream: stream(2) })
    expect(moments(draws, 0).mean).toBeCloseTo(1, 1)
    expect(moments(draws, 0).variance).toBeCloseTo(0.25, 1)
  })

  it('ULA is biased by O(h) and MALA is not', () => {
    const t = gaussianTarget([0], [[1]])
    const h = 0.5
    const run2 = (alg: Algorithm<ChainStart, { x: Vector } & Status>) =>
      moments(sampleChains(alg, { x0: [0] }, { chains: 4, steps: 5000, stream: stream(8), warmup: 100 }).draws, 0)
    // ULA's stationary variance is σ²/(1 − h/(2σ²)) = 4/3.
    expect(run2(unadjustedLangevin(t, { stepSize: h })).variance).toBeCloseTo(4 / 3, 1)
    expect(run2(mala(t, { stepSize: h })).variance).toBeCloseTo(1, 1)
  })

  it('SGLD samples a Gaussian-mean posterior', () => {
    // y_i ~ N(θ, 1), prior N(0, 10²): posterior N(Σy/(N + 0.01), 1/(N + 0.01)).
    const ys = toFlat(normals(stream('data'), 100, 1, 1))
    const N = ys.length
    const post = ys.reduce((a, b) => a + b, 0) / (N + 0.01)
    const alg = sgld(
      {
        dim: 1,
        size: N,
        gradLogPrior: (th) => [-toFlat(th)[0] / 100],
        gradLogLikelihood: (th, i) => [ys[i] - toFlat(th)[0]],
      },
      { batchSize: 20, stepSize: 2e-3 },
    )
    const { draws } = sampleChains(alg, { x0: [0] }, { chains: 2, steps: 4000, stream: stream(4), warmup: 500 })
    expect(moments(draws, 0).mean).toBeCloseTo(post, 1)
    expect(moments(draws, 0).variance).toBeGreaterThan(0.005)
    expect(moments(draws, 0).variance).toBeLessThan(0.03)
  })

  it('autodiff supplies the gradient when the target has none', () => {
    const t: LogDensity = {
      kind: 'log-density',
      dim: 2,
      normalised: false,
      logDensity: (x) => mul(-0.5, sum(square(x as Value))),
    }
    const withGrad = hmc(t, { stepSize: 0.2, steps: 5 })
    const s = run(withGrad, { x0: [1, 1] }, 3, { stream: stream(1) })
    const ref = run(
      hmc({ ...t, grad: (x) => toFlat(x).map((v) => -v) }, { stepSize: 0.2, steps: 5 }),
      { x0: [1, 1] },
      3,
      {
        stream: stream(1),
      },
    )
    expect(toFlat(s.x)).toEqual(toFlat(ref.x).map((v) => expect.closeTo(v, 12)) as unknown as number[])
  })
})

describe('internals', () => {
  it('leapfrog conserves energy approximately and is reversible', () => {
    const lf = leapfrog(std2, [1, 0], [0.3, -0.7], { stepSize: 0.1, steps: 50 })
    const H = toFlat(lf.energies)
    expect(Math.max(...H) - Math.min(...H)).toBeLessThan(0.01)
    const P = toFlat(lf.positions)
    const M = toFlat(lf.momenta)
    const end = P.slice(-2)
    const back = leapfrog(
      std2,
      end,
      M.slice(-2).map((v) => -v),
      { stepSize: 0.1, steps: 50 },
    )
    const B = toFlat(back.positions)
    expect(B.slice(-2)[0]).toBeCloseTo(1, 10)
    expect(B.slice(-2)[1]).toBeCloseTo(0, 10)
  })

  it('HMC with a huge step flags divergences on the funnel', () => {
    const tr = trace(hmc(funnel(), { stepSize: 1.5, steps: 20 }), { x0: [-2, 0.1] }, 50, { stream: stream(1) })
    expect(tr.steps.at(-1)!.divergentCount).toBeGreaterThan(0)
  })

  it('divergent and rejected proposals are counted apart; a relative threshold scales with the dimension', () => {
    const run50 = (o: Parameters<typeof hmc>[1]) =>
      trace(hmc(funnel(), { stepSize: 0.6, steps: 20, ...o }), { x0: [-2, 0.1] }, 60, { stream: stream(3) })
    const loose = run50({})
    const tight = run50({ divergenceThreshold: { relative: 2 } })
    expect(loose.final.divergenceLimit).toBe(1000)
    expect(tight.final.divergenceLimit).toBe(4)
    for (const tr of [loose, tight])
      tr.steps.forEach((st) => {
        expect(st.acceptedCount + st.rejectedCount + st.divergentCount).toBeGreaterThanOrEqual(st.t)
        // Every step is exactly one of: accepted, an ordinary rejection, or a divergence (accepted or not).
        const accDiv = tr.steps.filter((u, k) => k > 0 && k <= st.t && u.accepted && u.divergent).length
        expect(st.acceptedCount + st.rejectedCount + st.divergentCount - accDiv).toBe(st.t)
      })
    // Same draws: the tighter limit flags at least as many, and turns ordinary rejections into divergences.
    expect(tight.final.divergentCount).toBeGreaterThanOrEqual(loose.final.divergentCount)
    expect(tight.final.divergentCount).toBeGreaterThan(0)
    expect(tight.final.rejectedCount).toBeLessThanOrEqual(loose.final.rejectedCount)
    expect(() => hmc(funnel(), { divergenceThreshold: { relative: 0 } })).toThrow(/positive/)
    const n = run(nuts(funnel(), { stepSize: 0.6, divergenceThreshold: { relative: 2 } }), { x0: [-2, 0.1] }, 30, {
      stream: stream(4),
    })
    expect(n.divergenceLimit).toBe(4)
    expect(n.acceptedCount + n.rejectedCount + n.divergentCount).toBeGreaterThanOrEqual(30)
  })

  it('NUTS, multinomial and slice, recovers the moments of a correlated, badly scaled Gaussian', () => {
    // Σ with standard deviations 1, 3, 0.5 and correlation 0.6 between the first two.
    const sd = [1, 3, 0.5]
    const cov = [
      [1, 0.6 * 3, 0],
      [0.6 * 3, 9, 0],
      [0, 0, 0.25],
    ]
    const target = gaussianTarget([1, -2, 0.5], cov)
    for (const variant of ['multinomial', 'slice'] as const) {
      const { draws } = sampleChains(
        nuts(target, { stepSize: 0.3, variant }),
        { x0: [0, 0, 0] },
        { chains: 4, steps: 1000, stream: stream(21), warmup: 100 },
      )
      const [m, n] = draws.shape
      const x = toFlat(draws)
      ;[1, -2, 0.5].forEach((mu, k) => {
        const { mean, variance } = moments(draws, k)
        expect(Math.abs(mean - mu), `${variant} mean ${k}`).toBeLessThan(0.15 * sd[k])
        expect(Math.abs(variance / sd[k] ** 2 - 1), `${variant} variance ${k}`).toBeLessThan(0.12)
      })
      const m0 = moments(draws, 0).mean
      const m1 = moments(draws, 1).mean
      let c = 0
      for (let i = 0; i < m * n; i++) c += (x[i * 3] - m0) * (x[i * 3 + 1] - m1)
      expect(Math.abs(c / (m * n) / 3 - 0.6), `${variant} correlation`).toBeLessThan(0.06)
    }
  })

  it('NUTS acceptance statistic is the mean of min(1, exp(H₀ − H)) over the tree, in both variants', () => {
    for (const variant of ['multinomial', 'slice'] as const) {
      const tr = trace(nuts(std2, { stepSize: 0.5, variant }), { x0: [0.3, 0] }, 30, { stream: stream(5) })
      for (const st of tr.steps.slice(1)) {
        const times = toFlat(st.trajectoryTimes)
        const H = toFlat(st.energies)
        const H0 = H[times.indexOf(0)]
        let sum = 0
        for (let i = 0; i < H.length; i++) if (times[i] !== 0) sum += Math.min(1, Math.exp(H0 - H[i]))
        expect(st.acceptStat).toBeCloseTo(sum / st.leapfrogSteps, 12)
        expect(st.leapfrogSteps).toBe(H.length - 1)
      }
    }
    expect(nuts(std2).name).toBe('nuts')
    expect(nuts(std2, { variant: 'slice' }).name).toBe('nuts-slice')
  })

  it('NUTS builds trees and stops on a U-turn', () => {
    const s = run(nuts(std2, { stepSize: 0.2 }), { x0: [0, 0] }, 20, { stream: stream(2) })
    expect(s.treeDepth).toBeGreaterThan(0)
    expect(s.trajectory.shape[0]).toBe(s.leapfrogSteps + 1)
    expect(s.hitMaxDepth).toBe(false)
  })

  describe('dual-averaging step-size adaptation', () => {
    const std5 = gaussianTarget(
      [0, 0, 0, 0, 0],
      [0, 1, 2, 3, 4].map((i) => [0, 1, 2, 3, 4].map((j) => (i === j ? [1, 2, 0.5, 1, 3][i] : 0))),
    )
    const mean = (v: readonly number[]) => v.reduce((a, b) => a + b, 0) / v.length

    it.each([
      ['nuts', nuts(std5, { stepSize: 2, adapt: { warmup: 300 } })],
      ['nuts-slice', nuts(std5, { stepSize: 2, adapt: { warmup: 300 }, variant: 'slice' })],
      ['hmc', hmc(std5, { stepSize: 2, steps: 10, adapt: { warmup: 300, targetAcceptance: 0.7 } })],
    ] as const)('%s reaches the target acceptance and freezes ε at ε̄ after warmup', (name, alg) => {
      const tr = trace(alg as Algorithm<ChainStart, NutsState | HmcState>, { x0: [1, 1, 1, 1, 1] }, 1000, {
        stream: stream(3),
      })
      const states = tr.steps
      const target = name === 'hmc' ? 0.7 : 0.8
      // Warmup moves ε away from the poor start (2); afterwards one ε is used throughout and equals ε̄ at warmup's end.
      const after = states.slice(301)
      const eps = after[0].stepSize
      expect(eps).toBeLessThan(1.5)
      expect(after.every((st) => st.stepSize === eps && !st.adapting)).toBe(true)
      expect(eps).toBe(states[300].stepSizeBar)
      expect(states[299].adapting).toBe(true)
      // Dual averaging drives the mean acceptance statistic over warmup to δ; the averaged ε̄ is a little more cautious
      // than the late iterates, so acceptance after warmup sits at or somewhat above δ.
      const late = states.slice(151, 301).map((st) => st.acceptStat)
      expect(Math.abs(mean(late) - target)).toBeLessThan(0.08)
      const kept = mean(after.map((st) => st.acceptStat))
      expect(kept).toBeGreaterThan(target - 0.05)
      expect(kept).toBeLessThan(target + 0.15)
    })

    it('is off by default and checks its options', () => {
      const s = run(nuts(std2, { stepSize: 0.3 }), { x0: [0, 0] }, 20, { stream: stream(2) })
      expect(s.stepSize).toBe(0.3)
      expect(s.nextStepSize).toBe(0.3)
      expect(s.adapting).toBe(false)
      expect(s.acceptStat).toBeGreaterThan(0)
      expect(s.acceptStat).toBeLessThanOrEqual(1)
      expect(() => nuts(std2, { adapt: { warmup: 10, targetAcceptance: 1 } })).toThrow(/targetAcceptance/)
      expect(() => hmc(std2, { adapt: { warmup: -1 } })).toThrow(/warmup/)
      expect(() => nuts(std2, { adapt: { warmup: 10, kappa: 0.5 } })).toThrow(/kappa/)
    })

    it('follows the dual-averaging recursion of Hoffman and Gelman (2014, eq. 6)', () => {
      // Replay the recursion from the recorded acceptance statistics.
      const tr = trace(
        nuts(std2, { stepSize: 1, adapt: { warmup: 20, gamma: 0.05, t0: 10, kappa: 0.75 } }),
        { x0: [0, 0] },
        25,
        {
          stream: stream(9),
        },
      )
      const mu = Math.log(10)
      let hBar = 0
      let logBar = 0
      let next = 1
      for (let m = 1; m <= 25; m++) {
        const st = tr.steps[m]
        expect(st.stepSize).toBeCloseTo(next, 12)
        if (m <= 20) {
          hBar = (1 - 1 / (m + 10)) * hBar + (0.8 - st.acceptStat) / (m + 10)
          const logEps = mu - (Math.sqrt(m) / 0.05) * hBar
          logBar = m ** -0.75 * logEps + (1 - m ** -0.75) * logBar
          next = m < 20 ? Math.exp(logEps) : Math.exp(logBar)
        }
        expect(st.hBar).toBeCloseTo(hBar, 12)
        expect(st.nextStepSize).toBeCloseTo(next, 12)
      }
    })
  })

  it('Gibbs moves are axis-parallel', () => {
    const s = run(gibbs(bivariateGaussianConditionals(0.9)), { x0: [2, -2] }, 3, { stream: stream(1) })
    const m = toFlat(s.moves)
    expect(m[2 + 1]).toBe(m[1]) // first update changes x₀ only
    expect(m[4]).toBe(m[2]) // second changes x₁ only
  })

  it('random-walk acceptance falls as the scale grows', () => {
    const rate = (scale: number) =>
      run(randomWalkMetropolis(banana(), { scale }), { x0: [0, 0] }, 2000, { stream: stream(1) }).acceptanceRate
    expect(rate(0.1)).toBeGreaterThan(rate(1))
    expect(rate(1)).toBeGreaterThan(rate(5))
  })
})

describe('protocol', () => {
  const algs: [string, Algorithm<ChainStart, { x: Vector } & Status>][] = [
    ['rwm', randomWalkMetropolis(banana())],
    ['hmc', hmc(banana(), { stepSize: 0.1, steps: 5 })],
    ['nuts', nuts(banana())],
    ['nuts (slice)', nuts(banana(), { variant: 'slice' })],
    ['mala', mala(banana())],
    ['ula', unadjustedLangevin(banana())],
    ['slice', sliceSampler(banana())],
    ['gibbs', gibbs(bivariateGaussianConditionals(0.8), { scan: 'random' })],
    [
      'metropolis-hastings (Gaussian steps)',
      metropolisHastings(banana(), (x, st) => ({
        proposal: toFlat(x).map((v, i) => v + 0.5 * (toFlat(normals(st, 2))[i] as number)),
        logProposalRatio: 0,
      })),
    ],
    [
      'independence',
      independenceMetropolis(std2, { sample: (s) => [normal(s, 0, 2), normal(s, 0, 2)], logDensity: () => 0 }),
    ],
  ]
  it('sgld satisfies the Algorithm protocol', () => {
    const ys = [0.3, 1.1, -0.2, 0.8, 0.5, 1.4]
    const model = {
      dim: 1,
      size: ys.length,
      gradLogPrior: (th: Vector) => [-toFlat(th)[0] / 100],
      gradLogLikelihood: (th: Vector, i: number) => [ys[i] - toFlat(th)[0]],
    }
    checkProtocol(sgld(model, { batchSize: 2, stepSize: 1e-2 }), { x0: [0] }, { steps: 12, random: true })
  })

  it.each(algs)('%s satisfies the Algorithm protocol and draws from its streams', (_, alg) => {
    checkProtocol(alg, { x0: [0.2, 0.1] }, { steps: 20, random: true, record: { x0: (s) => toFlat(s.x)[0] } })
  })
})

describe('sequential Monte Carlo', () => {
  it.each(resamplingSchemes)('%s resampling is unbiased and sorted', (scheme) => {
    const w = [0.1, 0.4, 0.05, 0.45]
    const counts = [0, 0, 0, 0]
    const R = 2000
    for (let r = 0; r < R; r++) for (const i of toFlat(resample(stream(r), w, scheme, 10))) counts[i]++
    counts.forEach((c, i) => expect(c / (10 * R)).toBeCloseTo(w[i], 2))
    const one = toFlat(resample(stream(1), w, scheme, 10))
    expect([...one].sort((a, b) => a - b)).toEqual(one)
    expect(resample(stream(1), w, scheme).dtype).toBe('int32')
  })

  it('systematic resampling keeps ⌊Nw⌋ copies', () => {
    const c = toFlat(resample(stream(3), [0.5, 0.25, 0.25], 'systematic', 8))
    expect(c.filter((i) => i === 0).length).toBe(4)
  })

  it('ESS from log-weights', () => {
    expect(importanceEffectiveSampleSize([0, 0, 0, 0], { log: true })).toBeCloseTo(4, 12)
    expect(importanceEffectiveSampleSize([0, -1000, -1000], { log: true })).toBeCloseTo(1, 12)
  })

  it('particle filter on a linear-Gaussian model matches the Kalman filter', () => {
    // xₜ = 0.9xₜ₋₁ + N(0, 0.5²), x₀ ~ N(0, 1), yₜ = xₜ + N(0, 0.4²).
    const ys: number[] = []
    let x = normal(stream('x0'), 0, 1)
    for (let t = 0; t < 30; t++) {
      if (t > 0) x = 0.9 * x + normal(child(stream('tr'), t), 0, 0.5)
      ys.push(x + normal(child(stream('ob'), t), 0, 0.4))
    }
    const model = {
      dim: 1,
      sampleInitial: (s: ReturnType<typeof stream>) => normal(s, 0, 1),
      sampleTransition: (v: Vector, _t: number, s: ReturnType<typeof stream>) => normal(s, 0.9 * toFlat(v)[0], 0.5),
      logObservation: (y: number, v: Vector) =>
        -0.5 * ((y - toFlat(v)[0]) / 0.4) ** 2 - Math.log(0.4 * Math.sqrt(2 * Math.PI)),
    }
    const tr = trace(particleFilter(model, { particles: 3000 }), { observations: ys }, 100, {
      stream: stream(1),
      record: { mean: (s) => s.mean },
    })
    expect(tr.meta.stopped).toBe('done')
    // Kalman filter reference.
    let m = 0
    let P = 1
    let logZ = 0
    const kf: number[] = []
    ys.forEach((y, t) => {
      if (t > 0) {
        m = 0.9 * m
        P = 0.81 * P + 0.25
      }
      const S = P + 0.16
      logZ += -0.5 * Math.log(2 * Math.PI * S) - (0.5 * (y - m) ** 2) / S
      const K = P / S
      m += K * (y - m)
      P *= 1 - K
      kf.push(m)
    })
    const means = toFlat(tr.series.mean).slice(1)
    means.forEach((v, t) => expect(Math.abs(v - kf[t])).toBeLessThan(0.08))
    expect(tr.steps.at(-1)!.logEvidence).toBeCloseTo(logZ, 0)
    checkProtocol(
      particleFilter(model, { particles: 50 }),
      { observations: ys.slice(0, 8) },
      { steps: 8, random: true },
    )
  })

  it('tempered SMC estimates the evidence of a conjugate Gaussian model', () => {
    // θ ~ N(0, 1), y ~ N(θ, 0.5²) with y = 1.2: Z = N(1.2 | 0, 1.25).
    const y = 1.2
    const s = run(
      temperedSmc(
        {
          dim: 1,
          samplePrior: (st) => normal(st, 0, 1),
          logPrior: (v) => -0.5 * toFlat(v)[0] ** 2 - 0.5 * Math.log(2 * Math.PI),
          logLikelihood: (v) => -0.5 * ((y - toFlat(v)[0]) / 0.5) ** 2 - Math.log(0.5 * Math.sqrt(2 * Math.PI)),
        },
        { particles: 2000 },
      ),
      undefined,
      50,
      { stream: stream(2) },
    )
    expect(s.beta).toBe(1)
    expect(s.logEvidence).toBeCloseTo(-0.5 * Math.log(2 * Math.PI * 1.25) - (0.5 * y * y) / 1.25, 1)
    const p = toFlat(s.particles)
    const mean = p.reduce((a, b) => a + b, 0) / p.length
    expect(mean).toBeCloseTo(y / 1.25, 1)
  })

  it('tempered SMC satisfies the Algorithm protocol', () => {
    const model = {
      dim: 1,
      samplePrior: (st: ReturnType<typeof stream>) => normal(st, 0, 1),
      logPrior: (v: Vector) => -0.5 * toFlat(v)[0] ** 2,
      logLikelihood: (v: Vector) => -0.5 * ((1.2 - toFlat(v)[0]) / 0.5) ** 2,
    }
    checkProtocol(temperedSmc(model, { particles: 100 }), undefined, { steps: 6, random: true })
  })
})

describe('diagnostics against ArviZ', () => {
  type Case = {
    draws: number[][]
    ess_bulk: number
    ess_mean: number
    ess_tail: number
    iact: number
    rhat_rank: number
    rhat_split: number
    rhat_basic: number
    mcse_mean: number
    mcse_quantile: number[]
    acf: number[]
  }
  const cases = fixture<Record<string, Case>>('inference/stochastic')
  it.each(Object.entries(cases))('%s', (_, c) => {
    const x = c.draws
    expect(effectiveSampleSize(x)).toBeCloseTo(c.ess_bulk, 8)
    expect(effectiveSampleSize(x, { method: 'mean' })).toBeCloseTo(c.ess_mean, 8)
    expect(effectiveSampleSize(x, { method: 'tail' })).toBeCloseTo(c.ess_tail, 8)
    expect(integratedAutocorrelationTime(x)).toBeCloseTo(c.iact, 8)
    expect(splitRhat(x)).toBeCloseTo(c.rhat_rank, 10)
    expect(splitRhat(x, { method: 'split' })).toBeCloseTo(c.rhat_split, 10)
    if (x.length > 1) expect(splitRhat(x, { method: 'basic' })).toBeCloseTo(c.rhat_basic, 10)
    expect(monteCarloStandardError(x)).toBeCloseTo(c.mcse_mean, 10)
    ;[0.05, 0.5, 0.95].forEach((p, k) =>
      expect(monteCarloStandardError(x, { quantile: p })).toBeCloseTo(c.mcse_quantile[k], 10),
    )
    const acf = toFlat(autocorrelation(x[0], { maxLag: c.acf.length - 1 }))
    expect(summarise(x).rhat).toBeCloseTo(c.rhat_rank, 10)
    acf.forEach((v, k) => expect(v).toBeCloseTo(c.acf[k], 10))
  })

  it('per-parameter diagnostics for m×n×d draws, and edge cases', () => {
    const { draws } = sampleChains(
      randomWalkMetropolis(std2),
      { x0: [0, 0] },
      { chains: 2, steps: 200, stream: stream(1) },
    )
    expect(effectiveSampleSize(draws)).toHaveProperty('shape', [2])
    expect(effectiveSampleSize([1, 1, 1, 1, 1, 1, 1, 1])).toBe(8)
    expect(effectiveSampleSize([1, 2, 3])).toBeNaN()
    expect(summarise([toFlat(normals(stream(1), 400)), toFlat(normals(stream(2), 400))]).rhat).toBeLessThan(1.02)
  })

  it('shifted chains give a large R̂ and a small ESS', () => {
    const a = toFlat(normals(stream(1), 500))
    const b = toFlat(normals(stream(2), 500)).map((v) => v + 3)
    expect(splitRhat([a, b])).toBeGreaterThan(1.5)
    expect(effectiveSampleSize([a, b])).toBeLessThan(50)
  })
})

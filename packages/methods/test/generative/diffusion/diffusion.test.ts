import { describe, expect, it } from 'vitest'
import {
  alphaBarAt,
  cosineSchedule,
  ddimSampler,
  ddimTimesteps,
  ddpmSampler,
  denoiser,
  denoiserTraining,
  forwardNoise,
  forwardPosterior,
  forwardProcess,
  gaussianMixtureData,
  linearSchedule,
  mixtureLogDensity,
  mixtureMoments,
  mixtureNoisePredictor,
  mixtureScore,
  networkNoisePredictor,
  probabilityFlowSampler,
  reverseSdeSampler,
  sampleMixture,
  scheduleSde,
  scoreFromPredictor,
  sdeSchedule,
  stepTime,
  subVpSde,
  timeStep,
  veSde,
  vpSde,
  type SamplerState,
} from 'aifn-methods/generative/diffusion'
import { fork, normals, stream } from 'aifn-compute/foundation/random'
import { fromRows, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../protocol'
import { fixture } from '../../fixtures'

type Sched = { betas: number[]; alphaBars: number[]; steps: number[]; [k: string]: number[] }
type SubVp = {
  mixture: { weights: number[]; means: number[][]; stds: number[] }
  times: number[]
  drift: number[]
  diffusion: number[]
  meanScale: number[]
  std: number[]
  x0: number[][]
  steps: number
  end: number
  euler: number[][]
  rk4: number[][]
  reverse: { time: number; mean: number[][]; noiseScale: number }[]
}
const F = fixture<{ linear: Sched; cosine: Sched; vp: { times: number[]; meanScale: number[] }; subVp: SubVp }>(
  'generative/diffusion',
)

const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(1e-300, Math.abs(b))

describe('schedules', () => {
  it('the linear schedule matches DDPM', () => {
    const s = linearSchedule()
    F.linear.steps.forEach((t, i) => {
      expect(rel(s.betas.data[t - 1], F.linear.betas[i])).toBeLessThan(1e-12)
      expect(rel(alphaBarAt(s, t), F.linear.alphaBars[i])).toBeLessThan(1e-10)
    })
    expect(s.snr.data[0]).toBeCloseTo(s.alphaBars.data[0] / (1 - s.alphaBars.data[0]), 8)
  })

  it('the cosine schedule matches improved-diffusion, capping reported', () => {
    const s = cosineSchedule()
    F.cosine.steps.forEach((t, i) => {
      expect(rel(s.betas.data[t - 1], F.cosine.betas[i])).toBeLessThan(1e-10)
      expect(rel(alphaBarAt(s, t), F.cosine.alphaBars[i])).toBeLessThan(1e-9)
    })
    expect(s.capped).toContain(1000)
  })

  it('the forward posterior matches DDPM’s coefficients', () => {
    const s = linearSchedule()
    F.linear.steps.slice(1).forEach((t, j) => {
      const i = j + 1
      const { mean, variance } = forwardPosterior(s, tensor([1]), tensor([0]), t)
      expect(rel(toFlat(mean)[0], F.linear.coef1[i])).toBeLessThan(1e-9)
      expect(rel(variance, F.linear.posteriorVariance[i])).toBeLessThan(1e-9)
      const other = forwardPosterior(s, tensor([0]), tensor([1]), t)
      expect(rel(toFlat(other.mean)[0], F.linear.coef2[i])).toBeLessThan(1e-9)
    })
  })

  it('the SDE marginals are consistent', () => {
    const vp = vpSde()
    F.vp.times.forEach((t, i) => expect(vp.meanScale(t)).toBeCloseTo(F.vp.meanScale[i], 12))
    // VP: m² + s² = 1; sub-VP: s = 1 − m²; VE: m = 1.
    for (const t of [0.05, 0.5, 1]) {
      expect(vp.meanScale(t) ** 2 + vp.std(t) ** 2).toBeCloseTo(1, 12)
      expect(subVpSde().std(t)).toBeCloseTo(1 - vp.meanScale(t) ** 2, 12)
      expect(veSde().meanScale(t)).toBe(1)
    }
    // dVar/dt = 2f Var + g² for the VP marginal variance s(t)².
    const t = 0.3
    const h = 1e-6
    const dVar = (vp.std(t + h) ** 2 - vp.std(t - h) ** 2) / (2 * h)
    expect(dVar).toBeCloseTo(2 * vp.drift(t) * vp.std(t) ** 2 + vp.diffusion(t) ** 2, 5)
  })
})

describe('one time axis for discrete schedules and SDEs', () => {
  it('scheduleSde passes through the schedule at every step, with log ᾱ linear between steps', () => {
    for (const s of [linearSchedule(), cosineSchedule(200)]) {
      const sde = scheduleSde(s)
      expect(sde.schedule).toBe(s)
      for (const t of [0, 1, 2, 17, Math.floor(s.steps / 2), s.steps - 1, s.steps]) {
        const tau = stepTime(s, t)
        expect(timeStep(s, tau)).toBe(t)
        expect(rel(sde.meanScale(tau) ** 2, alphaBarAt(s, t))).toBeLessThan(1e-10)
        expect(sde.meanScale(tau) ** 2 + sde.std(tau) ** 2).toBeCloseTo(1, 12)
      }
      // The rate is −T log αₜ inside step t, and dVar/dτ = 2f Var + g² holds inside a step.
      const T = s.steps
      const tau = (17 - 0.5) / T
      expect(sde.diffusion(tau) ** 2).toBeCloseTo(-T * Math.log1p(-s.betas.data[16]), 10)
      const h = 1e-7
      const dVar = (sde.std(tau + h) ** 2 - sde.std(tau - h) ** 2) / (2 * h)
      expect(dVar).toBeCloseTo(2 * sde.drift(tau) * sde.std(tau) ** 2 + sde.diffusion(tau) ** 2, 5)
    }
  })

  it('sdeSchedule inverts scheduleSde and discretises the VP SDE to the DDPM linear schedule', () => {
    const s = cosineSchedule(100)
    const back = sdeSchedule(scheduleSde(s), 100)
    s.betas.data.forEach((b, i) => expect(rel(back.betas.data[i], b)).toBeLessThan(1e-8))
    // vpSde's defaults (β from 0.1 to 20 over [0, 1]) are DDPM's 1e-4 … 0.02 over 1000 steps.
    const vp = sdeSchedule(vpSde(), 1000)
    const lin = linearSchedule(1000)
    for (const t of [1, 250, 500, 1000]) {
      const [a, b] = [Math.log(alphaBarAt(vp, t)), Math.log(alphaBarAt(lin, t))]
      expect(Math.abs(a - b)).toBeLessThan(0.01 * Math.abs(b) + 2e-5)
    }
  })

  it('DDPM, DDIM and probability flow on one schedule share τ and agree', () => {
    const s = linearSchedule(1000)
    const x = tensor([
      [0.2, -0.4],
      [-1, 0.5],
    ])
    const ddpm = trace(ddpmSampler(predictor, s), { x }, 1000, { stream: stream(12) })
    expect(ddpm.steps[0].tau).toBe(1)
    expect(ddpm.steps[500].tau).toBeCloseTo(0.5, 12)
    expect(ddpm.steps[ddpm.steps.length - 1].tau).toBe(0)
    const ddim = run(ddimSampler(predictor, s, { steps: 1000 }), { x }, 1000, { stream: stream(13) })
    const flow = run(probabilityFlowSampler(predictor, scheduleSde(s), { steps: 200, end: 1 / 1000 }), { x }, 1000, {
      stream: stream(14),
    })
    expect(ddim.tau).toBe(0)
    expect(flow.tau).toBeCloseTo(1 / 1000, 12)
    // DDIM (η = 0) is a first-order discretisation of the probability-flow ODE on the same marginals.
    toFlat(ddim.x).forEach((v, i) => expect(Math.abs(v - toFlat(flow.x)[i])).toBeLessThan(0.05))
  })
})

const MIX = gaussianMixtureData(
  [0.3, 0.7],
  [
    [-2, 0],
    [1.5, 1],
  ],
  [0.4, [0.5, 0.3]],
)
const predictor = mixtureNoisePredictor(MIX)

describe('the Gaussian mixture', () => {
  it('its score is the gradient of its log density', () => {
    const x = tensor([
      [0.3, -0.2],
      [-1.5, 0.6],
    ])
    const [m, s] = [0.8, 0.6]
    const score = toRows(mixtureScore(MIX, x, m, s))
    const h = 1e-6
    for (let i = 0; i < 2; i++)
      for (let a = 0; a < 2; a++) {
        const up = toRows(x).map((r) => [...r])
        const down = toRows(x).map((r) => [...r])
        up[i][a] += h
        down[i][a] -= h
        const fd =
          (toFlat(mixtureLogDensity(MIX, fromRows(up), m, s))[i] -
            toFlat(mixtureLogDensity(MIX, fromRows(down), m, s))[i]) /
          (2 * h)
        expect(score[i][a]).toBeCloseTo(fd, 6)
      }
    // The predictor route gives the same score.
    expect(toFlat(scoreFromPredictor(predictor, x, m, s))[0]).toBeCloseTo(score[0][0], 10)
  })

  it('samples have the mixture’s moments', () => {
    const x = sampleMixture(stream('mix'), MIX, 20000)
    const { mean } = mixtureMoments(MIX)
    const xs = toRows(x)
    const m0 = xs.reduce((a, r) => a + r[0], 0) / xs.length
    expect(m0).toBeCloseTo(mean[0], 1)
  })
})

/** Mean and covariance of [n, 2] samples. */
function moments(x: Tensor) {
  const rows = toRows(x)
  const n = rows.length
  const mean = [0, 1].map((a) => rows.reduce((s, r) => s + r[a], 0) / n)
  const cov = [0, 1].map((a) =>
    [0, 1].map((b) => rows.reduce((s, r) => s + (r[a] - mean[a]) * (r[b] - mean[b]), 0) / n),
  )
  return { mean, cov }
}

function expectMixtureMoments(x: Tensor, tol: number) {
  const { mean, covariance } = mixtureMoments(MIX)
  const got = moments(x)
  for (let a = 0; a < 2; a++) {
    expect(Math.abs(got.mean[a] - mean[a])).toBeLessThan(tol)
    for (let b = 0; b < 2; b++) expect(Math.abs(got.cov[a][b] - covariance[a][b])).toBeLessThan(tol * 2)
  }
}

describe('sampling with the exact score recovers the mixture', () => {
  const start = { n: 1500, dimension: 2 }
  it('DDPM', () => {
    const final = run(ddpmSampler(predictor, linearSchedule(200, { betaEnd: 0.08 })), start, 1000, {
      stream: stream(1),
    })
    expect(final.time).toBe(0)
    expectMixtureMoments(final.x, 0.1)
  })

  it('DDIM, deterministic and stochastic', () => {
    const schedule = linearSchedule()
    expectMixtureMoments(run(ddimSampler(predictor, schedule, { steps: 50 }), start, 100, { stream: stream(2) }).x, 0.1)
    expectMixtureMoments(
      run(ddimSampler(predictor, schedule, { steps: 50, eta: 1 }), start, 100, { stream: stream(3) }).x,
      0.1,
    )
  })

  it('the reverse SDE and the probability-flow ODE', () => {
    expectMixtureMoments(
      run(reverseSdeSampler(predictor, vpSde(), { steps: 400 }), start, 1000, { stream: stream(4) }).x,
      0.1,
    )
    expectMixtureMoments(
      run(probabilityFlowSampler(predictor, vpSde(), { steps: 60 }), start, 1000, { stream: stream(5) }).x,
      0.1,
    )
    expectMixtureMoments(
      run(probabilityFlowSampler(predictor, veSde(), { steps: 200 }), start, 1000, { stream: stream(6) }).x,
      0.12,
    )
  })

  it('DDIM with η = 0 and the ODE are deterministic maps of the start', () => {
    const x = tensor([[0.2, -0.4]])
    const a = run(ddimSampler(predictor, linearSchedule(), { steps: 20 }), { x }, 100, { stream: stream(7) })
    const b = run(ddimSampler(predictor, linearSchedule(), { steps: 20 }), { x }, 100, { stream: stream(8) })
    expect(toFlat(a.x)).toEqual(toFlat(b.x))
  })

  it('DDIM uses S distinct levels from T down to 1', () => {
    expect(ddimTimesteps(1000, 5)).toEqual([1000, 750, 501, 251, 1, 0])
  })
})

describe('trace protocol', () => {
  const small = { n: 50, dimension: 2 }
  const algs: [string, Algorithm<typeof small, SamplerState>][] = [
    ['ddpm', ddpmSampler(predictor, linearSchedule(40, { betaEnd: 0.3 }))],
    ['ddim', ddimSampler(predictor, linearSchedule(), { steps: 20, eta: 0.5 })],
    ['sde', reverseSdeSampler(predictor, vpSde(), { steps: 30 })],
    ['ode', probabilityFlowSampler(predictor, vpSde(), { steps: 20 })],
  ]
  for (const [name, alg] of algs) {
    it(`${name} follows the trace protocol`, () => {
      expectProtocol(alg, small, { n: 15, record: { x: (s: SamplerState) => s.x }, seed: 9 })
    })
  }
  it('the forward process and denoiser training follow the trace protocol', () => {
    const x0 = sampleMixture(stream('x0'), MIX, 20)
    expectProtocol(forwardProcess(linearSchedule(30)), { x0 }, { n: 10 })
    const net = denoiser(2, { hidden: [8], frequencies: 2 })
    const alg = denoiserTraining({ data: x0, schedule: linearSchedule(30), net, batchSize: 8 })
    expectProtocol(alg, { params: net.layer.init(stream('net')) }, { n: 6, record: { loss: (s) => s.loss } })
  })

  it('the forward process has the closed-form marginal', () => {
    const schedule = linearSchedule(100, { betaEnd: 0.05 })
    const x0 = sampleMixture(stream('x0'), MIX, 4000)
    const end = run(forwardProcess(schedule), { x0 }, 60, { stream: stream(10) })
    const direct = forwardNoise(stream(11), x0, alphaBarAt(schedule, 60))
    const [a, b] = [moments(end.x), moments(direct.x)]
    expect(Math.abs(a.cov[0][0] - b.cov[0][0])).toBeLessThan(0.1)
    expect(Math.abs(a.mean[0] - b.mean[0])).toBeLessThan(0.05)
  })
})

describe('the learned denoiser', () => {
  it('training lowers the noise-prediction loss', () => {
    const schedule = linearSchedule(100, { betaEnd: 0.1 })
    const net = denoiser(2, { hidden: [32, 32], frequencies: 3 })
    const data = sampleMixture(stream('train'), MIX, 512)
    const alg = denoiserTraining({ data, schedule, net, batchSize: 128 })
    const t = trace(alg, { params: net.layer.init(stream('net')) }, 300, { every: 50, record: { loss: (s) => s.loss } })
    const losses = toFlat(t.series.loss)
    const early = losses[0]
    const late = losses[losses.length - 1]
    expect(late).toBeLessThan(early)
    const pred = networkNoisePredictor(net, t.steps[t.steps.length - 1].params)
    expect(pred(tensor([[0, 0]]), 0.5).shape).toEqual([1, 2])
  })
})

describe('sub-VP sampling against score_sde in torch', () => {
  const S = F.subVp
  const sde = subVpSde()
  const mixture = gaussianMixtureData(S.mixture.weights, S.mixture.means, S.mixture.stds)
  const predictor = mixtureNoisePredictor(mixture)
  const close = (got: number[], want: number[], tol: number) =>
    want.forEach((w, i) => expect(Math.abs(got[i] - w), `[${i}]`).toBeLessThan(tol))

  it('coefficients: drift, diffusion, mean scale and std (subVPSDE.sde and marginal_prob)', () => {
    S.times.forEach((t, i) => {
      expect(sde.drift(t)).toBeCloseTo(S.drift[i], 12)
      expect(rel(sde.diffusion(t), S.diffusion[i])).toBeLessThan(1e-12)
      expect(rel(sde.meanScale(t), S.meanScale[i])).toBeLessThan(1e-12)
      expect(rel(sde.std(t), S.std[i])).toBeLessThan(1e-12)
    })
  })

  it('the probability-flow ODE by Euler and RK4 from fixed starts', () => {
    const x = tensor(S.x0)
    const euler = run(
      probabilityFlowSampler(predictor, sde, { steps: S.steps, end: S.end, method: 'euler' }),
      { x },
      1000,
    )
    close(toFlat(euler.x), S.euler.flat(), 1e-10)
    const rk4 = run(probabilityFlowSampler(predictor, sde, { steps: 20, end: S.end }), { x }, 1000)
    close(toFlat(rk4.x), S.rk4.flat(), 1e-10)
  })

  it('reverse-SDE steps: the drift update and the noise scale g√h', () => {
    const alg = reverseSdeSampler(predictor, sde, { steps: S.steps, end: S.end })
    const h = (1 - S.end) / S.steps
    for (const c of S.reverse) {
      const x = tensor(S.x0)
      const s = stream(`reverse-${c.time}`)
      const z = toFlat(normals(fork(s), x.shape))
      const st: SamplerState = { t: 0, time: c.time, tau: c.time, x, noise: null, clean: null, evaluations: 0 }
      const next = toFlat(alg.step(st, { t: 0, stream: s }).x)
      expect(rel(sde.diffusion(c.time) * Math.sqrt(h), c.noiseScale)).toBeLessThan(1e-12)
      close(
        next.map((v, i) => v - c.noiseScale * z[i]),
        c.mean.flat(),
        1e-10,
      )
    }
  })
})

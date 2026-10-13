import { describe, expect, it } from 'vitest'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { rbf } from 'aifn-compute/learning/kernels'
import { fitMultiFidelityGp, gaussianProcessFunctions } from 'aifn-methods/learning/gaussian-processes'
import { fixture } from '../../fixtures'

type Ref = { rho: number[]; lml: number; mean: number[][]; sd: number[][] }
type Mf = {
  forrester: {
    xl: number[]
    xh: number[]
    yl: number[]
    yh: number[]
    xs: number[]
    kernels: [number, number][]
    noise: number
    fixed: Ref
    gls: Ref
  }
  three: { x: number[][]; y: number[][]; xs: number[]; kernels: [number, number][]; noise: number; gls: Ref }
}
const F = fixture<{ multiFidelity: Mf }>('learning/gaussian-processes').multiFidelity

const flat = (t: Parameters<typeof toFlat>[0]) => Array.from(toFlat(t) as ArrayLike<number>)

/** Stacks per-level inputs and targets into one data set with fidelity labels. */
function stack(xs: number[][], ys: number[][]) {
  return {
    x: tensor(xs.flat()),
    z: xs.flatMap((x, t) => x.map(() => t)),
    y: tensor(ys.flat()),
  }
}

function expectRef(gp: ReturnType<typeof fitMultiFidelityGp>, ref: Ref, xs: number[]) {
  ref.rho.forEach((r, k) => expect(gp.rho[k]).toBeCloseTo(r, 8))
  expect(gp.logMarginalLikelihood).toBeCloseTo(ref.lml, 5)
  ref.mean.forEach((mean, t) => {
    const p = gp.predict(tensor(xs), t)
    flat(p.mean).forEach((v, i) => expect(v).toBeCloseTo(mean[i], 5))
    flat(p.sd).forEach((v, i) => expect(Math.abs(v - ref.sd[t][i])).toBeLessThan(1e-4))
  })
}

describe('fitMultiFidelityGp', () => {
  const f = F.forrester
  const data = stack([f.xl, f.xh], [f.yl, f.yh])
  const kernels = f.kernels.map(([lengthscale, variance]) => rbf({ lengthscale, variance }))

  it('matches numpy recursive co-kriging with fixed hyperparameters: a fixed scale and the GLS scale', () => {
    const fixed = fitMultiFidelityGp(data.x, data.z, data.y, { kernels, rho: [2], noise: f.noise, fitKernels: false })
    expectRef(fixed, f.fixed, f.xs)
    const gls = fitMultiFidelityGp(data.x, data.z, data.y, { kernels, noise: f.noise, fitKernels: false })
    expectRef(gls, f.gls, f.xs)
  })

  it('matches numpy on three fidelities with a non-nested design', () => {
    const t = F.three
    const d = stack(t.x, t.y)
    const k3 = t.kernels.map(([lengthscale, variance]) => rbf({ lengthscale, variance }))
    const gp = fitMultiFidelityGp(d.x, d.z, d.y, { kernels: k3, noise: t.noise, fitKernels: false })
    expect(gp.fidelities).toBe(3)
    expectRef(gp, t.gls, t.xs)
    // The covariance of f_0 and f_2 is rho_0 rho_1 times the variance of f_0.
    const v0 = flat(gp.predict(tensor(t.xs), 0).sd).map((s) => s * s)
    flat(gp.covariance(tensor(t.xs), 2, 0)).forEach((c, i) => expect(c).toBeCloseTo(gp.rho[0] * gp.rho[1] * v0[i], 10))
  })

  it('Forrester: fitted kernels recover rho = 2 (high = 2 (low - 10 (x - 0.5) - 5)) from four expensive points', () => {
    const gp = fitMultiFidelityGp(data.x, data.z, data.y)
    expect(gp.rho[0]).toBeGreaterThan(1.7)
    expect(gp.rho[0]).toBeLessThan(2.3)
    const xs = [0.1, 0.25, 0.5, 0.75, 0.9]
    const high = (v: number) => (6 * v - 2) ** 2 * Math.sin(12 * v - 4)
    const p = gp.predict(tensor(xs), 1)
    // The high-fidelity range is about 22; the fit is within a small fraction of it.
    flat(p.mean).forEach((v, i) => expect(Math.abs(v - high(xs[i]))).toBeLessThan(1))
    // A GP on the four high-fidelity points alone is far off at the global minimum (x = 0.7572).
    const alone = fitMultiFidelityGp(tensor(f.xh), [0, 0, 0, 0], tensor(f.yh))
    expect(Math.abs(flat(alone.predict(tensor([0.75]), 0).mean)[0] - high(0.75))).toBeGreaterThan(2)
  })

  it('reports bad input', () => {
    expect(() => fitMultiFidelityGp(tensor([0, 1]), [0, 2], tensor([1, 2]))).toThrow(/every fidelity/)
    expect(() => fitMultiFidelityGp(tensor([0, 1]), [0], tensor([1, 2]))).toThrow(/inputs/)
    expect(() => fitMultiFidelityGp(tensor([0, 1]), [0, 1], tensor([1, 2]), { rho: [1, 2] })).toThrow(/scales/)
    expect(() =>
      fitMultiFidelityGp(tensor([0, 1]), [0, 1], tensor([1, 2]), { noise: 'fit', fitKernels: false }),
    ).toThrow(/fitKernels/)
    const gp = fitMultiFidelityGp(tensor([0, 1]), [0, 1], tensor([1, 2]), { fitKernels: false })
    expect(() => gp.predict(tensor([0.5]), 2)).toThrow(/fidelity 2/)
  })

  it('is registered', () => {
    expect(gaussianProcessFunctions.fitMultiFidelityGp.info.role).toBe('fit')
  })
})

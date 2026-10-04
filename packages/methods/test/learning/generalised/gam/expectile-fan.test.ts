import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { linspace, reshape, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { curve1d } from 'aifn-methods/data/synthetic'
import {
  expectileFan,
  expectileGam,
  gam,
  gamLinkBand,
  gamProblem,
  s,
  smoothingPath,
  smoothingProfile,
} from 'aifn-methods/learning/generalised/gam'

const d = curve1d(stream('fan'), { case: 'sine', n: 200 })
const data = { x: d.x as Tensor, y: d.y as Tensor }
const grid = reshape(linspace(0, 1, 21), [21, 1])
const terms = [s(0, { k: 12 })]

describe('expectileFan', () => {
  it('keeps every LAWS state per τ, ending at the expectileGam fit, with λ, EDF and the curve on the grid', () => {
    const fan = expectileFan({ terms, method: 'fixed', lambda: 0.5 }, data, [0.2, 0.9], { grid })
    expect(fan.taus).toEqual([0.2, 0.9])
    fan.runs.forEach((states, q) => {
      expect(states[0].t).toBe(0)
      const last = states[states.length - 1]
      expect(last.converged).toBe(true)
      expect(last.lambdas).toEqual([0.5])
      expect(last.edf).toBeGreaterThan(1)
      expect(Number.isNaN(last.criterion)).toBe(true)
      const model = expectileGam({ terms, method: 'fixed', lambda: 0.5, tau: fan.taus[q] }).fit(data)
      const curve = toFlat(last.curve!)
      toFlat(model.decide(grid)).forEach((v, i) => expect(curve[i]).toBeCloseTo(v, 8))
    })
    // The higher expectile lies above the lower one, and more points lie below it.
    const lo = fan.runs[0].at(-1)!
    const hi = fan.runs[1].at(-1)!
    expect(hi.below).toBeGreaterThan(lo.below)
    expect(toFlat(hi.fitted).every((v, i) => v > toFlat(lo.fitted)[i])).toBe(true)
  })

  it('records the REML criterion when λ is selected', () => {
    const fan = expectileFan({ terms }, data, [0.5])
    expect(Number.isFinite(fan.runs[0].at(-1)!.criterion)).toBe(true)
  })
})

describe('smoothingPath and gamLinkBand', () => {
  it('the EDF falls from P towards the penalty null space as λ grows; the criterion is the profile', () => {
    const problem = gamProblem({ terms, method: 'fixed' }, data)
    const logs = [-10, 0, 10, 20].map((v) => v * Math.LN10 * 0.5)
    const path = smoothingPath(problem, logs, 'gcv')
    for (let i = 1; i < logs.length; i++) expect(path.edf[i]).toBeLessThan(path.edf[i - 1])
    // Second-order penalty: a straight line (intercept and slope) survives an infinite λ.
    expect(path.edf[logs.length - 1]).toBeCloseTo(2, 1)
    expect(path.criterion).toEqual(smoothingProfile(problem, logs, 'gcv'))
  })

  it('η ± se at new inputs: η matches the model and the se exceeds the smooth term’s alone', () => {
    const model = gam({ terms }).fit(data)
    const band = gamLinkBand(model, grid)
    toFlat(model.forward(grid)).forEach((v, i) => expect(toFlat(band.fit)[i]).toBeCloseTo(v, 10))
    const smooth = toFlat(model.partial(0, grid).se)
    toFlat(band.se).forEach((v, i) => expect(v).toBeGreaterThan(smooth[i]))
  })
})

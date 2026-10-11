import { describe, expect, it } from 'vitest'
import { burgers, burgersSolution, heat, pinnRun, type PinnRun } from 'aifn-methods/neural/pinn'
import { DomainError } from 'aifn-compute/foundation/errors'

const last = (gen: Generator<PinnRun, PinnRun>): { last: PinnRun; count: number } => {
  let r: PinnRun | undefined
  let count = 0
  for (const s of gen) {
    r = s
    count++
  }
  return { last: r!, count }
}

describe('burgersSolution', () => {
  const nu = 0.01 / Math.PI
  it('starts from −sin(πx), is odd and is zero at the ends', () => {
    expect(burgersSolution(0.3, 0)).toBeCloseTo(-Math.sin(0.3 * Math.PI), 14)
    for (const t of [0.1, 0.5, 1]) {
      expect(Math.abs(burgersSolution(1, t))).toBeLessThan(1e-10)
      expect(Math.abs(burgersSolution(-1, t))).toBeLessThan(1e-10)
      expect(burgersSolution(0.4, t)).toBeCloseTo(-burgersSolution(-0.4, t), 10)
    }
  })

  it('satisfies the equation by finite differences away from the shock', () => {
    const h = 1e-4
    for (const [x, t] of [
      [0.5, 0.3],
      [-0.7, 0.6],
      [0.3, 0.9],
    ]) {
      const u = burgersSolution(x, t, nu)
      const ut = (burgersSolution(x, t + h, nu) - burgersSolution(x, t - h, nu)) / (2 * h)
      const ux = (burgersSolution(x + h, t, nu) - burgersSolution(x - h, t, nu)) / (2 * h)
      const uxx = (burgersSolution(x + h, t, nu) - 2 * u + burgersSolution(x - h, t, nu)) / (h * h)
      expect(Math.abs(ut + u * ux - nu * uxx)).toBeLessThan(1e-4)
    }
  })

  it('steepens into a near-shock at the origin', () => {
    // The steepest slope, −152.00516 at x = 0 and t = 1.6037/π (Basdevant et al., 1986, the reference value).
    const h = 1e-6
    const slope = (burgersSolution(h, 1.6037 / Math.PI) - burgersSolution(-h, 1.6037 / Math.PI)) / (2 * h)
    expect(slope).toBeCloseTo(-152.00516, 3)
  })

  it('refuses a non-positive viscosity or a negative time', () => {
    expect(() => burgersSolution(0, 1, 0)).toThrow(DomainError)
    expect(() => burgersSolution(0, -1)).toThrow(DomainError)
    expect(() => burgers(-1)).toThrow(DomainError)
    expect(() => heat(0)).toThrow(DomainError)
  })
})

describe('pinnRun', () => {
  const small = { hidden: [12, 12], collocation: 128, initialPoints: 24, boundaryPoints: 12, grid: [21, 11] } as const

  it('solves the heat equation from the equation alone', () => {
    const { last: run, count } = last(pinnRun(heat(), { ...small, adamSteps: 200, lbfgsSteps: 150, every: 100 }))
    expect(run.finished).toBe(true)
    expect(run.error).toBeNull()
    expect(count).toBeGreaterThan(2)
    expect(run.loss.length).toBe(run.residualLoss.length)
    expect(run.loss.at(-1)!).toBeLessThan(run.loss[0] / 100)
    expect(run.error2[0]).toBeGreaterThan(0.5)
    expect(run.error2.at(-1)!).toBeLessThan(0.05)
    expect(run.prediction.length).toBe(21 * 11)
    expect(run.collocation.length).toBe(256)
  }, 60_000)

  it('is deterministic in its seed', () => {
    const opts = { ...small, adamSteps: 5, lbfgsSteps: 3 }
    const a = last(pinnRun(burgers(), opts)).last
    const b = last(pinnRun(burgers(), opts)).last
    const c = last(pinnRun(burgers(), { ...opts, seed: 1 })).last
    expect(Array.from(a.loss)).toEqual(Array.from(b.loss))
    expect(a.loss[0]).not.toBe(c.loss[0])
  })

  it('reports an error in its last snapshot', () => {
    const broken = {
      ...heat(),
      initial: () => NaN,
      residual: () => {
        throw new Error('no residual')
      },
    }
    const { last: run } = last(pinnRun(broken, { ...small, adamSteps: 3, lbfgsSteps: 0 }))
    expect(run.finished).toBe(true)
    expect(run.error).toBe('no residual')
  })
})

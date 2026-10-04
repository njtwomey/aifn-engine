import { describe, expect, it } from 'vitest'
import {
  distributionalFamily,
  distributionalLinks,
  quantileResidual,
  type DistributionalFamilyName,
} from 'aifn-compute/probability/likelihoods'

const CASES: { name: DistributionalFamilyName; theta: number[][] }[] = [
  {
    name: 'normal',
    theta: [
      [1.5, 0.7],
      [-2, 3],
    ],
  },
  {
    name: 'student-t',
    theta: [
      [0.5, 1.2, 4],
      [2, 0.3, 15],
    ],
  },
  {
    name: 'box-cox-cole-green',
    theta: [
      [10, 0.15, -0.8],
      [5, 0.3, 1.2],
      [8, 0.2, 0],
    ],
  },
  {
    name: 'gamma',
    theta: [
      [2, 0.5],
      [10, 1.3],
    ],
  },
  { name: 'poisson', theta: [[3.5], [20]] },
]

/** E[g(Y)] by the midpoint rule over m quantiles (exact enough for smooth g of a continuous law). */
const expectation = (q: (p: number) => number, g: (y: number) => number, m = 20000) => {
  let s = 0
  for (let i = 0; i < m; i++) s += g(q((i + 0.5) / m))
  return s / m
}

describe('distributional families', () => {
  for (const { name, theta } of CASES) {
    const f = distributionalFamily(name)
    describe(f.abbreviation, () => {
      it('scores match finite differences of the log-density', () => {
        for (const th of theta) {
          for (const p of [0.1, 0.5, 0.93]) {
            const y = f.quantile(p, th)
            f.parameters.forEach((_, k) => {
              const h = 1e-5 * Math.max(1, Math.abs(th[k]))
              const up = th.map((v, j) => (j === k ? v + h : v))
              const dn = th.map((v, j) => (j === k ? v - h : v))
              const fd = (f.logPdf(y, up) - f.logPdf(y, dn)) / (2 * h)
              expect(f.score(k, y, th)).toBeCloseTo(fd, 4)
            })
          }
        }
      })

      it('the quantile function inverts the cdf', () => {
        for (const th of theta)
          for (const p of [0.03, 0.25, 0.5, 0.9, 0.97]) {
            const y = f.quantile(p, th)
            if (f.support === 'non-negative-integers') {
              expect(f.cdf(y, th)).toBeGreaterThanOrEqual(p)
              expect(f.cdf(y - 1, th)).toBeLessThan(p)
            } else expect(f.cdf(y, th)).toBeCloseTo(p, 8)
          }
      })

      if (f.support !== 'non-negative-integers')
        it('the expected score is zero and E[score²] = −E[∂²ℓ/∂θ²] (exact families)', () => {
          for (const th of theta) {
            const q = (p: number) => f.quantile(p, th)
            f.parameters.forEach((_, k) => {
              const mean = expectation(q, (y) => f.score(k, y, th))
              const info = expectation(q, (y) => f.score(k, y, th) ** 2)
              const scale = Math.sqrt(-f.expectedSecond(k, th))
              expect(Math.abs(mean) / scale).toBeLessThan(2e-2)
              // BCCG's expected second derivatives are gamlss.dist's approximations (truncation ignored). The midpoint rule
              // loses about 1% in the t's tails for ν; scipy's quad confirms the t's information to 1e-9.
              if (name !== 'box-cox-cole-green')
                expect(Math.abs(info / -f.expectedSecond(k, th) - 1)).toBeLessThan(3e-2)
            })
          }
        })
    })
  }

  it('BCCG with ν = 1 is a normal with mean μ and sd μσ (truncated at 0)', () => {
    const f = distributionalFamily('box-cox-cole-green')
    const th = [10, 0.1, 1]
    expect(f.quantile(0.5, th)).toBeCloseTo(10, 8)
    expect(f.quantile(0.975, th)).toBeCloseTo(10 + 1.959964 * 1, 4)
  })

  it('BCCG ν < 1 skews right: the upper centile is further from the median than the lower', () => {
    const f = distributionalFamily('box-cox-cole-green')
    const th = [10, 0.2, -1]
    expect(f.quantile(0.97, th) - f.quantile(0.5, th)).toBeGreaterThan(f.quantile(0.5, th) - f.quantile(0.03, th))
  })

  it('quantile residuals of draws are standard normal', () => {
    const f = distributionalFamily('student-t')
    const th = [1, 2, 5]
    const r = Array.from({ length: 999 }, (_, i) => quantileResidual(f, f.quantile((i + 0.5) / 999, th), th))
    expect(r[499]).toBeCloseTo(0, 8)
    expect(r[974]).toBeCloseTo(1.96, 1)
  })

  it('links default to gamlss.dist and reject links a parameter does not take', () => {
    const f = distributionalFamily('box-cox-cole-green')
    expect(distributionalLinks(f).map((l) => l.name)).toEqual(['identity', 'log', 'identity'])
    expect(() => distributionalLinks(f, { nu: 'log' })).toThrow()
  })
})

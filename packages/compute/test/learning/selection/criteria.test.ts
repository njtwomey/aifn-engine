/**
 * aifn-compute/learning/selection, criteria and rules: `informationCriteria` against statsmodels' OLS (`aic`, `bic`)
 * and `eval_measures.aicc` over a polynomial-degree sweep, `mallowsCp` in both forms, `softThresholdSure` against the
 * closed form and `sure` with a finite-difference divergence, `lassoSure` by hand, and `oneStandardError` on
 * scikit-learn's GridSearchCV fold scores and on this package's `gridSearch` and `crossValidate`.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { fromData, full, mean, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { real, space } from 'aifn-compute/foundation/space'
import type { Supervised } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { crossValidate, gridSearch, kFold } from 'aifn-compute/learning/validate'
import {
  informationCriteria,
  lassoSure,
  mallowsCp,
  oneStandardError,
  softThresholdSure,
  sure,
} from 'aifn-compute/learning/selection'
import { fixture } from '../../fixtures'

interface Fixture {
  informationCriteria: {
    n: number
    sigma2: number
    rows: {
      llf: number
      k: number
      aic: number
      bic: number
      aicc: number
      rss: number
      cp: number
      cpClassical: number
    }[]
  }
  softThresholdSure: { y: number[]; sigma2: number; lambdas: number[]; sure: number[] }
  oneStandardError: {
    lambdas: number[]
    scores: number[][]
    mean: number[]
    se: number[]
    best: number
    threshold: number
    eligible: number[]
    chosen: number
  }
}

const F = fixture<Fixture>('learning/selection')

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i])))
}

describe('informationCriteria', () => {
  const { n, rows } = F.informationCriteria
  it('matches statsmodels elementwise over a degree sweep', () => {
    const ic = informationCriteria({ logLik: rows.map((r) => r.llf), k: rows.map((r) => r.k), n })
    close(
      toFlat(ic.aic),
      rows.map((r) => r.aic),
      1e-12,
    )
    close(
      toFlat(ic.bic),
      rows.map((r) => r.bic),
      1e-12,
    )
    close(
      toFlat(ic.aicc),
      rows.map((r) => r.aicc),
      1e-12,
    )
  })

  it('gives numbers for numbers, NaN AICc when n <= k + 1, and is differentiable in the log-likelihood', () => {
    const one = informationCriteria({ logLik: rows[2].llf, k: rows[2].k, n })
    expect(one.aic).toBeCloseTo(rows[2].aic, 12)
    expect(typeof one.bic).toBe('number')
    expect(informationCriteria({ logLik: -3, k: 4, n: 5 }).aicc).toBeNaN()
    expect(grad((l: number) => informationCriteria({ logLik: l, k: 3, n: 10 }).bic)(1)).toBe(-2)
  })
})

describe('mallowsCp', () => {
  it('both forms over a degree sweep, and they rank alike', () => {
    const { n, sigma2, rows } = F.informationCriteria
    const cp = mallowsCp({ rss: rows.map((r) => r.rss), df: rows.map((r) => r.k), sigma2, n })
    close(
      toFlat(cp.cp),
      rows.map((r) => r.cp),
      1e-12,
    )
    close(
      toFlat(cp.classical),
      rows.map((r) => r.cpClassical),
      1e-12,
    )
    const order = (v: number[]) => [...v.keys()].sort((a, b) => v[a] - v[b])
    expect(order(toFlat(cp.cp))).toEqual(order(toFlat(cp.classical)))
  })
})

describe('sure', () => {
  const { y, sigma2, lambdas, sure: expected } = F.softThresholdSure
  it('soft thresholding: the closed form at every threshold', () => {
    close(toFlat(softThresholdSure(y, lambdas, sigma2)), expected, 1e-12)
    expect(softThresholdSure(y, lambdas[3], sigma2)).toBeCloseTo(expected[3], 12)
  })

  it('soft thresholding through the general formula, with a finite-difference divergence', () => {
    const soft = (v: number[], l: number) => v.map((x) => Math.sign(x) * Math.max(Math.abs(x) - l, 0))
    const l = 1.1
    const fit = soft(y, l)
    const h = 1e-6
    let divergence = 0
    y.forEach((_, i) => {
      const up = [...y]
      up[i] += h
      divergence += (soft(up, l)[i] - fit[i]) / h
    })
    expect(sure({ y, fit, divergence, sigma2 })).toBeCloseTo(softThresholdSure(y, l, sigma2), 4)
  })

  it('is unbiased for the risk of a linear shrinker (Monte Carlo over fixed draws)', () => {
    // y ~ N(mu, I); fit = c y, divergence = c n. E SURE = E ||c y - mu||^2 = c^2 n + (1 - c)^2 ||mu||^2.
    const mu = [1, -2, 0.5, 0, 3]
    const c = 0.6
    let total = 0
    const reps = 4000
    let state = 12345
    const gauss = () => {
      // Box-Muller on a small linear congruential generator, enough for a mean.
      state = (state * 1103515245 + 12345) % 2 ** 31
      const u1 = (state + 1) / 2 ** 31
      state = (state * 1103515245 + 12345) % 2 ** 31
      const u2 = state / 2 ** 31
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    }
    for (let r = 0; r < reps; r++) {
      const draw = mu.map((m) => m + gauss())
      total += sure({ y: draw, fit: draw.map((v) => c * v), divergence: c * mu.length, sigma2: 1 })
    }
    const risk = c * c * mu.length + (1 - c) ** 2 * mu.reduce((a, m) => a + m * m, 0)
    expect(Math.abs(total / reps - risk)).toBeLessThan(0.15)
  })

  it('lasso: df counts nonzero coefficients plus the intercept', () => {
    const x = tensor([
      [1, 0, 0.5],
      [0, 1, 0.2],
      [1, 1, 0.1],
      [2, 0, 0.3],
    ])
    const y = [1, -1, 0.5, 2]
    const coef = tensor([
      [1, 0.8],
      [0, -0.9],
      [0, 0],
    ])
    const s = lassoSure({ x, y, coef, intercept: [0.1, 0.2], sigma2: 0.5 })
    expect(toFlat(s.df as Tensor)).toEqual([2, 3])
    const fit = (b: number[], b0: number) =>
      [
        [1, 0, 0.5],
        [0, 1, 0.2],
        [1, 1, 0.1],
        [2, 0, 0.3],
      ].map((r) => b0 + r[0] * b[0] + r[1] * b[1] + r[2] * b[2])
    const byHand = (b: number[], b0: number, df: number) => sure({ y, fit: fit(b, b0), divergence: df, sigma2: 0.5 })
    close(toFlat(s.sure as Tensor), [byHand([1, 0, 0], 0.1, 2), byHand([0.8, -0.9, 0], 0.2, 3)], 1e-12)
    const single = lassoSure({ x, y, coef: tensor([1, 0, 0]), sigma2: 0.5 })
    expect(single.df).toBe(1)
    expect(single.sure).toBeCloseTo(byHand([1, 0, 0], 0, 1), 12)
  })
})

describe('oneStandardError', () => {
  const c = F.oneStandardError
  it('matches the rule on GridSearchCV fold scores (ridge on diabetes, larger penalties simpler)', () => {
    const rule = oneStandardError(
      { scores: tensor(c.scores), direction: 'higher' },
      { simplicity: c.lambdas.map((l) => -l) },
    )
    close(toFlat(rule.mean), c.mean, 1e-12)
    close(toFlat(rule.se), c.se, 1e-12)
    expect(rule.best).toBe(c.best)
    expect(rule.threshold).toBeCloseTo(c.threshold, 9)
    expect(Array.from(toFlat(rule.eligible))).toEqual(c.eligible)
    expect(rule.chosen).toBe(c.chosen)
    expect(rule.chosen).not.toBe(rule.best)
  })

  it('the same scores negated, lower is better: the same choice', () => {
    const negated = c.scores.map((r) => r.map((v) => -v))
    const rule = oneStandardError(tensor(negated), { simplicity: (i) => -c.lambdas[i] })
    expect(rule.direction).toBe('lower')
    expect(rule.best).toBe(c.best)
    expect(rule.chosen).toBe(c.chosen)
  })

  type Shrunk = { decide: (x: Tensor) => Tensor }
  const shrunkMean = ({ shrink }: { shrink: number }) => ({
    name: 'shrunk mean',
    fit: (d: Supervised<Tensor, Tensor>): Shrunk => {
      const m = (1 - shrink) * (mean(d.y) as number)
      return { decide: (x: Tensor) => full([x.shape[0]], m) }
    },
  })
  const mse = meanSquaredError
  const data = {
    kind: 'dataset' as const,
    x: fromData(new Float64Array(12), [12, 1]),
    y: tensor([3, 5, 4, 6, 2, 5, 4, 3, 6, 5, 4, 4]),
  }

  it('reads a gridSearch result and a list of crossValidate results alike', () => {
    const shrinks = space({ shrink: real(0, 0.3, { default: 0 }) })
    const result = gridSearch(shrunkMean, shrinks, { metrics: [mse], points: 4, refit: false }).run(
      data,
      kFold({ k: 4 }),
    )
    const simplicity = (i: number) => -result.rows[i].params.shrink
    const fromSearch = oneStandardError(result, { simplicity })
    const fromCv = oneStandardError(
      result.rows.map((r) => crossValidate(shrunkMean(r.params), data, kFold({ k: 4 }), [mse])),
      { simplicity },
    )
    expect(fromSearch.best).toBe(result.best.index)
    expect(toFlat(fromSearch.mean)).toEqual(result.rows.map((r) => r.mean))
    close(
      toFlat(fromSearch.se),
      result.rows.map((r) => r.std / 2),
      1e-12,
    )
    expect(fromCv.chosen).toBe(fromSearch.chosen)
    expect(fromCv.direction).toBe('lower')
    expect(result.rows[fromSearch.chosen].params.shrink).toBeGreaterThanOrEqual(result.best.params.shrink)
  })

  it('throws for one fold or a simplicity of the wrong length', () => {
    expect(() => oneStandardError(tensor([[1], [2]]))).toThrow(/two folds/)
    expect(() =>
      oneStandardError(
        tensor([
          [1, 2],
          [2, 3],
        ]),
        { simplicity: [1] },
      ),
    ).toThrow(/simplicities/)
  })
})

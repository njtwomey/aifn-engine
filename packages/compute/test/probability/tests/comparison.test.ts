/**
 * The tests for comparing learning algorithms (fixtures/gen/probability/tests.py `comparison`): McNemar's test against
 * statsmodels' `mcnemar` (exact, χ² with and without the continuity correction, one-sided exact from scipy's
 * binomtest), Friedman's test against scipy's `friedmanchisquare` with the Iman–Davenport F and Demšar's average ranks,
 * the studentized range and the Nemenyi critical difference against scipy's `studentized_range` and Demšar's (2006)
 * table, and the corrected resampled t, 5×2cv t, combined F and Bayesian correlated t against their formulas.
 */
import { describe, expect, it } from 'vitest'
import { tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import {
  bayesianCorrelatedTTest,
  combinedFTest,
  correctedResampledTTest,
  fiveByTwoCvTest,
  friedmanTest,
  mcnemarTest,
  nemenyiCriticalDifference,
  oneSampleTTest,
  rejectionRegion,
  studentizedRangeNull,
  type Alternative,
} from 'aifn-compute/probability/tests'
import { fixture } from '../../fixtures'

type Num = number
type McnemarCase = {
  table: Num[][]
  exact: { p: Num; less: Num; greater: Num }
  chi2: { statistic: Num; p: Num }
  chi2Uncorrected: { statistic: Num; p: Num }
}
type Fx = {
  comparison: {
    mcnemarTest: { cases: McnemarCase[]; predictions: { a: Num[]; b: Num[]; y: Num[]; table: Num[][] } }
    correctedResampledTTest: {
      d: Num[]
      trainSize: Num
      testSize: Num
      alternative: Alternative
      level: Num
      statistic: Num
      df: Num
      p: Num
      ci: [Num, Num]
      mean: Num
    }[]
    fiveByTwoCvTest: { d: Num[][]; t: Num; p: Record<Alternative, Num>; F: Num; pF: Num }[]
    friedmanTest: {
      scores: Num[][]
      statistic: Num
      p: Num
      averageRanks: Num[]
      averageRanksAscending: Num[]
      F: Num
      pF: Num
    }[]
    studentizedRange: {
      isf: { k: Num; alpha: Num; q: Num }[]
      isfFinite: { k: Num; df: Num; alpha: Num; q: Num }[]
      values: { k: Num; df: Num; q: Num; cdf: Num; sf: Num }[]
    }
    nemenyiCriticalDifference: { k: Num; N: Num; alpha: Num; cd: Num }[]
    bayesianCorrelatedTTest: {
      d: Num[]
      rho: Num
      rope: Num
      mean: Num
      scale: Num
      df: Num
      pLeft: Num
      pRope: Num
      pRight: Num
    }[]
  }
}
const F = fixture<Fx>('probability/tests').comparison

/** |a − b| ≤ tol·|b| (relative), or ≤ tol when b is 0; NaN and infinities must match exactly. */
function close(a: Num, b: Num, tol = 1e-12) {
  if (Number.isNaN(b) || !Number.isFinite(b)) return expect(a).toBe(b)
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(Math.abs(b), b === 0 ? 1 : 0))
}

describe('mcnemarTest', () => {
  it.each(F.mcnemarTest.cases.map((c) => [JSON.stringify(c.table), c] as const))('table %s', (_, c) => {
    const [b, n] = [c.table[0][1], c.table[0][1] + c.table[1][0]]
    const exact = mcnemarTest(c.table)
    expect(exact.statistic).toBe(b)
    expect(exact.null.name).toBe('Binomial')
    expect(exact.null.params).toEqual({ n, p: 0.5 })
    close(exact.pValue, c.exact.p, 1e-12)
    close(mcnemarTest(c.table, { alternative: 'less' }).pValue, c.exact.less, 1e-12)
    close(mcnemarTest(c.table, { alternative: 'greater' }).pValue, c.exact.greater, 1e-12)
    const chi = mcnemarTest(c.table, { exact: false })
    close(chi.statistic, c.chi2.statistic)
    close(chi.pValue, c.chi2.p, 1e-12)
    const raw = mcnemarTest(c.table, { exact: false, correction: false })
    close(raw.statistic, c.chi2Uncorrected.statistic)
    close(raw.pValue, c.chi2Uncorrected.p, 1e-12)
    expect(exact.n).toBe(c.table[0][0] + n + c.table[1][1])
    close(exact.estimate!, b / n)
  })

  it('from predictions builds the table of agreement and agrees with the table form', () => {
    const { a, b, y, table } = F.mcnemarTest.predictions
    const r = mcnemarTest(a, b, y)
    expect(toRows(r.table)).toEqual(table)
    expect(r.pValue).toBe(mcnemarTest(table).pValue)
    // Labels may be strings, and tensors are read like arrays.
    const name = (v: number) => ['cat', 'dog', 'fox'][v]
    expect(mcnemarTest(a.map(name), b.map(name), y.map(name), { exact: false }).statistic).toBe(
      mcnemarTest(table, { exact: false }).statistic,
    )
    expect(mcnemarTest(tensor(a), tensor(b), tensor(y)).pValue).toBe(r.pValue)
  })

  it('reports failures rather than guessing', () => {
    expect(() => mcnemarTest([1, 2], [1, 2], [1, 2, 3])).toThrow(/equal lengths/)
    expect(() => mcnemarTest([[1, 2, 3]])).toThrow(/2 × 2/)
    expect(() =>
      mcnemarTest([
        [1, -2],
        [3, 4],
      ]),
    ).toThrow(/non-negative/)
    expect(() =>
      mcnemarTest([
        [1, 2.5],
        [3, 4],
      ]),
    ).toThrow(/whole-number/)
    expect(() =>
      mcnemarTest(
        [
          [1, 2],
          [3, 4],
        ],
        { exact: false, alternative: 'less' },
      ),
    ).toThrow(/two-sided/)
    // The χ² form takes non-integer counts, as statsmodels does.
    close(
      mcnemarTest(
        [
          [1, 2.5],
          [3.5, 4],
        ],
        { exact: false, correction: false },
      ).statistic,
      1 / 6,
    )
  })
})

describe('correctedResampledTTest', () => {
  it.each(F.correctedResampledTTest.map((c, i) => [`${i} ${c.alternative} ${c.level}`, c] as const))(
    'case %s',
    (_, c) => {
      const r = correctedResampledTTest(c.d, {
        trainSize: c.trainSize,
        testSize: c.testSize,
        alternative: c.alternative,
        level: c.level,
      })
      close(r.statistic, c.statistic)
      expect(r.df).toBe(c.df)
      close(r.pValue, c.p, 1e-10)
      close(r.estimate!, c.mean)
      close(r.ci!.lower, c.ci[0], 1e-10)
      close(r.ci!.upper, c.ci[1], 1e-10)
      expect(r.null.name).toBe('StudentT')
    },
  )

  it('is more conservative than the naive paired t-test, and equal to it as the test fraction goes to 0', () => {
    const d = F.correctedResampledTTest[0].d
    const naive = oneSampleTTest(d)
    expect(correctedResampledTTest(d, { trainSize: 900, testSize: 100 }).pValue).toBeGreaterThan(naive.pValue)
    close(correctedResampledTTest(d, { trainSize: 1, testSize: 1e-300 }).statistic, naive.statistic)
    expect(() => correctedResampledTTest(d, { trainSize: 0, testSize: 1 })).toThrow(/positive/)
    expect(() => correctedResampledTTest([0.1], { trainSize: 9, testSize: 1 })).toThrow(/at least 2/)
  })
})

describe('fiveByTwoCvTest and combinedFTest', () => {
  it.each(F.fiveByTwoCvTest.map((c, i) => [i, c] as const))('case %s', (_, c) => {
    for (const alt of ['two-sided', 'less', 'greater'] as const) {
      const r = fiveByTwoCvTest(c.d, { alternative: alt })
      close(r.statistic, c.t)
      expect(r.df).toBe(5)
      close(r.pValue, c.p[alt], 1e-10)
      close(r.estimate!, c.d[0][0])
    }
    const f = combinedFTest(c.d)
    close(f.statistic, c.F)
    expect(f.df).toEqual([10, 5])
    close(f.pValue, c.pF, 1e-10)
    expect(fiveByTwoCvTest(c.d).combinedF.statistic).toBe(f.statistic)
  })

  it('needs a 5 × 2 matrix of finite differences', () => {
    expect(() => fiveByTwoCvTest([[0.1, 0.2]])).toThrow(/5 × 2/)
    expect(() => combinedFTest(Array.from({ length: 5 }, () => [0.1, NaN]))).toThrow(/finite/)
  })
})

describe('friedmanTest', () => {
  it.each(F.friedmanTest.map((c) => [`${c.scores.length} × ${c.scores[0].length}`, c] as const))(
    '%s scores',
    (_, c) => {
      const r = friedmanTest(c.scores)
      close(r.statistic, c.statistic)
      close(r.pValue, c.p, 1e-10)
      expect(r.df).toBe(c.scores[0].length - 1)
      expect(Array.from(toFlat(r.averageRanks))).toEqual(c.averageRanks)
      expect(Array.from(toFlat(friedmanTest(c.scores, { higherIsBetter: false }).averageRanks))).toEqual(
        c.averageRanksAscending,
      )
      expect(friedmanTest(c.scores, { higherIsBetter: false }).statistic).toBe(r.statistic)
      close(r.imanDavenport.statistic, c.F)
      close(r.imanDavenport.pValue, c.pF, 1e-10)
      const [k, N] = [c.scores[0].length, c.scores.length]
      expect(r.imanDavenport.df).toEqual([k - 1, (k - 1) * (N - 1)])
    },
  )

  it('agreement in every data set gives the largest statistic and an infinite Iman–Davenport F', () => {
    const r = friedmanTest([
      [0.9, 0.8, 0.7],
      [0.6, 0.5, 0.4],
      [0.95, 0.9, 0.85],
    ])
    expect(r.statistic).toBe(3 * 2)
    expect(r.imanDavenport.statistic).toBe(Infinity)
    expect(r.imanDavenport.pValue).toBe(0)
    expect(Array.from(toFlat(r.averageRanks))).toEqual([1, 2, 3])
    expect(() => friedmanTest([[1, 2, 3]])).toThrow(/two data sets/)
    expect(() => friedmanTest([[1], [2]])).toThrow(/two algorithms/)
  })
})

describe('the studentized range and the Nemenyi critical difference', () => {
  // Demšar (2006), Table 5(a): q_0.05 for k = 2 ... 10, the studentized range quantile over √2 (rounded).
  const demsar = [1.96, 2.343, 2.569, 2.728, 2.85, 2.949, 3.031, 3.102, 3.164]

  it("reproduces Demšar's table of q_0.05", () => {
    demsar.forEach((q, i) => {
      const got = (studentizedRangeNull(i + 2).isf(0.05) as number) / Math.SQRT2
      expect(Math.abs(got - q)).toBeLessThan(1e-3)
    })
  })

  it.each(F.studentizedRange.isf.map((c) => [`k = ${c.k}, alpha = ${c.alpha}`, c] as const))(
    'upper points with infinite degrees of freedom, %s',
    (_, c) => close(studentizedRangeNull(c.k).isf(c.alpha) as number, c.q, 1e-12),
  )

  it('the range of two normals is sqrt 2 times a half-normal', () => {
    const law = studentizedRangeNull(2)
    close(law.isf(0.05) as number, 1.959963984540054 * Math.SQRT2, 1e-12)
    close(law.cdf(1) as number, 2 * 0.7602499389065233 - 1, 1e-13)
  })

  it.each(F.studentizedRange.values.map((c) => [`k = ${c.k}, df = ${c.df}, q = ${c.q}`, c] as const))(
    'cdf and survival function, %s',
    (_, c) => {
      const law = studentizedRangeNull(c.k, c.df)
      close(law.cdf(c.q) as number, c.cdf, 1e-9)
      close(law.survival(c.q) as number, c.sf, 1e-9)
    },
  )

  it('upper points with finite degrees of freedom', () => {
    for (const c of F.studentizedRange.isfFinite)
      close(studentizedRangeNull(c.k, c.df).isf(c.alpha) as number, c.q, 1e-9)
  })

  it.each(F.nemenyiCriticalDifference.map((c) => [`k = ${c.k}, N = ${c.N}, alpha = ${c.alpha}`, c] as const))(
    'critical difference, %s',
    (_, c) => close(nemenyiCriticalDifference(c.k, c.N, c.alpha), c.cd, 1e-12),
  )

  it('checks its arguments', () => {
    expect(() => studentizedRangeNull(1)).toThrow(/at least 2/)
    expect(() => studentizedRangeNull(3, 0)).toThrow(/positive/)
    expect(() => nemenyiCriticalDifference(3, 0)).toThrow(/at least 1/)
    expect(() => nemenyiCriticalDifference(3, 10, 1)).toThrow(/\(0, 1\)/)
    expect(rejectionRegion({ null: studentizedRangeNull(3), tail: 'upper' }, 0.05)[0][0]).toBe(
      studentizedRangeNull(3).isf(0.05),
    )
  })
})

describe('bayesianCorrelatedTTest', () => {
  it.each(F.bayesianCorrelatedTTest.map((c, i) => [i, c] as const))('case %s', (_, c) => {
    const r = bayesianCorrelatedTTest(c.d, { rho: c.rho, rope: c.rope })
    close(r.mean, c.mean)
    close(r.scale, c.scale)
    expect(r.df).toBe(c.df)
    close(r.pLeft, c.pLeft, 1e-10)
    close(r.pRope, c.pRope, 1e-9)
    close(r.pRight, c.pRight, 1e-10)
    close(r.pLeft + r.pRope + r.pRight, 1, 1e-14)
    const { df, loc, scale } = r.posterior!.params as { df: Num; loc: Num; scale: Num }
    expect([df, scale]).toEqual([c.df, r.scale])
    close(loc, c.mean)
  })

  it('takes an asymmetric rope, and a point mass when every difference is equal', () => {
    const d = F.bayesianCorrelatedTTest[0].d
    const sym = bayesianCorrelatedTTest(d, { rho: 0.1, rope: 0.01 })
    const ends = bayesianCorrelatedTTest(d, { rho: 0.1, rope: [-0.01, 0.01] })
    expect(ends.pRope).toBe(sym.pRope)
    const same = bayesianCorrelatedTTest([0.002, 0.002, 0.002], { rho: 0.1, rope: 0.01 })
    expect([same.posterior, same.pLeft, same.pRope, same.pRight]).toEqual([null, 0, 1, 0])
    expect(() => bayesianCorrelatedTTest(d, { rho: 1 })).toThrow(/\[0, 1\)/)
    expect(() => bayesianCorrelatedTTest(d, { rho: 0.1, rope: [0.1, -0.1] })).toThrow(/rope/)
  })
})

/**
 * aifn-compute/probability/tests against second references (fixtures/gen/probability/tests.py `references`): the t-tests,
 * proportion tests and χ² goodness of fit against statsmodels, the independence tests against statsmodels'
 * contingency tables and the G statistic's definition, the log-rank test against lifelines, and the exact, rank and
 * Kolmogorov–Smirnov tests against scipy on fresh inputs (ties, zeros, empty cells, lopsided samples).
 */
import { describe, expect, it } from 'vitest'
import { Normal } from 'aifn-compute/probability/distributions'
import {
  binomialTest,
  chiSquareGoodnessOfFit,
  chiSquareIndependence,
  fisherExact,
  gTestGoodnessOfFit,
  gTestIndependence,
  ksTest,
  logRankTest,
  mannWhitneyU,
  oneSampleTTest,
  pairedTTest,
  pooledTTest,
  twoProportionZTest,
  welchTTest,
  wilcoxonSignedRank,
  zTest,
  type Alternative,
  type RankMethod,
  type TestResult,
} from 'aifn-compute/probability/tests'
import { fixture } from '../../fixtures'

type Interval2 = [number, number]
type TRow = {
  alternative: Alternative
  level: number
  mu: number
  statistic: number
  p: number
  df: number
  ci: Interval2
}
type Fx = {
  references: {
    x: number[]
    y: number[]
    x2: number[]
    oneSampleTTest: TRow[]
    pairedTTest: TRow[]
    pooledTTest: TRow[]
    welchTTest: TRow[]
    zTest: {
      alternative: Alternative
      sigma: number
      sigmaY: number | null
      two: boolean
      mu: number
      statistic: number
      p: number
    }[]
    binomialTest: { k: number; n: number; p0: number; alternative: Alternative; p: number; ci: Interval2 }[]
    twoProportionZTest: {
      k1: number
      n1: number
      k2: number
      n2: number
      alternative: Alternative
      z: number
      p: number
      wald?: Interval2
    }[]
    chiSquareGoodnessOfFit: GofRow[]
    gTestGoodnessOfFit: GofRow[]
    chiSquareIndependence: IndRow[]
    gTestIndependence: IndRow[]
    fisherExact: { table: number[][]; alternative: Alternative; p: number }[]
    ksTest: {
      u: number[]
      v: number[]
      cases: {
        sample: 'one' | 'two'
        loc?: number
        scale?: number
        alternative: Alternative
        method: 'exact' | 'asymp'
        statistic: number
        p: number
      }[]
    }
    mannWhitneyU: {
      name: string
      x: number[]
      y: number[]
      method: RankMethod
      continuity: boolean
      alternative: Alternative
      U: number
      p: number
    }[]
    wilcoxonSignedRank: {
      name: string
      d: number[]
      method: RankMethod
      correction: boolean
      alternative: Alternative
      plus: number
      p: number
    }[]
    logRankTest: { time: number[]; event: number[]; group: number[]; statistic: number; p: number; df: number }[]
  }
}
type GofRow = { observed: number[]; probabilities: number[] | null; ddof: number; statistic: number; p: number }
type IndRow = { table: number[][]; plain: [number, number, number]; yates?: [number, number, number] }

const R = fixture<Fx>('probability/tests').references

/** |a − b| ≤ tol·max(1, |b|); infinities must match exactly. */
function close(a: number, b: number, tol = 1e-9) {
  if (!Number.isFinite(b)) return expect(a).toBe(b)
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)))
}
const label = (r: { alternative: Alternative }, i: number) => `${i}: ${r.alternative}`

function checkT(r: TestResult, row: TRow) {
  close(r.statistic, row.statistic)
  close(r.pValue, row.p)
  close(r.df as number, row.df)
  close(r.ci!.lower, row.ci[0])
  close(r.ci!.upper, row.ci[1])
}

describe('t-tests against statsmodels', () => {
  const opts = (row: TRow) => ({ alternative: row.alternative, level: row.level, mu: row.mu })
  it.each(R.oneSampleTTest.map((row, i) => [label(row, i), row] as const))('one sample %s', (_, row) =>
    checkT(oneSampleTTest(R.x, opts(row)), row),
  )
  it.each(R.pairedTTest.map((row, i) => [label(row, i), row] as const))('paired %s', (_, row) =>
    checkT(pairedTTest(R.x2, R.y, opts(row)), row),
  )
  // statsmodels' tconfint_diff is the interval for the difference itself, whatever the null value.
  it.each(R.pooledTTest.map((row, i) => [label(row, i), row] as const))('pooled %s', (_, row) =>
    checkT(pooledTTest(R.x, R.y, opts(row)), row),
  )
  it.each(R.welchTTest.map((row, i) => [label(row, i), row] as const))('Welch %s', (_, row) =>
    checkT(welchTTest(R.x, R.y, opts(row)), row),
  )
})

describe('z-test with a known σ', () => {
  it.each(R.zTest.map((row, i) => [label(row, i), row] as const))('%s', (_, row) => {
    const r = zTest(R.x, {
      alternative: row.alternative,
      mu: row.mu,
      sigma: row.sigma,
      ...(row.two ? { y: R.y } : {}),
      ...(row.sigmaY !== null ? { sigmaY: row.sigmaY } : {}),
    })
    close(r.statistic, row.statistic)
    close(r.pValue, row.p)
  })
})

describe('proportions against scipy and statsmodels', () => {
  it.each(R.binomialTest.map((row, i) => [`${row.k}/${row.n} p₀ ${row.p0} ${row.alternative} (${i})`, row] as const))(
    'binomial test %s',
    (_, row) => {
      const r = binomialTest(row.k, row.n, { p: row.p0, alternative: row.alternative })
      close(r.pValue, row.p)
      close(r.ci!.lower, row.ci[0], 1e-8)
      close(r.ci!.upper, row.ci[1], 1e-8)
    },
  )
  it.each(R.twoProportionZTest.map((row, i) => [label(row, i), row] as const))('two proportions %s', (_, row) => {
    const r = twoProportionZTest(row.k1, row.n1, row.k2, row.n2, { alternative: row.alternative })
    close(r.statistic, row.z)
    close(r.pValue, row.p)
    if (row.wald) {
      close(r.ci!.lower, row.wald[0])
      close(r.ci!.upper, row.wald[1])
    }
  })
})

describe('χ² and G tests', () => {
  const gof = (row: GofRow) => ({ ddof: row.ddof, ...(row.probabilities ? { probabilities: row.probabilities } : {}) })
  it.each(R.chiSquareGoodnessOfFit.map((row, i) => [i, row] as const))(
    'χ² goodness of fit (statsmodels) %s',
    (_, row) => {
      const r = chiSquareGoodnessOfFit(row.observed, gof(row))
      close(r.statistic, row.statistic)
      close(r.pValue, row.p)
    },
  )
  it.each(R.gTestGoodnessOfFit.map((row, i) => [i, row] as const))('G goodness of fit %s', (_, row) => {
    const r = gTestGoodnessOfFit(row.observed, gof(row))
    close(r.statistic, row.statistic)
    close(r.pValue, row.p)
  })
  const ind = (f: typeof chiSquareIndependence, row: IndRow) => {
    for (const [correction, ref] of [
      [false, row.plain],
      [true, row.yates],
    ] as const) {
      if (!ref) continue
      const r = f(row.table, { correction })
      close(r.statistic, ref[0])
      close(r.pValue, ref[1])
      expect(r.df).toBe(ref[2])
    }
  }
  it.each(R.chiSquareIndependence.map((row, i) => [i, row] as const))('χ² independence (statsmodels) %s', (_, row) =>
    ind(chiSquareIndependence, row),
  )
  it.each(R.gTestIndependence.map((row, i) => [i, row] as const))('G independence %s', (_, row) =>
    ind(gTestIndependence, row),
  )
})

describe("Fisher's exact test", () => {
  it.each(R.fisherExact.map((row, i) => [`${JSON.stringify(row.table)} ${row.alternative} (${i})`, row] as const))(
    '%s',
    (_, row) => close(fisherExact(row.table, { alternative: row.alternative }).pValue, row.p),
  )
})

describe('Kolmogorov–Smirnov', () => {
  const K = R.ksTest
  it.each(K.cases.map((c, i) => [`${c.sample} ${c.alternative} ${c.method} (${i})`, c] as const))('%s', (_, c) => {
    const ref = c.sample === 'one' ? (x: number) => Normal(c.loc!, c.scale!).cdf(x) as number : K.v
    const r = ksTest(K.u, ref, { alternative: c.alternative, method: c.method })
    close(r.statistic, c.statistic)
    close(r.pValue, c.p, 1e-7)
  })
})

describe('rank tests against scipy', () => {
  it.each(R.mannWhitneyU.map((c, i) => [`${c.name} ${c.alternative} (${i})`, c] as const))(
    'Mann–Whitney %s',
    (_, c) => {
      const r = mannWhitneyU(c.x, c.y, { alternative: c.alternative, method: c.method, continuity: c.continuity })
      close(r.statistic, c.U)
      close(r.pValue, c.p)
    },
  )
  it.each(R.wilcoxonSignedRank.map((c, i) => [`${c.name} ${c.alternative} (${i})`, c] as const))(
    'Wilcoxon %s',
    (_, c) => {
      const r = wilcoxonSignedRank(c.d, null, {
        alternative: c.alternative,
        method: c.method,
        correction: c.correction,
      })
      close(r.statistic, c.plus)
      close(r.pValue, c.p)
    },
  )
  it('exact with ties throws instead of using the no-ties law', () => {
    expect(() => mannWhitneyU([1, 2, 2], [2, 3], { method: 'exact' })).toThrow(/no ties/)
    expect(() => wilcoxonSignedRank([1, -1, 2, 0], null, { method: 'exact' })).toThrow(/no ties/)
  })
})

describe('log-rank against lifelines', () => {
  it.each(R.logRankTest.map((c, i) => [`${c.df + 1} groups (${i})`, c] as const))('%s', (_, c) => {
    const r = logRankTest(c.time, c.event, c.group)
    close(r.statistic, c.statistic)
    close(r.pValue, c.p)
    expect(r.df).toBe(c.df)
  })
})

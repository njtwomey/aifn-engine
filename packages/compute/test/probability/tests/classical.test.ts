/**
 * aifn-compute/probability/tests against scipy.stats (fixtures/gen/probability/tests.py): every test's statistic, p-value
 * and interval on several cases, with ties, small samples and one-sided alternatives; the null law and the rejection
 * region agree with the p-value; the F distribution against scipy.stats.f.
 */
import { describe, expect, it } from 'vitest'
import { normalCdf } from 'aifn-compute/numerics/special'
import { FisherSnedecor, Normal, StudentT } from 'aifn-compute/probability/distributions'
import {
  benjaminiHochberg,
  benjaminiYekutieli,
  binomialTest,
  bonferroni,
  chiSquareGoodnessOfFit,
  chiSquareIndependence,
  cohensD,
  differenceOfMeansInterval,
  differenceOfProportionsInterval,
  fisherExact,
  gTestGoodnessOfFit,
  gTestIndependence,
  grubbs,
  hedgesG,
  hochberg,
  holm,
  ksTest,
  ljungBox,
  mannWhitneyU,
  meanInterval,
  oddsRatio,
  oneSampleTTest,
  oneWayAnova,
  pairedTTest,
  pooledTTest,
  proportionInterval,
  pValueOf,
  rejectionRegion,
  shapiroWilk,
  smirnovSf,
  twoProportionZTest,
  welchTTest,
  wilcoxonSignedRank,
  zTest,
  type Alternative,
  type ProportionIntervalMethod,
  type TestResult,
} from 'aifn-compute/probability/tests'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type Num = number
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F = fixture<any>('probability/tests')
// Parameterised cases over the untyped fixture rows.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const each = (cases: readonly (readonly [string | number, any])[]) => it.each(cases as [string, any][])

/** |a − b| ≤ tol·|b| (relative), or ≤ tol when b is 0; infinities must match exactly. */
function close(a: Num, b: Num, tol = 1e-9) {
  if (!Number.isFinite(b)) return expect(a).toBe(b)
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(Math.abs(b), b === 0 ? 1 : 0))
}

const closeArray = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-9) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < b.length; i++) close(a[i], b[i], tol)
}

/** The protocol's invariants: p is the null law's tail at the statistic, and p ≤ α exactly when inside the region. */
function consistent(r: TestResult) {
  expect(r.kind).toBe('test-result')
  for (const alpha of [0.01, 0.05, 0.2]) {
    const region = rejectionRegion(r, alpha)
    const inside = region.some(([lo, hi]) => r.statistic >= lo - 1e-9 && r.statistic <= hi + 1e-9)
    // Exact laws reject on the boundary itself; skip p-values within rounding of α.
    if (Math.abs(r.pValue - alpha) > 1e-6) expect(inside).toBe(r.pValue <= alpha)
  }
}

describe('t-tests and intervals (scipy ttest_1samp, ttest_rel, ttest_ind)', () => {
  const { x, y, small } = F.t
  each(
    F.t.cases.map((c: { test: string; alternative: string; level: number }): [string, any] => [
      `${c.test} ${c.alternative} ${c.level}`,
      c,
    ]),
  )('%s', (_, c) => {
    const o = { alternative: c.alternative as Alternative, level: c.level }
    const r =
      c.test === 'oneSample'
        ? oneSampleTTest(x, { ...o, mu: 0.3 })
        : c.test === 'oneSampleSmall'
          ? oneSampleTTest(small, o)
          : c.test === 'paired'
            ? pairedTTest(x.slice(0, 9), y, o)
            : c.test === 'pooled'
              ? pooledTTest(x, y, o)
              : welchTTest(x, y, o)
    close(r.statistic, c.statistic, 1e-12)
    close(r.df as number, c.df, 1e-12)
    close(r.pValue, c.p, 1e-9)
    close(r.ci!.lower, c.ci[0], 1e-9)
    close(r.ci!.upper, c.ci[1], 1e-9)
    consistent(r)
  })
  it('effect sizes: Cohen d, Hedges g (exact correction)', () => {
    close(cohensD(x, y), F.t.cohensD, 1e-12)
    close(hedgesG(x, y), F.t.hedgesG, 1e-12)
    close(cohensD(x, null, { mu: 0.3 }), F.t.cohensDOne, 1e-12)
    close(oneSampleTTest(x, { mu: 0.3 }).effectSize!.value, F.t.cohensDOne, 1e-12)
  })
  it('the z-test with a known σ, and the intervals', () => {
    const r = zTest(x, { mu: 0.1, sigma: 1.1 })
    close(r.statistic, F.t.z.statistic, 1e-12)
    close(r.pValue, F.t.z.p, 1e-10)
    close(zTest(x, { mu: 0.1, sigma: 1.1, alternative: 'greater' }).pValue, F.t.z.pGreater, 1e-10)
    closeArray([r.ci!.lower, r.ci!.upper], F.t.z.ci, 1e-12)
    const m = meanInterval(x, { level: 0.9 })
    const ref = F.t.cases.find(
      (c: { test: string; alternative: string; level: number }) =>
        c.test === 'oneSample' && c.alternative === 'two-sided' && c.level === 0.9,
    )
    closeArray([m.lower, m.upper], ref.ci, 1e-9)
    const w = differenceOfMeansInterval(x, y)
    const refW = F.t.cases.find(
      (c: { test: string; alternative: string; level: number }) =>
        c.test === 'welch' && c.alternative === 'two-sided' && c.level === 0.95,
    )
    closeArray([w.lower, w.upper], refW.ci, 1e-9)
    consistent(r)
  })
  it('null laws are Student t with the test degrees of freedom', () => {
    const r = welchTTest(x, y)
    expect(r.null.name).toBe('StudentT')
    close(r.pValue, 2 * (StudentT(r.df as number).survival(Math.abs(r.statistic)) as number), 1e-12)
  })
})

describe('binomial test and proportion intervals (scipy binomtest, proportion_ci)', () => {
  const METHODS: ProportionIntervalMethod[] = ['clopper-pearson', 'wilson', 'wilson-cc']
  each(
    F.binomial.cases.map((c: { k: number; n: number; alternative: string }): [string, any] => [
      `k=${c.k} n=${c.n} ${c.alternative}`,
      c,
    ]),
  )('%s', (_, c) => {
    const r = binomialTest(c.k, c.n, { p: c.p, alternative: c.alternative })
    close(r.pValue, c.pValue, 1e-9)
    consistent(r)
    for (const method of METHODS)
      for (const level of [0.95, 0.9]) {
        const ref = c.ci[`${method}@${level}`]
        const got = proportionInterval(c.k, c.n, { method, level, alternative: c.alternative })
        close(got.lower, ref[0], 1e-9)
        close(got.upper, ref[1], 1e-9)
      }
  })
  it('Wald, two-proportion z and Newcombe intervals (formulas)', () => {
    const w = proportionInterval(7, 20, { method: 'wald' })
    closeArray([w.lower, w.upper], F.binomial.wald, 1e-12)
    const t = F.binomial.twoProportion
    const r = twoProportionZTest(t.k1, t.n1, t.k2, t.n2)
    close(r.statistic, t.z, 1e-12)
    close(r.pValue, t.p, 1e-10)
    closeArray([r.ci!.lower, r.ci!.upper], t.wald, 1e-12)
    const nc = differenceOfProportionsInterval(t.k1, t.n1, t.k2, t.n2, { method: 'newcombe' })
    closeArray([nc.lower, nc.upper], t.newcombe, 1e-9)
  })
})

describe('χ² and G tests (scipy chisquare, power_divergence, chi2_contingency)', () => {
  const g = F.tables.gof
  it('goodness of fit', () => {
    const cases: [TestResult, number[]][] = [
      [chiSquareGoodnessOfFit(g.observed), g.uniform],
      [chiSquareGoodnessOfFit(g.observed, { expected: g.expected }), g.withExpected],
      [chiSquareGoodnessOfFit(g.observed, { ddof: 1 }), g.ddof],
      [gTestGoodnessOfFit(g.observed), g.g],
      [gTestGoodnessOfFit(g.observed, { expected: g.expected }), g.gExpected],
    ]
    for (const [r, ref] of cases) {
      close(r.statistic, ref[0], 1e-12)
      close(r.pValue, ref[1], 1e-10)
      consistent(r)
    }
    expect(() => chiSquareGoodnessOfFit(g.observed, { expected: [1, 2, 3, 4, 5, 6] })).toThrow()
  })
  each(F.tables.contingency.map((c: { table: number[][] }): [string, any] => [JSON.stringify(c.table), c]))(
    'independence %s',
    (_, c) => {
      for (const correction of [true, false]) {
        const p = chiSquareIndependence(c.table, { correction })
        const gt = gTestIndependence(c.table, { correction })
        const rp = c[`pearson/${correction ? 'True' : 'False'}`] ?? c[`pearson/${correction}`]
        const rg = c[`log-likelihood/${correction ? 'True' : 'False'}`] ?? c[`log-likelihood/${correction}`]
        close(p.statistic, rp[0], 1e-12)
        close(p.pValue, rp[1], 1e-10)
        expect(p.df).toBe(rp[2])
        close(gt.statistic, rg[0], 1e-12)
        close(gt.pValue, rg[1], 1e-10)
        consistent(p)
      }
      close(chiSquareIndependence(c.table).effectSize!.value, c.cramer, 1e-12)
    },
  )
})

describe("Fisher's exact test and odds ratios (scipy fisher_exact, contingency.odds_ratio)", () => {
  each(F.tables.fisher.map((c: { table: number[][] }): [string, any] => [JSON.stringify(c.table), c]))('%s', (_, c) => {
    for (const alternative of ['two-sided', 'less', 'greater'] as const) {
      const r = fisherExact(c.table, { alternative })
      const ref = c[alternative]
      close(r.pValue, ref.p, 1e-9)
      close(r.effectSize!.value, ref.or, 1e-6)
      close(r.ci!.lower, ref.ci[0], 1e-6)
      close(r.ci!.upper, ref.ci[1], 1e-6)
      consistent(r)
    }
    const s = oddsRatio(c.table, { kind: 'sample' })
    close(s.value, c.sample.or, 1e-12)
    if (c.table.flat().every((v: number) => v > 0)) {
      close(s.ci.lower, c.sample.ci[0], 1e-10)
      close(s.ci.upper, c.sample.ci[1], 1e-10)
    }
  })
})

describe('rank tests (scipy mannwhitneyu, wilcoxon)', () => {
  each(
    F.ranks.mannWhitney.map((c: { name: string; alternative: string }): [string, any] => [
      `${c.name} ${c.alternative}`,
      c,
    ]),
  )('Mann–Whitney %s', (_, c) => {
    const r = mannWhitneyU(c.x, c.y, { alternative: c.alternative, method: c.method, continuity: c.continuity })
    close(r.statistic, c.U, 1e-12)
    close(r.pValue, c.p, 1e-9)
    if (c.method === 'exact') consistent(r)
  })
  each(
    F.ranks.wilcoxon.map((c: { name: string; alternative: string }): [string, any] => [
      `${c.name} ${c.alternative}`,
      c,
    ]),
  )('Wilcoxon %s', (_, c) => {
    const r = wilcoxonSignedRank(c.x, c.y, { alternative: c.alternative, method: c.method, correction: c.correction })
    close(r.statistic, c.plus, 1e-12)
    close(r.pValue, c.p, 1e-9)
    if (c.method === 'exact') consistent(r)
  })
  it('auto picks the exact law without ties and small samples', () => {
    expect(mannWhitneyU([1, 2, 3], [4, 5, 6, 7]).method).toMatch(/exact/)
    expect(mannWhitneyU([1, 2, 2], [4, 5, 6, 7]).method).toMatch(/normal/)
    // U = 0 for complete separation: the exact two-sided p is 2/C(7, 3).
    close(mannWhitneyU([1, 2, 3], [4, 5, 6, 7]).pValue, 2 / 35, 1e-12)
  })
})

describe('Kolmogorov–Smirnov: one-sided and the Smirnov law (scipy ks_1samp, ks_2samp, smirnov)', () => {
  const Phi = (v: number) => normalCdf(v) as number
  it('one-sided one- and two-sample tests', () => {
    for (const alternative of ['greater', 'less'] as const) {
      const one = ksTest(F.ks.x, Phi, { alternative })
      close(one.statistic, F.ks.one[alternative][0], 1e-12)
      close(one.pValue, F.ks.one[alternative][1], 1e-9)
      const two = ksTest(F.ks.x, F.ks.y, { alternative })
      close(two.statistic, Math.abs(F.ks.two[alternative][0]), 1e-12)
      close(two.pValue, F.ks.two[alternative][1], 1e-9)
    }
  })
  it('smirnovSf', () => {
    for (const [d, n, p] of F.ks.smirnov) close(smirnovSf(d, n), p, 1e-9)
  })
  it('the rejection region is the Kolmogorov quantile', () => {
    const r = ksTest(F.ks.x, Phi)
    const [[lo]] = rejectionRegion(r, 0.05)
    // kstwo.isf(0.05, 10) = 0.40925
    close(lo, 0.4092460847, 1e-6)
  })
})

describe('Shapiro–Wilk (scipy shapiro, single precision)', () => {
  each(F.shapiro.map((c: { x: number[] }): [number, any] => [c.x.length, c]))('n = %i', (_, c) => {
    const r = shapiroWilk(c.x)
    close(r.statistic, c.W, 2e-6)
    close(r.pValue, c.p, 2e-4)
  })
})

describe('Ljung–Box, Grubbs, ANOVA (formulas; scipy f_oneway)', () => {
  it('Ljung–Box and Box–Pierce', () => {
    for (const c of F.ljungBox.cases) {
      const r = ljungBox(F.ljungBox.x, { lags: c.lags, fitted: c.fitted })
      close(r.statistic, c.Q, 1e-10)
      close(r.pValue, c.p, 1e-9)
      const b = ljungBox(F.ljungBox.x, { lags: c.lags, fitted: c.fitted, boxPierce: true })
      close(b.statistic, c.boxPierce, 1e-10)
      close(b.pValue, c.pBoxPierce, 1e-9)
    }
  })
  it('Grubbs: statistic, Bonferroni p-value and the tabulated critical value', () => {
    for (const alternative of ['two-sided', 'greater', 'less'] as const) {
      const r = grubbs(F.grubbs.x, { alternative })
      close(r.statistic, F.grubbs[alternative].G, 1e-12)
      close(r.pValue, F.grubbs[alternative].p, 1e-7)
    }
    expect(grubbs(F.grubbs.x).value).toBe(3.9)
    const [[crit]] = rejectionRegion(grubbs(F.grubbs.x), 0.05)
    close(crit, F.grubbs.critical10, 1e-6)
    close(crit, 2.29, 2e-3) // Grubbs (1969), n = 10, two-sided 5%
  })
  it('one-way ANOVA', () => {
    const r = oneWayAnova(F.anova.groups)
    close(r.statistic, F.anova.F, 1e-12)
    close(r.pValue, F.anova.p, 1e-9)
    close(r.effectSize!.value, F.anova.eta2, 1e-12)
    expect(r.null.name).toBe('FisherSnedecor')
    consistent(r)
  })
})

describe('multiple testing (statsmodels formulas; scipy false_discovery_control)', () => {
  const m = F.multiple
  it('adjusted p-values and rejections', () => {
    const procs = { bonferroni, holm, hochberg, benjaminiHochberg, benjaminiYekutieli }
    for (const [key, f] of Object.entries(procs)) {
      const r = f(m.p, { alpha: 0.05 })
      closeArray(toFlat(r.adjusted), m[key], 1e-12)
      expect(toFlat(r.rejected)).toEqual(m[key].map((q: number) => (q <= 0.05 ? 1 : 0)))
    }
    closeArray(toFlat(benjaminiHochberg(m.p).adjusted), m.scipyBH, 1e-12)
    closeArray(toFlat(benjaminiYekutieli(m.p).adjusted), m.scipyBY, 1e-12)
  })
  it('ordering: Bonferroni ≥ Holm ≥ Hochberg', () => {
    const b = toFlat(bonferroni(m.p).adjusted)
    const h = toFlat(holm(m.p).adjusted)
    const g = toFlat(hochberg(m.p).adjusted)
    b.forEach((v, i) => {
      expect(v).toBeGreaterThanOrEqual(h[i] - 1e-15)
      expect(h[i]).toBeGreaterThanOrEqual(g[i] - 1e-15)
    })
  })
})

describe('the protocol', () => {
  it('pValueOf: the likelihood rule equals scipy binomtest', () => {
    const r = binomialTest(7, 20, { p: 0.3 })
    close(pValueOf(r.null, 7, 'likelihood'), r.pValue, 1e-15)
  })
  it('rejection regions of continuous laws are quantiles', () => {
    const law = Normal(0, 1)
    const [[, lo], [hi]] = rejectionRegion({ null: law, tail: 'both' }, 0.05)
    close(lo, -1.959963984540054, 1e-12)
    close(hi, 1.959963984540054, 1e-12)
  })
  it('input errors are reported', () => {
    expect(() => oneSampleTTest([1])).toThrow()
    expect(() => oneSampleTTest([1, NaN, 2])).toThrow()
    expect(() => binomialTest(3, 2)).toThrow()
    expect(() => pairedTTest([1, 2, 3], [1, 2])).toThrow()
  })
})

describe('FisherSnedecor against scipy.stats.f', () => {
  each(F.f.map((c: { d1: number; d2: number }): [string, any] => [`F(${c.d1}, ${c.d2})`, c]))('%s', (_, c) => {
    const f = FisherSnedecor(c.d1, c.d2)
    c.x.forEach((x: number, i: number) => {
      close(f.logProb(x) as number, c.logpdf[i], 1e-10)
      close(f.cdf(x) as number, c.cdf[i], 1e-10)
      close(f.survival(x) as number, c.sf[i], 1e-10)
    })
    c.p.forEach((p: number, i: number) => {
      // Far tails inherit the inverse incomplete beta's ~5·10⁻⁸ relative accuracy.
      close(f.quantile(p) as number, c.ppf[i], 1e-7)
      close(f.isf(p) as number, c.isf[i], 1e-7)
    })
    close(f.mean() as number, c.mean, 1e-12)
    close(f.variance() as number, c.var, 1e-12)
    close(f.entropy() as number, c.entropy, 1e-10)
  })
})

describe('the registry', () => {
  it('every test is a function named by its key, whose null family is registered', async () => {
    const { testRegistry } = await import('aifn-compute/probability/tests')
    const { distributionRegistry } = await import('aifn-compute/probability/distributions')
    expect(Object.keys(testRegistry).length).toBeGreaterThanOrEqual(25)
    for (const [key, f] of Object.entries(testRegistry)) {
      expect(f.name).toBe(key)
      expect(f.info.kind).toBe('test')
      expect(f.info.null === 'exact' || f.info.null in distributionRegistry).toBe(true)
      expect(f.info.alternatives.length).toBeGreaterThan(0)
    }
  })
})

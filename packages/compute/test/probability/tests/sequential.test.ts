/**
 * Sequential tests and survival estimators: group-sequential boundaries against boundaries solved independently with
 * scipy's multivariate normal cdf and against published tables; the SPRT, mSPRT and confidence sequence by their
 * formulas and by simulation (type-I error and coverage at every n); CUSUM's recursion and average run lengths
 * against Montgomery's table; Kaplan–Meier against scipy.stats.ecdf, Nelson–Aalen by its definition, and the log-rank
 * test against scipy.stats.logrank.
 */
import { describe, expect, it } from 'vitest'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { Normal } from 'aifn-compute/probability/distributions'
import {
  confidenceSequence,
  constantBoundaries,
  cusum,
  cusumAverageRunLength,
  groupSequentialBoundaries,
  groupSequentialTest,
  kaplanMeier,
  logRankTest,
  msprt,
  nelsonAalen,
  normalMixtureLogLikelihoodRatio,
  normalMixtureRadius,
  spentAlpha,
  sprt,
  waldBoundaries,
} from 'aifn-compute/probability/tests'
import { fixture } from '../../fixtures'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F = fixture<any>('probability/tests')
// Parameterised cases over the untyped fixture rows.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const each = (cases: readonly (readonly [string | number, any])[]) => it.each(cases as [string, any][])
const close = (a: number, b: number, tol: number) =>
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1e-300, Math.abs(b)))
const draws = (seed: string, n: number, mean = 0, sd = 1) =>
  Float64Array.from(toFlat(normals(stream(seed), n, mean, sd)))

describe('group-sequential boundaries', () => {
  each(
    F.boundaries.spending.map((c: { family: string; sides: number }): [string, any] => [
      `${c.family} ${c.sides}-sided`,
      c,
    ]),
  )('alpha spending %s matches the multivariate-normal solution', (_, c) => {
    const b = groupSequentialBoundaries(c.information, {
      alpha: c.alpha,
      spending: { family: c.family, rho: 2 },
      sides: c.sides,
    })
    Array.from(b.z).forEach((z, k) => close(z, c.z[k], 2e-5))
    close(
      b.crossing.reduce((a, v) => a + v, 0),
      c.alpha,
      1e-6,
    )
  })
  it("Lan–DeMets O'Brien–Fleming, K = 5 (ldbounds, gsDesign)", () => {
    const b = groupSequentialBoundaries(5)
    Array.from(b.z).forEach((z, k) => close(z, F.boundaries.ldObf5[k], 5e-4))
  })
  it("Pocock and O'Brien–Fleming constants, K = 5 (Jennison and Turnbull, Tables 2.1 and 2.3)", () => {
    close(constantBoundaries('pocock', 5).z[0], F.boundaries.pocock5, 5e-4)
    const o = constantBoundaries('obrien-fleming', 5)
    close(o.z[4], F.boundaries.obrienFleming5, 5e-4)
    close(o.z[0], F.boundaries.obrienFleming5 * Math.sqrt(5), 5e-4)
  })
  it('spending functions spend α by t = 1', () => {
    for (const family of ['obrien-fleming', 'pocock', 'power'] as const) {
      close(spentAlpha({ family }, 0.05, 1), 0.05, 1e-12)
      expect(spentAlpha({ family }, 0.05, 0)).toBe(0)
      expect(spentAlpha({ family }, 0.05, 0.3)).toBeLessThan(spentAlpha({ family }, 0.05, 0.6))
    }
  })
  it('the test stops at the first look whose |Z| crosses', () => {
    const b = groupSequentialBoundaries([20, 40, 60])
    const x = draws('gs', 60, 0.9)
    const s = run(groupSequentialTest(x, { looks: [20, 40, 60], boundaries: b, sigma: 1 }), undefined, 100)
    expect(s.rejected).toBe(true)
    expect([20, 40, 60]).toContain(s.n)
    const n = s.n
    const m = x.slice(0, n).reduce((a, v) => a + v, 0) / n
    close(s.z, m * Math.sqrt(n), 1e-12)
  })
})

describe('SPRT', () => {
  it("Wald's boundaries and the log-likelihood ratio", () => {
    const { lower, upper } = waldBoundaries(0.05, 0.2)
    close(lower, Math.log(0.2 / 0.95), 1e-15)
    close(upper, Math.log(0.8 / 0.05), 1e-15)
    const x = [0.3, 1.2, 0.8, -0.1]
    const h0 = Normal(0, 1)
    const h1 = Normal(1, 1)
    const tr = trace(sprt(x, { h0, h1 }), undefined, 10, { keep: 'all' })
    let llr = 0
    x.forEach((v, i) => {
      llr += v - 0.5 // log N(v; 1, 1) − log N(v; 0, 1)
      close(tr.steps[i + 1].llr, llr, 1e-12)
    })
  })
  it('holds the type-I error near α under H₀ and stops sooner under H₁', () => {
    const h0 = Normal(0, 1)
    const h1 = Normal(0.5, 1)
    let reject = 0
    let length0 = 0
    let length1 = 0
    const R = 1000
    for (let r = 0; r < R; r++) {
      const s0 = run(sprt(draws(`sprt0-${r}`, 400), { h0, h1 }), undefined, 400)
      if (s0.decision === 'reject-null') reject++
      length0 += s0.t
      length1 += run(sprt(draws(`sprt1-${r}`, 400, 0.5), { h0, h1 }), undefined, 400).t
    }
    // Wald: the error is at most α/(1 − β) = 0.0625; with overshoot it is below α.
    expect(reject / R).toBeLessThan(0.0625)
    expect(length1 / R).toBeLessThan(40)
    expect(length0 / R).toBeLessThan(40)
  })
})

describe('mSPRT and confidence sequences (normal mixture)', () => {
  const m = { sigma: 1.3, tau: 0.4 }
  it('the mixture likelihood ratio and radius by their formulas', () => {
    const n = 37
    const mean = 0.21
    const v = m.sigma ** 2 + n * m.tau ** 2
    close(
      normalMixtureLogLikelihoodRatio(n, mean, 0.05, m),
      0.5 * Math.log(m.sigma ** 2 / v) + (n * n * m.tau ** 2 * (mean - 0.05) ** 2) / (2 * m.sigma ** 2 * v),
      1e-14,
    )
    // At θ₀ = x̄ ± radius the likelihood ratio is exactly 1/α.
    const r = normalMixtureRadius(n, 0.05, m)
    close(normalMixtureLogLikelihoodRatio(n, mean, mean + r, m), Math.log(20), 1e-12)
  })
  it('the always-valid p-value only decreases and the test stops at p ≤ α', () => {
    const tr = trace(msprt(draws('msprt', 300, 0.5), { ...m, sigma: 1 }), undefined, 300, { keep: 'all' })
    const p = tr.steps.map((s) => s.pValue)
    p.slice(1).forEach((v, i) => expect(v).toBeLessThanOrEqual(p[i]))
    const last = tr.steps[tr.steps.length - 1]
    expect(last.rejected).toBe(true)
    expect(last.pValue).toBeLessThanOrEqual(0.05)
  })
  it('peeking at every n: the confidence sequence covers at all times and mSPRT holds α', () => {
    let missed = 0
    let rejected = 0
    const R = 500
    const N = 300
    for (let r = 0; r < R; r++) {
      const x = draws(`cs-${r}`, N, 0.7, 1)
      const s = run(confidenceSequence(x, { sigma: 1, tau: 0.5, mu0: 0.7 }), undefined, N)
      if (s.lower > 0.7 || s.upper < 0.7) missed++
      if (run(msprt(x, { sigma: 1, tau: 0.5, theta0: 0.7 }), undefined, N).rejected) rejected++
    }
    // Both are bounded by α = 0.05 (Ville); allow sampling noise (sd ≈ 0.01 at R = 500).
    expect(missed / R).toBeLessThan(0.07)
    expect(rejected / R).toBeLessThan(0.07)
  })
})

describe('CUSUM', () => {
  it('the tabular recursion with resets', () => {
    const x = [0.2, 1.4, 2.1, 1.8, 2.6, -0.3, 0.1]
    const tr = trace(cusum(x, { k: 0.5, h: 3 }), undefined, 10, { keep: 'all' })
    let up = 0
    let lo = 0
    x.forEach((v, i) => {
      up = Math.max(0, up + v - 0.5)
      lo = Math.max(0, lo - v - 0.5)
      const alarm = up > 3 || lo > 3
      expect(tr.steps[i + 1].alarm).toBe(alarm)
      if (alarm) [up, lo] = [0, 0]
      close(tr.steps[i + 1].upper + 1, up + 1, 1e-14)
    })
    expect(tr.final.firstAlarm).toBe(4)
  })
  it('average run lengths: Montgomery (2009), Table 9.3 (k = ½, two-sided)', () => {
    close(cusumAverageRunLength({ h: 4, sides: 'both' }), 168, 0.01)
    close(cusumAverageRunLength({ h: 5, sides: 'both' }), 465, 0.01)
    close(cusumAverageRunLength({ h: 4, shift: 1, sides: 'both' }), 8.38, 0.01)
    close(cusumAverageRunLength({ h: 5, shift: 1, sides: 'both' }), 10.4, 0.01)
    // Siegmund's approximation by its formula.
    const b = 5 + 1.166
    close(cusumAverageRunLength({ h: 5, method: 'siegmund' }), (Math.exp(b) - b - 1) / 0.5, 1e-12)
  })
})

describe('survival (scipy ecdf, logrank)', () => {
  const S = F.survival
  it('Kaplan–Meier with Greenwood intervals, linear and log–log', () => {
    for (const interval of ['linear', 'log-log'] as const) {
      const km = kaplanMeier(S.time, S.event, { interval })
      expect(toFlat(km.time)).toEqual(S.km.time)
      toFlat(km.survival).forEach((v, i) => close(v, S.km.survival[i], 1e-12))
      toFlat(km.lower).forEach((v, i) => close(v, S.km[interval][0][i], 1e-9))
      toFlat(km.upper).forEach((v, i) => close(v, S.km[interval][1][i], 1e-9))
    }
    const median = S.km.time[S.km.survival.findIndex((v: number) => v <= 0.5)] ?? NaN
    expect(kaplanMeier(S.time, S.event).median).toBe(median)
  })
  it('Nelson–Aalen', () => {
    const na = nelsonAalen(S.time, S.event)
    for (const [key, ref] of [
      ['cumulativeHazard', 'H'],
      ['standardError', 'se'],
      ['lower', 'lower'],
      ['upper', 'upper'],
    ] as const)
      toFlat(na[key]).forEach((v, i) => close(v, S.na[ref][i], 1e-12))
  })
  it('log-rank: two groups (scipy) and three (the k-sample formula)', () => {
    const L = S.logrank
    const r = logRankTest(L.time, L.event, L.group)
    close(r.statistic, L.chi2, 1e-10)
    close(r.pValue, L.p, 1e-9)
    const r3 = logRankTest(L.time, L.event, L.group3)
    close(r3.statistic, L.chi2_3, 1e-10)
    close(r3.pValue, L.p3, 1e-9)
    expect(r3.df).toBe(2)
  })
})

describe('streams are data: each sequential algorithm terminates at the end of its data', () => {
  it.each([
    ['sprt', () => sprt([0.1, 0.2], { h0: Normal(0, 1), h1: Normal(0.01, 1) })],
    ['msprt', () => msprt([0.1, 0.2], { sigma: 1, tau: 1 })],
    ['confidenceSequence', () => confidenceSequence([0.1, 0.2], { sigma: 1, tau: 1 })],
    ['cusum', () => cusum([0.1, 0.2])],
  ] as const)('%s', (_, make) => {
    const s = run(make() as Parameters<typeof run>[0], undefined, 10, { stream: child(stream(1), 'x') })
    expect(s.t).toBe(2)
  })
})

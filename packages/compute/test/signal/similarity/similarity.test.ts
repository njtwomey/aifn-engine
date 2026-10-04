/**
 * Time-series similarity against stumpy (matrix profile, MASS) and plain-numpy references (banded DTW, LB_Keogh, PAA,
 * SAX, MINDIST); SCRIMP++ converging to STOMP; motif and discord picks on planted structure; lower-bound laws.
 */
import { describe, expect, it } from 'vitest'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import {
  discords,
  distanceProfile,
  dtw,
  keoghEnvelope,
  lbKeogh,
  lbKim,
  matrixProfile,
  motifs,
  paa,
  sax,
  saxBreakpoints,
  saxMinDist,
  scrimpProfile,
  scrimpSteps,
  slidingMeanStd,
  zNormalise,
} from 'aifn-compute/signal/similarity'
import { fixture } from '../../fixtures'

const F = fixture('signal/similarity') as {
  series: number[]
  m: number
  profile: number[]
  index: number[]
  query: number[]
  mass: number[]
  massRaw: number[]
  dtw: {
    x: number[]
    y: number[]
    window: number
    cost: 'squared' | 'absolute'
    distance: number
    accumulated: number[][]
  }[]
  keogh: { x: number[]; y: number[]; window: number; upper: number[]; lower: number[]; lbKeogh: number }
  sax: {
    x: number[]
    y: number[]
    paa7: number[]
    paa10: number[]
    breakpoints6: number[]
    word: number[]
    word2: number[]
    mindist: number
    euclidean: number
  }
}

const near = (got: ArrayLike<number>, want: ArrayLike<number>, digits = 8) =>
  Array.from(want).forEach((v, i) =>
    Number.isFinite(v) ? expect(got[i]).toBeCloseTo(v, digits) : expect(got[i]).toBe(v),
  )

describe('distance and matrix profiles against stumpy', () => {
  it('MASS equals stumpy.mass, z-normalised and raw', () => {
    near(toFlat(distanceProfile(F.query, F.series)), F.mass, 7)
    near(toFlat(distanceProfile(F.query, F.series, { normalise: false })), F.massRaw, 7)
  })
  it('STOMP equals stumpy.stump (profile and index)', () => {
    const mp = matrixProfile(F.series, F.m)
    near(toFlat(mp.profile), F.profile, 7)
    expect(Array.from(toFlat(mp.index))).toEqual(F.index)
  })
  it('SCRIMP++ is anytime: never rises, starts close after PreSCRIMP, and ends exactly at STOMP', () => {
    const steps = scrimpSteps(F.series, F.m, { diagonalsPerStep: 40 })
    const t = trace(steps, undefined, 1000, { keep: 'all' })
    expect(t.final.converged).toBe(true)
    near(toFlat(t.final.profile), F.profile, 7)
    for (let s = 1; s < t.steps.length; s++) {
      const prev = toFlat(t.steps[s - 1].profile)
      toFlat(t.steps[s].profile).forEach((v, i) => expect(v).toBeLessThanOrEqual(prev[i] + 1e-12))
    }
    // PreSCRIMP alone already finds the planted motif's distance.
    const first = toFlat(t.steps[0].profile)
    expect(first[80]).toBeCloseTo(F.profile[80], 6)
    expect(scrimpProfile(t.final, F.m).profile).toBe(t.final.profile)
    const without = run(scrimpSteps(F.series, F.m, { prescrimp: false, diagonalsPerStep: 100 }), undefined, 1000)
    near(toFlat(without.profile), F.profile, 7)
  })
  it('the top motif is the planted pair and the top discord the planted spike', () => {
    const mp = matrixProfile(F.series, F.m)
    const [top] = motifs(mp, { count: 2 })
    expect(Math.abs(top.a - 80)).toBeLessThanOrEqual(3)
    expect(Math.abs(top.b - 400)).toBeLessThanOrEqual(3)
    const [d] = discords(mp)
    expect(d.at).toBeGreaterThan(260 - F.m)
    expect(d.at).toBeLessThan(290)
    expect(motifs(mp, { count: 5 }).length).toBe(5)
  })
  it('sliding statistics and z-normalisation', () => {
    const { mean, std } = slidingMeanStd(F.series, F.m)
    const w = F.series.slice(10, 10 + F.m)
    const mu = w.reduce((a, b) => a + b, 0) / F.m
    expect(toFlat(mean)[10]).toBeCloseTo(mu, 12)
    expect(toFlat(std)[10]).toBeCloseTo(Math.sqrt(w.reduce((a, b) => a + (b - mu) ** 2, 0) / F.m), 12)
    const z = toFlat(zNormalise(w))
    expect(z.reduce((a, b) => a + b, 0)).toBeCloseTo(0, 12)
    expect(z.reduce((a, b) => a + b * b, 0) / F.m).toBeCloseTo(1, 12)
  })
})

describe('dynamic time warping', () => {
  it('equals the O(nm) recursion with a Sakoe–Chiba band (distance and table)', () => {
    for (const c of F.dtw) {
      const r = dtw(c.x, c.y, { window: c.window, cost: c.cost })
      expect(r.distance).toBeCloseTo(c.distance, 10)
      ;(toRows(r.accumulated) as number[][]).forEach((row, i) => near(row, c.accumulated[i], 9))
      // The path is monotone and continuous from corner to corner, inside the band.
      expect(r.path[0]).toEqual([0, 0])
      expect(r.path.at(-1)).toEqual([c.x.length - 1, c.y.length - 1])
      for (let k = 1; k < r.path.length; k++) {
        const di = r.path[k][0] - r.path[k - 1][0]
        const dj = r.path[k][1] - r.path[k - 1][1]
        expect(di >= 0 && dj >= 0 && di + dj >= 1 && di <= 1 && dj <= 1).toBe(true)
        expect(Math.abs(r.path[k][0] - r.path[k][1])).toBeLessThanOrEqual(c.window)
      }
    }
  })
  it('a zero-width band is the Euclidean distance; a wider band never costs more', () => {
    const c = F.dtw[0]
    const euclid = Math.sqrt(c.x.reduce((s, v, i) => s + (v - c.y[i]) ** 2, 0))
    expect(dtw(c.x, c.y, { window: 0 }).distance).toBeCloseTo(euclid, 12)
    expect(dtw(c.x, c.y, { window: 10 }).distance).toBeLessThanOrEqual(dtw(c.x, c.y, { window: 3 }).distance)
  })
  it('LB_Keogh and LB_Kim match the reference and lower-bound DTW', () => {
    const k = F.keogh
    const env = keoghEnvelope(k.y, k.window)
    near(toFlat(env.upper), k.upper, 12)
    near(toFlat(env.lower), k.lower, 12)
    expect(lbKeogh(k.x, k.y, k.window)).toBeCloseTo(k.lbKeogh, 12)
    for (const w of [0, 3, 8]) {
      const d = dtw(k.x, k.y, { window: w }).distance
      expect(lbKeogh(k.x, k.y, w)).toBeLessThanOrEqual(d + 1e-12)
      expect(lbKim(k.x, k.y)).toBeLessThanOrEqual(d + 1e-12)
      expect(lbKeogh(k.x, k.y, w, { cost: 'absolute' })).toBeLessThanOrEqual(
        dtw(k.x, k.y, { window: w, cost: 'absolute' }).distance + 1e-12,
      )
    }
  })
})

describe('PAA and SAX', () => {
  it('PAA with whole and fractional frames, breakpoints, words and MINDIST match the reference', () => {
    const s = F.sax
    near(toFlat(paa(s.x, 7)), s.paa7, 12)
    near(toFlat(paa(s.x, 10)), s.paa10, 12)
    near(toFlat(saxBreakpoints(6)), s.breakpoints6, 10)
    const a = sax(s.x, { segments: 8, alphabet: 6 })
    const b = sax(s.y, { segments: 8, alphabet: 6 })
    expect(Array.from(toFlat(a.symbols))).toEqual(s.word)
    expect(Array.from(toFlat(b.symbols))).toEqual(s.word2)
    expect(a.word).toBe(s.word.map((v) => String.fromCharCode(97 + v)).join(''))
    const md = saxMinDist(a.symbols, b.symbols, s.x.length, 6)
    expect(md).toBeCloseTo(s.mindist, 10)
    expect(md).toBeLessThanOrEqual(s.euclidean)
  })
})

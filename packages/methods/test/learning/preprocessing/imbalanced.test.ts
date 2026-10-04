import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import {
  adasyn,
  adasynWeights,
  borderlineSmote,
  borderStatus,
  randomOverSample,
  randomUnderSample,
  removeTomekLinks,
  smote,
  type Resampled,
} from 'aifn-methods/learning/preprocessing'
import { fixture } from '../../fixtures'

type F = {
  x: number[][]
  y: number[]
  danger: number[]
  noise: number[]
  adasyn_counts: number[]
  tomek_kept: number[]
  counts: { smote: number[]; borderline: number[]; adasyn: number[] }
}
const f = fixture<F>('learning/preprocessing')
const x = fromData(Float64Array.from(f.x.flat()), [f.x.length, 2])
const y = f.y

const counts = (r: Resampled) => {
  const c = [0, 0]
  for (const v of toFlat(r.y)) c[v]++
  return c
}

/** Every synthetic row lies on the segment between its two minority parents. */
function onSegments(r: Resampled) {
  const X = toFlat(r.x)
  const n0 = f.x.length
  for (let k = 0; k < r.synthetic.base.length; k++) {
    const row = n0 + k
    const b = r.synthetic.base[k]
    const nb = r.synthetic.neighbour[k]
    const u = r.synthetic.gap[k]
    expect(y[b]).toBe(1)
    expect(y[nb]).toBe(1)
    expect(u).toBeGreaterThanOrEqual(0)
    expect(u).toBeLessThan(1)
    for (let j = 0; j < 2; j++) expect(X[row * 2 + j]).toBeCloseTo(f.x[b][j] + u * (f.x[nb][j] - f.x[b][j]), 12)
  }
}

describe('imbalanced resampling (imbalanced-learn)', () => {
  it('random over- and under-sampling balance the classes', () => {
    expect(counts(randomOverSample(stream(1), x, y))).toEqual([150, 150])
    const under = randomUnderSample(stream(1), x, y)
    expect(counts(under)).toEqual([30, 30])
    expect(new Set(under.origin).size).toBe(60)
  })
  it('SMOTE matches imbalanced-learn’s counts and interpolates between minority neighbours', () => {
    const r = smote(stream(2), x, y)
    expect(counts(r)).toEqual(f.counts.smote)
    onSegments(r)
  })
  it('classifies borderline rows as imbalanced-learn does', () => {
    const b = borderStatus(x, y, 1, 10)
    expect(b.status.map((s) => (s === 'danger' ? 1 : 0))).toEqual(f.danger)
    expect(b.status.map((s) => (s === 'noise' ? 1 : 0))).toEqual(f.noise)
    const r = borderlineSmote(stream(3), x, y)
    expect(counts(r)).toEqual(f.counts.borderline)
    onSegments(r)
    const danger = new Set(Array.from(b.rows).filter((_, i) => b.status[i] === 'danger'))
    for (const base of r.synthetic.base) expect(danger.has(base)).toBe(true)
  })
  it('allocates ADASYN’s synthetic rows as imbalanced-learn does', () => {
    expect(Array.from(adasynWeights(x, y, 1, 5).counts)).toEqual(f.adasyn_counts)
    const r = adasyn(stream(4), x, y)
    expect(counts(r)).toEqual(f.counts.adasyn)
    onSegments(r)
  })
  it('removes the same Tomek links as imbalanced-learn', () => {
    const r = removeTomekLinks(x, y)
    expect(Array.from(r.origin)).toEqual(f.tomek_kept)
  })
})

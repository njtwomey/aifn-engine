/**
 * `semiSupervisedSplit`: the site's two variants are reproduced exactly (the per-class draw of the self-training,
 * graph-based and in-practice notes; the global draw of the consistency and generative notes, inlined below as it was
 * written), and the counts of the site's self-checks hold.
 */
import { describe, expect, it } from 'vitest'
import { child, permutation, stream, type Stream } from 'aifn-compute/foundation/random'
import { float64Data, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { moons, semiSupervisedSplit } from 'aifn-methods/data/synthetic'

/** The site's variant A (per-class child streams), inlined. */
function variantA(s: Stream, values: number[], perClass: number): Int32Array {
  const byClass = new Map<number, number[]>()
  values.forEach((k, i) => byClass.set(k, [...(byClass.get(k) ?? []), i]))
  const labels = new Int32Array(values.length).fill(-1)
  for (const [k, rows] of [...byClass.entries()].sort((a, b) => a[0] - b[0])) {
    const order = toFlat(permutation(child(s, 'class', k), rows.length))
    for (const r of order.slice(0, perClass)) labels[rows[r]] = k
  }
  return labels
}

/** The site's variant B (one global permutation), inlined: the labelled rows in increasing order. */
function variantB(s: Stream, y: number[], perClass: number): number[] {
  const n = y.length
  const K = Math.max(...y) + 1
  const want = new Array(K).fill(perClass)
  const order = float64Data(permutation(s, n))
  const taken = new Array(K).fill(0)
  const labelled: number[] = []
  for (let a = 0; a < n; a++) {
    const i = order[a]
    if (taken[y[i]] < want[y[i]]) {
      taken[y[i]]++
      labelled.push(i)
    }
  }
  return labelled.sort((a, b) => a - b)
}

describe('semiSupervisedSplit', () => {
  it('keeps 4 labels per class of 100 moons, agreeing with y, and hides 92 (the self-training check)', () => {
    const y = Array.from(toFlat(moons(stream(1), { n: 100 }).y))
    const split = semiSupervisedSplit(stream(2), y, 4)
    expect([0, 1].map((k) => split.labelled.filter((i) => y[i] === k).length)).toEqual([4, 4])
    expect(split.labelled.every((i) => split.labels[i] === y[i])).toBe(true)
    expect(split.unlabelled.length).toBe(92)
    expect(Array.from(split.labelledY)).toEqual(split.labelled.map((i) => y[i]))
  })

  it('reproduces variant A exactly, from a tensor or an array', () => {
    for (const seed of [1, 7, 42]) {
      const d = moons(stream(seed), { n: 60 })
      const y = Array.from(toFlat(d.y))
      for (const m of [0, 3, 40]) {
        const want = variantA(child(stream(seed), 'split'), y, m)
        expect(Array.from(semiSupervisedSplit(child(stream(seed), 'split'), d.y, m).labels)).toEqual(Array.from(want))
        expect(Array.from(semiSupervisedSplit(child(stream(seed), 'split'), y, m).labels)).toEqual(Array.from(want))
      }
    }
  })

  it("reproduces variant B exactly with draw 'global', and splits x (the consistency check: 8 and 92 × 2)", () => {
    const d = moons(stream('check/moons'), { n: 100, noise: 0.1 })
    const y = Array.from(toFlat(d.y))
    const sp = semiSupervisedSplit(stream('check/split'), d.y, 4, { x: d.x, draw: 'global', strict: true })
    expect(sp.labelledY.length).toBe(8)
    expect(float64Data(sp.unlabelledX as Tensor).length).toBe(92 * 2)
    expect(Array.from(sp.labelledY).filter((c) => c === 0).length).toBe(4)
    expect(sp.labelled).toEqual(variantB(stream('check/split'), y, 4))
    const x = toFlat(d.x)
    expect(Array.from(toFlat(sp.labelledX))).toEqual(sp.labelled.flatMap((i) => [x[2 * i], x[2 * i + 1]]))
    expect(Array.from(sp.unlabelledTruth)).toEqual(sp.unlabelled.map((i) => y[i]))
  })

  it('a short class keeps all its points, or throws with strict; bad input throws', () => {
    const y = [0, 0, 0, 1]
    const split = semiSupervisedSplit(stream(3), y, [1, 2])
    expect(split.labelledY.filter((c) => c === 1).length).toBe(1)
    expect(split.labelled.length).toBe(2)
    expect(() => semiSupervisedSplit(stream(3), y, 2, { strict: true })).toThrow(/class 1/)
    expect(() => semiSupervisedSplit(stream(3), [0, -1], 1)).toThrow(/non-negative/)
    expect(() => semiSupervisedSplit(stream(3), y, [1])).toThrow(/counts/)
    expect(() => semiSupervisedSplit(stream(3), y, 1.5)).toThrow(/perClass/)
  })
})

import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { digitGlyphs, digits } from 'aifn-methods/data/synthetic'
import { dbnRun, dbnUp, hiddenProbabilities, rbm, type DbnRun } from 'aifn-methods/generative/boltzmann'

const last = <T>(g: Generator<T, T>): T => {
  let r = g.next()
  let v = r.value
  while (!r.done) {
    v = r.value
    r = g.next()
  }
  return r.value ?? v
}

describe('deep belief network (greedy layer-wise)', () => {
  const data = digits(stream('dbn data'), { perClass: 20, flip: 0.05, noise: 0 })
  const run: DbnRun = last(
    dbnRun(data, { layers: [32, 12], epochs: 150, fineTuneEpochs: 200, labelsPerClass: 1, seed: 3 }),
  )

  it('the up pass of a one-layer stack is the RBM conditional', () => {
    const m = rbm(stream('one'), 6, 4)
    const v = [1, 0, 1, 1, 0, 0]
    expect(Array.from(dbnUp({ layers: [m] }, v)[0])).toEqual(Array.from(hiddenProbabilities(m, v)))
  })

  it('each layer lowers its reconstruction error, and the top RBM raises its exact likelihood', () => {
    for (const err of run.reconstructionError) expect(err[err.length - 1]).toBeLessThan(0.6 * err[1])
    const ll = run.logLikelihood[1].filter((v) => !Number.isNaN(v))
    expect(ll[ll.length - 1]).toBeGreaterThan(ll[0] + 1)
    expect(Number.isNaN(run.logLikelihood[0][0])).toBe(true)
  })

  it('samples from the stack look like digits (near a glyph in Hamming distance)', () => {
    const glyphs = toFlat(digitGlyphs())
    const nearest = (s: Float64Array, offset: number) => {
      let best = Infinity
      for (let d = 0; d < 10; d++) {
        let h = 0
        for (let p = 0; p < 35; p++) h += +(s[offset + p] > 0.5) !== glyphs[d * 35 + p] ? 1 : 0
        best = Math.min(best, h)
      }
      return best
    }
    const mean = (shot: Float64Array) => {
      let t = 0
      for (let c = 0; c < shot.length / 35; c++) t += nearest(shot, c * 35)
      return (t * 35) / shot.length
    }
    const first = run.checkpoints[0].samples
    const final = run.checkpoints[run.checkpoints.length - 1].samples
    expect(mean(final)).toBeLessThan(4)
    expect(mean(final)).toBeLessThan(mean(first) - 3)
  })

  it('fine-tunes from one label per class, and pretraining beats random weights', () => {
    const ft = run.fineTune!
    expect(ft.labelled).toBe(10)
    expect(ft.pretrained.length).toBe(201)
    expect(ft.pretrained[200]).toBeGreaterThan(0.8)
    expect(ft.pretrained[200]).toBeGreaterThanOrEqual(ft.random[200])
    expect(run.done).toBe(true)
  })

  it('is deterministic in the seed', () => {
    const a = last(dbnRun(data, { layers: [8], epochs: 5, fineTuneEpochs: 3, seed: 1 }))
    const b = last(dbnRun(data, { layers: [8], epochs: 5, fineTuneEpochs: 3, seed: 1 }))
    expect(a.checkpoints.at(-1)!.samples).toEqual(b.checkpoints.at(-1)!.samples)
    expect(a.fineTune).toEqual(b.fineTune)
  })
})

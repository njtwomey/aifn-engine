import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { fewShotEpisode, prototypeLogits, prototypicalRun, type PrototypicalRun } from 'aifn-methods/learning/transfer'

const last = <T>(g: Generator<T, T>): T => {
  let r = g.next()
  let v = r.value
  while (!r.done) {
    v = r.value
    r = g.next()
  }
  return r.value ?? v
}

describe('prototypical networks', () => {
  it('logits are minus squared distances to the class means (brute force)', () => {
    const e = fewShotEpisode(stream('episode'), { ways: 3, shots: 4, queries: 2 })
    const sx = toFlat(e.support.x)
    const sy = toFlat(e.support.y)
    const qx = toFlat(e.query.x)
    const z = toFlat(unwrap(prototypeLogits((x) => x, { x: e.support.x, y: sy }, e.query.x, 3)) as Tensor)
    for (let i = 0; i < 6; i++)
      for (let k = 0; k < 3; k++) {
        let mx = 0
        let my = 0
        for (let j = 0; j < sy.length; j++)
          if (sy[j] === k) {
            mx += sx[2 * j] / 4
            my += sx[2 * j + 1] / 4
          }
        expect(z[i * 3 + k]).toBeCloseTo(-((qx[2 * i] - mx) ** 2) - (qx[2 * i + 1] - my) ** 2, 10)
      }
  })

  it('a class is a direction: support points of a class share an angle, not a radius', () => {
    const e = fewShotEpisode(stream('dir'), { ways: 4, shots: 20, queries: 1, angularNoise: 0 })
    const x = toFlat(e.support.x)
    for (let i = 0; i < 80; i++) {
      const c = Math.floor(i / 20)
      expect(Math.cos(Math.atan2(x[2 * i + 1], x[2 * i]) - e.angles[c])).toBeCloseTo(1, 10)
    }
  })

  it('episodic training learns a metric for unseen classes that beats raw-input prototypes', () => {
    const run: PrototypicalRun = last(prototypicalRun({ episodes: 600, seed: 2, testEpisodes: 60 }))
    const acc = run.checkpoints.at(-1)!.accuracy
    expect(acc).toBeGreaterThan(run.rawAccuracy + 0.1)
    expect(acc).toBeGreaterThan(0.85)
    expect(run.finished).toBe(true)
  })
})

import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { type ClassificationTruth } from 'aifn-methods/data'
import { gaussianRing } from 'aifn-methods/data/synthetic'
import { mixtureLogDensityOf, squareGrid } from 'aifn-methods/generative'
import { ganRun, modeCoverage, optimalDiscriminator, type GanRun } from 'aifn-methods/generative/gan'

const ring = gaussianRing(stream(0), { n: 800 })
const model = (ring.meta.truth as ClassificationTruth).model

describe('GAN diagnostics', () => {
  it('real points hit every mode; points from one mode hit one', () => {
    const fresh = gaussianRing(stream(1), { n: 800 })
    const all = modeCoverage(model, fresh.x, { real: ring.x })
    expect(all.hit).toBe(8)
    expect(all.quality).toBeGreaterThan(0.95)
    const one = gaussianRing(stream(2), { n: [800, 0, 0, 0, 0, 0, 0, 0] })
    const collapsed = modeCoverage(model, one.x, { real: ring.x })
    expect(collapsed.hit).toBe(1)
    expect(collapsed.quality).toBeGreaterThan(0.95)
    expect(collapsed.perMode[0]).toBeGreaterThan(750)
  })

  it('law: D* ≈ 1/2 near the data when the generator samples the data', () => {
    const g = squareGrid(2.5, 30)
    const ld = mixtureLogDensityOf(model, g.points)
    const fresh = gaussianRing(stream(3), { n: 3000 }).x
    const { value } = optimalDiscriminator(ld, fresh, g.points)
    // On the modes (where p_data is high) the KDE of data samples matches p_data within its smoothing.
    const near = Array.from(value).filter((_, i) => ld[i] > 0)
    expect(near.length).toBeGreaterThan(0)
    for (const v of near) expect(v).toBeGreaterThan(0.2)
    const avg = near.reduce((a, b) => a + b, 0) / near.length
    expect(avg).toBeGreaterThan(0.25)
    expect(avg).toBeLessThan(0.75)
    expect(Array.from(value).every((v) => Number.isNaN(v) || (v >= 0 && v <= 1))).toBe(true)
  })

  it('ganRun streams snapshots and ends with every checkpoint, deterministically', () => {
    const go = () => {
      let last: GanRun | undefined
      let count = 0
      for (const s of ganRun(ring, {
        steps: 20,
        batchSize: 16,
        hidden: [8, 8],
        checkpoints: 4,
        grid: 8,
        samples: 32,
        arrows: 8,
        seed: 3,
      })) {
        last = s
        count++
      }
      return { last: last!, count }
    }
    const a = go()
    expect(a.count).toBeGreaterThan(1)
    expect(a.last.finished).toBe(true)
    expect(a.last.checkpoints.map((c) => c.step)).toEqual([0, 5, 10, 15, 20])
    expect(a.last.criticLoss.length).toBe(20)
    expect(a.last.checkpoints[0].field.length).toBe(64)
    expect(a.last.checkpoints[0].optimal).not.toBeNull()
    expect(a.last.checkpoints[0].coverage!.modes).toBe(8)
    expect(Array.from(go().last.generatorLoss)).toEqual(Array.from(a.last.generatorLoss))
    expect(toFlat(fromData(a.last.checkpoints[4].samples, [32, 2])).every(Number.isFinite)).toBe(true)
  })
})

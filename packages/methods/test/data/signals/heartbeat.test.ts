/**
 * The synthetic ECG: the traces of the AI Field Notes heartbeat figures, pinned (the generator moved here from the site
 * and must keep its output); the beat shapes; the parts of the trace adding up; registration.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { defaults } from 'aifn-compute/foundation/space'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { datasetRegistry, generate } from 'aifn-methods/data'
import { syntheticBeat, syntheticEcg, type SyntheticEcg } from 'aifn-methods/data/signals'

describe('syntheticEcg', () => {
  // The site's figures draw from stream(`dictionary-learning/heartbeat/${seed}`); these values are its traces.
  it('reproduces the heartbeat figures exactly', () => {
    const a = syntheticEcg(stream('dictionary-learning/heartbeat/1'), {
      n: 900,
      ectopic: 0.25,
      noise: 0.08,
      wander: 0.2,
    })
    const y = Array.from(toFlat(a.y))
    expect(y.slice(0, 4)).toEqual([0.25343557078476925, 0.21564919397294272, 0.1644020052079149, 0.11055458533796454])
    expect(y.slice(-2)).toEqual([-0.03761880153445171, -0.044549427464058454])
    expect(y.reduce((s, v) => s + v, 0)).toBeCloseTo(93.19527581843899, 10)
    expect(a.beats.slice(0, 5)).toEqual([
      { start: 0, ectopic: false },
      { start: 60, ectopic: false },
      { start: 123, ectopic: true },
      { start: 177, ectopic: false },
      { start: 238, ectopic: false },
    ])
    expect([a.beats.length, a.beats.filter((b) => b.ectopic).length]).toEqual([15, 3])
    const b = syntheticEcg(stream('dictionary-learning/heartbeat/3'), {
      n: 1500,
      ectopic: 0.1,
      noise: 0.08,
      wander: 0.2,
    })
    expect([b.beats.length, b.beats.filter((x) => x.ectopic).length]).toEqual([26, 4])
    expect(Array.from(toFlat(b.y)).reduce((s, v) => s + v, 0)).toBeCloseTo(108.59963456067139, 10)
  })

  it('is the clean trace plus noise, with beats of 54 to 66 samples back to back', () => {
    const ecg = syntheticEcg(stream(7), { n: 2000, noise: 0 })
    expect(Array.from(toFlat(ecg.y))).toEqual(Array.from(toFlat(ecg.clean)))
    ecg.beats.forEach((beat, i) => {
      if (i === 0) expect(beat.start).toBe(0)
      else expect(beat.start - ecg.beats[i - 1].start).toBeGreaterThanOrEqual(54)
      if (i > 0) expect(beat.start - ecg.beats[i - 1].start).toBeLessThanOrEqual(66)
    })
    expect(ecg.signal.fs).toBe(60)
    expect(toFlat(ecg.x)[60]).toBe(1)
  })

  it('draws ectopic beats at about the asked fraction', () => {
    const ecg = syntheticEcg(stream(9), { n: 30000, ectopic: 0.3 })
    const share = ecg.beats.filter((b) => b.ectopic).length / ecg.beats.length
    expect(share).toBeGreaterThan(0.25)
    expect(share).toBeLessThan(0.35)
  })

  it('is registered', () => {
    const entry = datasetRegistry.syntheticEcg
    expect(entry.info).toMatchObject({ task: 'sequence', output: 'dataset', random: true })
    const made = generate(entry, stream(1), defaults(entry.info.knobs)) as SyntheticEcg
    expect(made.y.shape).toEqual([900])
  })
})

describe('syntheticBeat', () => {
  it('peaks at the R spike, earlier and narrower than the ectopic beat', () => {
    const normal = syntheticBeat(60)
    const ectopic = syntheticBeat(60, true)
    const peak = (b: number[]) => b.indexOf(Math.max(...b))
    expect(peak(normal)).toBe(21) // u = 0.35
    expect(peak(ectopic)).toBeGreaterThan(peak(normal))
    const wide = (b: number[]) => b.filter((v) => v > Math.max(...b) / 2).length
    expect(wide(ectopic)).toBeGreaterThan(2 * wide(normal))
  })
})

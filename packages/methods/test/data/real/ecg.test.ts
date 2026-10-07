import { createHash } from 'node:crypto'
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { defaults } from 'aifn-compute/foundation/space'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { ECG_BEAT_CLASSES, ecgBeats, ecgDatasetRegistry, mitBihEcg } from 'aifn-methods/data/real/ecg'
import { generate, type Dataset } from 'aifn-methods/data'
import { expectInfo } from '../../registry'
import * as vendored from '../../../src/data/real/ecg/record'

describe('mitBihEcg', () => {
  const record = mitBihEcg()

  it('records its provenance and open licence', () => {
    expect(record.meta.source).toMatch(/Moody and Mark \(2001\)/)
    expect(record.meta.source).toMatch(/Open Data Commons Attribution License v1\.0/)
    expect(record.meta.sha256).toBe('f20ad3365fb9b7f845d0e5c48b6fe67081377ee466c3a220b7f69f35c8958baf')
    // The vendored samples are byte for byte what scripts/ecg.py wrote (guards against hand edits).
    expect(createHash('sha256').update(vendored.SAMPLES).digest('hex')).toBe(
      'd7cb3b6a4512ca14d2f89be52675301fd437988164018ad53e12df375a7cf440',
    )
    // 108000 int16 samples: 216000 bytes, 288000 base64 characters.
    expect(vendored.SAMPLES.length).toBe(288_000)
  })

  it("holds SciPy's five minutes at 360 Hz, in mV", () => {
    expect(record.signal.fs).toBe(360)
    const x = toFlat(record.signal.data)
    expect(x.length).toBe(108_000)
    // scipy.datasets.electrocardiogram()[:3] is (ADC − 1024) / 200 of 975, 981, 987.
    expect(Array.from(x.slice(0, 3))).toEqual([-0.245, -0.215, -0.185])
    expect(Math.min(...x)).toBeCloseTo((327 - 1024) / 200, 12)
    expect(Math.max(...x)).toBeCloseTo((1754 - 1024) / 200, 12)
  })

  it('carries the reference beat annotations of the span', () => {
    const counts: Record<string, number> = {}
    for (const b of record.beats) counts[b.symbol] = (counts[b.symbol] ?? 0) + 1
    expect(counts).toEqual({ N: 358, V: 93, F: 56, Q: 2 })
    expect(record.beats.every((b, i) => i === 0 || b.sample > record.beats[i - 1].sample)).toBe(true)
    expect(record.beats.find((b) => b.symbol === 'V')?.label).toBe(ECG_BEAT_CLASSES.indexOf('ventricular ectopic'))
    // Annotations sit on the QRS: the largest deflection from the local median within 150 ms of a beat is within 30 ms
    // of its fiducial point, for nearly every beat.
    const x = toFlat(record.signal.data)
    const near = record.beats.filter((b) => {
      const lo = Math.max(0, b.sample - 54)
      const window = Array.from(x.slice(lo, b.sample + 55))
      const median = [...window].sort((p, q) => p - q)[window.length >> 1]
      const deflection = window.map((v) => Math.abs(v - median))
      return Math.abs(lo + deflection.indexOf(Math.max(...deflection)) - b.sample) <= 11
    })
    expect(near.length / record.beats.length).toBeGreaterThan(0.9)
  })
})

describe('mitBihEcg decimated', () => {
  it('is at 360/q Hz with the beats moved to that rate', () => {
    const full = mitBihEcg()
    const low = mitBihEcg({ decimate: 4 })
    expect(low.signal.fs).toBe(90)
    expect(toFlat(low.signal.data).length).toBe(27_000)
    expect(low.beats.map((b) => b.sample)).toEqual(full.beats.map((b) => Math.round(b.sample / 4)))
    expect(low.beats.map((b) => b.time)).toEqual(full.beats.map((b) => b.time))
    // An R peak survives the lowpass: the decimated signal near each normal beat matches the full one there.
    const x = toFlat(full.signal.data)
    const y = toFlat(low.signal.data)
    const normal = full.beats.filter((b) => b.symbol === 'N' && b.sample > 400 && b.sample < 107_600)
    const err = normal.map((b) => Math.abs(y[Math.round(b.sample / 4)] - x[b.sample]))
    expect(err.sort((p, q) => p - q)[err.length >> 1]).toBeLessThan(0.25)
    expect(() => mitBihEcg({ decimate: 0 })).toThrow(/decimate/)
  })
})

describe('ecgBeats', () => {
  it('cuts one window per beat that fits, labelled by AAMI class', () => {
    const beats = ecgBeats()
    expect(beats.x.shape).toEqual([508, 235])
    expect(beats.meta.featureNames?.[90]).toBe('0.0 ms')
    const y = Array.from(toFlat(beats.y!))
    expect(y.filter((c) => c === 2).length).toBe(93)
    // Row r is the signal around beat r: its centre sample is the signal at the beat.
    const record = mitBihEcg()
    const x = toFlat(record.signal.data)
    const first = record.beats.find((b) => b.sample >= 90)!
    expect(toRows(beats.x)[0][90]).toBe(x[first.sample])
    expect(toFlat(beats.t!)[0]).toBeCloseTo(first.time, 12)
  })

  it('centres each window on its median when asked', () => {
    const rows = toRows(ecgBeats({ before: 0.1, after: 0.1, centre: true }).x)
    for (const row of rows.slice(0, 20)) expect([...row].sort((a, b) => a - b)[row.length >> 1]).toBe(0)
  })

  it('rejects an empty window', () => {
    expect(() => ecgBeats({ before: 0, after: 0 })).toThrow(/not both zero/)
  })

  it('is registered', () => {
    expectInfo(ecgDatasetRegistry, 'dataset')
    expect(ecgDatasetRegistry.mitBihEcg.info).toMatchObject({ output: 'series', random: false })
    const entry = ecgDatasetRegistry.ecgBeats
    const made = generate(entry, stream(1), defaults(entry.info.knobs)) as Dataset
    expect(made.x.shape).toEqual([508, 235])
  })
})

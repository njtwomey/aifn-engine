import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, fromData } from 'aifn-compute/foundation/tensor'
import {
  decodeSequence,
  encodeSequence,
  modularArithmetic,
  modularValue,
  SEQUENCE_TASKS,
  sequenceTasks,
  sequenceTaskTruth,
} from 'aifn-methods/data/synthetic'

describe('sequenceTasks', () => {
  it('has the requested split sizes and one row width', () => {
    for (const task of SEQUENCE_TASKS) {
      const d = sequenceTasks(stream(task), { task, n: 50, testN: 20, minLength: 3, maxLength: 5 })
      expect(d.train.tokens.shape, task).toEqual([50, d.width])
      expect(d.test.targets.shape, task).toEqual([20, d.width])
      expect(d.train.prompts.length, task).toBe(50)
    }
  })

  it('is deterministic by seed and differs across seeds', () => {
    const a = sequenceTasks(stream(1), { n: 30 })
    const b = sequenceTasks(stream(1), { n: 30 })
    const c = sequenceTasks(stream(2), { n: 30 })
    expect(a.train.prompts).toEqual(b.train.prompts)
    expect(toFlat(a.train.tokens)).toEqual(toFlat(b.train.tokens))
    expect(a.train.prompts).not.toEqual(c.train.prompts)
  })

  it('writes rows as ^ prompt = answer . with weights on the answer and end mark', () => {
    const d = sequenceTasks(stream(3), { task: 'reverse', n: 20 })
    const [n, L] = d.train.tokens.shape
    const tokens = toFlat(d.train.tokens)
    const targets = toFlat(d.train.targets)
    const weights = toFlat(d.train.weights)
    for (let i = 0; i < n; i++) {
      const row = Array.from(tokens.slice(i * L, (i + 1) * L))
      expect(decodeSequence(row)).toBe(`^${d.train.prompts[i]}=${d.train.answers[i]}.`)
      expect(d.train.answers[i]).toBe([...d.train.prompts[i]].reverse().join(''))
      // Targets are the row shifted left; the weighted targets spell the answer and the end mark.
      expect(Array.from(targets.slice(i * L, (i + 1) * L - 1))).toEqual(row.slice(1))
      const scored = row.slice(1).filter((_, j) => weights[i * L + j] === 1)
      expect(decodeSequence(scored)).toBe(`${d.train.answers[i]}.`)
    }
  })

  it('draws test lengths above the training range for length generalisation', () => {
    const d = sequenceTasks(stream(4), { task: 'copy', n: 40, testN: 40, minLength: 2, maxLength: 4, testLength: 7 })
    expect(Math.max(...d.train.lengths)).toBeLessThanOrEqual(4)
    expect(Math.min(...d.test.lengths)).toBeGreaterThanOrEqual(5)
    expect(Math.max(...d.test.lengths)).toBeLessThanOrEqual(7)
  })

  it('answers each task exactly', () => {
    const answer = (task: (typeof SEQUENCE_TASKS)[number], prompt: string) => sequenceTaskTruth(task).answer(prompt)
    expect(answer('copy', 'abca')).toBe('abca')
    expect(answer('reverse', 'abc')).toBe('cba')
    expect(answer('sort', 'dbca')).toBe('abcd')
    expect(answer('dyck1', '(()(')).toBe('))')
    expect(answer('dyck2', '([])')).toBe('')
    expect(() => answer('dyck2', '(]')).toThrow()
    expect(answer('dyck2', '[(')).toBe(')]')
    expect(answer('addition', '58+67')).toBe('125')
    expect(answer('induction', 'cadbea')).toBe('d')
  })

  it('generates prompts that its truth accepts, for every task', () => {
    for (const task of SEQUENCE_TASKS) {
      const d = sequenceTasks(stream(5), { task, n: 40, minLength: 3, maxLength: 6 })
      d.train.prompts.forEach((p, i) => expect(d.truth.answer(p), `${task}: ${p}`).toBe(d.train.answers[i]))
      if (task === 'dyck1') expect(d.train.prompts.every((p) => /^[()]+$/.test(p))).toBe(true)
    }
  })

  it('scores outputs exactly, ignoring text after the end mark', () => {
    const truth = sequenceTaskTruth('reverse')
    expect(truth.score('abc', 'cba.')).toEqual({ exact: true, tokenAccuracy: 1 })
    expect(truth.score('abc', 'cba.xyz')).toEqual({ exact: true, tokenAccuracy: 1 })
    expect(truth.score('abc', 'cab.')).toEqual({ exact: false, tokenAccuracy: 0.5 })
    expect(truth.score('abc', 'cb')).toEqual({ exact: false, tokenAccuracy: 0.5 })
  })

  it('round-trips the vocabulary and rejects unknown characters', () => {
    expect(decodeSequence(encodeSequence('^ab(=)].'))).toBe('^ab(=)].')
    expect(() => encodeSequence('z')).toThrow()
  })
})

describe('modularArithmetic', () => {
  it('covers the table and splits it by fraction, disjointly', () => {
    const d = modularArithmetic(stream(6), { p: 13, fraction: 0.3 })
    expect(d.table.x.shape).toEqual([169, 2])
    expect(d.train.x.shape[0]).toBe(Math.round(0.3 * 169))
    expect(d.train.x.shape[0] + d.test.x.shape[0]).toBe(169)
    const rows = new Set([...toFlat(d.train.rows), ...toFlat(d.test.rows)])
    expect(rows.size).toBe(169)
    const div = modularArithmetic(stream(6), { p: 13, op: '/' })
    expect(div.table.x.shape[0]).toBe(13 * 12)
    expect(() => modularArithmetic(stream(6), { p: 12, op: '/' })).toThrow()
  })

  it('is deterministic by seed', () => {
    const a = modularArithmetic(stream(7), { p: 11 })
    const b = modularArithmetic(stream(7), { p: 11 })
    const c = modularArithmetic(stream(8), { p: 11 })
    expect(toFlat(a.train.rows)).toEqual(toFlat(b.train.rows))
    expect(toFlat(a.train.rows)).not.toEqual(toFlat(c.train.rows))
  })

  it('labels every pair by the operation, and the truth decides the same', () => {
    for (const op of ['+', '-', '*', '/'] as const) {
      const d = modularArithmetic(stream(9), { p: 7, op })
      const x = toFlat(d.table.x)
      const y = toFlat(d.table.y)
      y.forEach((v, i) => {
        const [a, b] = [x[2 * i], x[2 * i + 1]]
        expect(v).toBe(modularValue(op, a, b, 7))
        if (op === '/') expect((v * b) % 7).toBe(a)
      })
      expect(toFlat(d.truth.decide(d.table.x))).toEqual(y)
      expect(d.truth.bayesRisk).toBe(0)
    }
  })

  it('carries an orthonormal Fourier basis and measures spectra on it', () => {
    for (const p of [7, 8]) {
      const { truth } = modularArithmetic(stream(10), { p })
      const B = toFlat(truth.fourierBasis)
      for (let i = 0; i < p; i++)
        for (let j = 0; j < p; j++) {
          let dot = 0
          for (let a = 0; a < p; a++) dot += B[a * p + i] * B[a * p + j]
          expect(dot).toBeCloseTo(i === j ? 1 : 0, 10)
        }
    }
    // A table of cos(2π·3a/p) and sin(2π·3a/p) has all its power at frequency 3.
    const p = 11
    const { truth } = modularArithmetic(stream(11), { p })
    const table = fromData(
      Float64Array.from({ length: 2 * p }, (_, q) => {
        const a = Math.floor(q / 2)
        return q % 2 === 0 ? Math.cos((2 * Math.PI * 3 * a) / p) : Math.sin((2 * Math.PI * 3 * a) / p)
      }),
      [p, 2],
    )
    const power = truth.spectrum(table)
    expect(power.length).toBe(6)
    expect(power[3]).toBeCloseTo(1, 10)
    expect(truth.frequencies).toEqual([0, 1, 2, 3, 4, 5])
  })
})

import { describe, expect, it } from 'vitest'
import { stream, uniform } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { langevinParticles, persistentLangevin, chainBuffer } from 'aifn-compute/inference/stochastic'

/** The score of N(μ, σ²I) in 2-d, batched. */
const gaussianScore = (mu: [number, number], sd: number) => (x: Tensor) => {
  const v = toFlat(x)
  return fromData(
    Float64Array.from(v, (xi, k) => -(xi - mu[k % 2]) / (sd * sd)),
    x.shape,
  )
}
const fresh = (s: ReturnType<typeof stream>, n: number) => uniform(s, -3, 3, { shape: [n, 2] }) as Tensor

describe('langevinParticles', () => {
  it('law: on N(μ, σ²) the particles reach mean μ and ULA variance σ²/(1 − α/(2σ²))', () => {
    const mu: [number, number] = [1, -2]
    const sd = 0.5
    const alpha = 0.02
    const x0 = fromData(new Float64Array(2 * 4000), [4000, 2])
    const end = run(langevinParticles(gaussianScore(mu, sd), { stepSize: alpha }), { x: x0 }, 300, {
      stream: stream(1),
    })
    const v = toFlat(end.x)
    const m = [0, 0]
    for (let i = 0; i < 4000; i++) for (let k = 0; k < 2; k++) m[k] += v[2 * i + k] / 4000
    let s2 = 0
    for (let i = 0; i < 4000; i++) s2 += (v[2 * i] - m[0]) ** 2 / 3999
    expect(m[0]).toBeCloseTo(1, 1)
    expect(m[1]).toBeCloseTo(-2, 1)
    const exact = (sd * sd) / (1 - alpha / (2 * sd * sd))
    expect(Math.abs(s2 - exact) / exact).toBeLessThan(0.08)
  })

  it('without noise it is gradient ascent on log π, clipped to the bound', () => {
    const x0 = fromData(Float64Array.of(5, 5), [1, 2])
    const end = run(langevinParticles(gaussianScore([0, 0], 1), { stepSize: 0.1, noise: 0, bound: 2 }), { x: x0 }, 1)
    // 5 + 0.1·(−5) = 4.5, clipped to 2.
    expect(Array.from(toFlat(end.x))).toEqual([2, 2])
  })
})

describe('persistentLangevin', () => {
  const score = gaussianScore([0, 0], 1)
  it('is deterministic in its stream and writes the drawn slots back', () => {
    const buffer = chainBuffer(stream('b'), 50, fresh)
    const a = persistentLangevin(score, buffer, stream(7), 10, { steps: 5, stepSize: 0.05, fresh })
    const b = persistentLangevin(score, buffer, stream(7), 10, { steps: 5, stepSize: 0.05, fresh })
    expect(Array.from(toFlat(a.x))).toEqual(Array.from(toFlat(b.x)))
    const slots = toFlat(a.slots)
    const next = toFlat(a.buffer.samples)
    const x = toFlat(a.x)
    // The last write to a slot wins when a slot is drawn twice.
    const lastRow = new Map<number, number>()
    slots.forEach((slot, i) => lastRow.set(slot, i))
    for (const [slot, i] of lastRow) expect([next[2 * slot], next[2 * slot + 1]]).toEqual([x[2 * i], x[2 * i + 1]])
    // Untouched slots keep their values.
    const old = toFlat(buffer.samples)
    for (let j = 0; j < 50; j++) if (!lastRow.has(j)) expect(next[2 * j]).toBe(old[2 * j])
  })

  it('restarts every chain with ρ = 1 and none with ρ = 0', () => {
    const buffer = chainBuffer(stream('b'), 20, fresh)
    const all = persistentLangevin(score, buffer, stream(3), 15, { steps: 0, reinitialise: 1, fresh })
    expect(Array.from(all.restarted).every((r) => r === 1)).toBe(true)
    const none = persistentLangevin(score, buffer, stream(3), 15, { steps: 0, reinitialise: 0, fresh })
    expect(Array.from(none.restarted).every((r) => r === 0)).toBe(true)
    // With no steps and no restarts the draws are the stored chains.
    const stored = toFlat(buffer.samples)
    const slots = toFlat(none.slots)
    const x = toFlat(none.x)
    slots.forEach((slot, i) => expect(x[2 * i]).toBe(stored[2 * slot]))
  })
})

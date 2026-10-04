/** `dormandPrinceRows`: each row's solve, steps and work equal `dormandPrince` on that row alone. */
import { describe, expect, it } from 'vitest'
import { dormandPrince, dormandPrinceRows } from 'aifn-compute/dynamics/ode'
import { run } from 'aifn-compute/foundation/trace'
import { fromData, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

/** A row-wise field: a damped rotation whose rate differs by row, plus a time-dependent push. */
const rate = [0.5, 3, 12, 40]
const fieldRow = (r: number, t: number, a: number, b: number): [number, number] => [
  -rate[r] * b - 0.3 * a + Math.sin(3 * t),
  rate[r] * a - 0.3 * b,
]

describe('dormandPrinceRows', () => {
  const x0 = Float64Array.of(1, 0, 0.5, 0.5, -1, 0.2, 0.3, -0.7)
  const tols = [
    { rtol: 1e-3, atol: 1e-6 },
    { rtol: 1e-6, atol: 1e-9 },
  ]
  for (const tol of tols)
    it(`matches dormandPrince row by row at rtol ${tol.rtol}`, () => {
      const sol = dormandPrinceRows(
        (t, x, rows) => {
          const out = new Float64Array(x.length)
          rows.forEach((r, j) => out.set(fieldRow(r, t[j], x[2 * j], x[2 * j + 1]), 2 * j))
          return out
        },
        x0,
        2,
        { tEnd: 2, ...tol },
      )
      for (let r = 0; r < rate.length; r++) {
        const alone = run(
          dormandPrince(
            (t, x) => {
              const v = toFlat(unwrap(x as Value) as Tensor)
              return fromData(Float64Array.from(fieldRow(r, t as number, v[0], v[1])), [2])
            },
            { tEnd: 2, ...tol },
          ),
          { x0: [x0[2 * r], x0[2 * r + 1]] },
          10_000,
        )
        expect(sol.evaluations[r]).toBe(alone.evaluations)
        expect(sol.steps[r]).toBe(alone.t)
        expect(sol.rejected[r]).toBe(alone.rejected)
        const v = toFlat(unwrap(alone.x as Value) as Tensor)
        expect(sol.x[2 * r]).toBeCloseTo(v[0], 12)
        expect(sol.x[2 * r + 1]).toBeCloseTo(v[1], 12)
      }
      // Faster rotations take more work.
      expect(sol.evaluations[3]).toBeGreaterThan(sol.evaluations[0])
    })

  it('integrates backwards and handles an empty interval', () => {
    const decay = dormandPrinceRows((_t, x) => x.map((v) => -v), Float64Array.of(1, 2), 1, {
      t0: 1,
      tEnd: 0,
      rtol: 1e-8,
      atol: 1e-10,
    })
    expect(decay.x[0]).toBeCloseTo(Math.E, 6)
    expect(decay.x[1]).toBeCloseTo(2 * Math.E, 6)
    const none = dormandPrinceRows((_t, x) => x, Float64Array.of(1), 1, { t0: 1, tEnd: 1 })
    expect(none.evaluations[0]).toBe(0)
  })

  it('resumes a solve without restarting it', () => {
    const f = (_t: Float64Array, x: Float64Array, rows: Int32Array) => {
      const out = new Float64Array(x.length)
      rows.forEach((r, j) => out.set([-rate[r] * x[2 * j + 1], rate[r] * x[2 * j]], 2 * j))
      return out
    }
    const tol = { rtol: 1e-7, atol: 1e-9 }
    const first = dormandPrinceRows(f, x0, 2, { tEnd: 0.5, ...tol })
    const fresh = dormandPrinceRows(f, first.x, 2, { t0: 0.5, tEnd: 1, ...tol })
    const resumed = dormandPrinceRows(f, first.x, 2, { t0: 0.5, tEnd: 1, ...tol, resume: first })
    for (let r = 0; r < rate.length; r++) {
      // Exact rotation by rate·1 radians.
      const [a, b] = [x0[2 * r], x0[2 * r + 1]]
      const c = Math.cos(rate[r])
      const s = Math.sin(rate[r])
      expect(resumed.x[2 * r]).toBeCloseTo(c * a - s * b, 5)
      expect(resumed.x[2 * r + 1]).toBeCloseTo(s * a + c * b, 5)
    }
    // The slow row skips the restart's two evaluations and the short steps of a fresh start.
    expect(resumed.evaluations[0]).toBeLessThan(fresh.evaluations[0])
    // A row whose step is NaN starts afresh.
    const mixed = dormandPrinceRows(f, first.x, 2, {
      t0: 0.5,
      tEnd: 1,
      ...tol,
      resume: {
        nextStepSize: Float64Array.from(first.nextStepSize, (v, i) => (i === 0 ? NaN : v)),
        derivative: first.derivative,
      },
    })
    expect(mixed.evaluations[0]).toBe(fresh.evaluations[0])
    expect(mixed.evaluations[1]).toBe(resumed.evaluations[1])
  })
})

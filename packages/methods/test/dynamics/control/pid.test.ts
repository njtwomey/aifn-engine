import { describe, expect, it } from 'vitest'
import { pidLoop } from 'aifn-methods/dynamics/control'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { transferFunction } from 'aifn-compute/systems'
import { expectProtocol } from '../../protocol'

// A first-order lag 1/(s + 1).
const plant = transferFunction([1], [1, 1])

describe('pidLoop', () => {
  it('proportional control of a first-order lag settles at the closed-loop gain kp/(1 + kp)', () => {
    const s = run(pidLoop(plant, { kp: 4 }, { dt: 0.01 }), {}, 2000)
    expect(s.y).toBeCloseTo(4 / 5, 6)
    expect(s.e).toBeCloseTo(1 / 5, 6)
    expect(s.time).toBeCloseTo(20, 9)
    expect(s.t).toBe(2000)
  })

  it('integral action removes the steady-state error, also under a load disturbance', () => {
    const s = run(pidLoop(plant, { kp: 2, ki: 1 }, { dt: 0.01, disturbance: 0.5 }), {}, 4000)
    expect(s.y).toBeCloseTo(1, 5)
    // At rest the plant (gain 1) gives y = u + d, so u = 0.5, all of it from the integrator (e = 0).
    expect(s.u).toBeCloseTo(0.5, 5)
    expect(s.integral).toBeCloseTo(0.5, 5)
  })

  it('actuator limits clip the input and the clamp keeps the integrator from winding up', () => {
    const opts = { dt: 0.01, setpoint: 1, uMin: -1.2, uMax: 1.2 }
    const tr = trace(pidLoop(plant, { kp: 5, ki: 5 }, { ...opts, antiWindup: 'clamp' }), {}, 1500, {
      record: { u: (s) => s.u, y: (s) => s.y, sat: (s) => (s.saturated ? 1 : 0) },
    })
    const u = toFlat(tr.series.u)
    expect(Math.max(...u)).toBeLessThanOrEqual(1.2)
    expect(toFlat(tr.series.sat).some((v) => v === 1)).toBe(true)
    const none = trace(pidLoop(plant, { kp: 5, ki: 5 }, { ...opts, antiWindup: 'none' }), {}, 1500, {
      record: { y: (s) => s.y },
    })
    // Windup overshoots further than the clamped loop.
    expect(Math.max(...toFlat(none.series.y))).toBeGreaterThan(Math.max(...toFlat(tr.series.y)))
    expect(toFlat(tr.series.y).at(-1)).toBeCloseTo(1, 4)
  })

  it('stops at tEnd and follows the trace protocol', () => {
    const tr = trace(
      pidLoop(plant, { kp: 1, ki: 0.5, kd: 0.1, filter: 0.05 }, { dt: 0.1, tEnd: 1, delay: 0.2 }),
      {},
      100,
    )
    expect(tr.meta.stopped).toBe('done')
    expect(tr.final.time).toBeCloseTo(1, 9)
    expectProtocol(
      pidLoop(plant, { kp: 1, ki: 0.5, kd: 0.1, filter: 0.05 }, { dt: 0.1, delay: 0.2 }),
      {},
      {
        n: 20,
        record: { y: (s) => s.y },
      },
    )
  })
})

/**
 * aifn-compute/nn/sequence HiPPO memories: LegS row 3 against the closed form, LegT's input column, a unit step's exact
 * LegS coefficients by projection, online state and reconstruction, and online LegT and LegS against the windowed
 * projection on a smooth input (the HiPPO note's checks).
 */
import { describe, expect, it } from 'vitest'
import { slice, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  hippoBasis,
  hippoLegS,
  hippoLegT,
  hippoOnline,
  hippoProject,
  hippoReconstruct,
  hippoWindow,
} from 'aifn-compute/nn/sequence'

const maxError = (got: number[], want: number[]) => Math.max(...got.map((g, i) => Math.abs(g - want[i])))
const [s3, s5, s7] = [Math.sqrt(3), Math.sqrt(5), Math.sqrt(7)]
const dt = 1e-3
const T = 2000
const step = Array.from({ length: T }, (_, k) => ((k + 1) * dt > 1 ? 1 : 0))
const smooth = Array.from({ length: T }, (_, k) => Math.sin(6 * (k + 1) * dt) + 0.5 * Math.cos(15 * (k + 1) * dt))

describe('HiPPO', () => {
  it('LegS row 3 and the LegT input column match their closed forms', () => {
    expect(maxError(toRows(hippoLegS(4).A)[3], [-s7, -s7 * s3, -s7 * s5, -4])).toBeLessThan(1e-12)
    expect(maxError(toFlat(hippoLegT(3, 2).B), [0.5, s3 / 2, s5 / 2])).toBeLessThan(1e-12)
    const A = toRows(hippoLegT(3, 1).A)
    expect(A[0]).toEqual([-1, s3, -s5])
    expect(A[2][0]).toBeCloseTo(-s5, 14)
  })

  it('remembers a unit step: exact LegS coefficients (0.5, 0.433, 0, -0.165) at t = 2', () => {
    const exact = [0.5, s3 / 4, 0, -s7 / 16]
    expect(maxError(toFlat(hippoProject(step, dt, 2, { measure: 'legs', order: 4 })), exact)).toBeLessThan(2e-3)
    const states = hippoOnline(step, dt, { measure: 'legs', order: 4 })
    expect(states.shape).toEqual([T, 4])
    const last = toFlat(slice(states, T - 1) as Tensor)
    expect(maxError(last, exact)).toBeLessThan(5e-3)
    expect(maxError(toFlat(hippoReconstruct(last, 'legs', 2, 1, [2, 0])), [0.8125, 0.1875])).toBeLessThan(1e-2)
  })

  it('tracks the windowed projection online: LegT (N = 8, theta = 0.5) and LegS (N = 8) at t = 1.5', () => {
    const legt = hippoOnline(smooth, dt, { measure: 'legt', order: 8, theta: 0.5 })
    const projT = toFlat(hippoProject(smooth, dt, 1.5, { measure: 'legt', order: 8, theta: 0.5 }))
    expect(maxError(toFlat(slice(legt, 1499) as Tensor), projT)).toBeLessThan(2e-2)
    const legs = hippoOnline(smooth, dt, { measure: 'legs', order: 8 })
    const projS = toFlat(hippoProject(smooth, dt, 1.5, { measure: 'legs', order: 8 }))
    expect(maxError(toFlat(slice(legs, 1499) as Tensor), projS)).toBeLessThan(2e-3)
  })

  it('maps the window to [-1, 1]: the basis at the newest point is sqrt(2n + 1)', () => {
    expect(hippoWindow('legt', 3, 0.5)).toEqual([2.5, 3])
    expect(maxError(toFlat(hippoBasis('legt', 3, 3, 0.5, 3) as Tensor), [1, s3, s5])).toBeLessThan(1e-14)
  })
})

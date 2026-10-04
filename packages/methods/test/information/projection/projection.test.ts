import { describe, expect, it } from 'vitest'
import { normalProjection } from 'aifn-methods/information/projection'
import { Mixture, Normal } from 'aifn-compute/probability/distributions'

const close = (a: number, b: number, tol = 1e-12) =>
  expect(Math.abs(a - b) / Math.max(1, Math.abs(b))).toBeLessThan(tol)

describe('info: KL projections onto the normal family', () => {
  it('forward matches moments; reverse seeks the mode nearest the start', () => {
    const p = Mixture([0.5, 0.5], [Normal(-2, 0.5), Normal(2, 0.5)])
    const forward = normalProjection(p, 'forward', { init: { loc: 1.5, scale: 0.5 } })
    expect(forward.converged).toBe(true)
    close(forward.loc, 0, 1e-5)
    close(forward.scale, Math.sqrt(0.25 + 4), 1e-5)
    const right = normalProjection(p, 'reverse', { init: { loc: 1.5, scale: 1 } })
    const left = normalProjection(p, 'reverse', { init: { loc: -1.5, scale: 1 } })
    expect(right.path.length).toBeGreaterThan(2)
    close(right.loc, 2, 1e-3)
    close(right.scale, 0.5, 1e-3)
    close(left.loc, -2, 1e-3)
    // One mode of an equal mixture of well-separated components: KL(q ‖ p) ≈ log 2.
    close(right.objective, Math.LN2, 1e-3)
  })
})

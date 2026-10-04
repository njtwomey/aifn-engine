import { describe, expect, test } from 'vitest'
import { ackermann, closedLoopPoles, dlqr, lqr } from 'aifn-compute/dynamics/control'
import { kleinmanIteration, riccatiDoubling, riccatiMatrixSign, riccatiRecursion } from 'aifn-compute/numerics/linalg'
import { toComplexFlat, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'
import { checkProtocol } from '../../protocol'

type Mat = number[][]
type Riccati = { A: Mat; B: Mat; Q: Mat; R: Mat; P: Mat }
const F = fixture<{ care: Riccati; cartpole: Riccati; dare: Riccati; dare_stable: Riccati }>('systems')

const close = (a: Tensor | null, b: Mat | number[], tol: number) => {
  expect(a).not.toBeNull()
  const x = toFlat(a!)
  const y = (b as (number | number[])[]).flat()
  expect(x.length).toBe(y.length)
  const scale = Math.max(1, ...y.map(Math.abs))
  x.forEach((v, i) => expect(Math.abs(v - y[i])).toBeLessThan(tol * scale))
}
const sorted = (z: Tensor) => toComplexFlat(z).sort((u, v) => u.re - v.re || u.im - v.im)

describe('LQR', () => {
  test('CARE by Kleinman and by the sign function matches scipy', () => {
    for (const c of [F.care, F.cartpole]) {
      const k = lqr({ A: c.A, B: c.B }, c.Q, c.R)
      expect(k.converged).toBe(true)
      close(k.P, c.P, 1e-9)
      const s = lqr({ A: c.A, B: c.B }, c.Q, c.R, { method: 'sign' })
      expect(s.converged).toBe(true)
      close(s.P, c.P, 1e-8)
      expect(k.closedLoop.dtype).toBe('complex128')
      expect(Math.max(...toComplexFlat(k.closedLoop).map((z) => z.re))).toBeLessThan(0)
    }
  })

  test('DARE by doubling and by the recursion matches scipy', () => {
    for (const c of [F.dare, F.dare_stable]) {
      const d = dlqr({ A: c.A, B: c.B }, c.Q, c.R)
      expect(d.converged).toBe(true)
      close(d.P, c.P, 1e-9)
    }
    const r = dlqr({ A: F.dare.A, B: F.dare.B }, F.dare.Q, F.dare.R, { method: 'recursion' })
    expect(r.converged).toBe(true)
    close(r.P, F.dare.P, 1e-8)
    toComplexFlat(r.closedLoop).forEach((z) => expect(Math.hypot(z.re, z.im)).toBeLessThan(1))
  })

  test('the double integrator has the textbook gain', () => {
    const r = lqr(
      {
        A: [
          [0, 1],
          [0, 0],
        ],
        B: [[0], [1]],
      },
      [
        [1, 0],
        [0, 1],
      ],
      [[1]],
    )
    close(r.K, [[1, Math.sqrt(3)]], 1e-9)
    const p = closedLoopPoles(
      [
        [0, 1],
        [0, 0],
      ],
      [[0], [1]],
      r.K,
    )
    const s = sorted(p)
    expect(s[0].re).toBeCloseTo(-Math.sqrt(3) / 2, 9)
    expect(Math.abs(s[0].im)).toBeCloseTo(0.5, 9)
  })

  test('an uncontrollable unstable mode is reported, not hidden', () => {
    const res = lqr(
      {
        A: [
          [1, 0],
          [0, -1],
        ],
        B: [[0], [1]],
      },
      [
        [1, 0],
        [0, 1],
      ],
      [[1]],
    )
    expect(res.converged).toBe(false)
    expect(res.failure).toBe('not stabilisable')
  })

  test('the Riccati solvers satisfy the Algorithm protocol', () => {
    const record = { t: (s: { t: number }) => s.t, residual: (s: { residual: number }) => s.residual }
    checkProtocol(kleinmanIteration(F.care), undefined, { steps: 6, record })
    checkProtocol(riccatiMatrixSign(F.care), undefined, { steps: 6, record })
    checkProtocol(riccatiRecursion(F.dare), undefined, { steps: 6, record })
    checkProtocol(riccatiDoubling(F.dare), undefined, { steps: 5, record })
  })
})

describe('pole placement', () => {
  test('Ackermann places the requested poles', () => {
    const plant = { A: F.cartpole.A, B: F.cartpole.B }
    const p = ackermann(plant, [
      { re: -1, im: 1 },
      { re: -1, im: -1 },
      { re: -2, im: 0 },
      { re: -3, im: 0 },
    ])
    expect(p.controllable).toBe(true)
    expect(p.closedLoop.dtype).toBe('complex128')
    const e = sorted(closedLoopPoles(plant.A, plant.B, p.K!))
    const want = [
      { re: -3, im: 0 },
      { re: -2, im: 0 },
      { re: -1, im: -1 },
      { re: -1, im: 1 },
    ]
    e.forEach((z, i) => {
      expect(z.re).toBeCloseTo(want[i].re, 8)
      expect(z.im).toBeCloseTo(want[i].im, 8)
    })
    // φ(s) = (s² + 2s + 2)(s + 2)(s + 3) = s⁴ + 7s³ + 18s² + 22s + 12.
    close(p.characteristic, [1, 7, 18, 22, 12], 1e-12)
    const bad = ackermann(
      {
        A: [
          [-1, 0],
          [0, -1],
        ],
        B: [[1], [1]],
      },
      [-2, -3],
    )
    expect(bad.controllable).toBe(false)
    expect(bad.K).toBeNull()
  })
})

import { describe, expect, it } from 'vitest'
import {
  bisection,
  minimizeScalar,
  brent,
  broyden,
  continuation,
  findRoot,
  fixedPoint,
  newtonHomotopy,
  newtonRoot,
  newtonSystem,
  regulaFalsi,
  secant,
  solveSystem,
  type SystemWithJacobian,
} from 'aifn-compute/numerics/roots'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { extend, run, seek, trace } from 'aifn-compute/foundation/trace'
import { checkProtocol } from '../../protocol'

const cubic = (x: number) => x * x * x - 2 * x - 5 // Newton's own example; root 2.0945514815423265
const ROOT = 2.0945514815423265

describe('scalar roots', () => {
  it('every method finds the root of x³ − 2x − 5', () => {
    const bracket = { lo: 2, hi: 3 }
    expect(run(bisection(cubic), bracket, 100).x).toBeCloseTo(ROOT, 11)
    expect(run(regulaFalsi(cubic), bracket, 100).x).toBeCloseTo(ROOT, 12)
    expect(run(brent(cubic), bracket, 100).x).toBeCloseTo(ROOT, 12)
    expect(run(secant(cubic), { x0: 2, x1: 3 }, 100).x).toBeCloseTo(ROOT, 12)
    const d = (x: number) => ({ value: cubic(x), derivative: 3 * x * x - 2 })
    expect(run(newtonRoot(d), { x0: 2 }, 100).x).toBeCloseTo(ROOT, 12)
    expect(run(newtonRoot(d, { damped: true }), { x0: 0.5 }, 100).x).toBeCloseTo(ROOT, 12)
  })
  it('Brent uses far fewer evaluations than bisection and reports its step kinds', () => {
    const b = trace(brent(cubic), { lo: 0, hi: 5 }, 100)
    const s = b.steps.at(-1)!
    expect(s.converged).toBe(true)
    expect(s.evaluations).toBeLessThan(15)
    expect(new Set(b.steps.map((q) => q.method)).has('inverse-quadratic')).toBe(true)
  })
  it('reports a bracket without a sign change', () => {
    const r = findRoot((x) => x * x + 1, [-1, 1])
    expect(r.converged).toBe(false)
    expect(r.failure).toBe('no sign change')
  })
  it('protocol: seek equals run; extend equals a longer trace', () => {
    const alg = bisection(cubic)
    const record = { x: (s: { x: number }) => s.x }
    const a = trace(alg, { lo: 2, hi: 3 }, 20, { record })
    expect(seek(alg, { lo: 2, hi: 3 }, 9).x).toBe(run(alg, { lo: 2, hi: 3 }, 9).x)
    expect(toFlat(extend(trace(alg, { lo: 2, hi: 3 }, 8, { record }), alg, 12).series.x)).toEqual(toFlat(a.series.x))
  })
})

describe('systems', () => {
  // x² + y² = 4, xy = 1.
  const F: SystemWithJacobian = (v) => {
    const [x, y] = toFlat(v)
    return {
      value: [x * x + y * y - 4, x * y - 1],
      jacobian: [
        [2 * x, 2 * y],
        [y, x],
      ],
    }
  }
  const check = (x: number[]) => {
    expect(Math.abs(x[0] ** 2 + x[1] ** 2 - 4)).toBeLessThan(1e-9)
    expect(Math.abs(x[0] * x[1] - 1)).toBeLessThan(1e-9)
  }
  it('Newton, damped Newton and Broyden solve it', () => {
    check(toFlat(run(newtonSystem(F), { x0: [2, 0.3] }, 50).x))
    check(toFlat(solveSystem(F, [2, 0.3]).x))
    const values = (v: Parameters<SystemWithJacobian>[0]) => F(v).value
    const b = solveSystem(values, [2, 0.3], { method: 'broyden' })
    expect(b.converged).toBe(true)
    check(toFlat(b.x))
    expect(run(broyden(values), { x0: [2, 0.3] }, 100).converged).toBe(true)
  })
  it('fixed-point iteration on cos converges with contraction ≈ |sin x*|', () => {
    const s = run(
      fixedPoint((x) => toFlat(x).map(Math.cos)),
      { x0: [1] },
      200,
    )
    expect(s.converged).toBe(true)
    expect(toFlat(s.x)[0]).toBeCloseTo(0.7390851332151607, 11)
    const t = run(
      fixedPoint((x) => toFlat(x).map(Math.cos), { ftol: 0 }),
      { x0: [1] },
      30,
    )
    expect(t.contraction).toBeCloseTo(Math.sin(0.7390851332151607), 3)
    const d = run(
      fixedPoint((x) => toFlat(x).map((v) => 2 * v + 1), { patience: 5 }),
      { x0: [1] },
      100,
    )
    expect(d.failure).toBe('diverging')
  })
  it('continuation follows the Newton homotopy to a root', () => {
    const s = run(continuation(newtonHomotopy(F, [3, 0.5])), { x0: [3, 0.5] }, 200)
    expect(s.converged).toBe(true)
    check(toFlat(s.x))
  })
})

describe('the Algorithm protocol', () => {
  const F: SystemWithJacobian = (v) => {
    const [x, y] = toFlat(v)
    return {
      value: [x * x + y * y - 4, x * y - 1],
      jacobian: [
        [2 * x, 2 * y],
        [y, x],
      ],
    }
  }
  const d = (x: number) => ({ value: cubic(x), derivative: 3 * x * x - 2 })
  it('scalar methods', () => {
    const record = { x: (s: { x: number }) => s.x }
    checkProtocol(bisection(cubic), { lo: 0, hi: 5 }, { steps: 12, record })
    checkProtocol(regulaFalsi(cubic), { lo: 0, hi: 5 }, { steps: 12, record })
    checkProtocol(brent(cubic), { lo: 0, hi: 5 }, { steps: 12, record })
    checkProtocol(secant(cubic), { x0: 2, x1: 3 }, { steps: 6, record })
    checkProtocol(newtonRoot(d, { damped: true }), { x0: 0.5 }, { steps: 6, record })
  })
  it('systems, fixed points and continuation', () => {
    checkProtocol(newtonSystem(F), { x0: [2, 0.3] }, { steps: 5 })
    checkProtocol(
      broyden((v) => F(v).value),
      { x0: [2, 0.3] },
      { steps: 6 },
    )
    checkProtocol(
      fixedPoint((x) => toFlat(x).map(Math.cos)),
      { x0: [1] },
      { steps: 10 },
    )
    checkProtocol(continuation(newtonHomotopy(F, [3, 0.5])), { x0: [3, 0.5] }, { steps: 10 })
  })
})

describe('minimizeScalar', () => {
  const quartic = (x: number) => (x - 2) ** 4 + (x - 2) ** 2 + 1

  it('Brent finds a smooth minimum from a bracket, expanding downhill past it', () => {
    const r = minimizeScalar(quartic)
    expect(r.converged).toBe(true)
    expect(r.bracketed).toBe(true)
    expect(r.x).toBeCloseTo(2, 7)
    expect(r.value).toBeCloseTo(1, 12)
    expect(minimizeScalar((x) => Math.cos(x), { bracket: [3, 3.5] }).x).toBeCloseTo(Math.PI, 7)
  })

  it('golden section converges too, in more steps', () => {
    const brent = minimizeScalar(quartic, { bounds: [0, 5] })
    const golden = minimizeScalar(quartic, { bounds: [0, 5], method: 'golden' })
    expect(golden.converged).toBe(true)
    expect(golden.x).toBeCloseTo(2, 6)
    expect(golden.steps).toBeGreaterThan(brent.steps)
  })

  it('stays within bounds and returns the endpoint for a monotone function', () => {
    const r = minimizeScalar((x) => x, { bounds: [1, 4] })
    expect(r.x).toBeGreaterThanOrEqual(1)
    expect(r.x).toBeCloseTo(1, 6)
    expect(() => minimizeScalar((x) => x, { bounds: [4, 1] })).toThrow(/bounds/)
  })

  it('reports a function that decreases without bound, and a step budget that runs out', () => {
    const r = minimizeScalar((x) => -x)
    expect(r.bracketed).toBe(false)
    expect(r.converged).toBe(false)
    expect(minimizeScalar(quartic, { bounds: [0, 5], maxSteps: 3 }).converged).toBe(false)
  })

  it('matches scipy.optimize.minimize_scalar (Brent) on a standard problem', () => {
    // scipy: minimize_scalar(lambda x: (x - 2) * x * (x + 2) ** 2) → x = 1.28077640…, f = −9.91484…
    const r = minimizeScalar((x) => (x - 2) * x * (x + 2) ** 2)
    expect(r.x).toBeCloseTo(1.2807764064044151, 7)
    expect(r.value).toBeCloseTo(-9.914949590828147, 10)
  })
})

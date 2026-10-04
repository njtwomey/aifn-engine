import { describe, expect, it } from 'vitest'
import { adaptiveBdf, solveIvp, type Rhs } from 'aifn-compute/dynamics/ode'
import { get, mul, neg, square, stack, sub, add, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { fixture } from '../../fixtures'
import { checkProtocol } from '../../protocol'

type BdfCase = {
  span: [number, number]
  y0: number[]
  rtol: number
  atol: number[]
  t: number[]
  x: number[][]
  nfev: number
  njev: number
  nlu: number
  reference: number[]
}
const fx = fixture<{ bdf: Record<string, BdfCase> }>('dynamics/ode').bdf

const robertson: Rhs = (_t, y) => {
  const [a, b, c] = [get(y, 0), get(y, 1), get(y, 2)]
  const bc = mul(1e4, mul(b, c))
  const bb = mul(3e7, square(b))
  return stack([add(mul(-0.04, a), bc), sub(sub(mul(0.04, a), bc), bb), bb])
}
const robertsonJacobian = (_t: number, y: Tensor) => {
  const [a, b, c] = toFlat(y)
  void a
  return [
    [-0.04, 1e4 * c, 1e4 * b],
    [0.04, -1e4 * c - 6e7 * b, -1e4 * b],
    [0, 6e7 * b, 0],
  ]
}
const MU = 1000
const vanDerPol: Rhs = (_t, y) => {
  const [a, b] = [get(y, 0), get(y, 1)]
  return stack([b, sub(mul(mul(MU, sub(1, square(a))), b), a)])
}
const vanDerPolJacobian = (_t: number, y: Tensor) => {
  const [a, b] = toFlat(y)
  return [
    [0, 1],
    [-2 * MU * a * b - 1, MU * (1 - a * a)],
  ]
}

const problems = {
  robertson: [robertson, robertsonJacobian],
  van_der_pol: [vanDerPol, vanDerPolJacobian],
  van_der_pol_tight: [vanDerPol, vanDerPolJacobian],
} as const

describe('adaptive BDF (variable order 1–5)', () => {
  it.each(Object.keys(problems) as (keyof typeof problems)[])('matches scipy BDF step for step (%s)', (key) => {
    const ref = fx[key]
    const [f, jac] = problems[key]
    const alg = adaptiveBdf(f, { tEnd: ref.span[1], rtol: ref.rtol, atol: ref.atol, jacobian: jac })
    const tr = trace(alg, { x0: ref.y0, t0: ref.span[0] }, 100_000)
    expect(tr.meta.stopped).toBe('done')
    // Robertson's sequence is reproduced exactly. Van der Pol at μ = 1000 has ~10³ sharp relaxation jumps, and the
    // last-bit differences between numpy's and this arithmetic eventually move a step boundary there, so the
    // sequences agree step for step through the first jumps (t < 700) and in work to a few percent.
    const exact = key === 'robertson'
    const compared = exact ? ref.t.length : ref.t.findIndex((t) => t > 700)
    if (exact) expect(tr.steps.length).toBe(ref.t.length)
    else expect(Math.abs(tr.steps.length / ref.t.length - 1)).toBeLessThan(0.03)
    tr.steps.slice(0, compared).forEach((s, k) => {
      expect(Math.abs(s.time - ref.t[k])).toBeLessThan(1e-7 * (1 + Math.abs(ref.t[k])))
      toFlat(s.x).forEach((v, i) => expect(Math.abs(v - ref.x[k][i])).toBeLessThan(1e-6 * (1 + Math.abs(ref.x[k][i]))))
    })
    const last = tr.final
    if (exact) {
      expect(last.evaluations).toBe(ref.nfev)
      expect(last.jacobianEvaluations).toBe(ref.njev)
      expect(last.factorisations).toBe(ref.nlu)
    } else expect(Math.abs(last.evaluations / ref.nfev - 1)).toBeLessThan(0.05)
  })

  it.each(Object.keys(problems) as (keyof typeof problems)[])(
    'reaches the Radau reference within its tolerance, with an autodiff Jacobian (%s)',
    (key) => {
      const ref = fx[key]
      const [f] = problems[key]
      const s = run(adaptiveBdf(f, { tEnd: ref.span[1], rtol: ref.rtol, atol: ref.atol }), { x0: ref.y0 }, 100_000)
      expect(s.time).toBe(ref.span[1])
      toFlat(s.x).forEach((v, i) =>
        expect(Math.abs(v - ref.reference[i])).toBeLessThan(
          200 * (ref.atol[i] + ref.rtol * Math.abs(ref.reference[i])),
        ),
      )
    },
  )

  it('climbs from order 1 to a high order on Robertson, keeps mass and takes few steps on a stiff problem', () => {
    const ref = fx.robertson
    const tr = trace(
      adaptiveBdf(robertson, { tEnd: ref.span[1], rtol: ref.rtol, atol: ref.atol }),
      { x0: ref.y0 },
      100_000,
      { record: { order: (s) => s.order } },
    )
    const orders = new Set(toFlat(tr.series.order))
    expect(Math.max(...orders)).toBeGreaterThanOrEqual(4)
    expect(orders.has(1)).toBe(true)
    // Mass is conserved: y₁ + y₂ + y₃ = 1 (a linear invariant that every linear multistep method keeps).
    const sums = tr.steps.map((s) => toFlat(s.x).reduce((a, b) => a + b, 0))
    sums.forEach((v) => expect(v).toBeCloseTo(1, 10))
    // Dormand–Prince would need ~10⁶ steps for t up to 1e5 (stability limit ~1e-3).
    expect(tr.steps.length).toBeLessThan(1000)
  })

  it('solveIvp runs it as method "bdf"; the classical BDF variant and backwards integration work', () => {
    const sol = solveIvp((_t, x) => neg(x), [0, 5], [1], { method: 'bdf', rtol: 1e-8, atol: 1e-10 })
    expect(toRows(sol.x).at(-1)![0]).toBeCloseTo(Math.exp(-5), 7)
    const classic = run(
      adaptiveBdf((_t, x) => neg(x), { tEnd: 5, rtol: 1e-8, atol: 1e-10, variant: 'bdf' }),
      { x0: [1] },
      10_000,
    )
    expect(toFlat(classic.x)[0]).toBeCloseTo(Math.exp(-5), 7)
    const back = run(
      adaptiveBdf((_t, x) => neg(x), { tEnd: 0, rtol: 1e-8, atol: 1e-10 }),
      { x0: [Math.exp(-2)], t0: 2 },
      10_000,
    )
    expect(toFlat(back.x)[0]).toBeCloseTo(1, 6)
  })

  it('reports a step-size underflow instead of looping on a blow-up', () => {
    const blowUp: Rhs = (_t, x) => square(x)
    const s = run(adaptiveBdf(blowUp, { tEnd: 2 }), { x0: [1] }, 100_000)
    expect(s.failure).not.toBeNull()
    expect(s.time).toBeLessThan(1.0001)
  })

  it('satisfies the Algorithm protocol', () => {
    checkProtocol(adaptiveBdf(vanDerPol, { tEnd: 1 }), { x0: [2, 0] }, { steps: 16, record: { time: (s) => s.time } })
  })
})

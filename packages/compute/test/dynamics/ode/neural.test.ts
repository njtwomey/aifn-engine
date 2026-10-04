/**
 * Differentiable ODE flows for neural ODEs: backprop and adjoint gradients against the analytic sensitivity of
 * x′ = −kx, against central differences on a small tanh field, and against each other; the exact and Hutchinson
 * Jacobian traces against tr(A); the CNF log-density change against the exact change of variables of a linear flow;
 * solve reports (NFE) and the adjoint's reconstruction error with checkpoints.
 */
import { describe, expect, it } from 'vitest'
import { grad, gradCheck } from 'aifn-compute/foundation/autodiff'
import { stream, child } from 'aifn-compute/foundation/random'
import {
  add,
  matmul,
  mul,
  neg,
  reshape,
  slice,
  sum,
  tanh,
  tensor,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { dormandPrince, type OdeSolveInfo } from 'aifn-compute/dynamics/ode'
import { augmentedDynamics, jacobianTrace, odeFlow, traceProbe } from 'aifn-compute/dynamics/ode'
import { run } from 'aifn-compute/foundation/trace'
import { fixture } from '../../fixtures'

const flat = (v: unknown): number[] => (typeof v === 'number' ? [v] : Array.from(toFlat(unwrap(v as Value) as Tensor)))

/** A tanh field f(x) = tanh(x W₁ + b) W₂ on [B, 2] with θ packed as one vector of 2·3 + 3 + 3·2 = 15 entries. */
const field = (_t: number, x: Value, p: Value): Value => {
  const w1 = reshape(slice(p, [0, 6]), [2, 3])
  const b = slice(p, [6, 9])
  const w2 = reshape(slice(p, [9, 15]), [3, 2])
  return matmul(tanh(add(matmul(x, w1), b)), w2)
}
const theta = tensor([0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.1, -0.2, 0.05, 0.7, -0.4, 0.3, 0.9, -0.5, 0.2])
const x0 = tensor([
  [0.3, -0.7],
  [1.1, 0.4],
])

describe('odeFlow', () => {
  for (const gradient of ['backprop', 'adjoint'] as const)
    for (const method of ['rk4', 'dormand-prince'] as const)
      it(`${gradient} · ${method}: d/dk Σ x(1) of x′ = −kx is −Σx₀ e^{−k}`, () => {
        const flow = odeFlow((_t, x, k) => mul(neg(k), x), [0, 1], { gradient, method, rtol: 1e-8, atol: 1e-10 })
        const g = grad((k: Value) => sum(flow(tensor([1, 2]), k)[1]))(0.5) as number
        expect(g).toBeCloseTo(-3 * Math.exp(-0.5), 6)
      })

  it('backprop through RK4 matches central differences of the discrete solution (θ and x₀)', () => {
    const flow = odeFlow(field, [0, 1], { method: 'rk4', stepSize: 0.25 })
    const loss = ([p, x]: [Value, Value]) =>
      sum(
        mul(
          flow(x, p)[1],
          tensor([
            [1, -2],
            [0.5, 1],
          ]),
        ),
      )
    const report = gradCheck(loss, [theta, x0] as [Value, Value], { rtol: 1e-6, atol: 1e-8 })
    expect(report.ok).toBe(true)
  })

  it('adjoint and backprop agree to the solver accuracy; backprop through Dormand–Prince matches differences', () => {
    const loss = (gradient: 'backprop' | 'adjoint') => (p: Value) =>
      sum(mul(odeFlow(field, [0, 1], { gradient, method: 'dormand-prince', rtol: 1e-9, atol: 1e-11 })(x0, p)[1], 1))
    const a = flat(grad(loss('adjoint'))(theta))
    const b = flat(grad(loss('backprop'))(theta))
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 6))
    // With the step sizes held, the discrete Dormand–Prince solution is smooth in θ near the base point.
    expect(gradCheck(loss('backprop'), theta, { rtol: 1e-4, atol: 1e-6, eps: 1e-7 }).maxAbsError).toBeLessThan(1e-4)
  })

  it('samples several times, in either direction, and reports the work of each solve', () => {
    const infos: OdeSolveInfo[] = []
    const flow = odeFlow((_t, x) => neg(x), [1, 0.5, 0], {
      method: 'rk4',
      stepSize: 0.05,
      onSolve: (i) => infos.push(i),
    })
    const xs = flow(tensor([1]), 0)
    expect(flat(xs[2])[0]).toBeCloseTo(Math.E, 6)
    expect(infos.map((i) => [i.phase, i.steps, i.evaluations])).toEqual([
      ['forward', 10, 40],
      ['forward', 10, 40],
    ])
  })

  it('the adjoint reports its backward solves; checkpoints shrink the reconstruction error of x₀', () => {
    // x′ = xA with A = R diag(−40, 1) Rᵀ: the fast direction decays by e⁻⁴⁰ forwards, so integrating back from x(1)
    // amplifies the rounding and solver error in it by e⁴⁰; restarting from kept states bounds the amplification per segment.
    const c = Math.cos(0.6)
    const s = Math.sin(0.6)
    const R = [
      [c, -s],
      [s, c],
    ]
    const D = [-40, 1]
    const M = [0, 1].map((i) => [0, 1].map((j) => R[i][0] * D[0] * R[j][0] + R[i][1] * D[1] * R[j][1]))
    const reconstruction = (checkpoints: number) => {
      const infos: OdeSolveInfo[] = []
      const flow = odeFlow((_t, x, m) => matmul(x, m), [0, 1], {
        gradient: 'adjoint',
        method: 'rk4',
        stepSize: 0.005,
        checkpoints,
        onSolve: (i) => infos.push(i),
      })
      grad((m: Value) => sum(flow(tensor([[1, 0.5]]), m)[1]))(tensor(M))
      const back = infos.filter((i) => i.phase === 'backward')
      expect(back.length).toBe(checkpoints)
      const end = back.at(-1)!.x
      return Math.hypot(end[0] - 1, end[1] - 0.5)
    }
    const [one, ten] = [reconstruction(1), reconstruction(10)]
    expect(ten).toBeLessThan(1e-3 * one)
  })
})

describe('odeFlow against torchdiffeq', () => {
  // torchdiffeq's odeint_adjoint (dopri5, rtol 1e-11) on the same tanh field and weights (fixtures/gen/dynamics/ode.py).
  const ref = (fixture('dynamics/ode') as { neural_ode: Record<string, number[] | number[][]> }).neural_ode
  for (const gradient of ['adjoint', 'backprop'] as const)
    it(`${gradient}: x(1), ∂L/∂θ and ∂L/∂x₀ match`, () => {
      const flow = odeFlow(field, [0, 1], { gradient, method: 'dormand-prince', rtol: 1e-10, atol: 1e-12 })
      const W = tensor(ref.weights as number[][])
      const loss = ([p, x]: [Value, Value]) => sum(mul(flow(x, p)[1], W))
      const x1 = flat(flow(tensor(ref.x0 as number[][]), tensor(ref.theta as number[]))[1])
      x1.forEach((v, i) => expect(v).toBeCloseTo((ref.x1 as number[][]).flat()[i], 8))
      const [gp, gx] = grad(loss)([tensor(ref.theta as number[]), tensor(ref.x0 as number[][])] as [Value, Value]) as [
        Value,
        Value,
      ]
      flat(gp).forEach((v, i) => expect(v).toBeCloseTo((ref.grad_theta as number[])[i], 7))
      flat(gx).forEach((v, i) => expect(v).toBeCloseTo((ref.grad_x0 as number[][]).flat()[i], 7))
    })
})

describe('jacobianTrace', () => {
  const A = tensor([
    [0.5, 2, -1],
    [0.3, -1.5, 0.7],
    [1, 0.2, 0.25],
  ])
  // Rows x ↦ x A: the Jacobian of each row is Aᵀ, with trace tr(A) = −0.75.
  const linear = (x: Value) => matmul(x, A)
  const x = tensor([
    [1, 2, 3],
    [-1, 0.5, 2],
  ])

  it('exact: tr(A) for every row', () => {
    expect(flat(jacobianTrace(linear, x).trace)).toEqual([-0.75, -0.75].map((v) => expect.closeTo(v, 12)))
  })

  it('Hutchinson: εᵀAᵀε for the probe given, and unbiased over probes (Rademacher and Gaussian)', () => {
    const probe = tensor([
      [1, -1, 1],
      [-1, -1, 1],
    ])
    const r = jacobianTrace(linear, x, { estimator: 'hutchinson', probe })
    const quad = (e: number[]) => {
      let s = 0
      const a = flat(A)
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += e[i] * a[j * 3 + i] * e[j]
      return s
    }
    expect(flat(r.trace)[0]).toBeCloseTo(quad([1, -1, 1]), 12)
    expect(flat(r.trace)[1]).toBeCloseTo(quad([-1, -1, 1]), 12)
    for (const kind of ['rademacher', 'gaussian'] as const) {
      const big = tensor(Array.from({ length: 4000 }, () => [0.1, 0.2, 0.3]))
      const est = flat(
        jacobianTrace(linear, big, { estimator: 'hutchinson', probe: traceProbe(stream(3), [4000, 3], kind) }).trace,
      )
      const m = est.reduce((a, b) => a + b, 0) / est.length
      expect(m).toBeCloseTo(-0.75, 0)
    }
  })

  it('differentiates under grad (second order through the field)', () => {
    const p = theta
    const report = gradCheck((q: Value) => sum(jacobianTrace((y) => field(0, y, q), x0).trace), p, {
      rtol: 1e-6,
      atol: 1e-8,
    })
    expect(report.ok).toBe(true)
  })
})

describe('augmentedDynamics', () => {
  it('a linear flow x′ = xA: Δ(1) = −tr(A), the exact change of variables, both estimators and both gradients', () => {
    const A = tensor([
      [-0.4, 1.2],
      [-0.8, 0.9],
    ])
    for (const logDensity of ['exact', 'hutchinson'] as const)
      for (const gradient of ['backprop', 'adjoint'] as const) {
        const probe = traceProbe(child(stream(1), 'probe'), [2, 2])
        const aug = augmentedDynamics((_t, y, a) => matmul(y, a), { dim: 2, logDensity, probe })
        const flow = odeFlow(aug.rhs, [0, 1], { gradient, method: 'dormand-prince', rtol: 1e-10, atol: 1e-12 })
        const z = flow(aug.pack(x0), A)[1]
        const { logDensityChange } = aug.unpack(z)
        // For a linear field the Hutchinson estimate εᵀAᵀε equals tr(A) whenever ε is ±1 (cross terms aside, it is
        // tr(A) + Σ_{j≠k} εⱼεₖ A_kj); check against that exact per-row value.
        const e = flat(probe)
        const a = flat(A)
        const expected =
          logDensity === 'exact'
            ? [-0.5, -0.5]
            : [0, 1].map((r) => -(a[0] + a[3] + e[2 * r] * e[2 * r + 1] * (a[1] + a[2])))
        flat(logDensityChange).forEach((v, i) => expect(v).toBeCloseTo(expected[i], 8))
        // d Δ(1)/dA = −I for the exact trace.
        if (logDensity === 'exact') {
          const g = flat(grad((m: Value) => sum(aug.unpack(flow(aug.pack(x0), m)[1]).logDensityChange!))(A))
          g.forEach((v, i) => expect(v).toBeCloseTo([-2, 0, 0, -2][i], 6))
        }
      }
  })

  it('integrates the kinetic energy and the Frobenius estimate', () => {
    // x′ = −x from x₀: ∫₀¹ ‖x‖² dt = ‖x₀‖²(1 − e⁻²)/2; J = −I, so ‖εᵀJ‖² = ‖ε‖² = d for Rademacher ε.
    const probe = traceProbe(stream(2), [2, 2])
    const aug = augmentedDynamics((_t, y) => neg(y), { dim: 2, kinetic: true, jacobianFrobenius: true, probe })
    const z = odeFlow(aug.rhs, [0, 1], { method: 'dormand-prince', rtol: 1e-10, atol: 1e-12 })(aug.pack(x0), 0)[1]
    const parts = aug.unpack(z)
    const norms = [0.3 ** 2 + 0.7 ** 2, 1.1 ** 2 + 0.4 ** 2]
    flat(parts.kinetic).forEach((v, i) => expect(v).toBeCloseTo((norms[i] * (1 - Math.exp(-2))) / 2, 8))
    flat(parts.jacobianFrobenius).forEach((v) => expect(v).toBeCloseTo(2, 10))
  })
})

describe('dormandPrince on traced values', () => {
  it('still solves on raw values as before', () => {
    const s = run(
      dormandPrince((_t, x) => neg(x), { tEnd: 1, rtol: 1e-8, atol: 1e-10 }),
      { x0: [1] },
      1000,
    )
    expect(flat(s.x)[0]).toBeCloseTo(Math.exp(-1), 7)
  })
})

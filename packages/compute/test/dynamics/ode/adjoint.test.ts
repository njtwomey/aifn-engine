/**
 * The adjoint method (`odeAdjoint`) against the exact sensitivity of a linear system (autodiff through `expm`), the
 * discrete gradient of `unrolled` through tensor RK4, and central differences of the solver, on a damped oscillator
 * and on Lotka–Volterra; checkpoints against a system whose state cannot be recovered by integrating backwards.
 */
import { describe, expect, it } from 'vitest'
import { grad, jvp } from 'aifn-compute/foundation/autodiff'
import {
  get,
  matmul,
  mul,
  neg,
  stack,
  sub,
  sum,
  tensor,
  toFlat,
  unwrap,
  type Tensor,
  type Traced,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { run, unrolled } from 'aifn-compute/foundation/trace'
import { expm } from 'aifn-compute/numerics/linalg'
import { odeAdjoint, rungeKutta, type ParametricRhs } from 'aifn-compute/dynamics/ode'

const flat = (v: unknown): number[] =>
  typeof v === 'number' ? [v] : Array.isArray(v) ? v : Array.from(toFlat(v as Tensor))

function central(f: (v: number[]) => number, at: number[], eps = 1e-6): number[] {
  return at.map((_, i) => {
    const up = [...at]
    const down = [...at]
    up[i] += eps
    down[i] -= eps
    return (f(up) - f(down)) / (2 * eps)
  })
}

/** ∂L/∂x₀ and ∂L/∂θ of L = Σ wᵢ xᵢ(T) by the adjoint method. */
function adjointGradients(f: ParametricRhs, T: number, x0: Tensor, theta: Tensor, w: Tensor, options = {}) {
  const flow = odeAdjoint(f, [0, T], options)
  return grad((x: Value, th: Value) => sum(mul(w, flow(x, th))), { argnums: [0, 1] })(x0, theta) as [Tensor, Tensor]
}

/** The same gradients through every RK4 step (`unrolled`), and L on raw values for differences. */
function discrete(f: ParametricRhs, T: number, h: number, x0: Tensor, theta: Tensor, w: Tensor) {
  const steps = Math.round(T / h)
  const solver = (th: Value) => rungeKutta((t, x) => f(t, x, th), 'rk4', { stepSize: h, tEnd: T })
  const loss = (x: Value, th: Value) => sum(mul(w, unrolled(solver, { x0: x as never }, steps, { params: th }).x))
  const raw = (x: number[], th: number[]) => unwrap(sum(mul(w, run(solver(tensor(th)), { x0: x }, steps).x))) as number
  const [gx, gt] = grad(loss, { argnums: [0, 1] })(x0, theta) as [Tensor, Tensor]
  return { gx, gt, raw }
}

const close = (a: unknown, b: unknown, digits: number, label: string) =>
  flat(a).forEach((v, i) => expect(v, `${label}[${i}]`).toBeCloseTo(flat(b)[i], digits))

describe('odeAdjoint', () => {
  // A damped oscillator x′ = A(θ)x, A = [[0, 1], [−k, −c]], θ = (k, c).
  const oscillator: ParametricRhs = (_t, x, th) =>
    stack([get(x, 1), sub(neg(mul(get(th, 0), get(x, 0))), mul(get(th, 1), get(x, 1)))])
  const T = 2
  const x0 = tensor([1, -0.5])
  const theta = tensor([2, 0.3])
  const w = tensor([0.7, -1.3])
  const exact = (x: Value, th: Value) => {
    const A = stack([stack([0, 1]), stack([neg(get(th, 0)), neg(get(th, 1))])])
    return sum(mul(w, matmul(expm(mul(T, A) as Traced).value, x)))
  }

  it('matches the exact sensitivity of a linear system, the discrete gradient and finite differences', () => {
    const [ax, at] = adjointGradients(oscillator, T, x0, theta, w, { stepSize: 0.01 })
    const [ex, et] = grad(exact, { argnums: [0, 1] })(x0, theta) as [Tensor, Tensor]
    close(ax, ex, 8, 'x₀ vs expm')
    close(at, et, 8, 'θ vs expm')
    const d = discrete(oscillator, T, 0.01, x0, theta, w)
    close(ax, d.gx, 7, 'x₀ vs unrolled')
    close(at, d.gt, 7, 'θ vs unrolled')
    close(
      at,
      central((v) => d.raw(flat(x0), v), flat(theta)),
      7,
      'θ vs differences',
    )
  })

  it('Lotka–Volterra: agrees with unrolled RK4 and finite differences', () => {
    // x′ = αx − βxy, y′ = δxy − γy, θ = (α, β, δ, γ).
    const lv: ParametricRhs = (_t, z, th) => {
      const x = get(z, 0)
      const y = get(z, 1)
      const xy = mul(x, y)
      return stack([sub(mul(get(th, 0), x), mul(get(th, 1), xy)), sub(mul(get(th, 2), xy), mul(get(th, 3), y))])
    }
    const z0 = tensor([1.2, 0.8])
    const th = tensor([1.1, 0.6, 0.4, 0.9])
    const wz = tensor([1, 0.5])
    const [ax, at] = adjointGradients(lv, 3, z0, th, wz, { stepSize: 0.005 })
    const d = discrete(lv, 3, 0.005, z0, th, wz)
    close(ax, d.gx, 7, 'z₀ vs unrolled')
    close(at, d.gt, 7, 'θ vs unrolled')
    close(
      ax,
      central((v) => d.raw(v, flat(th)), flat(z0)),
      6,
      'z₀ vs differences',
    )
    close(
      at,
      central((v) => d.raw(flat(z0), v), flat(th)),
      6,
      'θ vs differences',
    )
    // Dormand–Prince for both solves, at tight tolerances.
    const [px, pt] = adjointGradients(lv, 3, z0, th, wz, { method: 'dormand-prince', rtol: 1e-10, atol: 1e-12 })
    close(px, d.gx, 6, 'z₀ dormand-prince')
    close(pt, d.gt, 6, 'θ dormand-prince')
  })

  it('checkpoints restore a state that backward integration loses', () => {
    // A fast mode (rate ≈ 25) next to a slow one: x(T) holds the fast component only to rounding, so integrating x
    // backwards from x(T) amplifies that rounding by e^{25T}. Restarting from checkpoints bounds the growth.
    // x′ = [[−θ₀, 1], [1, −θ₁]] x: the adjoint of the slow mode persists, so ∂L/∂θ₁ = −∫ a₂ x₂ dt needs x₂ near t₀.
    const stiff: ParametricRhs = (_t, x, th) =>
      stack([sub(get(x, 1), mul(get(th, 0), get(x, 0))), sub(get(x, 0), mul(get(th, 1), get(x, 1)))])
    const th = tensor([0.5, 25])
    const xs = tensor([1, 1])
    const exactStiff = (x: Value, t: Value) => {
      const A = stack([stack([neg(get(t, 0)), 1]), stack([1, neg(get(t, 1))])])
      return sum(mul(w, matmul(expm(mul(T, A) as Traced).value, x)))
    }
    const [, et] = grad(exactStiff, { argnums: [0, 1] })(xs, th) as [Tensor, Tensor]
    const [, plain] = adjointGradients(stiff, T, xs, th, w, { stepSize: 0.002 })
    const [, kept] = adjointGradients(stiff, T, xs, th, w, { stepSize: 0.002, checkpoints: 40 })
    close(kept, et, 7, 'with checkpoints')
    const error = (g: Tensor) => Math.max(...flat(g).map((v, i) => Math.abs(v - flat(et)[i])))
    expect(error(plain)).toBeGreaterThan(1e3 * error(kept))
  })

  it('takes a scalar parameter, and forward mode agrees with the reverse rule', () => {
    const decay = odeAdjoint((_t, x, k) => mul(neg(k), x), [0, 1], { stepSize: 0.01 })
    const L = (k: Value) => sum(decay(tensor([1, 2]), k))
    const g = grad(L)(2) as number
    expect(g).toBeCloseTo(-3 * Math.exp(-2), 8)
    expect(jvp(L, 2, 1).tangent as number).toBeCloseTo(g, 10)
  })

  it('refuses bad options', () => {
    expect(() => odeAdjoint((_t, x) => x, [0, 1], { checkpoints: 0 })).toThrow()
    expect(() => odeAdjoint((_t, x) => x, [1, 1])).toThrow()
  })
})

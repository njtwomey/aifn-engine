/**
 * Differentiating an ODE solution through the solver's steps: `unrolled` over `rungeKutta` gives the gradient of a
 * function of x(T) with respect to the initial state and to the parameters the right-hand side closes over. Checked
 * against central differences of the same discrete solver, and against the exact sensitivity of the linear system,
 * the gradient through x(T) = expm(AT) x₀.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { NotDifferentiableError } from 'aifn-compute/foundation/errors'
import {
  matmul,
  mul,
  neg,
  stack,
  sub,
  sum,
  tensor,
  toFlat,
  unwrap,
  get,
  type Tensor,
  type Traced,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { run, unrolled } from 'aifn-compute/foundation/trace'
import { expm } from 'aifn-compute/numerics/linalg'
import { rungeKutta } from 'aifn-compute/dynamics/ode'

// A damped oscillator x′ = A(θ)x with A = [[0, 1], [−k, −c]], θ = (k, c); loss Σ wᵢ xᵢ(T).
const T = 2
const h = 0.01
const steps = Math.round(T / h)
const w = tensor([0.7, -1.3])
const rhs = (x: Value, theta: Value): Value =>
  stack([get(x, 1), sub(neg(mul(get(theta, 0), get(x, 0))), mul(get(theta, 1), get(x, 1)))])
const solver = (theta: Value) => rungeKutta((_t, x) => rhs(x, theta), 'rk4', { stepSize: h, tEnd: T })
const loss = (x: Value) => sum(mul(w, x))

const theta0 = tensor([2, 0.3])
const x0 = tensor([1, -0.5])
const flat = (v: unknown) => (typeof v === 'number' ? [v] : toFlat(v as Tensor))

/** L(x(T)) from the solver run on raw values, for central differences. */
const lossRaw = (x: number[], theta: number[]) => unwrap(loss(run(solver(tensor(theta)), { x0: x }, steps).x)) as number

function central(f: (v: number[]) => number, at: number[], eps = 1e-6): number[] {
  return at.map((_, i) => {
    const up = [...at]
    const down = [...at]
    up[i] += eps
    down[i] -= eps
    return (f(up) - f(down)) / (2 * eps)
  })
}

/** The exact gradient of L(expm(A(θ)T) x₀), by autodiff through `expm`. */
const exact = (x: Value, theta: Value) => {
  const A = stack([stack([0, 1]), stack([neg(get(theta, 0)), neg(get(theta, 1))])])
  return loss(matmul(expm(mul(T, A) as Traced).value, x))
}

describe('gradients through rungeKutta', () => {
  for (const checkpointEvery of [undefined, 20]) {
    it(`with respect to x₀ and θ match finite differences and the exact sensitivity (checkpointEvery ${checkpointEvery})`, () => {
      const solve = (x: Value, theta: Value) =>
        loss(unrolled(solver, { x0: x as never }, steps, { params: theta, checkpointEvery }).x)
      const [gx, gtheta] = grad(solve, { argnums: [0, 1] })(x0, theta0) as [Tensor, Tensor]
      const fdX = central((v) => lossRaw(v, flat(theta0)), flat(x0))
      const fdTheta = central((v) => lossRaw(flat(x0), v), flat(theta0))
      flat(gx).forEach((g, i) => expect(g).toBeCloseTo(fdX[i], 7))
      flat(gtheta).forEach((g, i) => expect(g).toBeCloseTo(fdTheta[i], 7))
      // RK4 with h = 0.01 is within O(h⁴) of the exact flow, and so are its derivatives.
      const [ex, etheta] = grad(exact, { argnums: [0, 1] })(x0, theta0) as [Tensor, Tensor]
      flat(gx).forEach((g, i) => expect(g).toBeCloseTo(flat(ex)[i], 6))
      flat(gtheta).forEach((g, i) => expect(g).toBeCloseTo(flat(etheta)[i], 6))
    })
  }

  it('second derivatives in θ agree with differences of the gradient', () => {
    const dk = (theta: Value) =>
      get(grad((th: Value) => loss(unrolled(solver, { x0 }, 50, { params: th }).x))(theta) as Value, 0)
    const d2 = flat(grad(dk)(theta0))
    const fd = central((v) => unwrap(dk(tensor(v))) as number, flat(theta0), 1e-5)
    d2.forEach((g, i) => expect(g).toBeCloseTo(fd[i], 5))
  })

  it('a right-hand side on raw arrays is refused, not silently zero', () => {
    // The primal values read on raw arrays: the result no longer depends on the trace.
    const raw = rungeKutta(
      (_t, x) => {
        const v = toFlat(unwrap(x) as Tensor)
        return [v[1], -v[0]]
      },
      'rk4',
      { stepSize: h },
    )
    expect(() => grad((x: Value) => loss(unrolled(raw, { x0: x as never }, 3).x))(x0)).toThrow(NotDifferentiableError)
  })

  it('every explicit tableau differentiates', () => {
    for (const method of ['euler', 'heun', 'midpoint'] as const) {
      const alg = (theta: Value) => rungeKutta((_t, x) => rhs(x, theta), method, { stepSize: 0.05 })
      const g = flat(grad((th: Value) => loss(unrolled(alg, { x0 }, 20, { params: th }).x))(theta0))
      const fd = central((v) => unwrap(loss(run(alg(tensor(v)), { x0: flat(x0) }, 20).x)) as number, flat(theta0))
      g.forEach((gi, i) => expect(gi, method).toBeCloseTo(fd[i], 7))
    }
  })
})

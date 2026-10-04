/**
 * Test problems for the optim tests, written here so the compute tests do not depend on `aifn-methods`'s test surfaces:
 * Rosenbrock's valley and a rotated quadratic bowl as bare value-and-gradient functions with their Hessians, and the
 * Rosenbrock function as an `Objective` written with primitives (gradients by autodiff).
 */
import type { Objective } from 'aifn-compute/foundation/contracts'
import { add, get, mul, square, sub, toFlat, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'

export type Problem = {
  /** f and ∇f. */
  objective: (x: Vector) => { value: number; grad: number[] }
  value: (x: Vector) => number
  hessian: (x: Vector) => number[][]
  start: number[]
  minimum: number[]
}

/** Rosenbrock's function (1 − x)² + 100(y − x²)², minimum 0 at (1, 1), started at (−1.2, 1). */
export function rosenbrock(): Problem {
  const value = (v: Vector) => {
    const [x, y] = toFlat(v)
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2
  }
  return {
    value,
    objective: (v) => {
      const [x, y] = toFlat(v)
      return { value: value(v), grad: [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)] }
    },
    hessian: (v) => {
      const [x, y] = toFlat(v)
      return [
        [2 - 400 * (y - 3 * x * x), -400 * x],
        [-400 * x, 200],
      ]
    },
    start: [-1.2, 1],
    minimum: [1, 1],
  }
}

/**
 * ½ xᵀAx with A = R diag(1, condition) Rᵀ and R the rotation by `angle`: eigenvalues 1 and `condition`, minimum 0 at
 * the origin, started at (−2, 1.5).
 */
export function bowl({ condition = 10, angle = 0 }: { condition?: number; angle?: number } = {}): Problem {
  const [c, s] = [Math.cos(angle), Math.sin(angle)]
  const A = [
    [c * c + condition * s * s, (1 - condition) * c * s],
    [(1 - condition) * c * s, s * s + condition * c * c],
  ]
  const Ax = (x: number[]) => A.map((row) => row[0] * x[0] + row[1] * x[1])
  const value = (v: Vector) => {
    const x = toFlat(v)
    const g = Ax(x)
    return 0.5 * (g[0] * x[0] + g[1] * x[1])
  }
  return {
    value,
    objective: (v) => ({ value: value(v), grad: Ax(toFlat(v)) }),
    hessian: () => A,
    start: [-2, 1.5],
    minimum: [0, 0],
  }
}

/** Rosenbrock's function as an `Objective` built from primitives, so methods differentiate it by autodiff. */
export const rosenbrockObjective: Objective = {
  kind: 'objective',
  name: 'rosenbrock',
  dim: 2,
  value: (v: Tensor) => {
    const x = get(v, 0)
    const y = get(v, 1)
    return add(square(sub(1, x)), mul(100, square(sub(y, square(x)))))
  },
}

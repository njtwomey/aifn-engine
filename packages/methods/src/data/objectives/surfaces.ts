/**
 * Test surfaces for optimisers (moved from `aifn-compute/optim`: they are data, not methods), each with its value, gradient, Hessian, known minima, a conventional start
 * and a plotting domain. Formulas from Jamil & Yang (2013), "A literature survey of benchmark functions for global
 * optimisation problems", and Surjanovic & Bingham, "Virtual Library of Simulation Experiments" (sfu.ca/~ssurjano).
 */

import type { ObjectiveFn } from 'aifn-compute/optim'
import { dense, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { ObjectiveInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

type F64 = dense.F64
const { mat, toF64, vec } = dense

/** A test function with everything a figure or a test needs. */
export type TestFunction = {
  name: string
  dimension: number
  /** f(x). */
  value: (x: Vector) => number
  /** f(x) and ∇f(x), for the gradient-based methods. */
  objective: ObjectiveFn
  /** ∇²f(x) (n×n). */
  hessian: (x: Vector) => Matrix
  /** The global minimisers (all of them where there are finitely many and they are known). */
  minima: Vector[]
  /** f at the global minimisers. */
  minimumValue: number
  /** A conventional starting point. */
  start: Vector
  /** A box that shows the interesting structure, per coordinate: `lo[i] ≤ x_i ≤ hi[i]`. */
  lo: Vector
  hi: Vector
}

type Kernel = { value: (x: F64) => number; grad: (x: F64) => F64; hessian: (x: F64) => F64 }

function build(
  name: string,
  n: number,
  kernel: Kernel,
  extra: { minima: number[][]; minimumValue: number; start: number[]; lo: number[]; hi: number[] },
): TestFunction {
  const read = (x: Vector) => {
    const a = toF64(x, name)
    if (a.length !== n) throw new ShapeError(name, `${name}: expected a point of length ${n}, got ${a.length}`)
    return a
  }
  return {
    name,
    dimension: n,
    value: (x) => kernel.value(read(x)),
    objective: (x) => {
      const a = read(x)
      return { value: kernel.value(a), grad: vec(kernel.grad(a)) }
    },
    hessian: (x) => mat(kernel.hessian(read(x)), n, n),
    minima: extra.minima.map((m) => vec(Float64Array.from(m))),
    minimumValue: extra.minimumValue,
    start: vec(Float64Array.from(extra.start)),
    lo: vec(Float64Array.from(extra.lo)),
    hi: vec(Float64Array.from(extra.hi)),
  }
}

/**
 * The Rosenbrock function (Rosenbrock, 1960), generalised to n ≥ 2 dimensions as
 * f(x) = Σᵢ b(x_{i+1} − x_i²)² + (a − x_i)², with a curved, flat-bottomed valley. Minimum 0 at (a, a², …) for a = 1
 * (all ones); defaults a = 1, b = 100, n = 2, start (−1.2, 1, −1.2, 1, …).
 */
export function rosenbrock({ a = 1, b = 100, n = 2 }: { a?: number; b?: number; n?: number } = {}): TestFunction {
  if (n < 2) throw new DomainError('rosenbrock', 'rosenbrock: needs n ≥ 2')
  const kernel: Kernel = {
    value: (x) => {
      let s = 0
      for (let i = 0; i < n - 1; i++) s += b * (x[i + 1] - x[i] * x[i]) ** 2 + (a - x[i]) ** 2
      return s
    },
    grad: (x) => {
      const g = new Float64Array(n)
      for (let i = 0; i < n - 1; i++) {
        const r = x[i + 1] - x[i] * x[i]
        g[i] += -4 * b * x[i] * r - 2 * (a - x[i])
        g[i + 1] += 2 * b * r
      }
      return g
    },
    hessian: (x) => {
      const H = new Float64Array(n * n)
      for (let i = 0; i < n - 1; i++) {
        H[i * n + i] += 12 * b * x[i] * x[i] - 4 * b * x[i + 1] + 2
        H[i * n + i + 1] += -4 * b * x[i]
        H[(i + 1) * n + i] += -4 * b * x[i]
        H[(i + 1) * n + i + 1] += 2 * b
      }
      return H
    },
  }
  // The minimiser (a, a², a⁴…) is exact only for a = 1 when n > 2; for n = 2 it is (a, a²).
  const minimum = n === 2 ? [a, a * a] : Array.from({ length: n }, () => a)
  return build('rosenbrock', n, kernel, {
    minima: n === 2 || a === 1 ? [minimum] : [],
    minimumValue: 0,
    start: Array.from({ length: n }, (_, i) => (i % 2 === 0 ? -1.2 : 1)),
    lo: Array.from({ length: n }, () => -2),
    hi: Array.from({ length: n }, () => 3),
  })
}

/** Himmelblau's function (Himmelblau, 1972), f = (x² + y − 11)² + (x + y² − 7)²: four minima with value 0. */
export function himmelblau(): TestFunction {
  const kernel: Kernel = {
    value: ([x, y]) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
    grad: ([x, y]) => {
      const u = x * x + y - 11
      const v = x + y * y - 7
      return Float64Array.of(4 * x * u + 2 * v, 2 * u + 4 * y * v)
    },
    hessian: ([x, y]) => {
      const u = x * x + y - 11
      const v = x + y * y - 7
      const hxy = 4 * x + 4 * y
      return Float64Array.of(4 * u + 8 * x * x + 2, hxy, hxy, 2 + 4 * v + 8 * y * y)
    },
  }
  return build('himmelblau', 2, kernel, {
    minima: [
      [3, 2],
      [-2.805118086952745, 3.131312518250573],
      [-3.779310253377747, -3.28318599128617],
      [3.584428340330492, -1.848126526964404],
    ],
    minimumValue: 0,
    start: [0, 0],
    lo: [-5, -5],
    hi: [5, 5],
  })
}

/**
 * Beale's function, f = (1.5 − x + xy)² + (2.25 − x + xy²)² + (2.625 − x + xy³)²: minimum 0 at (3, 0.5) in a narrow
 * curved valley, with flat plateaus near the axes. Start (1, 1).
 */
export function beale(): TestFunction {
  const c = [1.5, 2.25, 2.625]
  const kernel: Kernel = {
    value: ([x, y]) => c.reduce((s, ci, k) => s + (ci - x + x * y ** (k + 1)) ** 2, 0),
    grad: ([x, y]) => {
      let gx = 0
      let gy = 0
      c.forEach((ci, k) => {
        const p = k + 1
        const r = ci - x + x * y ** p
        gx += 2 * r * (y ** p - 1)
        gy += 2 * r * x * p * y ** (p - 1)
      })
      return Float64Array.of(gx, gy)
    },
    hessian: ([x, y]) => {
      let hxx = 0
      let hxy = 0
      let hyy = 0
      c.forEach((ci, k) => {
        const p = k + 1
        const r = ci - x + x * y ** p
        const rx = y ** p - 1
        const ry = x * p * y ** (p - 1)
        const rxy = p * y ** (p - 1)
        const ryy = p > 1 ? x * p * (p - 1) * y ** (p - 2) : 0
        hxx += 2 * rx * rx
        hxy += 2 * (rx * ry + r * rxy)
        hyy += 2 * (ry * ry + r * ryy)
      })
      return Float64Array.of(hxx, hxy, hxy, hyy)
    },
  }
  return build('beale', 2, kernel, {
    minima: [[3, 0.5]],
    minimumValue: 0,
    start: [1, 1],
    lo: [-4.5, -4.5],
    hi: [4.5, 4.5],
  })
}

/**
 * A quadratic bowl f(x) = ½(x − c)ᵀA(x − c) with condition number κ = `condition`. In n dimensions A is diagonal with
 * eigenvalues spaced geometrically from 1 to κ; in 2-D it is rotated by `angle` radians, A = R diag(1, κ) Rᵀ.
 * Minimum 0 at c (default 0). Start (−2, 1.5, …) + c.
 */
export function quadraticBowl({
  condition = 10,
  n = 2,
  angle = 0,
  center,
}: { condition?: number; n?: number; angle?: number; center?: readonly number[] } = {}): TestFunction {
  const eig = Array.from({ length: n }, (_, i) => (n === 1 ? 1 : condition ** (i / (n - 1))))
  const A = new Float64Array(n * n)
  if (n === 2) {
    const [c, s] = [Math.cos(angle), Math.sin(angle)]
    // R diag(λ₁, λ₂) Rᵀ with R the rotation by `angle`.
    A[0] = eig[0] * c * c + eig[1] * s * s
    A[1] = A[2] = (eig[0] - eig[1]) * c * s
    A[3] = eig[0] * s * s + eig[1] * c * c
  } else for (let i = 0; i < n; i++) A[i * n + i] = eig[i]
  const c0 = Float64Array.from(center ?? new Array<number>(n).fill(0))
  const Ad = (x: F64) => {
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i] += A[i * n + j] * (x[j] - c0[j])
    return out
  }
  const kernel: Kernel = {
    value: (x) => 0.5 * Ad(x).reduce((s, v, i) => s + v * (x[i] - c0[i]), 0),
    grad: Ad,
    hessian: () => Float64Array.from(A),
  }
  return build('quadratic-bowl', n, kernel, {
    minima: [Array.from(c0)],
    minimumValue: 0,
    start: Array.from({ length: n }, (_, i) => (i % 2 === 0 ? -2 : 1.5) + c0[i]),
    lo: Array.from(c0, (c) => c - 3),
    hi: Array.from(c0, (c) => c + 3),
  })
}

/**
 * The Rastrigin function (Rastrigin, 1974), f(x) = An + Σ(x_i² − A cos 2πx_i): a bowl covered in a regular grid of
 * local minima; the global minimum is 0 at the origin. Defaults A = 10, n = 2, start (2.3, −1.7, …).
 */
export function rastrigin({ A = 10, n = 2 }: { A?: number; n?: number } = {}): TestFunction {
  const w = 2 * Math.PI
  const kernel: Kernel = {
    value: (x) => x.reduce((s, xi) => s + xi * xi - A * Math.cos(w * xi), A * n),
    grad: (x) => x.map((xi) => 2 * xi + A * w * Math.sin(w * xi)),
    hessian: (x) => {
      const H = new Float64Array(n * n)
      for (let i = 0; i < n; i++) H[i * n + i] = 2 + A * w * w * Math.cos(w * x[i])
      return H
    },
  }
  return build('rastrigin', n, kernel, {
    minima: [new Array<number>(n).fill(0)],
    minimumValue: 0,
    start: Array.from({ length: n }, (_, i) => (i % 2 === 0 ? 2.3 : -1.7)),
    lo: new Array<number>(n).fill(-5.12),
    hi: new Array<number>(n).fill(5.12),
  })
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const objective = definer<ObjectiveInfo>('objective', 'data/objectives')

objective(
  {
    key: 'rosenbrock',
    name: 'Rosenbrock',
    summary: 'A curved, narrow valley with its minimum at (a, a², …): slow for gradient descent.',
    params: space({
      a: real(-3, 3, { default: 1 }),
      b: real(1, 1000, { default: 100, scale: 'log' }),
      n: int(2, 20, { default: 2 }),
    }),
    dim: null,
    truth: true,
    notes: ['gradient-descent', 'newtons-method', 'quasi-newton-methods'],
  },
  rosenbrock,
)

objective(
  {
    key: 'himmelblau',
    name: 'Himmelblau',
    summary: 'A two-dimensional surface with four global minima of value 0.',
    params: space({}),
    dim: 2,
    truth: true,
    notes: ['gradient-descent'],
  },
  himmelblau,
)

objective(
  {
    key: 'beale',
    name: 'Beale',
    summary: 'A two-dimensional surface with sharp ridges and its minimum at (3, 0.5).',
    params: space({}),
    dim: 2,
    truth: true,
    notes: ['gradient-descent'],
  },
  beale,
)

objective(
  {
    key: 'quadraticBowl',
    name: 'Quadratic bowl',
    summary: 'A convex quadratic with condition number κ, rotated in two dimensions.',
    params: space({
      condition: real(1, 1000, { default: 10, label: 'κ', scale: 'log' }),
      n: int(1, 20, { default: 2 }),
      angle: real(-Math.PI, Math.PI, { default: 0 }),
    }),
    dim: null,
    truth: true,
    notes: ['condition-number', 'convergence-of-gradient-descent', 'momentum-and-nesterov'],
  },
  quadraticBowl,
)

objective(
  {
    key: 'rastrigin',
    name: 'Rastrigin',
    summary: 'A bowl covered in a regular grid of local minima; the global minimum is 0 at the origin.',
    params: space({ A: real(0, 50, { default: 10 }), n: int(1, 20, { default: 2 }) }),
    dim: null,
    truth: true,
    notes: ['gradient-descent'],
  },
  rastrigin,
)

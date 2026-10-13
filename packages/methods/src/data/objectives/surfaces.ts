/**
 * Test surfaces for optimisers, each with its value, gradient, Hessian, known minima, a conventional start and a
 * plotting domain (moved from `aifn-compute/optim`: they are data, not methods). Formulas from Jamil and Yang (2013),
 * "A literature survey of benchmark functions for global optimisation problems", and Surjanovic and Bingham, "Virtual
 * Library of Simulation Experiments" (sfu.ca/~ssurjano).
 *
 * Each factory returns a `TestFunction` whose value, gradient and Hessian are closed forms evaluated in float64; they
 * check that a point has the surface's dimension and throw `ShapeError` otherwise. The surfaces are registered as
 * objectives (kind `objective`), with their parameters as a space.
 */

import type { ObjectiveFn } from 'aifn-compute/optim'
import { dense, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { ObjectiveInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A float64 working array. */
type F64 = dense.F64
const { mat, toF64, vec } = dense

/** A test function with everything a figure or a test needs. */
export type TestFunction = {
  /** The surface's name, also used in error messages. */
  name: string
  /** The number $n$ of coordinates of a point. */
  dimension: number
  /** $f(\xvec)$. */
  value: (x: Vector) => number
  /** $f(\xvec)$ and $\nabla f(\xvec)$, for the gradient-based methods. */
  objective: ObjectiveFn
  /** $\nabla^2 f(\xvec)$ ($n \times n$). */
  hessian: (x: Vector) => Matrix
  /** The global minimisers (all of them where there are finitely many and they are known). */
  minima: Vector[]
  /** $f$ at the global minimisers. */
  minimumValue: number
  /** A conventional starting point. */
  start: Vector
  /** The lower corner of a box that shows the interesting structure: $\mathrm{lo}_i \le x_i \le \mathrm{hi}_i$. */
  lo: Vector
  /** The upper corner of that box. */
  hi: Vector
}

/** A surface's closed forms on a working array: the value, the gradient and the Hessian (row-major, $n^2$ values). */
type Kernel = { value: (x: F64) => number; grad: (x: F64) => F64; hessian: (x: F64) => F64 }

/**
 * Wrap a surface's closed forms as a `TestFunction`: every method checks that its point has $n$ coordinates (throwing
 * `ShapeError` otherwise) and converts the gradient and Hessian to a vector and an $n \times n$ matrix.
 *
 * @param name The surface's name, used as its `name` and in error messages.
 * @param n The dimension of a point.
 * @param kernel The value, gradient and Hessian on a float64 array of $n$ values.
 * @param extra The known minimisers (each $n$ values), the value there, the conventional start, and the lower and
 *   upper corners of the plotting box.
 * @returns The test function.
 */
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
 * The Rosenbrock function (Rosenbrock, 1960), generalised to $n \ge 2$ dimensions as
 * $f(\xvec) = \sum_{i=1}^{n-1} b(x_{i+1} - x_i^2)^2 + (a - x_i)^2$, with a curved, flat-bottomed valley. In two
 * dimensions the minimum is 0 at $(a, a^2)$; in more, with $a = 1$ it is 0 at $(1, \dots, 1)$, with $a = 0$
 * it is 0 at $(0, \dots, 0)$, and for other $a$ the minimum is positive and no minimiser is listed. The start is
 * $(-1.2, 1, -1.2, 1, \dots)$ and the plotting box $[-2, 3]^n$. Throws `DomainError` when $n < 2$.
 *
 * @param options The shape parameters and the dimension.
 * @param options.a The valley's offset $a$: the minimiser's first coordinate.
 * @param options.b The weight $b$ of the curved-valley term: the larger, the steeper the valley's walls.
 * @param options.n The dimension $n$, at least 2.
 * @returns The test function.
 *
 * @example The value at the minimum and at the conventional start
 * const f = rosenbrock()
 * print('f at the minimum', f.minima[0], ':', f.value(f.minima[0]))
 * print('f at the start', f.start, ':', f.value(f.start))
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
  // The global minimum value is 0 for n = 2 (at (a, a²)), for a = 1 (at (1…1)), and for a = 0 (at the origin).
  const knownZero = n === 2 || a === 1 || a === 0
  const minimum =
    n === 2 ? [a, a * a] : a === 1 ? Array.from({ length: n }, () => 1) : Array.from({ length: n }, () => 0)
  return build('rosenbrock', n, kernel, {
    minima: knownZero ? [minimum] : [],
    minimumValue: knownZero ? 0 : NaN,
    start: Array.from({ length: n }, (_, i) => (i % 2 === 0 ? -1.2 : 1)),
    lo: Array.from({ length: n }, () => -2),
    hi: Array.from({ length: n }, () => 3),
  })
}

/**
 * Himmelblau's function (Himmelblau, 1972), $f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$: four global minima of value
 * 0, one at $(3, 2)$ and three irrational. The start is the origin and the plotting box $[-5, 5]^2$.
 *
 * @returns The two-dimensional test function.
 *
 * @example The value at each of the four minima
 * const f = himmelblau()
 * print('minima:', f.minima)
 * print('f there:', f.minima.map((m) => f.value(m)))
 */
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
 * Beale's function, $f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2$: minimum 0 at
 * $(3, 0.5)$ in a narrow curved valley, with flat plateaus near the axes. The start is $(1, 1)$ and the plotting box
 * $[-4.5, 4.5]^2$.
 *
 * @returns The two-dimensional test function.
 *
 * @example The value and gradient at the minimum
 * const f = beale()
 * const { value, grad } = f.objective(f.minima[0])
 * print('f(3, 0.5) =', value, ' gradient:', grad)
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
 * A quadratic bowl $f(\xvec) = \tfrac12 (\xvec - \cvec)^\top \Amat (\xvec - \cvec)$ with condition number
 * $\kappa$ = `condition`. In $n$ dimensions $\Amat$ is diagonal with eigenvalues spaced geometrically from 1 to
 * $\kappa$ (just 1 when $n = 1$); in two it is rotated by `angle` radians, $\Amat = \Rmat \diag(1, \kappa) \Rmat^\top$.
 * Minimum 0 at $\cvec$. The start is $(-2, 1.5, -2, \dots) + \cvec$ and the plotting box $\cvec \pm 3$ in each
 * coordinate.
 *
 * @param options The conditioning, the dimension, the rotation and the centre.
 * @param options.condition The condition number $\kappa$ of $\Amat$, the ratio of its largest eigenvalue to its
 *   smallest.
 * @param options.n The dimension $n$.
 * @param options.angle The rotation of the eigenvectors, in radians (two dimensions only; ignored otherwise).
 * @param options.center The minimiser $\cvec$, $n$ values (default the origin). Throws `ShapeError` if its length
 *   does not match `n`.
 * @returns The test function.
 *
 * @example The Hessian holds the eigenvalues 1 and the condition number
 * const f = quadraticBowl({ condition: 100 })
 * print('Hessian:', f.hessian(f.start))
 * print('f at the minimum:', f.value(f.minima[0]), ' at the start:', f.value(f.start))
 */
export function quadraticBowl({
  condition = 10,
  n = 2,
  angle = 0,
  center,
}: { condition?: number; n?: number; angle?: number; center?: readonly number[] } = {}): TestFunction {
  if (center && center.length !== n)
    throw new ShapeError(
      'quadraticBowl',
      `quadraticBowl: center length (${center.length}) must match dimension n (${n})`,
    )
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
 * The Rastrigin function (Rastrigin, 1974), $f(\xvec) = An + \sum_i \left(x_i^2 - A \cos 2\pi x_i\right)$: a bowl
 * covered in a regular grid of local minima near the integer points; the global minimum is 0 at the origin. The start
 * is $(2.3, -1.7, 2.3, \dots)$ and the plotting box $[-5.12, 5.12]^n$.
 *
 * @param options The ripple amplitude and the dimension.
 * @param options.A The amplitude $A$ of the cosine ripple: 0 leaves the bowl $\lVert \xvec \rVert^2$.
 * @param options.n The dimension $n$.
 * @returns The test function.
 *
 * @example The global minimum, and a local one next to it
 * const f = rastrigin()
 * print('f at the origin:', f.value([0, 0]))
 * print('f near (1, 0):', f.value([0.995, 0]), ' gradient there:', f.objective([0.995, 0]).grad)
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

/**
 * The minimiser of a one-dimensional function on $[\mathrm{lo}, \mathrm{hi}]$: the best of 2001 equally spaced points,
 * polished by Newton's method on $f'$ when it is interior (the step is kept inside the neighbouring grid cells).
 *
 * @param kernel The function's closed forms; the gradient and Hessian are read as one value each.
 * @param lo The left end of the interval.
 * @param hi The right end.
 * @returns The minimiser.
 */
function minimise1d(kernel: Kernel, lo: number, hi: number): number {
  const n = 2001
  const h = (hi - lo) / (n - 1)
  let best = lo
  for (let i = 1; i < n; i++) {
    const x = lo + i * h
    if (kernel.value(Float64Array.of(x)) < kernel.value(Float64Array.of(best))) best = x
  }
  if (best === lo || best === hi) return best
  let x = best
  for (let it = 0; it < 50; it++) {
    const g = kernel.grad(Float64Array.of(x))[0]
    const H = kernel.hessian(Float64Array.of(x))[0]
    if (!(H > 0)) break
    const next = Math.min(Math.max(x - g / H, best - h), best + h)
    if (Math.abs(next - x) <= 1e-15 * Math.max(1, Math.abs(x))) break
    x = next
  }
  return x
}

/**
 * The one-dimensional test function of Forrester, Sóbester and Keane (2008), $f(x) = (6x - 2)^2 \sin(12x - 4)$ on
 * $[0, 1]$: a local minimum of $-0.986$ at $x = 0.1426$, a gentle hump, and the global minimum $-6.0207$ at
 * $x = 0.7572$ in a narrow well, a standard first example of Bayesian optimisation. The low-fidelity version
 * (`fidelity: 'low'`) is their cheap approximation $a f(x) + b(x - 0.5) - c$, with $a = 0.5$, $b = 10$ and $c = -5$
 * by default: correlated with $f$ but tilted, so its minimum ($0.665$ at $x = 0.0924$) is in the wrong place, the
 * trap of multi-fidelity optimisation. The minimiser is computed (on a fine grid, then by Newton's method), so it
 * holds for any $a$, $b$, $c$. The start is $0.5$ and the plotting box $[0, 1]$.
 *
 * @param options The fidelity and the coefficients of the low-fidelity version.
 * @param options.fidelity `'high'` (default) for $f$ itself, `'low'` for $a f(x) + b(x - 0.5) - c$.
 * @param options.a The scale $a$ of $f$ in the low-fidelity version.
 * @param options.b The slope $b$ of the low-fidelity version's linear tilt.
 * @param options.c The offset $c$ subtracted in the low-fidelity version.
 * @returns The one-dimensional test function.
 *
 * @example Both fidelities at a few points, and their minima
 * const hi = forrester()
 * const lo = forrester({ fidelity: 'low' })
 * for (const x of [0, 0.5, 0.75, 1])
 *   print('x =', x, ' f =', hi.value([x]).toFixed(4), ' low =', lo.value([x]).toFixed(4))
 * print('minimum', hi.minimumValue, 'at', hi.minima[0], '; low fidelity', lo.minimumValue, 'at', lo.minima[0])
 */
export function forrester({
  fidelity = 'high',
  a = 0.5,
  b = 10,
  c = -5,
}: { fidelity?: 'high' | 'low'; a?: number; b?: number; c?: number } = {}): TestFunction {
  const low = fidelity === 'low'
  const f = (x: number) => (6 * x - 2) ** 2 * Math.sin(12 * x - 4)
  const df = (x: number) => {
    const u = 6 * x - 2
    return 12 * u * Math.sin(12 * x - 4) + 12 * u * u * Math.cos(12 * x - 4)
  }
  const d2f = (x: number) => {
    const u = 6 * x - 2
    const [s, co] = [Math.sin(12 * x - 4), Math.cos(12 * x - 4)]
    return 72 * s + 288 * u * co - 144 * u * u * s
  }
  const kernel: Kernel = {
    value: ([x]) => (low ? a * f(x) + b * (x - 0.5) - c : f(x)),
    grad: ([x]) => Float64Array.of(low ? a * df(x) + b : df(x)),
    hessian: ([x]) => Float64Array.of(low ? a * d2f(x) : d2f(x)),
  }
  const xStar = minimise1d(kernel, 0, 1)
  return build(low ? 'forrester-low' : 'forrester', 1, kernel, {
    minima: [[xStar]],
    minimumValue: kernel.value(Float64Array.of(xStar)),
    start: [0.5],
    lo: [0],
    hi: [1],
  })
}

/** The constants of the Branin function: $b = 5.1 / 4\pi^2$, $c = 5 / \pi$, $r = 6$, $s = 10$, $t = 1 / 8\pi$. */
const BRANIN = { b: 5.1 / (4 * Math.PI ** 2), c: 5 / Math.PI, r: 6, s: 10, t: 1 / (8 * Math.PI) }

/**
 * The Branin (Branin–Hoo) function (Branin, 1972; in the form of Dixon and Szegő, 1978),
 * $f(\xvec) = (x_2 - b x_1^2 + c x_1 - r)^2 + s(1 - t)\cos x_1 + s$ with $b = 5.1 / 4\pi^2$, $c = 5 / \pi$, $r = 6$,
 * $s = 10$ and $t = 1 / 8\pi$, on $[-5, 10] \times [0, 15]$: three global minima of value $s t = 10 / 8\pi
 * \approx 0.397887$, at $(-\pi, 12.275)$, $(\pi, 2.275)$ and $(3\pi, 2.475)$, along a curved valley. A standard
 * two-dimensional benchmark of Bayesian optimisation. The start is the box's centre $(2.5, 7.5)$.
 *
 * @returns The two-dimensional test function.
 *
 * @example The three minima, and the value at the start
 * const f = branin()
 * print('minima:', f.minima)
 * print('f there:', f.minima.map((m) => f.value(m)), ' 10 / 8 pi =', 10 / (8 * Math.PI))
 * print('f at the start', f.start, ':', f.value(f.start))
 */
export function branin(): TestFunction {
  const { b, c, r, s, t } = BRANIN
  const kernel: Kernel = {
    value: ([x, y]) => (y - b * x * x + c * x - r) ** 2 + s * (1 - t) * Math.cos(x) + s,
    grad: ([x, y]) => {
      const g = y - b * x * x + c * x - r
      return Float64Array.of(2 * g * (c - 2 * b * x) - s * (1 - t) * Math.sin(x), 2 * g)
    },
    hessian: ([x, y]) => {
      const g = y - b * x * x + c * x - r
      const gx = c - 2 * b * x
      return Float64Array.of(2 * gx * gx - 4 * b * g - s * (1 - t) * Math.cos(x), 2 * gx, 2 * gx, 2)
    },
  }
  const at = (x: number) => [x, b * x * x - c * x + r]
  return build('branin', 2, kernel, {
    minima: [at(-Math.PI), at(Math.PI), at(3 * Math.PI)],
    minimumValue: s * t,
    start: [2.5, 7.5],
    lo: [-5, 0],
    hi: [10, 15],
  })
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Registers a surface of this file (kind `objective`, area `data/objectives`). */
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

objective(
  {
    key: 'forrester',
    name: 'Forrester',
    summary: 'A one-dimensional function with a deceptive local minimum, and a tilted low-fidelity version.',
    params: space({
      fidelity: oneOf(['high', 'low'], { default: 'high' }),
      a: real(0, 2, { default: 0.5 }),
      b: real(-20, 20, { default: 10 }),
      c: real(-10, 10, { default: -5 }),
    }),
    dim: 1,
    truth: true,
    cite: ['forrester2008'],
  },
  forrester,
)

objective(
  {
    key: 'branin',
    name: 'Branin',
    summary: 'A two-dimensional function with three global minima of value 0.397887 along a curved valley.',
    params: space({}),
    dim: 2,
    truth: true,
    cite: ['branin1972'],
  },
  branin,
)

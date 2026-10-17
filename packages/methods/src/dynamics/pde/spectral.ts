/**
 * Periodic spectral solutions on $[0, 1)$: real Fourier series and the exact solution of the periodic heat equation,
 * and a pseudo-spectral solver for the viscous Burgers equation, the reference outputs of operator learning.
 *
 * A `FourierSeries1d` is $f(x) = a_0 + \sum_{k=1}^{K} (a_k \cos 2\pi kx + b_k \sin 2\pi kx)$. The heat equation
 * $u_t = D u_{xx}$ multiplies mode $k$ by $e^{-D(2\pi k)^2 t}$ (`heatSolution`). `solveBurgersPeriodic` integrates
 * $u_t + (u^2/2)_x = \nu u_{xx}$ by a Fourier pseudo-spectral method with the 2/3 dealiasing rule and fourth-order
 * Runge–Kutta in the integrating-factor form, so the diffusion is integrated exactly and the step is limited by
 * advection only (Trefethen, 2000, "Spectral Methods in MATLAB", ch. 10; Kassam and Trefethen, 2005, "Fourth-order
 * time-stepping for stiff PDEs"). Its error against the Cole–Hopf solution `burgersSolution` of
 * `aifn-methods/neural/pinn` is about $5 \times 10^{-8}$ at 128 points. Unlike the finite-difference solvers of this
 * module (`heatEquation` and its neighbours on a Dirichlet or Neumann `Grid1`), these work on the periodic unit
 * interval only, and return results rather than stepping.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { irfft, rfft } from 'aifn-compute/foundation/fourier'
import { fromData, readonlyData, type Tensor } from 'aifn-compute/foundation/tensor'

/**
 * A real periodic function on $[0, 1)$, $f(x) = a_0 + \sum_{k=1}^{K} (a_k \cos 2\pi kx + b_k \sin 2\pi kx)$.
 */
export type FourierSeries1d = {
  /** The constant term $a_0$. */
  mean: number
  /** The cosine coefficients $a_1, \dots, a_K$. */
  cos: Float64Array
  /** The sine coefficients $b_1, \dots, b_K$. */
  sin: Float64Array
}

/**
 * A Fourier series evaluated at the points `xs`, term by term.
 *
 * @param f The series.
 * @param xs The points.
 * @returns $f(x)$ at each point.
 *
 * @example A sine and a constant
 * const f = { mean: 1, cos: Float64Array.of(0), sin: Float64Array.of(2) }
 * print('f =', evaluateSeries(f, periodicGrid(4)))
 */
export function evaluateSeries(f: FourierSeries1d, xs: ArrayLike<number>): Float64Array {
  const out = new Float64Array(xs.length)
  for (let i = 0; i < xs.length; i++) {
    let v = f.mean
    for (let k = 0; k < f.cos.length; k++) {
      const a = 2 * Math.PI * (k + 1) * xs[i]
      v += f.cos[k] * Math.cos(a) + f.sin[k] * Math.sin(a)
    }
    out[i] = v
  }
  return out
}

/**
 * The uniform periodic grid of $n$ points on $[0, 1)$, $x_j = j/n$.
 *
 * @param n The number of points.
 * @returns The points.
 *
 * @example Eight points
 * print('x =', periodicGrid(8))
 */
export function periodicGrid(n: number): Float64Array {
  return Float64Array.from({ length: n }, (_, j) => j / n)
}

/**
 * The exact solution of the periodic heat equation $u_t = D u_{xx}$ at time $t$ from the initial series $f$: mode $k$
 * decays by $e^{-D(2\pi k)^2 t}$ and the mean is kept.
 *
 * @param f The initial profile, as a series.
 * @param diffusivity The diffusivity $D$.
 * @param time The time $t$.
 * @returns The solution at time $t$, as a series.
 *
 * @example The first mode decays by exp(-0.1 * 4 pi^2 * 0.1) = 0.6738
 * const f = { mean: 0, cos: Float64Array.of(0, 0), sin: Float64Array.of(1, 1) }
 * print('sine coefficients at t = 0.1:', heatSolution(f, 0.1, 0.1).sin)
 */
export function heatSolution(f: FourierSeries1d, diffusivity: number, time: number): FourierSeries1d {
  const decay = (k: number) => Math.exp(-diffusivity * (2 * Math.PI * k) ** 2 * time)
  return {
    mean: f.mean,
    cos: f.cos.map((c, k) => c * decay(k + 1)),
    sin: f.sin.map((c, k) => c * decay(k + 1)),
  }
}

/**
 * The viscous Burgers equation $u_t + (u^2/2)_x = \nu u_{xx}$ on the periodic domain $[0, 1)$, for a batch of
 * initial profiles at once: Fourier pseudo-spectral in space with the 2/3 rule on the nonlinear term,
 * integrating-factor RK4 in time (the diffusion is integrated exactly, so the step is limited by advection only).
 * The spectra are batched `rfft` and `irfft` calls along the grid. About 1 s for 80 profiles of 128 points at
 * $\nu = 0.02$, $t = 0.5$: a plain function of a tensor and numbers, so a figure can run it in the compute worker.
 * Throws `DomainError` for non-finite profiles or an odd grid.
 *
 * @param u0 The initial profiles on the uniform grid $x_j = j/M$, $[\text{count}, M]$, $M$ even.
 * @param options The viscosity, the end time and the number of steps.
 * @param options.viscosity The viscosity $\nu$.
 * @param options.time The end time $t$.
 * @param options.steps The number of time steps (default: enough that
 *   $\Delta t \le 0.4\,\Delta x / \max\lvert u_0 \rvert$, at least 50; at most 100,000 either way).
 * @returns The profiles at time $t$, $[\text{count}, M]$.
 *
 * @example A sine steepens towards a shock
 * const x = periodicGrid(64)
 * const u0 = fromData(Float64Array.from(x, (v) => Math.sin(2 * Math.PI * v)), [1, 64])
 * const u = solveBurgersPeriodic(u0, { viscosity: 0.01, time: 0.15 })
 * print('max |u| before and after:', 1, max(abs(u)))
 */
export function solveBurgersPeriodic(
  u0: Tensor,
  { viscosity, time, steps }: { viscosity: number; time: number; steps?: number },
): Tensor {
  const [count, M] = u0.shape
  if (M % 2 !== 0)
    throw new DomainError('solveBurgersPeriodic', `solveBurgersPeriodic: the grid must be even (got ${M})`)
  const H = M / 2 + 1
  const start = Float64Array.from(readonlyData(u0) as Float64Array)
  let peak = 0
  for (const a of start) peak = Math.max(peak, Math.abs(a))
  if (!Number.isFinite(peak))
    throw new DomainError('solveBurgersPeriodic', 'solveBurgersPeriodic: the initial profiles are not finite')
  // At most 100 000 steps, so a huge profile cannot make the loop run for ever.
  const n = Math.min(100_000, steps ?? Math.max(50, Math.ceil((time * peak) / (0.4 / M))))
  const h = time / n
  // The integrating factors exp(−ν(2πk)²h) and exp(−ν(2πk)²h/2), and the dealiasing mask k < M/3.
  const E = new Float64Array(H)
  const E2 = new Float64Array(H)
  const keep = new Float64Array(H)
  for (let k = 0; k < H; k++) {
    E[k] = Math.exp(-viscosity * (2 * Math.PI * k) ** 2 * h)
    E2[k] = Math.exp(-viscosity * (2 * Math.PI * k) ** 2 * (h / 2))
    keep[k] = 3 * k < M ? 1 : 0
  }
  const forward = (u: Float64Array) =>
    Float64Array.from(readonlyData(rfft(fromData(u, [count, M]), { axis: 1 })) as Float64Array)
  const inverse = (v: Float64Array) =>
    readonlyData(irfft(fromData(v, [count, H], 'complex128'), { axis: 1, n: M })) as Float64Array
  // h N(v), N(v) = −(i 2πk / 2) FFT(u²) dealiased, for the spectrum v (interleaved re, im).
  const nonlinear = (v: Float64Array) => {
    const u = inverse(v)
    const sq = new Float64Array(u.length)
    for (let i = 0; i < u.length; i++) sq[i] = u[i] * u[i]
    const w = forward(sq)
    for (let i = 0; i < count; i++)
      for (let k = 0; k < H; k++) {
        const j = 2 * (i * H + k)
        const c = h * Math.PI * k * keep[k]
        const re = w[j]
        w[j] = c * w[j + 1]
        w[j + 1] = -c * re
      }
    return w
  }
  // fa ∘ v + scale · fb ∘ w for per-mode real factors fa, fb (null for 1).
  const combine = (v: Float64Array, fa: Float64Array | null, w: Float64Array, fb: Float64Array | null, scale = 1) => {
    const out = new Float64Array(v.length)
    for (let i = 0; i < count; i++)
      for (let k = 0; k < H; k++) {
        const j = 2 * (i * H + k)
        const a = fa ? fa[k] : 1
        const b = (fb ? fb[k] : 1) * scale
        out[j] = a * v[j] + b * w[j]
        out[j + 1] = a * v[j + 1] + b * w[j + 1]
      }
    return out
  }
  let v = forward(start)
  for (let step = 0; step < n; step++) {
    const a = nonlinear(v)
    const b = nonlinear(combine(v, E2, a, E2, 0.5))
    const c = nonlinear(combine(v, E2, b, null, 0.5))
    const d = nonlinear(combine(v, E, c, E2))
    const next = new Float64Array(v.length)
    for (let i = 0; i < count; i++)
      for (let k = 0; k < H; k++)
        for (let r = 0; r < 2; r++) {
          const j = 2 * (i * H + k) + r
          next[j] = E[k] * v[j] + (E[k] * a[j] + 2 * E2[k] * (b[j] + c[j]) + d[j]) / 6
        }
    v = next
  }
  return fromData(Float64Array.from(inverse(v)), [count, M])
}

/**
 * Method-of-lines finite differences on a uniform one-dimensional grid for the heat, transport (advection), wave and
 * Fokker–Planck equations, each a traceable `Algorithm` stepping in time.
 *
 * Every solver's state reports its stability number (the diffusion number $r = D \Delta t/\Delta x^2$ or the Courant
 * number $\nu = \lvert c \rvert \Delta t/\Delta x$), the scheme's limit and whether it holds, rather than refusing
 * an unstable step; a non-finite solution sets `diverged`. The heat and Fokker–Planck solvers share a $\theta$-method
 * on a tridiagonal operator, whose implicit system is factored once (LeVeque, 2007, "Finite Difference Methods for
 * Ordinary and Partial Differential Equations", §9–10; Morton and Mayers, 2005, "Numerical Solution of Partial
 * Differential Equations", 2nd ed., §2–4; Chang and Cooper, 1970, for the conservative Fokker–Planck flux form).
 */

import { lu, luSolve, type LU } from 'aifn-compute/numerics/linalg'
import { fromData, toFlat, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Status } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { DomainError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'

type F64 = Float64Array<ArrayBuffer>

/**
 * A uniform grid of `n` points on [`a`, `b`], both ends included (spacing $\Delta x = (b - a)/(n - 1)$; at least 3
 * points).
 */
export type Grid1 = { a: number; b: number; n: number }

/**
 * The points of a grid (length $n$). Throws `DomainError` for fewer than 3 points.
 *
 * @param options The grid.
 * @param options.a The left end $a$.
 * @param options.b The right end $b$.
 * @param options.n The number of points $n$, ends included.
 * @returns The points $a + (b - a) i/(n - 1)$, $i = 0, \dots, n - 1$.
 *
 * @example Five points on [0, 1]
 * print(gridPoints({ a: 0, b: 1, n: 5 }))
 */
export function gridPoints({ a, b, n }: Grid1): Vector {
  if (!(n >= 3)) throw new DomainError('gridPoints', 'gridPoints: a grid needs at least 3 points')
  return fromData(
    Float64Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1)),
    [n],
  )
}

/** An initial profile: values at the grid points, or a function of x sampled there. */
export type Profile = ArrayLike<number> | Tensor | ((x: number) => number)

/**
 * An initial profile as values at the grid points. Throws `ShapeError` when an array's length is not the grid's.
 *
 * @param profile A function of $x$ (sampled at the points) or $n$ values.
 * @param grid The grid.
 * @param where The caller's name, used in error messages.
 * @returns The $n$ values, as a new array.
 */
function sample(profile: Profile, grid: Grid1, where: string): F64 {
  const xs = toFlat(gridPoints(grid))
  if (typeof profile === 'function') return Float64Array.from(xs, profile)
  const v = Float64Array.from(
    (profile as Tensor).shape !== undefined ? toFlat(profile as Tensor) : Array.from(profile as ArrayLike<number>),
  )
  if (v.length !== grid.n)
    throw new ShapeError(where, `${where}: the initial profile has ${v.length} values for ${grid.n} points`)
  return v
}

/**
 * Boundary conditions, by `kind`: fixed values (`'dirichlet'`, the ends held at `left` and `right`, each defaulting to
 * the initial profile's end value), zero flux (`'neumann'`, reflecting) or `'periodic'`.
 */
export type Boundary = { kind: 'dirichlet'; left?: number; right?: number } | { kind: 'neumann' } | { kind: 'periodic' }

/** How a stability limit reads for the scheme in use. */
export type Stability = {
  /** The name of the number: `'r = DΔt/Δx²'`, `'ν = |c|Δt/Δx'`, … */
  number: string
  /** Its value for the grid and time step in use. */
  value: number
  /** The largest value for which the scheme is stable (Infinity for unconditionally stable schemes). */
  limit: number
  /** Whether `value` is within `limit`. */
  stable: boolean
}

/** The state of every PDE solver: `t` counts time steps (the runner's `Status`), `time` is the solution's time. */
export type PdeState = Status & {
  /** Time steps taken. */
  t: number
  /** The current time, $t \, \Delta t$. */
  time: number
  /** The solution at the grid points. */
  u: Vector
  /** The time step $\Delta t$. */
  dt: number
  /**
   * $\int u \, dx$ by the trapezoid rule (the total mass for densities; `fokkerPlanck` reports the sum
   * $\sum_i u_i \Delta x$ it conserves).
   */
  mass: number
  /** The stability number of the scheme in use, its limit and whether it holds. */
  stability: Stability
  /** True once the solution is not finite. */
  diverged: boolean
  /** `'not finite'` once the solution is not finite, else null. */
  failure: string | null
}

/** The options every solver shares: `tEnd`, the time at which the run is done (default never). */
type Common = { tEnd?: number }

/**
 * $\int u \, dx$ by the trapezoid rule, or the plain sum times $\Delta x$ on a periodic grid.
 *
 * @param u The values at the grid points.
 * @param dx The spacing $\Delta x$.
 * @param periodic Whether every point is a distinct point of a periodic grid (each then weighs $\Delta x$, not
 *   $\Delta x/2$ at the ends).
 * @returns The integral.
 */
function trapezoidMass(u: F64, dx: number, periodic: boolean): number {
  let s = 0
  for (let i = 0; i < u.length; i++) s += u[i]
  if (!periodic) s -= 0.5 * (u[0] + u[u.length - 1])
  return s * dx
}

/**
 * Whether every value is finite.
 *
 * @param u The values.
 * @returns True when none is NaN or infinite.
 */
const finite = (u: F64) => u.every(Number.isFinite)

/**
 * A tridiagonal operator $(\Lmat \uvec)_i = l_i u_{i-1} + d_i u_i + p_i u_{i+1}$ (`lo`, `di`, `up`), with periodic
 * wrap-around when `periodic`; otherwise the missing neighbours at the ends count as 0.
 */
type Tridiagonal = { lo: F64; di: F64; up: F64; periodic: boolean }

/**
 * $\Lmat \uvec$ for a tridiagonal operator.
 *
 * @param L The operator.
 * @param u The values $\uvec$ ($n$, as many as the operator's diagonal); not modified.
 * @returns $\Lmat \uvec$, as a new array.
 */
function apply(L: Tridiagonal, u: F64): F64 {
  const n = u.length
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const left = i > 0 ? u[i - 1] : L.periodic ? u[n - 1] : 0
    const right = i < n - 1 ? u[i + 1] : L.periodic ? u[0] : 0
    out[i] = L.lo[i] * left + L.di[i] * u[i] + L.up[i] * right
  }
  return out
}

/**
 * The LU factors of $\Imat - \theta \Delta t \Lmat$ (dense; $n$ is small in figures), reused at every step since
 * $\Delta t$ is fixed.
 *
 * @param L The operator $\Lmat$.
 * @param theta The implicitness $\theta \in (0, 1]$.
 * @param dt The time step $\Delta t$.
 * @returns The factors, with `singular` set when the system cannot be solved.
 */
function implicitFactor(L: Tridiagonal, theta: number, dt: number): LU {
  const n = L.di.length
  const M = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    M[i * n + i] = 1 - theta * dt * L.di[i]
    const l = i > 0 ? i - 1 : L.periodic ? n - 1 : -1
    const r = i < n - 1 ? i + 1 : L.periodic ? 0 : -1
    if (l >= 0) M[i * n + l] -= theta * dt * L.lo[i]
    if (r >= 0) M[i * n + r] -= theta * dt * L.up[i]
  }
  return lu(fromData(M, [n, n]))
}

/**
 * Time-stepping schemes for $\uvec' = \Lmat \uvec$: explicit (forward) Euler, implicit (backward) Euler, or
 * Crank–Nicolson.
 */
export type TimeScheme = 'explicit' | 'implicit' | 'crank-nicolson'

/** The implicitness $\theta$ of each scheme: 0 explicit, 1 implicit, $1/2$ Crank–Nicolson. */
const THETA: Record<TimeScheme, number> = { explicit: 0, implicit: 1, 'crank-nicolson': 0.5 }

/** A deterministic time stepper from an initial profile (the core of an `Algorithm` whose `init` samples a profile). */
type Stepper = {
  /** The algorithm's name. */
  name: string
  /** The state at time 0 from the sampled profile. */
  init(u0: F64): PdeState
  /** The state one time step on. */
  step(s: PdeState): PdeState
  /** Whether the run has reached its end time. */
  done(s: PdeState): boolean
}

/**
 * A $\theta$-method stepper for $\uvec' = \Lmat\uvec$, shared by the heat and Fokker–Planck solvers:
 * $(\Imat - \theta \Delta t \Lmat)\uvec^{k+1} = \uvec^k + (1 - \theta)\Delta t \Lmat\uvec^k$. Zero rows of
 * $\Lmat$ hold their values fixed (Dirichlet ends). Throws `DomainError` for an unknown scheme and `NumericalError`
 * when the implicit system is singular.
 *
 * @param name The algorithm's name, also used in error messages.
 * @param L The operator $\Lmat$.
 * @param scheme The time-stepping scheme, which sets $\theta$.
 * @param dt The time step $\Delta t$.
 * @param dx The grid spacing $\Delta x$, for the reported mass.
 * @param stability The stability number reported in every state.
 * @param tEnd The time at which the run is done, or undefined for never.
 * @returns The stepper.
 */
function thetaMethod(
  name: string,
  L: Tridiagonal,
  scheme: TimeScheme,
  dt: number,
  dx: number,
  stability: Stability,
  tEnd: number | undefined,
): Stepper {
  const theta = THETA[scheme]
  if (theta === undefined) throw new DomainError(name, `${name}: unknown scheme ${scheme}`)
  const factor = theta > 0 ? implicitFactor(L, theta, dt) : null
  if (factor?.singular) throw new NumericalError(name, `${name}: the implicit system is singular`, 'singular')
  const state = (time: number, u: F64, t: number): PdeState => ({
    t,
    time,
    u: fromData(u, [u.length]),
    dt,
    mass: trapezoidMass(u, dx, L.periodic),
    stability,
    diverged: !finite(u),
    failure: finite(u) ? null : 'not finite',
  })
  return {
    name,
    init: (u0) => state(0, Float64Array.from(u0), 0),
    step: (s) => {
      const u = s.u.data as F64
      const Lu = apply(L, u)
      const rhs = Float64Array.from(u, (v, i) => v + (1 - theta) * dt * Lu[i])
      const next = factor ? (Float64Array.from(toFlat(luSolve(factor, fromData(rhs, [rhs.length])))) as F64) : rhs
      return state(s.time + dt, next, s.t + 1)
    },
    done: (s) => tEnd !== undefined && s.time >= tEnd - 1e-12 * Math.max(1, Math.abs(tEnd)),
  }
}

/** Options for `heatEquation`. */
export type HeatOptions = Common & {
  /** The diffusivity $D > 0$ in $u_t = D u_{xx}$. */
  diffusivity: number
  /** The grid. */
  grid: Grid1
  /** The boundary conditions. */
  boundary: Boundary
  /** The time step $\Delta t > 0$. */
  dt: number
  /** Default `'crank-nicolson'`. */
  scheme?: TimeScheme
}

/**
 * The heat (diffusion) equation $u_t = D u_{xx}$ by the method of lines: the second difference
 * $D(u_{i-1} - 2u_i + u_{i+1})/\Delta x^2$ in space, then explicit Euler (FTCS; stable only when
 * $r = D\Delta t/\Delta x^2 \le 1/2$), implicit Euler or Crank–Nicolson (both unconditionally stable; Crank–Nicolson
 * is second order in time but lets high-frequency error oscillate for large $r$). Zero-flux (Neumann) ends use a
 * mirrored ghost point, so the mass is conserved. On a periodic grid all $n$ points are distinct, so the period is
 * $n \Delta x$. `init` takes `{ u0 }`, the initial profile. Throws `DomainError` unless $D$ and $\Delta t$ are
 * positive.
 *
 * @param options The diffusivity, grid, boundary, time step, scheme and end time.
 * @returns The solver as an `Algorithm`, one time step per step.
 *
 * @example A rod with cold ends: the sine mode decays exponentially
 * const grid = { a: 0, b: 1, n: 21 }
 * const heat = heatEquation({ diffusivity: 1, grid, boundary: { kind: 'dirichlet' }, dt: 0.01 })
 * const u0 = (x) => Math.sin(Math.PI * x)
 * for (const steps of [0, 10, 20]) {
 *   const s = run(heat, { u0 }, steps)
 *   print('t =', s.time, ' middle =', s.u.data[10], ' exact =', Math.exp(-(Math.PI ** 2) * s.time))
 * }
 *
 * @example The explicit scheme beyond its limit
 * const grid = { a: 0, b: 1, n: 21 }
 * const ftcs = heatEquation({ diffusivity: 1, grid, boundary: { kind: 'dirichlet' }, dt: 0.002, scheme: 'explicit' })
 * const s = run(ftcs, { u0: (x) => Math.sin(Math.PI * x) + 0.01 * Math.sin(19 * Math.PI * x) }, 60)
 * print(s.stability)
 * print('largest |u| after 60 steps =', Math.max(...s.u.data.map(Math.abs)))
 */
export function heatEquation(options: HeatOptions): Algorithm<{ u0: Profile }, PdeState> {
  const { diffusivity: D, grid, boundary, dt, scheme = 'crank-nicolson', tEnd } = options
  if (!(D > 0) || !(dt > 0))
    throw new DomainError('heatEquation', 'heatEquation: the diffusivity and dt must be positive')
  const n = grid.n
  const dx = (grid.b - grid.a) / (n - 1)
  const k = D / (dx * dx)
  const lo = new Float64Array(n).fill(k)
  const di = new Float64Array(n).fill(-2 * k)
  const up = new Float64Array(n).fill(k)
  if (boundary.kind === 'dirichlet') {
    for (const i of [0, n - 1]) lo[i] = di[i] = up[i] = 0
  } else if (boundary.kind === 'neumann') {
    // Ghost point u_{−1} = u_1 (and u_n = u_{n−2}), zero gradient at the ends.
    up[0] = 2 * k
    lo[n - 1] = 2 * k
    lo[0] = up[n - 1] = 0
  }
  const r = (D * dt) / (dx * dx)
  const limit = scheme === 'explicit' ? 0.5 : Infinity
  const stability = { number: 'r = DΔt/Δx²', value: r, limit, stable: r <= limit }
  const L = { lo, di, up, periodic: boundary.kind === 'periodic' }
  const stepper = thetaMethod(`heat-${scheme}`, L, scheme, dt, dx, stability, tEnd)
  return {
    name: stepper.name,
    init: ({ u0 }) => {
      const u = sample(u0, grid, 'heatEquation')
      if (boundary.kind === 'dirichlet') {
        u[0] = boundary.left ?? u[0]
        u[n - 1] = boundary.right ?? u[n - 1]
      }
      return stepper.init(u)
    },
    step: stepper.step,
    done: stepper.done,
  }
}

/** Options for `transportEquation`. */
export type TransportOptions = Common & {
  /** The constant velocity $c$ in $u_t + c u_x = 0$. */
  velocity: number
  /** The grid; when periodic, its last point repeats the first. */
  grid: Grid1
  /**
   * Periodic (default), or an inflow boundary holding the upstream end at its initial value (the downstream end is
   * extrapolated with zero gradient).
   */
  boundary?: 'periodic' | 'inflow'
  /** The time step $\Delta t$. */
  dt: number
  /** Default `'upwind'`. */
  scheme?: 'upwind' | 'lax-wendroff' | 'lax-friedrichs'
}

/**
 * The transport (advection) equation $u_t + c u_x = 0$ with Courant number $\nu = c\Delta t/\Delta x$: first-order
 * upwind ($u_i - \nu(u_i - u_{i-1})$ for $c > 0$; monotone, diffusive), Lax–Friedrichs
 * ($\tfrac{1}{2}(u_{i-1} + u_{i+1}) - \tfrac{1}{2}\nu(u_{i+1} - u_{i-1})$; more diffusive) and Lax–Wendroff
 * (second order; dispersive, with oscillations behind steep fronts). All are stable exactly when
 * $\lvert \nu \rvert \le 1$ (the CFL condition: the numerical domain of dependence must contain the true one).
 * `init` takes `{ u0 }`, the initial profile.
 *
 * @param options The velocity, grid, boundary, time step, scheme and end time.
 * @returns The solver as an `Algorithm`, one time step per step.
 *
 * @example A pulse once round a periodic domain: upwind smears it, Lax–Wendroff keeps its height
 * const u0 = (x) => Math.exp(-((x - 0.5) ** 2) / 0.005)
 * for (const scheme of ['upwind', 'lax-wendroff']) {
 *   const pde = transportEquation({ velocity: 1, grid: { a: 0, b: 1, n: 41 }, dt: 0.0125, scheme })
 *   const s = run(pde, { u0 }, 80)
 *   print(scheme, ' t =', s.time, ' peak =', Math.max(...s.u.data), ' mass =', s.mass)
 * }
 */
export function transportEquation(options: TransportOptions): Algorithm<{ u0: Profile }, PdeState> {
  const { velocity: c, grid, boundary = 'periodic', dt, scheme = 'upwind', tEnd } = options
  const n = grid.n
  const dx = (grid.b - grid.a) / (n - 1)
  const nu = (c * dt) / dx
  const stability = { number: 'ν = |c|Δt/Δx', value: Math.abs(nu), limit: 1, stable: Math.abs(nu) <= 1 }
  const periodic = boundary === 'periodic'
  // On a periodic grid the last point repeats the first, so the unknowns are points 0…n−2.
  const m = periodic ? n - 1 : n
  const state = (time: number, u: F64, t: number): PdeState => ({
    t,
    time,
    u: fromData(u, [n]),
    dt,
    mass: trapezoidMass(u, dx, false),
    stability,
    diverged: !finite(u),
    failure: finite(u) ? null : 'not finite',
  })
  return {
    name: `transport-${scheme}`,
    init: ({ u0 }) => {
      const u = sample(u0, grid, 'transportEquation')
      if (periodic) u[n - 1] = u[0]
      return state(0, u, 0)
    },
    step: (s) => {
      const u = s.u.data as F64
      const next = new Float64Array(n)
      const at = (i: number) => (periodic ? u[((i % m) + m) % m] : u[Math.min(n - 1, Math.max(0, i))])
      for (let i = 0; i < m; i++) {
        const [l, c0, r] = [at(i - 1), u[i], at(i + 1)]
        if (scheme === 'upwind') next[i] = nu >= 0 ? c0 - nu * (c0 - l) : c0 - nu * (r - c0)
        else if (scheme === 'lax-friedrichs') next[i] = 0.5 * (l + r) - 0.5 * nu * (r - l)
        else next[i] = c0 - 0.5 * nu * (r - l) + 0.5 * nu * nu * (r - 2 * c0 + l)
      }
      if (periodic) next[n - 1] = next[0]
      else if (nu >= 0) next[0] = u[0]
      else next[n - 1] = u[n - 1]
      return state(s.time + dt, next, s.t + 1)
    },
    done: (s) => tEnd !== undefined && s.time >= tEnd - 1e-12 * Math.max(1, Math.abs(tEnd)),
  }
}

/** Options for `waveEquation`. */
export type WaveOptions = Common & {
  /** The wave speed $c$ in $u_{tt} = c^2 u_{xx}$. */
  speed: number
  /** The grid; when periodic, its last point repeats the first. */
  grid: Grid1
  /** Fixed ends ($u = 0$, the default) or periodic. */
  boundary?: 'fixed' | 'periodic'
  /** The time step $\Delta t$. */
  dt: number
}

/** The state of `waveEquation`: the solution now and one step earlier, and the discrete energy. */
export type WaveState = PdeState & {
  /** The solution one time step earlier (at time 0, the virtual level implied by the Taylor start). */
  previous: Vector
  /**
   * $\tfrac{1}{2}\sum_i ((u_i - u_i^{\text{prev}})/\Delta t)^2 \Delta x
   * + \tfrac{1}{2}c^2\sum_i ((u_{i+1} - u_i)/\Delta x)^2 \Delta x$: nearly conserved when stable.
   */
  energy: number
}

/**
 * The wave equation $u_{tt} = c^2 u_{xx}$ by the leapfrog scheme
 * $u_i^{k+1} = 2u_i^k - u_i^{k-1} + \nu^2(u_{i+1}^k - 2u_i^k + u_{i-1}^k)$, $\nu = c\Delta t/\Delta x$, started
 * with the Taylor step $u^1 = u^0 + \Delta t \, v^0 + \tfrac{1}{2}\nu^2\delta^2 u^0$. Second order,
 * non-dissipative, and stable exactly when $\lvert \nu \rvert \le 1$; at $\nu = 1$ it is exact on the grid. `init`
 * takes `{ u0, v0 }` (displacement and velocity, the velocity 0 by default); fixed ends are set to 0.
 *
 * @param options The speed, grid, boundary, time step and end time.
 * @returns The solver as an `Algorithm`, one time step per step, whose state also carries the previous level and the
 *   discrete energy.
 *
 * @example A standing wave at Courant number 1, exact on the grid
 * const wave = waveEquation({ speed: 1, grid: { a: 0, b: 1, n: 21 }, dt: 0.05 })
 * const u0 = (x) => Math.sin(Math.PI * x)
 * for (const steps of [0, 10, 20]) {
 *   const s = run(wave, { u0 }, steps)
 *   print('t =', s.time, ' middle =', s.u.data[10], ' exact =', Math.cos(Math.PI * s.time), ' energy =', s.energy)
 * }
 */
export function waveEquation(options: WaveOptions): Algorithm<{ u0: Profile; v0?: Profile }, WaveState> {
  const { speed: c, grid, boundary = 'fixed', dt, tEnd } = options
  const n = grid.n
  const dx = (grid.b - grid.a) / (n - 1)
  const nu = (c * dt) / dx
  const stability = { number: 'ν = cΔt/Δx', value: Math.abs(nu), limit: 1, stable: Math.abs(nu) <= 1 }
  const periodic = boundary === 'periodic'
  const m = periodic ? n - 1 : n
  const lap = (u: F64, i: number) => {
    const l = periodic ? u[(i - 1 + m) % m] : u[i - 1]
    const r = periodic ? u[(i + 1) % m] : u[i + 1]
    return r - 2 * u[i] + l
  }
  const energy = (u: F64, prev: F64) => {
    let e = 0
    for (let i = 0; i < m; i++) e += 0.5 * ((u[i] - prev[i]) / dt) ** 2 * dx
    for (let i = 0; i + 1 < n; i++) e += 0.5 * c * c * ((u[i + 1] - u[i]) / dx) ** 2 * dx
    return e
  }
  const make = (time: number, u: F64, prev: F64, t: number): WaveState => ({
    t,
    time,
    u: fromData(u, [n]),
    previous: fromData(prev, [n]),
    dt,
    mass: trapezoidMass(u, dx, false),
    energy: energy(u, prev),
    stability,
    diverged: !finite(u),
    failure: finite(u) ? null : 'not finite',
  })
  const advance = (u: F64, prev: F64 | null, v: F64 | null): F64 => {
    const next = new Float64Array(n)
    const lo = periodic ? 0 : 1
    const hi = periodic ? m : n - 1
    for (let i = lo; i < hi; i++) {
      next[i] = prev ? 2 * u[i] - prev[i] + nu * nu * lap(u, i) : u[i] + dt * v![i] + 0.5 * nu * nu * lap(u, i)
    }
    if (periodic) next[n - 1] = next[0]
    return next
  }
  return {
    name: 'wave-leapfrog',
    init: ({ u0, v0 }) => {
      const u = sample(u0, grid, 'waveEquation')
      if (!periodic) u[0] = u[n - 1] = 0
      else u[n - 1] = u[0]
      const v = v0 === undefined ? new Float64Array(n) : sample(v0, grid, 'waveEquation')
      // Store the virtual previous level u^{−1} = u¹ − 2Δt v⁰ implied by the Taylor start, so that step() is uniform
      // and the reported energy is meaningful at t = 0.
      const u1 = advance(u, null, v)
      const prev = Float64Array.from(u1, (x, i) => x - 2 * dt * v[i])
      if (!periodic) prev[0] = prev[n - 1] = 0
      return make(0, u, prev, 0)
    },
    step: (s) => {
      const u = s.u.data as F64
      const next = advance(u, s.previous.data as F64, null)
      return make(s.time + dt, next, Float64Array.from(u), s.t + 1)
    },
    done: (s) => tEnd !== undefined && s.time >= tEnd - 1e-12 * Math.max(1, Math.abs(tEnd)),
  }
}

/** Options for `fokkerPlanck`. */
export type FokkerPlanckOptions = Common & {
  /** The drift $\mu(x)$ of $dX = \mu(X) \, dt + \sigma(X) \, dW$, evaluated midway between grid points. */
  drift: (x: number) => number
  /** The diffusion coefficient $D(x) = \sigma(x)^2/2$, evaluated at the grid points; must be non-negative. */
  diffusion: (x: number) => number
  /** The grid; its ends are reflecting walls. */
  grid: Grid1
  /** The time step $\Delta t$. */
  dt: number
  /** Default `'implicit'`. */
  scheme?: TimeScheme
}

/**
 * The Fokker–Planck (forward Kolmogorov) equation $p_t = -(\mu p)_x + (D p)_{xx}$ for the density of
 * $dX = \mu \, dt + \sigma \, dW$ with $D = \sigma^2/2$, in conservative flux form
 * $p_i' = -(J_{i+1/2} - J_{i-1/2})/\Delta x$ with the exponentially fitted flux of Scharfetter and Gummel (1969):
 * $J = (D/\Delta x)(B(-w) p_i - B(w) p_{i+1})$, $w = (\mu - D')\Delta x/D$, $B(z) = z/(e^z - 1)$, which keeps
 * densities non-negative, is exact for the stationary density when the coefficients are constant between grid points,
 * and reduces to upwinding as $D \to 0$. The flux is zero through both ends (reflecting walls), so the mass
 * $\sum_i p_i \Delta x$ is conserved exactly. The stability number reported is the largest over the cell interfaces
 * of $\Delta t (2 \max D/\Delta x^2 + \lvert \mu - D' \rvert/\Delta x)$, with limit 1 for the explicit scheme.
 * `init` takes `{ u0 }`, the initial density. Throws `DomainError` for a negative diffusion.
 *
 * @param options The drift, diffusion, grid, time step, scheme (default implicit) and end time.
 * @returns The solver as an `Algorithm`, one time step per step.
 *
 * @example An Ornstein–Uhlenbeck density relaxes to its stationary variance 1/2
 * // dX = -X dt + dW: drift -x, diffusion 1/2; stationary density N(0, 1/2).
 * const grid = { a: -4, b: 4, n: 81 }
 * const fp = fokkerPlanck({ drift: (x) => -x, diffusion: () => 0.5, grid, dt: 0.05 })
 * const u0 = (x) => Math.exp(-((x - 1) ** 2) / 0.08) / Math.sqrt(0.08 * Math.PI)
 * const xs = gridPoints(grid).data
 * for (const steps of [0, 20, 100]) {
 *   const s = run(fp, { u0 }, steps)
 *   const mean = s.u.data.reduce((acc, p, i) => acc + p * xs[i] * 0.1, 0)
 *   const variance = s.u.data.reduce((acc, p, i) => acc + p * (xs[i] - mean) ** 2 * 0.1, 0)
 *   print('t =', s.time, ' mass =', s.mass, ' mean =', mean, ' variance =', variance)
 * }
 */
export function fokkerPlanck(options: FokkerPlanckOptions): Algorithm<{ u0: Profile }, PdeState> {
  const { drift, diffusion, grid, dt, scheme = 'implicit', tEnd } = options
  const n = grid.n
  const dx = (grid.b - grid.a) / (n - 1)
  const xs = toFlat(gridPoints(grid))
  const D = xs.map(diffusion)
  if (D.some((d) => !(d >= 0)))
    throw new DomainError('fokkerPlanck', 'fokkerPlanck: the diffusion must be non-negative')
  // Each interior cell interface i+½ (between points i and i+1) carries a flux J = a·p_i + b·p_{i+1}.
  const lo = new Float64Array(n)
  const di = new Float64Array(n)
  const up = new Float64Array(n)
  let maxRate = 0
  for (let i = 0; i + 1 < n; i++) {
    const mu = drift(0.5 * (xs[i] + xs[i + 1]))
    // Write −(μp)_x + (Dp)_xx = −((μ − D′)p − D p_x)_x and use the exponentially fitted (Scharfetter–Gummel) flux
    // J = (D/Δx)(B(−w) p_i − B(w) p_{i+1}) with w = (μ − D′)Δx/D and B(z) = z/(eᶻ − 1): it is exact for the
    // stationary density of constant coefficients, keeps the density positive, and reduces to upwinding as D → 0.
    const Dh = 0.5 * (D[i] + D[i + 1])
    const v = mu - (D[i + 1] - D[i]) / dx
    let a: number
    let b: number
    if (Dh > 0) {
      const w = (v * dx) / Dh
      const B = (z: number) => (Math.abs(z) < 1e-8 ? 1 - z / 2 : z / Math.expm1(z))
      a = (Dh / dx) * B(-w)
      b = -(Dh / dx) * B(w)
    } else {
      a = Math.max(v, 0)
      b = Math.min(v, 0)
    }
    // p_i loses J/Δx, p_{i+1} gains J/Δx.
    di[i] -= a / dx
    up[i] -= b / dx
    lo[i + 1] += a / dx
    di[i + 1] += b / dx
    maxRate = Math.max(maxRate, (2 * Math.max(D[i], D[i + 1])) / (dx * dx) + Math.abs(v) / dx)
  }
  const value = dt * maxRate
  const limit = scheme === 'explicit' ? 1 : Infinity
  const stability = { number: 'Δt(2D/Δx² + |μ|/Δx)', value, limit, stable: value <= limit }
  const L = { lo, di, up, periodic: false }
  const stepper = thetaMethod(`fokker-planck-${scheme}`, L, scheme, dt, dx, stability, tEnd)
  // Mass for densities is Σ p_i Δx (the control-volume sum the scheme conserves), not the trapezoid rule.
  const withMass = (s: PdeState): PdeState => ({ ...s, mass: (s.u.data as F64).reduce((acc, v) => acc + v, 0) * dx })
  return {
    name: stepper.name,
    init: ({ u0 }) => withMass(stepper.init(sample(u0, grid, 'fokkerPlanck'))),
    step: (s) => withMass(stepper.step(s)),
    done: stepper.done,
  }
}

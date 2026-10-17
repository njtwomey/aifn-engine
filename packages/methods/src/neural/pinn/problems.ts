/**
 * Initial-boundary value problems in one space dimension for a physics-informed network to solve, each with its exact
 * solution: viscous Burgers' equation, whose solution steepens into a near-shock (the benchmark of Raissi, Perdikaris &
 * Karniadakis, 2019, §3.1), and the heat equation; and the problems of `pinnVariantRun`'s notes: Poisson's equation
 * with an imbalanced residual (`poissonProblem`), a steep interior layer (`layerProblem`) and convection at a speed
 * that defeats a plain network (`convectionProblem`).
 *
 * Every problem is a `FieldProblem`: steady or time-dependent, with the derivatives its residual reads (`needs`), its
 * coefficients by name (which `pinnVariantRun` may fit or ramp, and passes to the residual as values), and, for exact
 * conditions, a `lift` that meets them and a `distance` that vanishes on them (Lagaris et al., 1998; Sukumar and
 * Srivastava, 2022). `burgers` and `heat` are also `PinnProblem`s, the time-dependent problems `pinnRun` takes, whose
 * residual reads the derivatives alone; `heatProblem` is `heat`.
 *
 * A problem is a residual $\mathcal{N}[u]$ written in $u$ and its derivatives, so that the network is asked for
 * $\mathcal{N}[u] = 0$ inside the domain, an initial profile $u(x, t_0)$ and Dirichlet values at the two ends. Burgers'
 * solution is computed by the Cole–Hopf transform (Hopf, 1950; Cole, 1951), which turns the equation into the heat
 * equation and the solution into a ratio of two Gaussian integrals, evaluated here by the trapezoid rule.
 */

import { add, mul, sin, slice, square, sub, tanh, type Value } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A solution and its derivatives at the collocation points, each $[n, 1]$. A derivative is computed when a residual
 * first reads it, so a residual that never reads $u_x$ does not pay for it.
 */
export type PinnDerivatives = {
  /** $u$. */
  readonly u: Value
  /** $\partial u / \partial t$. */
  readonly ut: Value
  /** $\partial u / \partial x$. */
  readonly ux: Value
  /** $\partial^2 u / \partial x^2$. */
  readonly uxx: Value
}

/** An initial-boundary value problem for $u(x, t)$ on $[a, b] \times [t_0, t_1]$ with Dirichlet ends. */
export type PinnProblem = {
  /** A short name, for captions. */
  readonly name: string
  /** The equation, for captions: TeX. */
  readonly equation: string
  /** The space interval $[a, b]$. */
  readonly x: readonly [number, number]
  /** The time interval $[t_0, t_1]$. */
  readonly t: readonly [number, number]
  /** The residual $\mathcal{N}[u]$, zero where $u$ solves the equation, from $u$ and its derivatives. */
  readonly residual: (d: PinnDerivatives) => Value
  /** The initial profile $u(x, t_0)$. */
  readonly initial: (x: number) => number
  /** The values $u(a, t)$ and $u(b, t)$ at the two ends. */
  readonly boundary: (t: number) => readonly [number, number]
  /** The exact solution, when known, for measuring the error. */
  readonly solution?: (x: number, t: number) => number
}

/**
 * A solution and its derivatives at the points, each $[n, 1]$, with the points' coordinates: the input of a
 * `FieldProblem`'s residual. Derivatives a problem does not need are 0, as is $u_t$ for a steady problem.
 */
export type FieldDerivatives = PinnDerivatives & {
  /** The positions $x$, $[n, 1]$. */
  readonly x: Value
  /** The times $t$, $[n, 1]$, or null for a steady problem. */
  readonly t: Value | null
}

/** A boundary-value problem for $u(x)$, or an initial-boundary value problem for $u(x, t)$, with Dirichlet ends. */
export type FieldProblem = {
  /** A short name, for captions. */
  readonly name: string
  /** The space interval $[a, b]$. */
  readonly x: readonly [number, number]
  /** The time interval $[t_0, t_1]$; absent for a steady problem. */
  readonly t?: readonly [number, number]
  /** The derivatives the residual reads (each costs nested forward passes). */
  readonly needs: readonly ('ux' | 'uxx' | 'ut')[]
  /** The coefficients of the equation at their true values, by name. */
  readonly coefficients: Readonly<Record<string, number>>
  /** The residual $\mathcal{N}[u]$, zero where $u$ solves the equation, given the coefficients as values. */
  readonly residual: (d: FieldDerivatives, c: Readonly<Record<string, Value>>) => Value
  /** The initial profile $u(x, t_0)$ (time-dependent problems). */
  readonly initial?: (x: number) => number
  /** The values $u(a, t)$ and $u(b, t)$ ($t$ is ignored by steady problems). */
  readonly boundary: (t: number) => readonly [number, number]
  /** For hard conditions: a function of the points $\Xmat$ that meets every condition, in tensor primitives. */
  readonly lift?: (X: Value) => Value
  /** For hard conditions: a function of $\Xmat$ that is zero on the boundary and initial line, positive inside. */
  readonly distance?: (X: Value) => Value
  /** The exact solution, when known. */
  readonly solution?: (x: number, t: number) => number
}

/**
 * Column `k` of a matrix of points, $[n, 1]$.
 *
 * @param X The points, one per row.
 * @param k The coordinate: 0 for $x$, 1 for $t$.
 * @returns The column.
 */
function column(X: Value, k: number): Value {
  return slice(X, null, [k, k + 1])
}

/**
 * The solution of viscous Burgers' equation $u_t + u u_x = \nu u_{xx}$ on the line from $u(x, 0) = -\sin \pi x$, by the
 * Cole–Hopf transform: $u = -2\nu \phi_x / \phi$ with $\phi$ solving the heat equation, which gives
 *
 * $$ u(x, t) = -\frac{\int \sin \pi(x - \eta) \, f(x - \eta) \, e^{-\eta^2 / 4\nu t} \, d\eta}{\int f(x - \eta) \, e^{-\eta^2 / 4\nu t} \, d\eta}, \qquad f(y) = e^{-\cos(\pi y) / 2\pi\nu}. $$
 *
 * The integrals are taken by the trapezoid rule over the interval where the integrand is not negligible, with the
 * largest exponent factored out so that small $\nu$ does not overflow. The solution is odd in $x$ and 2-periodic, so it
 * is zero at $x = \pm 1$ for all $t$. Throws `DomainError` for $\nu \le 0$ or $t < 0$.
 *
 * @param x The position.
 * @param t The time, at least 0.
 * @param nu The viscosity $\nu$. Raissi et al. take $0.01/\pi$, which steepens the profile into a near-shock at
 *   $x = 0$ by $t \approx 0.4$.
 * @param nodes The number of trapezoid nodes.
 * @returns $u(x, t)$.
 *
 * @example The profile steepens at the origin and stays zero at the ends
 * for (const t of [0, 0.25, 0.5, 1]) {
 *   const row = [-1, -0.5, -0.05, 0, 0.05, 0.5, 1].map((x) => burgersSolution(x, t).toFixed(3))
 *   print(`t = ${t}:`, row.join('  '))
 * }
 */
export function burgersSolution(x: number, t: number, nu = 0.01 / Math.PI, nodes = 1201): number {
  if (!(nu > 0)) throw new DomainError('burgersSolution', 'burgersSolution: the viscosity must be positive')
  if (!(t >= 0)) throw new DomainError('burgersSolution', 'burgersSolution: the time must be at least 0')
  if (t === 0) return -Math.sin(Math.PI * x)
  const k = 1 / (2 * Math.PI * nu)
  // Where e^{−η²/4νt} has fallen e^{−40} below the largest swing of f's exponent (2k), the integrand is negligible.
  const half = Math.sqrt(4 * nu * t * (2 * k + 40))
  const h = (2 * half) / (nodes - 1)
  const exponent = (eta: number) => -k * Math.cos(Math.PI * (x - eta)) - (eta * eta) / (4 * nu * t)
  let top = -Infinity
  for (let i = 0; i < nodes; i++) top = Math.max(top, exponent(-half + i * h))
  let num = 0
  let den = 0
  for (let i = 0; i < nodes; i++) {
    const eta = -half + i * h
    const w = Math.exp(exponent(eta) - top)
    num += Math.sin(Math.PI * (x - eta)) * w
    den += w
  }
  return -num / den
}

/**
 * Viscous Burgers' equation $u_t + u u_x - \nu u_{xx} = 0$ on $[-1, 1] \times [0, 1]$ from $u(x, 0) = -\sin \pi x$
 * with $u(\pm 1, t) = 0$ (Raissi et al., 2019, §3.1). The nonlinear term steepens the profile into a near-shock at
 * $x = 0$, which a network has to resolve; its exact solution is `burgersSolution`.
 *
 * As a `FieldProblem` its coefficient is `nu`, and its exact conditions come from the lift $-\sin \pi x$ and the
 * distance $t(1 - x^2)$.
 *
 * @param nu The viscosity $\nu$; smaller makes the shock steeper and harder to fit.
 * @returns The problem.
 *
 * @example The domain, a value of the exact solution and the ends
 * const p = burgers()
 * print(p.name, ' on x in', p.x, ' t in', p.t)
 * print('u(0.5, 0.5) =', p.solution(0.5, 0.5), ' ends at t = 0.5:', p.boundary(0.5))
 */
export function burgers(nu = 0.01 / Math.PI): PinnProblem & FieldProblem {
  if (!(nu > 0)) throw new DomainError('burgers', 'burgers: the viscosity must be positive')
  return {
    name: "Burgers' equation",
    equation: 'u_t + u u_x = \\nu u_{xx}',
    x: [-1, 1],
    t: [0, 1],
    needs: ['ux', 'uxx', 'ut'],
    coefficients: { nu },
    residual: (d: PinnDerivatives, c?: Readonly<Record<string, Value>>) =>
      sub(add(d.ut, mul(d.u, d.ux)), mul(c?.nu ?? nu, d.uxx)),
    initial: (x) => -Math.sin(Math.PI * x),
    boundary: () => [0, 0],
    lift: (X) => mul(-1, sin(mul(column(X, 0), Math.PI))),
    distance: (X) => mul(column(X, 1), sub(1, square(column(X, 0)))),
    solution: (x, t) => burgersSolution(x, t, nu),
  }
}

/**
 * The heat equation $u_t - \kappa u_{xx} = 0$ on $[-1, 1] \times [0, 1]$ from $u(x, 0) = \sin \pi x$ with
 * $u(\pm 1, t) = 0$, whose solution $u = e^{-\kappa\pi^2 t} \sin \pi x$ decays without changing shape: the gentle
 * problem to start from. As a `FieldProblem` its coefficient is `kappa` (which `pinnVariantRun` can fit from
 * observations), and its exact conditions come from the lift $\sin \pi x$ and the distance $s\,t(1 - x^2)$, which
 * peaks at $s$.
 *
 * @param kappa The diffusivity $\kappa$.
 * @param distanceScale The peak $s$ of the distance function (hard conditions only).
 * @returns The problem.
 *
 * @example The solution has decayed by $e^{-\kappa\pi^2}$ at $t = 1$
 * const p = heat(0.1)
 * print('u(0.5, 0) =', p.solution(0.5, 0), ' u(0.5, 1) =', p.solution(0.5, 1))
 *
 * @example The lift meets the conditions and the distance vanishes on them
 * const p = heat(0.1)
 * const X = tensor([[-1, 0.3], [0.5, 0], [1, 0.7]])
 * print('lift:', p.lift(X), ' distance:', p.distance(X))
 */
export function heat(kappa = 0.1, distanceScale = 1): PinnProblem & FieldProblem {
  if (!(kappa > 0)) throw new DomainError('heat', 'heat: the diffusivity must be positive')
  return {
    name: 'Heat equation',
    equation: 'u_t = \\kappa u_{xx}',
    x: [-1, 1],
    t: [0, 1],
    needs: ['uxx', 'ut'],
    coefficients: { kappa },
    residual: (d: PinnDerivatives, c?: Readonly<Record<string, Value>>) => sub(d.ut, mul(c?.kappa ?? kappa, d.uxx)),
    initial: (x) => Math.sin(Math.PI * x),
    boundary: () => [0, 0],
    lift: (X) => sin(mul(column(X, 0), Math.PI)),
    distance: (X) => mul(distanceScale, mul(column(X, 1), sub(1, square(column(X, 0))))),
    solution: (x, t) => Math.exp(-kappa * Math.PI * Math.PI * t) * Math.sin(Math.PI * x),
  }
}

/**
 * The heat equation as `pinnVariantRun`'s notes name it: the same function as `heat`, so the problem is defined once
 * (on $[-1, 1] \times [0, 1]$ from $\sin \pi x$, coefficient `kappa`, lift $\sin \pi x$, distance $s\,t(1 - x^2)$).
 *
 * @example The same problem under both names
 * print('same function:', heatProblem === heat)
 * print('coefficients:', heatProblem(0.2).coefficients)
 */
export const heatProblem: typeof heat = heat

/**
 * Poisson's equation $-u'' = f$ on $[0, 1]$ with exact solution $u = \sin(k\pi x) + x$ for an integer $k$, so
 * $f = (k\pi)^2 \sin k\pi x$, $u(0) = 0$ and $u(1) = 1$. The residual is of size $(k\pi)^2$ while the boundary
 * misfit is of size 1: the imbalance that loss weights must correct. Lift $x$, distance $s\,x(1 - x)$ with $s = 4$ by
 * default, so that the distance peaks at 1.
 *
 * @param k The wavenumber $k$ of the solution.
 * @param distanceScale The scale $s$ of the distance function.
 * @returns The (steady) problem.
 *
 * @example The residual's scale against the boundary values
 * const p = poissonProblem(4)
 * print('(k pi)^2 =', (4 * Math.PI) ** 2, ' boundary:', p.boundary(0), ' u(0.125) =', p.solution(0.125, 0))
 */
export function poissonProblem(k = 4, distanceScale = 4): FieldProblem {
  const w = k * Math.PI
  return {
    name: 'poisson',
    x: [0, 1],
    needs: ['uxx'],
    coefficients: {},
    residual: (d) => add(d.uxx, mul(w * w, sin(mul(d.x, w)))),
    boundary: () => [0, 1],
    lift: (X) => column(X, 0),
    distance: (X) => mul(distanceScale, mul(column(X, 0), sub(1, column(X, 0)))),
    solution: (x) => Math.sin(w * x) + x,
  }
}

/**
 * $-u'' = f$ on $[0, 1]$ with exact solution $u = \tanh((x - 1/2)/\varepsilon)$, a layer of width about
 * $\varepsilon$ at the centre: $f = -u'' = 2u(1 - u^2)/\varepsilon^2$. Uniform collocation points rarely fall in the
 * layer. Lift the straight line between the end values, distance $4x(1 - x)$.
 *
 * @param epsilon The layer's width $\varepsilon$.
 * @returns The (steady) problem.
 *
 * @example The layer is nearly a step at epsilon = 0.02
 * const p = layerProblem(0.02)
 * print('u at 0.45, 0.5, 0.55:', [0.45, 0.5, 0.55].map((x) => p.solution(x, 0)))
 */
export function layerProblem(epsilon = 0.02): FieldProblem {
  const e = epsilon
  const exact = (x: number) => Math.tanh((x - 0.5) / e)
  return {
    name: 'layer',
    x: [0, 1],
    needs: ['uxx'],
    coefficients: {},
    residual: (d) => {
      const z = tanh(mul(sub(d.x, 0.5), 1 / e))
      return add(d.uxx, mul(2 / (e * e), mul(z, sub(1, square(z)))))
    },
    boundary: () => [exact(0), exact(1)],
    lift: (X) => add(exact(0), mul(column(X, 0), exact(1) - exact(0))),
    distance: (X) => mul(4, mul(column(X, 0), sub(1, column(X, 0)))),
    solution: (x) => exact(x),
  }
}

/**
 * Convection $u_t + \beta u_x = 0$ on $[0, 2\pi] \times [0, 1]$ from $u(x, 0) = \sin x$, with the exact values
 * $\sin(-\beta t)$ and $\sin(2\pi - \beta t)$ at the ends; exact $u = \sin(x - \beta t)$ (Krishnapriyan et al.,
 * 2021, §3.1, with Dirichlet ends in place of the periodic condition). A plain network fails above $\beta \approx 10$.
 *
 * @param beta The speed $\beta$, the problem's coefficient (which a curriculum may ramp).
 * @returns The problem.
 *
 * @example The exact solution travels at speed beta
 * const p = convectionProblem(10)
 * print('u(1, 0) =', p.solution(1, 0), ' u(1 + 10 t, t) at t = 0.3:', p.solution(4, 0.3))
 */
export function convectionProblem(beta = 10): FieldProblem {
  return {
    name: 'convection',
    x: [0, 2 * Math.PI],
    t: [0, 1],
    needs: ['ux', 'ut'],
    coefficients: { beta },
    residual: (d, c) => add(d.ut, mul(c.beta, d.ux)),
    initial: (x) => Math.sin(x),
    boundary: (t) => [Math.sin(-beta * t), Math.sin(2 * Math.PI - beta * t)],
    solution: (x, t) => Math.sin(x - beta * t),
  }
}

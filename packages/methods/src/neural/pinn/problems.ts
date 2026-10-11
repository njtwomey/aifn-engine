/**
 * Initial-boundary value problems in one space dimension for a physics-informed network to solve, each with its exact
 * solution: viscous Burgers' equation, whose solution steepens into a near-shock (the benchmark of Raissi, Perdikaris &
 * Karniadakis, 2019, §3.1), and the heat equation.
 *
 * A problem is a residual $\mathcal{N}[u]$ written in $u$ and its derivatives, so that the network is asked for
 * $\mathcal{N}[u] = 0$ inside the domain, an initial profile $u(x, t_0)$ and Dirichlet values at the two ends. Burgers'
 * solution is computed by the Cole–Hopf transform (Hopf, 1950; Cole, 1951), which turns the equation into the heat
 * equation and the solution into a ratio of two Gaussian integrals, evaluated here by the trapezoid rule.
 */

import { add, mul, sub, type Value } from 'aifn-compute/foundation/tensor'
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
 * @param nu The viscosity $\nu$; smaller makes the shock steeper and harder to fit.
 * @returns The problem.
 *
 * @example The domain, a value of the exact solution and the ends
 * const p = burgers()
 * print(p.name, ' on x in', p.x, ' t in', p.t)
 * print('u(0.5, 0.5) =', p.solution(0.5, 0.5), ' ends at t = 0.5:', p.boundary(0.5))
 */
export function burgers(nu = 0.01 / Math.PI): PinnProblem {
  if (!(nu > 0)) throw new DomainError('burgers', 'burgers: the viscosity must be positive')
  return {
    name: "Burgers' equation",
    equation: 'u_t + u u_x = \\nu u_{xx}',
    x: [-1, 1],
    t: [0, 1],
    residual: (d) => sub(add(d.ut, mul(d.u, d.ux)), mul(nu, d.uxx)),
    initial: (x) => -Math.sin(Math.PI * x),
    boundary: () => [0, 0],
    solution: (x, t) => burgersSolution(x, t, nu),
  }
}

/**
 * The heat equation $u_t - \kappa u_{xx} = 0$ on $[-1, 1] \times [0, 1]$ from $u(x, 0) = \sin \pi x$ with
 * $u(\pm 1, t) = 0$, whose solution $u = e^{-\kappa\pi^2 t} \sin \pi x$ decays without changing shape: the gentle
 * problem to start from.
 *
 * @param kappa The diffusivity $\kappa$.
 * @returns The problem.
 *
 * @example The solution has decayed by $e^{-\kappa\pi^2}$ at $t = 1$
 * const p = heat(0.1)
 * print('u(0.5, 0) =', p.solution(0.5, 0), ' u(0.5, 1) =', p.solution(0.5, 1))
 */
export function heat(kappa = 0.1): PinnProblem {
  if (!(kappa > 0)) throw new DomainError('heat', 'heat: the diffusivity must be positive')
  return {
    name: 'Heat equation',
    equation: 'u_t = \\kappa u_{xx}',
    x: [-1, 1],
    t: [0, 1],
    residual: (d) => sub(d.ut, mul(kappa, d.uxx)),
    initial: (x) => Math.sin(Math.PI * x),
    boundary: () => [0, 0],
    solution: (x, t) => Math.exp(-kappa * Math.PI * Math.PI * t) * Math.sin(Math.PI * x),
  }
}

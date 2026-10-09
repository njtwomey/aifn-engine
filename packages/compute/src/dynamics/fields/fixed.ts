/**
 * Fixed points of $\xvec' = \fvec(\xvec)$ and their linearisation: Newton's method (`aifn-compute/numerics/roots`) from
 * a grid of seeds, the Jacobian by autodiff, eigenvalues (`aifn-compute/numerics/linalg`'s general eigensolver), the
 * Hartman–Grobman classification, the stable and unstable manifolds of saddles, and orbital derivatives of Lyapunov
 * functions (Strogatz, 2015, "Nonlinear Dynamics and Chaos", 2nd ed., §5–6; Perko, 2001, "Differential Equations and
 * Dynamical Systems", §2.6–2.9; Khalil, 2002, "Nonlinear Systems", §4.1).
 *
 * By the Hartman–Grobman theorem the flow near a hyperbolic fixed point $\xvec^*$ (no eigenvalue of
 * $\Jmat = \partial \fvec / \partial \xvec$ on the imaginary axis) looks like that of $\xvec' = \Jmat\xvec$, so the
 * eigenvalues of $\Jmat$ decide its kind and stability; a non-hyperbolic point is reported as such.
 */

import { det } from 'aifn-compute/numerics/linalg'
import { eig, type Eigen } from 'aifn-compute/numerics/linalg'
import { solveSystem } from 'aifn-compute/numerics/roots'
import { dense, fromData, imagPart, realPart, toFlat, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Scalar, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { gradientAt, jacobianAt, type ScalarField, type VectorField } from './calculus'
import { streamline, type Box } from './flow'
import { ShapeError } from 'aifn-compute/foundation/errors'

const { toF64, toMatrixF64 } = dense

/** The kinds of fixed point. The planar names follow the trace–determinant plane. */
export type FixedPointKind =
  | 'stable node'
  | 'unstable node'
  | 'stable spiral'
  | 'unstable spiral'
  | 'saddle'
  | 'centre'
  | 'stable degenerate node'
  | 'unstable degenerate node'
  | 'star'
  | 'non-hyperbolic'

/** The linear stability of a fixed point. */
export type Classification = {
  /** The kind of fixed point, from the eigenvalues (and in the plane the trace and determinant). */
  kind: FixedPointKind
  /**
   * `stable` when every eigenvalue has negative real part, `unstable` when one has positive real part, `marginal`
   * otherwise (each judged against the tolerance of `classifyLinear`).
   */
  stability: 'stable' | 'unstable' | 'marginal'
  /** True when no eigenvalue lies on the imaginary axis: then the linearisation decides the local phase portrait. */
  hyperbolic: boolean
  /** The eigenvalues and eigenvectors of $\Jmat$ (complex128). */
  eigen: Eigen
  /** $\trace \Jmat$. */
  trace: Scalar
  /** $\det \Jmat$. */
  determinant: Scalar
}

/**
 * Classifies the linearisation $\xvec' = \Jmat\xvec$ of a fixed point from the eigenvalues of $\Jmat$ ($n \times n$).
 * An eigenvalue counts as on the imaginary axis when $\lvert \operatorname{Re} \lambda \rvert \le \epsilon$, with
 * $\epsilon$ = `tolerance` $\cdot (1 + \max_{ij} \lvert J_{ij} \rvert)$. Hyperbolic points (none on the axis) are
 * stable, unstable or saddles, named node or spiral by whether complex eigenvalues occur; a planar point with a
 * repeated eigenvalue is a star ($\Jmat = \lambda\Imat$) or degenerate node; a planar point with purely imaginary
 * eigenvalues is a centre of the linearisation (the nonlinear point may be a weak spiral). Any other non-hyperbolic
 * point is `'non-hyperbolic'`. A non-square matrix throws `ShapeError`.
 *
 * @param J The Jacobian $\Jmat$ at the fixed point ($n \times n$).
 * @param options How close to the imaginary axis counts as on it.
 * @param options.tolerance The relative tolerance $\epsilon$ above, also used to decide whether eigenvalues are
 *   complex or repeated (default $10^{-9}$).
 * @returns The kind, the stability, whether the point is hyperbolic, the eigen-decomposition, and the trace and
 *   determinant of $\Jmat$.
 *
 * @example The planar zoo
 * const kinds = {
 *   'diag(-1, -2)': [[-1, 0], [0, -2]],
 *   'rotation': [[0, 1], [-1, 0]],
 *   'damped rotation': [[-0.1, 1], [-1, -0.1]],
 *   'diag(1, -1)': [[1, 0], [0, -1]],
 *   'Jordan block': [[-1, 1], [0, -1]],
 *   '-I': [[-1, 0], [0, -1]],
 * }
 * for (const [name, J] of Object.entries(kinds)) print(name, '→', classifyLinear(J).kind)
 *
 * @example Trace, determinant and eigenvalues of a saddle
 * const c = classifyLinear([[0, 1], [1, 0]])
 * print(c.kind, c.stability, ' trace =', c.trace, ' det =', c.determinant)
 * print('eigenvalues =', c.eigen.values)
 */
export function classifyLinear(J: MatrixLike, { tolerance = 1e-9 }: { tolerance?: Scalar } = {}): Classification {
  const { data, m: n } = toMatrixF64(J, 'classifyLinear')
  if (data.length !== n * n) throw new ShapeError('classifyLinear', 'classifyLinear: expected a square matrix')
  const e = eig(fromData(Float64Array.from(data), [n, n]))
  const re = toFlat(realPart(e.values))
  const im = toFlat(imagPart(e.values))
  let scale = 0
  for (const v of data) scale = Math.max(scale, Math.abs(v))
  const eps = tolerance * (1 + scale)
  let traceJ = 0
  for (let i = 0; i < n; i++) traceJ += data[i * n + i]
  const determinant = det(fromData(Float64Array.from(data), [n, n]))
  const positive = re.filter((r) => r > eps).length
  const negative = re.filter((r) => r < -eps).length
  const hyperbolic = positive + negative === n
  const complex = im.some((v) => Math.abs(v) > eps)
  let kind: FixedPointKind
  let stability: Classification['stability']
  if (!hyperbolic) {
    stability = positive > 0 ? 'unstable' : 'marginal'
    kind = n === 2 && complex && positive === 0 ? 'centre' : 'non-hyperbolic'
  } else if (positive > 0 && negative > 0) {
    stability = 'unstable'
    kind = 'saddle'
  } else {
    const stable = negative === n
    stability = stable ? 'stable' : 'unstable'
    const repeated = n === 2 && !complex && Math.abs(re[0] - re[1]) <= Math.sqrt(eps) * (1 + Math.abs(re[0]))
    if (complex) kind = stable ? 'stable spiral' : 'unstable spiral'
    else if (repeated) {
      const offDiagonal = Math.abs(data[1]) + Math.abs(data[2]) + Math.abs(data[0] - data[3])
      kind = offDiagonal <= eps ? 'star' : stable ? 'stable degenerate node' : 'unstable degenerate node'
    } else kind = stable ? 'stable node' : 'unstable node'
  }
  return { kind, stability, hyperbolic, eigen: e, trace: traceJ, determinant }
}

/**
 * A fixed point with its linearisation: the point $\xvec^*$ (`point`) and the Jacobian
 * $\Jmat = \partial \fvec / \partial \xvec$ there (`jacobian`), with the classification of $\Jmat$.
 */
export type FixedPoint = Classification & { point: Vector; jacobian: Matrix }

/**
 * Linearises $\fvec$ at a fixed point $\xvec^*$: the Jacobian by autodiff and its classification by `classifyLinear`.
 * The point is not checked to be fixed.
 *
 * @param f The vector field, written with tensor primitives so that it can be differentiated.
 * @param point The fixed point $\xvec^*$ ($n$ values).
 * @returns The classification with the point and the Jacobian.
 *
 * @example The two equilibria of a pendulum
 * // q′ = p, p′ = −sin q: hanging down (0, 0) is a centre, upside down (π, 0) a saddle.
 * const f = (x) => stack([get(x, 1), neg(sin(get(x, 0)))])
 * const down = linearise(f, [0, 0])
 * const up = linearise(f, [Math.PI, 0])
 * print('down:', down.kind, ' J =', down.jacobian)
 * print('up:', up.kind, ' J =', up.jacobian)
 */
export function linearise(f: VectorField, point: VectorLike): FixedPoint {
  const J = jacobianAt(f, point)
  const p = toF64(point, 'linearise')
  return { ...classifyLinear(J), point: fromData(p, [p.length]), jacobian: J }
}

/** Options for `fixedPoints`. */
export type FixedPointOptions = {
  /** Newton starts per axis on the box (seeds at the centres of a regular grid of cells). Default 7. */
  seeds?: Size
  /** Extra starting points, one per row ($k \times n$). */
  starts?: MatrixLike
  /** Points closer than this are the same fixed point. Default $10^{-6}$ times the longest side of the box. */
  mergeTolerance?: Scalar
}

/**
 * The fixed points $\fvec(\xvec^*) = \zeros$ inside a box, each linearised and classified. Damped Newton
 * (`aifn-compute/numerics/roots`, Jacobian by autodiff) runs from a regular grid of seeds (`seeds` per axis, so
 * $\text{seeds}^n$ starts) and any given `starts`; converged points inside the box are merged within `mergeTolerance`.
 * A start from which Newton fails or throws is skipped. A fixed point whose basin of attraction for Newton misses every
 * seed is not found, so raise `seeds` for fields with many fixed points. Sorted by the first coordinate, then the
 * second.
 *
 * @param f The vector field, written with tensor primitives so that it can be differentiated.
 * @param box The region searched, one `[lo, hi]` interval per coordinate (its length is $n$).
 * @param options The seed grid, extra starts and merge tolerance.
 * @returns The fixed points found, each with its Jacobian and classification.
 *
 * @example The three equilibria of an unforced Duffing oscillator
 * // x′ = y, y′ = x − x³: a saddle at the origin between two centres at x = ±1.
 * const f = (x) => stack([get(x, 1), sub(get(x, 0), pow(get(x, 0), 3))])
 * for (const p of fixedPoints(f, [[-2, 2], [-2, 2]])) print(p.point, p.kind)
 */
export function fixedPoints(f: VectorField, box: Box, options: FixedPointOptions = {}): FixedPoint[] {
  const n = box.length
  const { seeds = 7 } = options
  const size = Math.max(...box.map(([lo, hi]) => hi - lo))
  const mergeTol = options.mergeTolerance ?? 1e-6 * size
  const starts: number[][] = []
  const total = seeds ** n
  for (let k = 0; k < total; k++) {
    let r = k
    const x = box.map(([lo, hi]) => {
      const i = r % seeds
      r = Math.floor(r / seeds)
      return lo + ((hi - lo) * (i + 0.5)) / seeds
    })
    starts.push(x)
  }
  if (options.starts) {
    const { data, m } = toMatrixF64(options.starts, 'fixedPoints', undefined, n)
    for (let i = 0; i < m; i++) starts.push(Array.from(data.subarray(i * n, (i + 1) * n)))
  }
  const F = (x: Vector) => ({ value: toF64(f(x), 'fixedPoints'), jacobian: jacobianAt(f, x) })
  const found: number[][] = []
  for (const x0 of starts) {
    let r
    try {
      r = solveSystem(F, x0, { method: 'damped-newton', maxSteps: 50, ftol: 1e-12 })
    } catch {
      continue
    }
    if (!r.converged) continue
    const x = toFlat(r.x)
    const inside = x.every((v, i) => v >= box[i][0] - mergeTol && v <= box[i][1] + mergeTol)
    if (!inside || found.some((y) => Math.hypot(...y.map((v, i) => v - x[i])) <= mergeTol)) continue
    found.push(x)
  }
  found.sort((a, b) => a[0] - b[0] || (a[1] ?? 0) - (b[1] ?? 0))
  return found.map((x) => linearise(f, x))
}

/**
 * The branches of a fixed point's invariant manifolds, each a curve ($m \times n$, one point per row) that starts at
 * the point and leaves it: `stable`, whose points tend to the fixed point as $t \to \infty$, and `unstable`, whose
 * points tend to it as $t \to -\infty$.
 */
export type Manifolds = { stable: Matrix[]; unstable: Matrix[] }

/**
 * The one-dimensional stable and unstable manifolds of a hyperbolic fixed point (both branches of each), traced from
 * $\xvec^* \pm \varepsilon\vvec$ along each real eigenvector $\vvec$, forwards in time for an unstable direction
 * ($\lambda > 0$) and backwards for a stable one ($\lambda < 0$), by `streamline`. For a saddle in the plane these are
 * the separatrices. Each branch stops on leaving `bounds`. Complex and zero eigenvalues give no branch.
 *
 * @param f The vector field the fixed point belongs to.
 * @param fixed The fixed point, with its eigenvectors, as `linearise` or `fixedPoints` returns it.
 * @param options How far from the point to start and how long to follow each branch.
 * @param options.eps The distance $\varepsilon$ from $\xvec^*$ along the eigenvector at which each branch starts
 *   (default $10^{-5}$).
 * @param options.t The time each branch is followed for (default 20).
 * @param options.steps The number of RK4 steps over that time (default 400).
 * @param options.bounds The box a branch stops on leaving; without it a branch runs the whole time.
 * @returns The stable and unstable branches, each starting at $\xvec^*$.
 *
 * @example The separatrices of a linear saddle
 * // x′ = x, y′ = −y: the unstable manifold is the x axis, the stable one the y axis.
 * const f = (x) => stack([get(x, 0), neg(get(x, 1))])
 * const saddle = linearise(f, [0, 0])
 * const { stable, unstable } = invariantManifolds(f, saddle, { bounds: [[-1, 1], [-1, 1]] })
 * print('unstable branches end at', unstable.map((c) => toRows(c).at(-1)))
 * print('stable branches end at', stable.map((c) => toRows(c).at(-1)))
 */
export function invariantManifolds(
  f: VectorField,
  fixed: FixedPoint,
  { eps = 1e-5, t = 20, steps = 400, bounds }: { eps?: Scalar; t?: Scalar; steps?: Size; bounds?: Box } = {},
): Manifolds {
  const re = toFlat(realPart(fixed.eigen.values))
  const im = toFlat(imagPart(fixed.eigen.values))
  const V = toFlat(realPart(fixed.eigen.vectors))
  const n = re.length
  const x = toFlat(fixed.point)
  const out: Manifolds = { stable: [], unstable: [] }
  re.forEach((lambda, k) => {
    if (im[k] !== 0 || lambda === 0) return
    const v = Array.from({ length: n }, (_, i) => V[i * n + k])
    for (const s of [1, -1]) {
      const start = x.map((xi, i) => xi + s * eps * v[i])
      const curve = streamline(f, start, {
        t,
        steps,
        direction: lambda > 0 ? 'forward' : 'backward',
        bounds,
        minSpeed: 0,
      })
      // Prepend the fixed point so the branch visibly leaves it; backward branches come reversed from streamline.
      const pts = toFlat(curve)
      const m = curve.shape[0]
      const ordered =
        lambda > 0 ? pts : Array.from({ length: m }, (_, r) => pts.slice((m - 1 - r) * n, (m - r) * n)).flat()
      const data = Float64Array.from([...x, ...ordered])
      ;(lambda > 0 ? out.unstable : out.stable).push(fromData(data, [m + 1, n]))
    }
  })
  return out
}

/**
 * The orbital derivative $\dot{V}(\xvec) = \nabla V(\xvec) \cdot \fvec(\xvec)$ of a candidate Lyapunov function along
 * the flow: negative where $V$ decreases along trajectories.
 *
 * @param V The candidate Lyapunov function, written with tensor primitives so that it can be differentiated.
 * @param f The vector field.
 * @param x The point at which $\dot{V}$ is evaluated ($n$ values).
 * @returns $\dot{V}(\xvec)$, a number.
 *
 * @example Energy decreases under damping
 * // V = (x² + y²)/2 for x′ = y, y′ = −x − y: V̇ = −y², zero on the x axis and negative off it.
 * const V = (x) => mul(0.5, sum(square(x)))
 * const f = (x) => stack([get(x, 1), sub(neg(get(x, 0)), get(x, 1))])
 * print('at (1, 0):', lyapunovDerivative(V, f, [1, 0]))
 * print('at (1, 2):', lyapunovDerivative(V, f, [1, 2]))
 */
export function lyapunovDerivative(V: ScalarField, f: VectorField, x: VectorLike): Scalar {
  const g = toFlat(gradientAt(V, x))
  const p = toF64(x, 'lyapunovDerivative')
  const fx = toF64(f(fromData(p, [p.length])), 'lyapunovDerivative')
  return g.reduce((s, v, i) => s + v * fx[i], 0)
}

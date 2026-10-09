/**
 * Alternating projections: a point in the intersection of closed convex sets $\Ccal_1 \cap \dots \cap \Ccal_m$, given
 * only the Euclidean projection onto each set. A cycle maps $\xvec \mapsto P_m(\dots P_2(P_1(\xvec)))$; when the
 * intersection is non-empty the cycles converge to a point of it (Boyd and Dattorro 2003, "Alternating projections",
 * EE392o notes; von Neumann 1950 for two subspaces, Bregman 1965 for convex sets). The limit is some point of the
 * intersection, not in general the projection of the start onto it; Dykstra's variant (Boyle and Dykstra 1986) adds a
 * correction per set and converges to that projection.
 *
 * Also the projection onto the affine set of fixed group sums $\{\xvec : \sum_{i \in G_k} x_i = b_k\}$, for disjoint
 * groups $G_k$: the shift $x_i \leftarrow x_i + (b_k - \sum_{j \in G_k} x_j)/\lvert G_k \rvert$ inside each group,
 * the set label-proportion constraints define.
 */

import type { Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, type Vector } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'

const { toF64, vec } = dense

/** A Euclidean projection onto a closed convex set. */
export type Projection = (x: Vector) => Vector

/** Options of alternating projections. */
export interface AlternatingProjectionsOptions {
  /** Stop once a cycle moves $\xvec$ by less than this in the max norm (default 1e-9). */
  tolerance?: number
  /** Dykstra's corrections, so the limit is the projection of the start onto the intersection (default false). */
  dykstra?: boolean
}

/** One state of alternating projections: the point after $t$ full cycles. */
export interface AlternatingProjectionsState extends Status {
  /** The point $\xvec_t$ after $t$ cycles (the start at $t = 0$). */
  readonly x: Vector
  /** $\lVert \xvec_t - \xvec_{t-1} \rVert_\infty$ (Infinity at $t = 0$). */
  readonly change: number
  /** Dykstra's corrections, one per set (empty without Dykstra). */
  readonly corrections: readonly Vector[]
}

/**
 * Alternating projections as steps: each step is one cycle through the projections in order (step 0 is the start).
 * Converged when a cycle moves the point by less than the tolerance; nothing else stops it, so an empty intersection
 * runs to the step limit. Throws `DomainError` when no projection is given.
 *
 * @param projections The Euclidean projections $P_1, \dots, P_m$ onto the sets, applied in this order in every cycle.
 * @param options The tolerance of the stopping test, and whether to apply Dykstra's corrections.
 * @returns The algorithm, to step with `run` or `trace` from `{ x0 }`.
 *
 * @example Cycles between a line and a disc close in on their intersection
 * // The line x₁ + x₂ = 1 cuts the disc of radius 0.8 about the origin in a chord.
 * const line = projectGroupSums([0, 0], [1])
 * const disc = projectBall(0.8)
 * const tr = trace(alternatingProjectionsSteps([line, disc]), { x0: [2, 0] }, 5)
 * for (const s of tr.steps) print(`cycle ${s.t}: x =`, s.x, 'change =', s.change)
 */
export function alternatingProjectionsSteps(
  projections: readonly Projection[],
  options: AlternatingProjectionsOptions = {},
): Algorithm<{ x0: VectorLike }, AlternatingProjectionsState> {
  if (projections.length === 0)
    throw new DomainError('alternatingProjectionsSteps', 'alternatingProjectionsSteps: give at least one projection')
  const { tolerance = 1e-9, dykstra = false } = options
  return {
    name: dykstra ? 'dykstraProjections' : 'alternatingProjections',
    init: ({ x0 }) => {
      const x = toF64(x0, 'alternatingProjectionsSteps')
      return {
        t: 0,
        x: vec(x),
        change: Infinity,
        corrections: dykstra ? projections.map(() => vec(new Float64Array(x.length))) : [],
      }
    },
    step: (s) => {
      let x = toF64(s.x, 'alternatingProjectionsSteps')
      const start = x
      const corrections: Vector[] = []
      projections.forEach((project, k) => {
        if (!dykstra) {
          x = toF64(project(vec(x)), 'alternatingProjectionsSteps')
          return
        }
        // Dykstra: project x + q_k, and keep q_k = (x + q_k) − P_k(x + q_k) for the next visit to set k.
        const q = toF64(s.corrections[k], 'alternatingProjectionsSteps')
        const y = Float64Array.from(x, (v, i) => v + q[i])
        const p = toF64(project(vec(y)), 'alternatingProjectionsSteps')
        corrections.push(vec(Float64Array.from(y, (v, i) => v - p[i])))
        x = p
      })
      let change = 0
      for (let i = 0; i < x.length; i++) change = Math.max(change, Math.abs(x[i] - start[i]))
      return { t: s.t + 1, x: vec(x), change, corrections, converged: change < tolerance }
    },
  }
}

/** The result of {@link alternatingProjections}. */
export interface AlternatingProjectionsResult {
  /** The point after the last cycle. */
  readonly x: Vector
  /** Cycles taken. */
  readonly cycles: number
  /** Whether the last cycle moved the point by less than the tolerance (false when `maxCycles` ran out first). */
  readonly converged: boolean
}

/**
 * Run alternating projections (or Dykstra's) from $\xvec_0$ until a cycle moves $\xvec$ by less than the tolerance,
 * or for `maxCycles` cycles. Throws `DomainError` when no projection is given.
 *
 * @param projections The Euclidean projections onto the sets, applied in this order in every cycle.
 * @param x0 The starting point $\xvec_0$.
 * @param options The tolerance and `dykstra` of `AlternatingProjectionsOptions`, and `maxCycles`, the most cycles to
 *   run (default 1000).
 * @returns The final point, the cycles taken and whether the run converged.
 *
 * @example Dykstra's corrections find the closest point of the intersection
 * // The box [0, 1]² meets the line x₁ + x₂ = 1 in a segment; the closest point of it to (2, 1) is (1, 0).
 * const box = projectBox(0, 1)
 * const line = projectGroupSums([0, 0], [1])
 * const plain = alternatingProjections([box, line], [2, 1])
 * const dykstra = alternatingProjections([box, line], [2, 1], { dykstra: true })
 * print('alternating: x =', plain.x, 'cycles =', plain.cycles)
 * print('Dykstra: x =', dykstra.x, 'cycles =', dykstra.cycles)
 */
export function alternatingProjections(
  projections: readonly Projection[],
  x0: VectorLike,
  options: AlternatingProjectionsOptions & { maxCycles?: number } = {},
): AlternatingProjectionsResult {
  const { maxCycles = 1000 } = options
  const s = run(alternatingProjectionsSteps(projections, options), { x0 }, maxCycles)
  return { x: s.x, cycles: s.t, converged: s.converged === true }
}

/**
 * Projection onto $\{\xvec : \sum_{i \in G_k} x_i = b_k \text{ for every group } k\}$: `groups[i]` is the group of
 * coordinate $i$ ($0, \dots, K - 1$), or $-1$ for a coordinate in no group (left unchanged); `targets[k]` is $b_k$.
 * The groups are disjoint, so the projection shifts every coordinate of group $k$ by the same amount,
 * $(b_k - \sum_{j \in G_k} x_j)/\lvert G_k \rvert$. Throws `DomainError` for a group label that is not $-1$ or in
 * $0, \dots, K - 1$; the projection throws `ShapeError` for a point whose length is not that of `groups`.
 *
 * @param groups The group of each coordinate, one entry per coordinate: an integer in $0, \dots, K - 1$, or $-1$.
 * @param targets The sums $b_0, \dots, b_{K-1}$, one per group; a group with no coordinates is ignored.
 * @returns The projection, a function from a point (of the length of `groups`) to its projection.
 *
 * @example Two groups with fixed sums, and a free coordinate
 * // Coordinates 0 and 1 must sum to 1, coordinates 2 and 3 to 2; coordinate 4 is free.
 * const project = projectGroupSums([0, 0, 1, 1, -1], [1, 2])
 * print('projection =', project(tensor([0.2, 0.2, 1, 2, 5])))
 */
export function projectGroupSums(groups: ArrayLike<number>, targets: VectorLike): Projection {
  const b = toF64(targets, 'projectGroupSums')
  const size = new Float64Array(b.length)
  for (let i = 0; i < groups.length; i++) {
    const g = groups[i]
    if (g === -1) continue
    if (!(Number.isInteger(g) && g >= 0 && g < b.length))
      throw new DomainError(
        'projectGroupSums',
        `projectGroupSums: group ${g} of coordinate ${i} is not −1 or in 0 … ${b.length - 1}`,
      )
    size[g]++
  }
  return (x) => {
    const v = toF64(x, 'projectGroupSums')
    if (v.length !== groups.length)
      throw new ShapeError('projectGroupSums', `projectGroupSums: ${v.length} coordinates for ${groups.length} groups`)
    const sum = new Float64Array(b.length)
    for (let i = 0; i < v.length; i++) if (groups[i] >= 0) sum[groups[i]] += v[i]
    return vec(
      v.map((vi, i) =>
        groups[i] >= 0 && size[groups[i]] > 0 ? vi + (b[groups[i]] - sum[groups[i]]) / size[groups[i]] : vi,
      ),
    )
  }
}

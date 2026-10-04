/**
 * Alternating projections: a point in the intersection of closed convex sets C₁ ∩ … ∩ C_m, given only the Euclidean
 * projection onto each set. A cycle maps x ↦ P_m(… P₂(P₁(x))); when the intersection is non-empty the cycles converge
 * to a point of it (Boyd and Dattorro 2003, "Alternating projections", EE392o notes; von Neumann 1950 for two
 * subspaces, Bregman 1965 for convex sets). The limit is some point of the intersection, not in general the projection
 * of the start onto it; Dykstra's variant (Boyle and Dykstra 1986) adds a correction per set and converges to that
 * projection.
 *
 * Also the projection onto the affine set of fixed group sums {x : Σ_{i ∈ G_k} x_i = b_k}, for disjoint groups G_k: the
 * shift x_i ← x_i + (b_k − Σ_{j ∈ G_k} x_j)/|G_k| inside each group, the set label-proportion constraints define.
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
  /** Stop once a cycle moves x by less than this in the max norm (default 1e-9). */
  tolerance?: number
  /** Dykstra's corrections, so the limit is the projection of the start onto the intersection (default false). */
  dykstra?: boolean
}

/** One state of alternating projections: the point after t full cycles. */
export interface AlternatingProjectionsState extends Status {
  readonly x: Vector
  /** ‖x_t − x_{t−1}‖_∞ (Infinity at t = 0). */
  readonly change: number
  /** Dykstra's corrections, one per set (empty without Dykstra). */
  readonly corrections: readonly Vector[]
}

/**
 * Alternating projections as steps: each step is one cycle through the projections in order (step 0 is the start).
 * Converged when a cycle moves the point by less than the tolerance.
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
  readonly x: Vector
  /** Cycles taken. */
  readonly cycles: number
  readonly converged: boolean
}

/** Run alternating projections (or Dykstra's) from x0 until a cycle moves x by less than the tolerance. */
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
 * Projection onto {x : Σ_{i ∈ G_k} x_i = b_k for every group k}: `groups[i]` is the group of coordinate i (0 … K − 1),
 * or −1 for a coordinate in no group (left unchanged); `targets[k]` is b_k. The groups are disjoint, so the projection
 * shifts every coordinate of group k by the same amount, (b_k − Σ_{j ∈ G_k} x_j)/|G_k|.
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

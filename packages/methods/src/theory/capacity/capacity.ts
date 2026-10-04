/**
 * Capacity of hypothesis classes on a finite sample. A class shatters points when it realises every one of their 2ⁿ
 * labellings; the VC dimension is the size of the largest set it shatters (Vapnik and Chervonenkis, 1971). Here the
 * labellings a class realises are enumerated exactly for three classes on the plane or the line:
 *
 * - half-planes sign(w·x + b): a labelling is realisable when the linear program "find w, b with yᵢ(w·xᵢ + b) ≥ 1" is
 *   feasible (VC dimension 3 in the plane);
 * - axis-aligned rectangles labelling their inside positive: realisable when the bounding box of the positives holds no
 *   negative (VC dimension 4);
 * - intervals on the line (first coordinate), positive inside: the same test in one dimension (VC dimension 2).
 *
 * The empirical Rademacher complexity R̂(H) = E_σ sup_h (1/n) Σ σᵢ h(xᵢ) of the realised labellings (as ±1) is estimated
 * by Monte Carlo over random signs σ, against Massart's finite-class bound √(2 ln |H| / n).
 */

import { child, units, type Stream } from 'aifn-compute/foundation/random'
import { dense, type MatrixLike } from 'aifn-compute/foundation/tensor'
import { linprog } from 'aifn-compute/optim/programming'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A hypothesis class of the demonstrations. */
export type ShatterClass = 'half-planes' | 'rectangles' | 'intervals'

/** Whether a ±1 labelling of points [n, 2] is realisable by the class. */
export function realisable(points: MatrixLike, labels: ArrayLike<number>, family: ShatterClass): boolean {
  const { data: X, m: n } = dense.toMatrixF64(points, 'realisable')
  const pos = Array.from({ length: n }, (_, i) => i).filter((i) => labels[i] > 0)
  const neg = Array.from({ length: n }, (_, i) => i).filter((i) => labels[i] <= 0)
  if (family === 'half-planes') {
    if (pos.length === 0 || neg.length === 0) return true
    // Variables (w₁, w₂, b), free; constraints −yᵢ(w·xᵢ + b) ≤ −1; objective 0 (feasibility).
    const A = Array.from({ length: n }, (_, i) => {
      const y = labels[i] > 0 ? 1 : -1
      return [-y * X[2 * i], -y * X[2 * i + 1], -y]
    })
    const r = linprog({ c: [0, 0, 0], A_ub: A, b_ub: new Array<number>(n).fill(-1), bounds: [null, null] })
    return r.status === 'optimal'
  }
  if (pos.length === 0) return true
  const dims = family === 'rectangles' ? 2 : 1
  const lo = [Infinity, Infinity]
  const hi = [-Infinity, -Infinity]
  for (const i of pos)
    for (let k = 0; k < dims; k++) {
      lo[k] = Math.min(lo[k], X[2 * i + k])
      hi[k] = Math.max(hi[k], X[2 * i + k])
    }
  return !neg.some((i) => {
    for (let k = 0; k < dims; k++) if (X[2 * i + k] < lo[k] || X[2 * i + k] > hi[k]) return false
    return true
  })
}

/** Every labelling of the points (bit i of the index is point i positive), with whether the class realises it. */
export function shatteringTable(
  points: MatrixLike,
  family: ShatterClass,
): { labellings: Int8Array[]; realised: boolean[]; count: number; shattered: boolean } {
  const n = dense.toMatrixF64(points, 'shatteringTable').m
  if (n > 14) throw new DomainError('shatteringTable', 'shatteringTable: at most 14 points (2ⁿ labellings)')
  const labellings: Int8Array[] = []
  const realised: boolean[] = []
  for (let mask = 0; mask < 1 << n; mask++) {
    const y = Int8Array.from({ length: n }, (_, i) => ((mask >> i) & 1 ? 1 : -1))
    labellings.push(y)
    realised.push(realisable(points, y, family))
  }
  const count = realised.filter(Boolean).length
  return { labellings, realised, count, shattered: count === 1 << n }
}

/**
 * The empirical Rademacher complexity of a finite set of ±1 labellings on n points, E_σ max_h (1/n) Σ σᵢ hᵢ, by
 * `draws` Monte Carlo sign vectors; with Massart's bound √(2 ln |H| / n) and the running estimate after each draw.
 */
export function empiricalRademacher(
  s: Stream,
  labellings: readonly ArrayLike<number>[],
  draws = 2000,
): { estimate: number; standardError: number; massart: number; running: Float64Array } {
  if (labellings.length === 0)
    throw new DomainError('empiricalRademacher', 'empiricalRademacher: needs at least one hypothesis')
  const n = labellings[0].length
  const running = new Float64Array(draws)
  let sum = 0
  let sumSq = 0
  for (let d = 0; d < draws; d++) {
    const u = units(child(s, 'signs', d), n)
    let best = -Infinity
    for (const h of labellings) {
      let v = 0
      for (let i = 0; i < n; i++) v += (u[i] < 0.5 ? -1 : 1) * h[i]
      if (v > best) best = v
    }
    sum += best / n
    sumSq += (best / n) ** 2
    running[d] = sum / (d + 1)
  }
  const estimate = sum / draws
  const variance = Math.max(0, sumSq / draws - estimate * estimate)
  return {
    estimate,
    standardError: Math.sqrt(variance / draws),
    massart: Math.sqrt((2 * Math.log(labellings.length)) / n),
    running,
  }
}

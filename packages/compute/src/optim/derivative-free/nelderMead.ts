/**
 * The Nelder–Mead simplex method (Nelder & Mead, 1965), derivative free, as specified by Lagarias, Reeds, Wright &
 * Wright (1998), "Convergence properties of the Nelder–Mead simplex method in low dimensions", with the optional
 * dimension-adaptive coefficients of Gao & Han (2012).
 */

import type { Matrix, Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { StartOptions } from '../options'
import type { IterateState, MatrixLike, StoppingOptions, ValueFunction } from 'aifn-compute/foundation/contracts'
import { DEFAULT_DIVERGE, divergedAt, evaluateValue } from '../options'
import { dense } from 'aifn-compute/foundation/tensor'

const { data, mat, toF64, toMatrixF64, vec } = dense
type F64 = dense.F64

/** What a Nelder–Mead step did to the simplex. */
export type NelderMeadOperation = 'init' | 'reflect' | 'expand' | 'contract-outside' | 'contract-inside' | 'shrink'

/** A point the step evaluated, labelled by its role. */
export type NelderMeadTrial = {
  kind: 'reflect' | 'expand' | 'contract-outside' | 'contract-inside' | 'shrink'
  point: Vector
  value: number
}

/** The state of `nelderMead`. `x` and `value` are the best vertex and its value. */
export type NelderMeadState = IterateState & {
  /** The n + 1 vertices as rows, sorted by value, best first. */
  simplex: Matrix
  /** f at each vertex, ascending. */
  values: Vector
  /** The centroid of the n best vertices, about which the worst was reflected on the last step. */
  centroid: Vector
  operation: NelderMeadOperation
  /** The points evaluated on the last step, in order. */
  trials: NelderMeadTrial[]
  /** max‖v_i − v_best‖∞, the simplex size used in the stopping test. */
  size: number
  /** max|f_i − f_best|, the value spread used in the stopping test. */
  spread: number
}

/** Options for `nelderMead`. */
export type NelderMeadOptions = Pick<StoppingOptions, 'divergeAbove'> & {
  /** Stop when the simplex size is at most `xTolerance` and the value spread at most `fTolerance`. Defaults 1e-8. */
  xTolerance?: number
  fTolerance?: number
  /**
   * The starting simplex, (n + 1)×n. Default: x₀ and n points each with one coordinate of x₀ scaled by 1.05 (or set
   * to 0.00025 when it is zero), as scipy does.
   */
  initialSimplex?: MatrixLike
  /** Use the coefficients of Gao & Han (2012), which adapt to the dimension. Default false. */
  adaptive?: boolean
}

/**
 * Nelder–Mead: each step replaces the worst vertex by its reflection through the centroid of the others, an expansion,
 * or an outside or inside contraction, or shrinks the simplex towards the best vertex (Lagarias et al., 1998, §2).
 * Coefficients ρ = 1, χ = 2, γ = ½, σ = ½ (or adaptive). `f` returns a number. `init` takes `{ x0 }`.
 */
export function nelderMead(
  f: ValueFunction,
  options: NelderMeadOptions = {},
): Algorithm<StartOptions, NelderMeadState> {
  const { xTolerance = 1e-8, fTolerance = 1e-8, adaptive = false, divergeAbove = DEFAULT_DIVERGE } = options
  const name = 'nelder-mead'

  const pack = (
    vertices: F64[],
    values: number[],
    centroid: F64,
    operation: NelderMeadOperation,
    trials: NelderMeadTrial[],
    t: number,
    evaluations: number,
  ): NelderMeadState => {
    // Stable sort, so that among equal values the incumbent vertices keep their order.
    const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b] || a - b)
    const n = vertices[0].length
    const simplex = new Float64Array((n + 1) * n)
    const sorted = new Float64Array(n + 1)
    order.forEach((k, r) => {
      simplex.set(vertices[k], r * n)
      sorted[r] = values[k]
    })
    let size = 0
    let spread = 0
    for (let r = 1; r <= n; r++) {
      spread = Math.max(spread, Math.abs(sorted[r] - sorted[0]))
      for (let j = 0; j < n; j++) size = Math.max(size, Math.abs(simplex[r * n + j] - simplex[j]))
    }
    const best = simplex.slice(0, n)
    return {
      t,
      x: vec(best),
      value: sorted[0],
      simplex: mat(simplex, n + 1, n),
      values: vec(sorted),
      centroid: vec(centroid),
      operation,
      trials,
      size,
      spread,
      evaluations,
      converged: size <= xTolerance && spread <= fTolerance,
      diverged: divergedAt(sorted[0], best, divergeAbove),
    }
  }

  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const n = x.length
      let vertices: F64[]
      if (options.initialSimplex) {
        const { data: s } = toMatrixF64(options.initialSimplex, name, n + 1, n)
        vertices = Array.from({ length: n + 1 }, (_, r) => s.slice(r * n, (r + 1) * n))
      } else {
        vertices = [Float64Array.from(x)]
        for (let k = 0; k < n; k++) {
          const v = Float64Array.from(x)
          v[k] = v[k] !== 0 ? 1.05 * v[k] : 0.00025
          vertices.push(v)
        }
      }
      const values = vertices.map((v) => evaluateValue(f, v))
      return pack(vertices, values, Float64Array.from(x), 'init', [], 0, n + 1)
    },
    step: (s) => {
      const n = s.x.shape[0]
      const rho = 1
      const chi = adaptive ? 1 + 2 / n : 2
      const gamma = adaptive ? 0.75 - 1 / (2 * n) : 0.5
      const sigma = adaptive ? 1 - 1 / n : 0.5
      const S = data(s.simplex)
      const values = Array.from(data(s.values))
      const vertices = Array.from({ length: n + 1 }, (_, r) => S.slice(r * n, (r + 1) * n))
      const worst = vertices[n]
      const centroid = new Float64Array(n)
      for (let r = 0; r < n; r++) for (let j = 0; j < n; j++) centroid[j] += vertices[r][j] / n
      // x̄ + c(x̄ − x_worst)
      const along = (c: number) => centroid.map((m, j) => m + c * (m - worst[j]))
      const trials: NelderMeadTrial[] = []
      const tryPoint = (kind: NelderMeadTrial['kind'], p: F64) => {
        const value = evaluateValue(f, p)
        trials.push({ kind, point: vec(p), value })
        return value
      }
      const replaceWorst = (p: F64, v: number, operation: NelderMeadOperation) => {
        vertices[n] = p
        values[n] = v
        return pack(vertices, values, centroid, operation, trials, s.t + 1, s.evaluations + trials.length)
      }

      const fBest = values[0]
      const fSecondWorst = values[n - 1]
      const fWorst = values[n]
      const xr = along(rho)
      const fr = tryPoint('reflect', xr)
      if (fr < fBest) {
        const xe = along(rho * chi)
        const fe = tryPoint('expand', xe)
        return fe < fr ? replaceWorst(xe, fe, 'expand') : replaceWorst(xr, fr, 'reflect')
      }
      if (fr < fSecondWorst) return replaceWorst(xr, fr, 'reflect')
      if (fr < fWorst) {
        const xc = along(rho * gamma)
        const fc = tryPoint('contract-outside', xc)
        if (fc <= fr) return replaceWorst(xc, fc, 'contract-outside')
      } else {
        const xcc = along(-gamma)
        const fcc = tryPoint('contract-inside', xcc)
        if (fcc < fWorst) return replaceWorst(xcc, fcc, 'contract-inside')
      }
      // Shrink towards the best vertex.
      const best = vertices[0]
      for (let r = 1; r <= n; r++) {
        vertices[r] = best.map((b, j) => b + sigma * (vertices[r][j] - b))
        values[r] = tryPoint('shrink', vertices[r])
      }
      return pack(vertices, values, centroid, 'shrink', trials, s.t + 1, s.evaluations + trials.length)
    },
    done: (s) => s.converged || s.diverged,
  }
}

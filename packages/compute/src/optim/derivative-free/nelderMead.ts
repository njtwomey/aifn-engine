/**
 * The Nelder–Mead simplex method (Nelder and Mead, 1965), which minimises $f : \reals^n \to \reals$ from its values
 * alone, as specified by Lagarias, Reeds, Wright and Wright (1998), "Convergence properties of the Nelder–Mead simplex
 * method in low dimensions", §2, with the optional dimension-adaptive coefficients of Gao and Han (2012).
 *
 * The method keeps $n + 1$ vertices sorted by value. Each step moves the worst vertex $\xvec_{n+1}$ along the line
 * through the centroid $\bar{\xvec}$ of the other $n$, to $\bar{\xvec} + c(\bar{\xvec} - \xvec_{n+1})$ with
 * $c = \rho$ (reflect), $\rho\chi$ (expand), $\rho\gamma$ (contract outside) or $-\gamma$ (contract inside), or
 * shrinks every vertex towards the best by $\sigma$. It stops when both the simplex and the spread of its values are
 * small.
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
  /** The move that proposed the point: one of the steps of `NelderMeadOperation` other than `'init'`. */
  kind: 'reflect' | 'expand' | 'contract-outside' | 'contract-inside' | 'shrink'
  /** The point, of length $n$. */
  point: Vector
  /** $f$ at the point. */
  value: number
}

/** The state of `nelderMead`. `x` and `value` are the best vertex and its value. */
export type NelderMeadState = IterateState & {
  /**
   * The $n + 1$ vertices as the rows of an $(n + 1) \times n$ matrix, sorted by value, best first (ties keep the
   * older vertex first).
   */
  simplex: Matrix
  /** $f$ at each vertex, ascending, in the order of the rows of `simplex`. */
  values: Vector
  /**
   * The centroid of the $n$ best vertices, about which the worst was moved on the last step (on `'init'`, the
   * starting point $\xvec_0$).
   */
  centroid: Vector
  /** What the last step did to the simplex. */
  operation: NelderMeadOperation
  /** The points evaluated on the last step, in order (none on `'init'`). */
  trials: NelderMeadTrial[]
  /**
   * $\max_i \lVert \vvec_i - \vvec_1 \rVert_\infty$ over the vertices $\vvec_i$, with $\vvec_1$ the best: the simplex
   * size of the stopping test.
   */
  size: number
  /** $\max_i \lvert f_i - f_1 \rvert$, the spread of the values of the stopping test. */
  spread: number
}

/** Options for `nelderMead`. */
export type NelderMeadOptions = Pick<StoppingOptions, 'divergeAbove'> & {
  /** Stop when the simplex size is at most `xTolerance` and the value spread at most `fTolerance`. Defaults 1e-8. */
  xTolerance?: number
  /** The largest value spread at which the method stops (with `xTolerance`). Default 1e-8. */
  fTolerance?: number
  /**
   * The starting simplex, $(n + 1) \times n$, one vertex per row ($n$ is the length of $\xvec_0$, which is still
   * required). Default: $\xvec_0$ and $n$ points each with one coordinate of $\xvec_0$ scaled by 1.05 (or set to
   * 0.00025 when it is zero), as scipy does.
   */
  initialSimplex?: MatrixLike
  /**
   * Use the coefficients of Gao and Han (2012), which adapt to the dimension: $\chi = 1 + 2/n$,
   * $\gamma = 3/4 - 1/(2n)$, $\sigma = 1 - 1/n$. They equal the standard ones when $n = 2$. Default false.
   */
  adaptive?: boolean
}

/**
 * Nelder–Mead: each step replaces the worst vertex by its reflection through the centroid of the others, an expansion,
 * or an outside or inside contraction, or shrinks the simplex towards the best vertex (Lagarias et al., 1998, §2).
 * Coefficients $\rho = 1$, $\chi = 2$, $\gamma = 1/2$, $\sigma = 1/2$ (or adaptive). `init` takes `{ x0 }`. Each
 * step costs one or two evaluations of $f$, or $n + 2$ when it shrinks. The method converges when the simplex size and
 * the value spread are both within tolerance, and flags divergence when the best value or vertex is not finite or
 * $\lvert f \rvert$ exceeds `divergeAbove`.
 *
 * @param f The objective: takes a point (a vector of length $n$) and returns $f(\xvec)$ as a number, or an object with
 *   a `value` field (any `grad` is ignored).
 * @param options The tolerances of the stopping test, the starting simplex and whether the coefficients adapt to the
 *   dimension.
 * @returns The algorithm, to step with `run` or `trace` from `{ x0 }`.
 *
 * @example Minimise a quadratic from its values alone
 * // The minimum is at (1, -0.5), where f = 0.
 * const f = (x) => {
 *   const [a, b] = toFlat(x)
 *   return (a - 1) ** 2 + 2 * (b + 0.5) ** 2
 * }
 * const s = run(nelderMead(f), { x0: [0, 0] }, 500)
 * print('x =', s.x)
 * print('f(x) =', s.value)
 * print('steps =', s.t)
 * print('evaluations =', s.evaluations)
 *
 * @example The moves of the first steps, from a simplex of your own
 * const f = (x) => {
 *   const [a, b] = toFlat(x)
 *   return (a - 1) ** 2 + 2 * (b + 0.5) ** 2
 * }
 * const tr = trace(nelderMead(f, { initialSimplex: [[0, 0], [2, 0], [0, 2]] }), { x0: [0, 0] }, 6)
 * print('operations =', tr.steps.map((s) => s.operation))
 * print('best values =', tr.steps.map((s) => s.value))
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

/**
 * `linprog`: one call for a linear program, by the simplex method (Dantzig, 1951; Bland, 1977) or Mehrotra's
 * interior-point method (Mehrotra, 1992, "On the implementation of a primal-dual interior point method") on the
 * homogeneous self-dual embedding. Both return the same `LinearProgramResult`, so the methods can be swapped.
 */

import { interiorPointSolve } from './interior'
import type { LinearProgram } from './lp'
import { simplexSolve, type LinearProgramResult, type LinprogOptions } from './simplex'

/**
 * Solve a linear program: minimise $\cvec^\top\xvec$ subject to $\Amat_{\text{ub}}\xvec \le \bvec_{\text{ub}}$,
 * $\Amat_{\text{eq}}\xvec = \bvec_{\text{eq}}$ and bounds (default $\xvec \ge \zeros$), as `scipy.optimize.linprog`.
 * `method` is `simplex` (default; two-phase tableau simplex, with Bland's rule unless `rule` says otherwise) or
 * `interior-point` (Mehrotra's predictor–corrector). At an optimum the result carries a duality report (duals in
 * scipy's convention, slacks, reduced costs, duality gap and complementary slackness). Infeasibility and
 * unboundedness are reported in `status` by both methods (the interior-point method certifies them through its
 * homogeneous self-dual embedding), not thrown; an ill-formed problem throws as `parseLP` does.
 *
 * @param problem The linear program, in the form of `scipy.optimize.linprog`.
 * @param options The method, tolerance and `maxSteps`, and the simplex pivot `rule` (ignored by the interior point).
 * @returns The solution, objective and duality report, or the status that explains why there is none.
 *
 * @example Maximise over a polygon
 * // Maximise x + y subject to x + 2y <= 4 and 3x + y <= 6, x, y >= 0: minimise -x - y. Both rows bind at the optimum.
 * const r = linprog({ c: [-1, -1], A_ub: [[1, 2], [3, 1]], b_ub: [4, 6] })
 * print('x =', r.x, ' objective =', r.objective)
 * print('duals of the rows =', r.report.duals.ineq)
 * print('duality gap =', r.report.dualityGap)
 *
 * @example The interior-point method reaches the same vertex
 * const r = linprog({ c: [-1, -1], A_ub: [[1, 2], [3, 1]], b_ub: [4, 6] }, { method: 'interior-point' })
 * print('x =', r.x, ' steps =', r.steps)
 *
 * @example Infeasible and unbounded programs are reported in the status
 * print('x + y <= -1, x, y >= 0:', linprog({ c: [1, 1], A_ub: [[1, 1]], b_ub: [-1] }).status)
 * const r = linprog({ c: [-1, 0], A_ub: [[1, -1]], b_ub: [1], bounds: [[0, null], [null, null]] })
 * print('maximise x, x - y <= 1, y free:', r.status, ' ray =', r.ray)
 */
export function linprog(problem: LinearProgram, options: LinprogOptions = {}): LinearProgramResult {
  if (options.method === 'interior-point')
    return interiorPointSolve(problem, { tolerance: options.tolerance, maxSteps: options.maxSteps })
  return simplexSolve(problem, options)
}

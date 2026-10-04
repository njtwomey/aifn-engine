/**
 * `linprog`: one call for a linear program, by the simplex method (Dantzig, 1951; Bland, 1977) or Mehrotra's
 * interior-point method (Mehrotra, 1992, "On the implementation of a primal-dual interior point method") on the
 * homogeneous self-dual embedding.
 */

import { interiorPointSolve } from './interior'
import type { LinearProgram } from './lp'
import { simplexSolve, type LinearProgramResult, type LinprogOptions } from './simplex'

/**
 * Solve a linear program: minimise cᵀx subject to A_ub x ≤ b_ub, A_eq x = b_eq and bounds (default x ≥ 0), as
 * `scipy.optimize.linprog`. `method` is `simplex` (default; two-phase tableau simplex with Bland's rule) or
 * `interior-point` (Mehrotra's predictor–corrector). At an optimum the result carries a duality report (duals in
 * scipy's convention, slacks, reduced costs, duality gap and complementary slackness). Infeasibility and
 * unboundedness are reported in `status` by both methods (the interior-point method certifies them through its
 * homogeneous self-dual embedding).
 */
export function linprog(problem: LinearProgram, options: LinprogOptions = {}): LinearProgramResult {
  if (options.method === 'interior-point')
    return interiorPointSolve(problem, { tolerance: options.tolerance, maxSteps: options.maxSteps })
  return simplexSolve(problem, options)
}

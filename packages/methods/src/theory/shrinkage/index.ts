/**
 * `aifn-methods/theory/shrinkage`: the James–Stein estimator of a normal mean and Stein's paradox, its risk below the
 * maximum-likelihood estimate's in three or more dimensions.
 *
 * - The estimator: `jamesStein` shrinks one observation $\xvec \sim \Gauss(\thetavec, \sigma^2\Imat_p)$ towards zero,
 *   a fixed target or its grand mean (Efron and Morris), with the positive-part option (`JamesSteinOptions`,
 *   `JamesSteinEstimate`, `ShrinkageTarget`).
 * - Its risk: `jamesSteinRisk` simulates the risk of the MLE, James–Stein and the positive part against
 *   $\norm{\thetavec - \tvec}$ with common random numbers (`JamesSteinRiskOptions`, `JamesSteinRisk`);
 *   `jamesSteinExactRisk` is the closed form $\sigma^2(p - (p - 2)^2\expect[1/\chi^2_p(\lambda)])$, $2\sigma^2$ at the
 *   target.
 *
 * The simulator takes a stream and draws from its child, so a seed fixes the study. `shrinkageFunctions` registers
 * the functions.
 */

export {
  jamesStein,
  jamesSteinExactRisk,
  jamesSteinRisk,
  type JamesSteinEstimate,
  type JamesSteinOptions,
  type JamesSteinRisk,
  type JamesSteinRiskOptions,
  type ShrinkageTarget,
} from './shrinkage'
export { shrinkageFunctions } from './registry'

/**
 * `aifn-methods/optimisation/multi-fidelity`: multi-fidelity hyperparameter search, successive halving, Hyperband
 * and ASHA, run over given learning curves so that every decision can be checked against the known answer.
 *
 * - Schedules, as plain data: `successiveHalvingSchedule` gives the rungs (configurations $n_k$ and budget $r_k$ per
 *   rung, $\lfloor n_k / \eta \rfloor$ kept) and `hyperbandSchedule` the brackets of Li et al. (2018), with their
 *   total budgets from scratch and when training resumes (`MultiFidelitySchedule`, `Bracket`, `Rung`).
 * - Running one: `multiFidelitySteps` steps through a schedule over a `CurveTable` or `CurveFunction`, one rung per
 *   step, or with `asynchronous` as ASHA, one finished evaluation per step on simulated workers. Each state holds the
 *   evaluations and promotions, the budget spent, the simulated time, the incumbent, and whether the true best
 *   configuration (by its final loss) is still in the race (`MultiFidelityState`).
 * - Extrapolation: `fitPowerLaw` fits $a + b\,t^{-c}$ to a partial learning curve by variable projection
 *   (`PowerLawFit`).
 *
 * Losses are lower-is-better throughout. Seeded families of curves with known final values, and curves that cross,
 * come from `learningCurves` in `aifn-methods/data/synthetic`. `multiFidelityAlgorithms` and `multiFidelityFunctions`
 * register the entries.
 */

export {
  hyperbandSchedule,
  successiveHalvingSchedule,
  type Bracket,
  type MultiFidelitySchedule,
  type Rung,
} from './schedules'
export {
  multiFidelitySteps,
  type CurveFunction,
  type CurveTable,
  type MultiFidelityEvaluation,
  type MultiFidelityIncumbent,
  type MultiFidelityJob,
  type MultiFidelityOptions,
  type MultiFidelityPromotion,
  type MultiFidelityState,
} from './run'
export { fitPowerLaw, type PowerLawFit } from './extrapolate'
export { multiFidelityAlgorithms, multiFidelityFunctions } from './registry'

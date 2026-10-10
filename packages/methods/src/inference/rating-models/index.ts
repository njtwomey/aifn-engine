/**
 * `aifn-methods/inference/rating-models`: skill ratings from games and answers, online and in batch.
 *
 * - Online ratings from paired results: `eloExpected` and `eloUpdate` for one game and `eloRatings` over a sequence;
 *   Glicko's `glickoG`, `glickoExpected` and `glickoUpdate`, Glicko-2's `glicko2Update`, and `glickoRatings` over
 *   rating periods. `ratingAgreement` and `runningLogLoss` score a run against the truth and the results.
 * - Batch comparison models fitted by MM: `bradleyTerry` for pairs and `plackettLuce` for rankings, with
 *   `bradleyTerryProbability` and `plackettLuceLogProbability`, and `prequentialBradleyTerry` to compare its
 *   predictions with an online rating's.
 * - Item response theory: `irtProbability`, `itemInformation`, and `fitIrt` (1PL or 2PL, by marginal maximum
 *   likelihood with abilities integrated out).
 * - TrueSkill: the closed-form two-player `trueSkillUpdate` with its `drawMargin`, expectation propagation over a fixed
 *   set of matches (`trueSkillEp`), and the same factor graph in the model language (`trueSkillModel`).
 * - Rating dynamics: every system behind one `OnlineRater` interface (`createRater`, from a `RaterSpec`) run on a
 *   `GameStream` by `rateStream`; `trueSkillThroughTime` as the smoother over the whole stream; `settlingGames`,
 *   `settledRatings` and `ratingScaleMap` to compare runs; and the settings of real chess sites (`lichessSpec` with
 *   `LICHESS_PROVISIONAL_RD`, `chessComSpec` with `chessComStarts` and `CHESS_COM_LEVELS`, `fideSpec` with `fideDp`).
 * - Ratings are on Elo's scale (400 points for odds of $10 : 1$, `ELO_SCALE` points per unit of log-odds, and
 *   `THURSTONE_BETA` the matching performance noise), except that `trueSkillUpdate`, `trueSkillEp` and
 *   `trueSkillModel` default to TrueSkill's own $\mu_0 = 25$ scale. A result `{ a, b, score }` is player `a`'s score:
 *   1, $\tfrac{1}{2}$ or 0. Histories are row-major, one row per step after a starting row.
 * - `ratingModelAlgorithms` and `ratingModelFunctions` register the functions with the notes they serve.
 */

export { drawMargin } from './examples'
export { trueSkillEp } from './examples'
export { trueSkillModel, trueSkillUpdate } from './examples'
export { type TrueSkillEpState } from './examples'
export { type Rating } from './examples'
export {
  eloExpected,
  eloRatings,
  eloUpdate,
  glicko2Update,
  glickoExpected,
  glickoG,
  glickoRatings,
  glickoUpdate,
  ratingAgreement,
  runningLogLoss,
  type EloOptions,
  type Glicko2Options,
  type GlickoGame,
  type GlickoRating,
  type GlickoRatingsOptions,
  type PairedResult,
  type RatingRun,
} from './elo'
export {
  bradleyTerry,
  bradleyTerryProbability,
  plackettLuce,
  plackettLuceLogProbability,
  prequentialBradleyTerry,
  type PrequentialOptions,
  type ComparisonFit,
  type MmOptions,
} from './paired'
export { fitIrt, irtProbability, itemInformation, type IrtFit, type IrtFitOptions } from './irt'
export {
  CHESS_COM_LEVELS,
  ELO_SCALE,
  LICHESS_PROVISIONAL_RD,
  THURSTONE_BETA,
  chessComSpec,
  chessComStarts,
  createRater,
  fideDp,
  fideSpec,
  lichessSpec,
  rateStream,
  ratingScaleMap,
  settledRatings,
  settlingGames,
  trueSkillThroughTime,
  type EloSpec,
  type FideSpec,
  type GameStream,
  type Glicko2Spec,
  type GlickoSpec,
  type KalmanSpec,
  type OnlineRater,
  type RaterSpec,
  type RatingTrace,
  type ScaleMap,
  type SkillEstimate,
  type SmoothedTrace,
  type TrueSkillSpec,
  type TrueSkillThroughTimeOptions,
} from './dynamics'
export { ratingModelAlgorithms, ratingModelFunctions } from './registry'

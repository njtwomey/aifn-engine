/**
 * `aifn-methods/inference/rating-models`: skill rating. Online ratings from paired results: Elo, Glicko and Glicko-2
 * (`elo.ts`); batch comparison models fitted by MM: Bradley–Terry and Plackett–Luce (`paired.ts`); item response
 * theory, 1PL and 2PL by penalised maximum likelihood (`irt.ts`); and TrueSkill: closed-form updates, expectation
 * propagation and its structure in the model language (`examples.ts`); and rating dynamics: every system as an online
 * rater on one game stream, TrueSkill Through Time as the smoother, and the chess sites' settings (`dynamics.ts`).
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

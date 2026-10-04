/**
 * The registry of `aifn-methods/inference/rating-models`.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as dynamics from './dynamics'
import * as elo from './elo'
import * as examples from './examples'
import * as irt from './irt'
import * as paired from './paired'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/rating-models')
const fn = definer<FunctionInfo>('function', 'inference/rating-models')
const notes = ['trueskill', 'skill-rating', 'assessing-skills']

algorithm(
  {
    key: 'trueSkillEp',
    name: 'TrueSkill by expectation propagation',
    summary: 'Skills from a sequence of matches, by EP over the match factor graphs.',
    problem: 'factor-graph',
    state: { iterate: 'means', flags: ['converged'] },
    notes: [...notes, 'expectation-propagation'],
    cite: ['herbrich2006'],
  },
  examples.trueSkillEp,
)
fn(
  {
    key: 'trueSkillUpdate',
    name: 'TrueSkill update',
    summary: 'The closed-form two-player update of skill means and variances after one match.',
    role: 'inference',
    notes,
    cite: ['herbrich2006'],
  },
  examples.trueSkillUpdate,
)
fn(
  { key: 'trueSkillModel', name: 'TrueSkill model', role: 'construction', notes, cite: ['herbrich2006'] },
  examples.trueSkillModel,
)
fn({ key: 'drawMargin', name: 'TrueSkill draw margin', role: 'property', notes: ['trueskill'] }, examples.drawMargin)

const ELO = ['elo-rating', 'paired-comparison-models']
const GLICKO = ['glicko-and-glicko-2', 'paired-comparison-models']
fn(
  {
    key: 'eloUpdate',
    name: 'Elo update',
    summary: 'Move both ratings by K times the surprise: score minus 1/(1 + 10^((r_b − r_a)/400)).',
    role: 'inference',
    notes: ELO,
    cite: ['elo1978'],
  },
  elo.eloUpdate,
)
fn({ key: 'eloExpected', name: 'Elo expected score', role: 'property', notes: ELO, cite: ['elo1978'] }, elo.eloExpected)
fn(
  {
    key: 'eloRatings',
    name: 'Elo ratings over a season',
    summary: 'Elo updates over a sequence of results, with the rating history and the predictive log loss.',
    role: 'estimator',
    notes: ELO,
    cite: ['elo1978'],
  },
  elo.eloRatings,
)
fn(
  {
    key: 'glickoUpdate',
    name: 'Glicko update',
    summary: 'One rating period: the rating moves by the surprise weighted by its deviation; the deviation shrinks.',
    role: 'inference',
    notes: GLICKO,
    cite: ['glickman1999'],
  },
  elo.glickoUpdate,
)
fn(
  {
    key: 'glicko2Update',
    name: 'Glicko-2 update',
    summary: 'One rating period with a volatility updated by the Illinois iteration, then the deviation and rating.',
    role: 'inference',
    notes: GLICKO,
    cite: ['glickman2001'],
  },
  elo.glicko2Update,
)
fn({ key: 'glickoG', name: 'Glicko g(RD)', role: 'property', notes: GLICKO, cite: ['glickman1999'] }, elo.glickoG)
fn(
  { key: 'glickoExpected', name: 'Glicko expected score', role: 'property', notes: GLICKO, cite: ['glickman1999'] },
  elo.glickoExpected,
)
fn(
  {
    key: 'glickoRatings',
    name: 'Glicko ratings over a season',
    summary: 'Glicko or Glicko-2 over results cut into rating periods, with rating and deviation histories.',
    role: 'estimator',
    notes: GLICKO,
    cite: ['glickman1999', 'glickman2001'],
  },
  elo.glickoRatings,
)
fn(
  {
    key: 'bradleyTerry',
    name: 'Bradley–Terry by MM',
    summary: 'Strengths with P(i beats j) = γ_i/(γ_i + γ_j), by Hunter’s minorisation–maximisation updates.',
    role: 'fit',
    notes: ['paired-comparison-models'],
    cite: ['bradley1952', 'hunter2004'],
  },
  paired.bradleyTerry,
)
fn(
  {
    key: 'bradleyTerryProbability',
    name: 'Bradley–Terry win probability',
    role: 'property',
    notes: ['paired-comparison-models'],
    cite: ['bradley1952'],
  },
  paired.bradleyTerryProbability,
)
fn(
  {
    key: 'plackettLuce',
    name: 'Plackett–Luce by MM',
    summary: 'Strengths from rankings read as successive choices, by Hunter’s MM updates.',
    role: 'fit',
    notes: ['plackett-luce-model'],
    cite: ['plackett1975', 'luce1959', 'hunter2004'],
  },
  paired.plackettLuce,
)
fn(
  {
    key: 'plackettLuceLogProbability',
    name: 'Plackett–Luce log-probability',
    role: 'property',
    notes: ['plackett-luce-model'],
    cite: ['plackett1975'],
  },
  paired.plackettLuceLogProbability,
)
fn(
  {
    key: 'prequentialBradleyTerry',
    name: 'Prequential Bradley–Terry',
    summary: 'Predict each result from a Bradley–Terry fit on the results before it, refitted periodically.',
    role: 'estimator',
    notes: ['paired-comparison-models'],
    cite: ['hunter2004'],
  },
  paired.prequentialBradleyTerry,
)
fn(
  {
    key: 'ratingAgreement',
    name: 'Rating agreement with the truth',
    summary: 'Spearman correlation between ratings and true skills after each result.',
    role: 'estimator',
    notes: ELO,
  },
  elo.ratingAgreement,
)
fn(
  {
    key: 'runningLogLoss',
    name: 'Running predictive log loss',
    summary: 'The mean log loss of a rating system’s predictions of the results so far.',
    role: 'estimator',
    notes: ELO,
  },
  elo.runningLogLoss,
)
const IRT = ['item-response-theory']
fn(
  {
    key: 'irtProbability',
    name: 'Item response function',
    summary: 'P(correct) = c + (1 − c) σ(a(θ − b)).',
    role: 'property',
    notes: IRT,
    cite: ['lord1968'],
  },
  irt.irtProbability,
)
fn(
  {
    key: 'itemInformation',
    name: 'Item information',
    summary: 'The Fisher information an item carries about ability, a²P(1 − P) for c = 0; it peaks at θ = b.',
    role: 'property',
    notes: IRT,
    cite: ['lord1968'],
  },
  irt.itemInformation,
)
fn(
  {
    key: 'fitIrt',
    name: '1PL / 2PL item response model',
    summary:
      'Difficulties and discriminations by marginal maximum likelihood over a θ grid (autodiff, L-BFGS); abilities by EAP.',
    role: 'fit',
    notes: IRT,
    cite: ['rasch1960', 'lord1968', 'bock1981'],
  },
  irt.fitIrt,
)

const DYN = ['skill-rating', 'elo-rating', 'glicko-and-glicko-2', 'trueskill']
fn(
  {
    key: 'createRater',
    name: 'Online rater',
    summary:
      'Elo, FIDE Elo, Glicko, Glicko-2, TrueSkill or a logistic Kalman filter behind one interface: predict, rate a game, report mean ± sd.',
    role: 'construction',
    notes: [...DYN, 'kalman-filter'],
    cite: ['elo1978', 'glickman1999', 'glickman2001', 'herbrich2006'],
  },
  dynamics.createRater,
)
fn(
  {
    key: 'rateStream',
    name: 'Ratings over a game stream',
    summary: 'Any online rater over the rounds of a stream: every player’s mean and uncertainty after each round.',
    role: 'estimator',
    notes: DYN,
    cite: ['elo1978', 'glickman1999', 'glickman2001', 'herbrich2006'],
  },
  dynamics.rateStream,
)
fn(
  {
    key: 'trueSkillThroughTime',
    name: 'TrueSkill Through Time',
    summary:
      'Skill chains with a Gaussian random walk per round, smoothed by EP over all games: each estimate uses the games before and after it.',
    role: 'estimator',
    notes: ['trueskill-through-time', 'trueskill', 'kalman-smoother', 'expectation-propagation'],
    cite: ['dangauthier2007', 'herbrich2006'],
  },
  dynamics.trueSkillThroughTime,
)
fn(
  {
    key: 'settlingGames',
    name: 'Games to settle',
    summary: 'Games played until a rating stays within ±50 (or a chosen tolerance) of its settled value.',
    role: 'estimator',
    notes: DYN,
  },
  dynamics.settlingGames,
)
fn(
  {
    key: 'settledRatings',
    name: 'Settled ratings',
    summary: 'Each player’s mean rating over the last quarter of a run.',
    role: 'estimator',
    notes: DYN,
  },
  dynamics.settledRatings,
)
fn(
  {
    key: 'ratingScaleMap',
    name: 'Mapping between rating scales',
    summary: 'The least-squares line, mean offset and spread between two systems’ ratings of the same players.',
    role: 'estimator',
    notes: DYN,
  },
  dynamics.ratingScaleMap,
)
fn(
  {
    key: 'chessComStarts',
    name: 'Chess.com starting ratings',
    summary:
      'Starting ratings from self-declared levels (400, 800, 1200, 1600), each player picking the nearest to a noisy self-estimate.',
    role: 'construction',
    notes: ['glicko-and-glicko-2'],
    cite: ['glickman1999'],
  },
  dynamics.chessComStarts,
)
fn(
  {
    key: 'fideDp',
    name: 'FIDE rating difference of a score',
    summary: 'dp = √2·200·Φ⁻¹(p) within ±800, the normal table FIDE uses for initial and performance ratings.',
    role: 'property',
    notes: ['elo-rating'],
    cite: ['elo1978'],
  },
  dynamics.fideDp,
)

/** The algorithms of the module. */
export const ratingModelAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  examples,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const ratingModelFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  examples,
  dynamics,
  elo,
  paired,
  irt,
) as Table<FunctionInfo>

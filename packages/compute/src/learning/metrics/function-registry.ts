/**
 * The functions of `aifn-compute/learning/metrics` that are not metrics (curves, tables, decompositions and intervals),
 * each registered with a name, a role and the notes it serves. The metrics themselves are in `registry.ts`.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as agreement from './agreement'
import * as classification from './classification'
import * as clustering from './clustering'
import * as confusion from './confusion'
import * as curves from './curves'
import * as distances from './distances'
import * as probabilistic from './probabilistic'
import * as ranking from './ranking'
import * as regression from './regression'
import * as uncertainty from './uncertainty'

/** Registers a function of this module as a `function` entry of module `learning/metrics`. */
const fn = definer<FunctionInfo>('function', 'learning/metrics')
const ROC = ['receiver-operating-characteristic-curve-and-area']
const PR = ['precision-recall-curve-and-average-precision']
const OP = ['operating-points-and-equal-error-rate']
const CAL = ['calibration-error']
const CI = ['metric-confidence-intervals']

// ── Confusion counts ─────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  { key: 'confusionMatrix', name: 'Confusion matrix', role: 'estimator', notes: ['confusion-matrix'] },
  confusion.confusionMatrix,
)
fn(
  { key: 'confusionMargins', name: 'Confusion margins', role: 'property', notes: ['confusion-matrix'] },
  confusion.confusionMargins,
)
fn(
  {
    key: 'binaryCounts',
    name: 'Binary counts (TP, FP, FN, TN)',
    role: 'estimator',
    notes: ['confusion-matrix', 'sensitivity-specificity-and-predictive-values'],
  },
  confusion.binaryCounts,
)
fn(
  {
    key: 'binaryRates',
    name: 'Binary rates',
    role: 'estimator',
    notes: ['sensitivity-specificity-and-predictive-values'],
  },
  confusion.binaryRates,
)
fn(
  { key: 'countsAtThreshold', name: 'Counts at a threshold', role: 'estimator', notes: [...ROC, ...OP] },
  confusion.countsAtThreshold,
)
fn(
  {
    key: 'precisionRecallFscoreSupport',
    name: 'Precision, recall, F-score and support',
    role: 'estimator',
    notes: ['precision-recall-and-f-score', 'averaging-multiclass-metrics'],
  },
  classification.precisionRecallFscoreSupport,
)
fn(
  { key: 'kappaFromTable', name: "Cohen's κ from a table", role: 'estimator', notes: ['cohens-kappa'] },
  classification.kappaFromTable,
)

// ── Curves and operating points ──────────────────────────────────────────────────────────────────────────────────────

fn(
  { key: 'rocCurve', name: 'ROC curve', role: 'estimator', returns: 'curve', notes: ROC, cite: ['fawcett2006'] },
  curves.rocCurve,
)
fn(
  {
    key: 'rocConvexHull',
    name: 'ROC convex hull',
    role: 'transform',
    returns: 'curve',
    notes: ['receiver-operating-characteristic-convex-hull', ...ROC],
  },
  curves.rocConvexHull,
)
fn(
  { key: 'precisionRecallCurve', name: 'Precision–recall curve', role: 'estimator', returns: 'curve', notes: PR },
  curves.precisionRecallCurve,
)
fn(
  { key: 'precisionRecallTrapezoid', name: 'Trapezoidal area under the PR curve', role: 'estimator', notes: PR },
  curves.precisionRecallTrapezoid,
)
fn(
  {
    key: 'precisionRecallGainCurve',
    name: 'Precision–recall–gain curve',
    role: 'estimator',
    returns: 'curve',
    notes: ['precision-recall-gain-curves'],
  },
  curves.precisionRecallGainCurve,
)
fn({ key: 'detCurve', name: 'DET curve', role: 'estimator', returns: 'curve', notes: OP }, curves.detCurve)
fn(
  {
    key: 'costCurve',
    name: 'Cost curve',
    role: 'estimator',
    returns: 'curve',
    notes: ['cost-curves'],
    cite: ['drummond2006'],
  },
  curves.costCurve,
)
fn(
  { key: 'gainCurve', name: 'Cumulative gain curve', role: 'estimator', returns: 'curve', notes: ROC },
  curves.gainCurve,
)
fn(
  {
    key: 'normalisedExpectedCost',
    name: 'Normalised expected cost',
    role: 'estimator',
    notes: ['cost-curves', 'threshold-choice-methods'],
  },
  curves.normalisedExpectedCost,
)
fn(
  { key: 'operatingPoint', name: 'Operating point', role: 'estimator', notes: [...OP, 'threshold-choice-methods'] },
  curves.operatingPoint,
)
fn(
  {
    key: 'youdenPoint',
    name: "Youden's operating point",
    role: 'estimator',
    notes: [...OP, 'threshold-choice-methods'],
  },
  curves.youdenPoint,
)
fn({ key: 'equalErrorRate', name: 'Equal error rate', role: 'estimator', notes: OP }, curves.equalErrorRate)
fn({ key: 'binormalRates', name: 'Binormal rates', role: 'property', notes: ROC }, curves.binormalRates)
fn(
  {
    key: 'binormalCurves',
    name: 'Binormal ROC and PR curves',
    role: 'property',
    returns: 'curve',
    notes: [...ROC, ...PR],
  },
  curves.binormalCurves,
)
fn(
  { key: 'binormalAuroc', name: 'Binormal AUROC', tex: '\\Phi(d / \\sqrt{2})', role: 'property', notes: ROC },
  curves.binormalAuroc,
)
fn(
  { key: 'binormalAveragePrecision', name: 'Binormal average precision', role: 'property', notes: PR },
  curves.binormalAveragePrecision,
)
fn(
  { key: 'binormalEqualErrorRate', name: 'Binormal equal error rate', role: 'property', notes: OP },
  curves.binormalEqualErrorRate,
)

// ── Probabilistic ────────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'reliabilityDiagram',
    name: 'Reliability diagram',
    role: 'estimator',
    notes: ['reliability-diagrams-and-consistency-bars', ...CAL],
  },
  probabilistic.reliabilityDiagram,
)
fn(
  {
    key: 'consistencyBars',
    name: 'Consistency bars',
    role: 'estimator',
    random: true,
    notes: ['reliability-diagrams-and-consistency-bars'],
  },
  probabilistic.consistencyBars,
)
fn(
  {
    key: 'brierDecomposition',
    name: 'Brier score decomposition',
    summary: 'Reliability, resolution and uncertainty (Murphy, 1973).',
    role: 'estimator',
    notes: ['log-loss-and-brier-score', ...CAL],
    cite: ['murphy1973'],
  },
  probabilistic.brierDecomposition,
)
fn(
  {
    key: 'pitValues',
    name: 'Probability integral transform values',
    role: 'transform',
    notes: ['quantile-calibration', 'probabilistic-forecasting'],
  },
  probabilistic.pitValues,
)

// ── Intervals and tests ──────────────────────────────────────────────────────────────────────────────────────────────

fn(
  { key: 'waldInterval', name: 'Wald interval for a proportion', role: 'estimator', notes: CI },
  uncertainty.waldInterval,
)
fn({ key: 'wilsonInterval', name: 'Wilson score interval', role: 'estimator', notes: CI }, uncertainty.wilsonInterval)
fn(
  {
    key: 'bootstrapMetric',
    name: 'Bootstrap interval for a metric',
    role: 'estimator',
    random: true,
    notes: [...CI, 'bootstrap'],
  },
  uncertainty.bootstrapMetric,
)
fn(
  {
    key: 'pairedBootstrap',
    name: 'Paired bootstrap comparison',
    role: 'test',
    random: true,
    notes: [...CI, 'bootstrap'],
  },
  uncertainty.pairedBootstrap,
)
fn(
  { key: 'aurocDeLong', name: "AUROC with DeLong's variance", role: 'estimator', notes: [...CI, ...ROC] },
  uncertainty.aurocDeLong,
)
fn({ key: 'delongTest', name: "DeLong's test", role: 'test', notes: [...CI, ...ROC] }, uncertainty.delongTest)

// ── Other tables and helpers ─────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'contingencyTable',
    name: 'Contingency table',
    role: 'estimator',
    notes: ['rand-index-and-adjusted-rand-index', 'information-theoretic-clustering-metrics'],
  },
  clustering.contingencyTable,
)
fn(
  {
    key: 'pairConfusion',
    name: 'Pair confusion matrix',
    role: 'estimator',
    notes: ['rand-index-and-adjusted-rand-index', 'fowlkes-mallows-index'],
  },
  clustering.pairConfusion,
)
fn(
  {
    key: 'homogeneityCompletenessV',
    name: 'Homogeneity, completeness and V-measure',
    role: 'estimator',
    notes: ['information-theoretic-clustering-metrics'],
  },
  clustering.homogeneityCompletenessV,
)
fn(
  {
    key: 'silhouetteSamples',
    name: 'Silhouette per sample',
    role: 'estimator',
    notes: ['internal-clustering-indices', 'choosing-the-number-of-clusters'],
  },
  clustering.silhouetteSamples,
)
fn(
  { key: 'chiSquareStatistic', name: 'Chi-square statistic of a table', role: 'test', notes: ['nominal-association'] },
  agreement.chiSquareStatistic,
)
fn(
  { key: 'hausdorffDistances', name: 'Directed Hausdorff distances', role: 'estimator', notes: ['hausdorff-distance'] },
  distances.hausdorffDistances,
)
fn(
  { key: 'orthogonalProcrustes', name: 'Orthogonal Procrustes', role: 'solver', notes: ['procrustes-analysis'] },
  distances.orthogonalProcrustes,
)
fn(
  {
    key: 'tweedieUnitDeviance',
    name: 'Tweedie unit deviance',
    role: 'estimator',
    notes: ['deviance-and-generalised-linear-model-diagnostics'],
  },
  regression.tweedieUnitDeviance,
)
fn(
  { key: 'gainFunction', name: 'Gain function', role: 'transform', notes: ['normalised-discounted-cumulative-gain'] },
  ranking.gainFunction,
)
fn(
  {
    key: 'positionDiscount',
    name: 'Position discount',
    role: 'transform',
    notes: ['normalised-discounted-cumulative-gain'],
  },
  ranking.positionDiscount,
)

/**
 * The functions of the module that are not metrics, keyed by name, each with its registry `info` (name, role and
 * notes).
 *
 * @example The registered functions and one entry
 * print('functions:', Object.keys(metricsFunctions).length)
 * print(metricsFunctions.confusionMatrix.info)
 */
export const metricsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>(
    'function',
    confusion,
    classification,
    curves,
    probabilistic,
    uncertainty,
    clustering,
    agreement,
    distances,
    regression,
    ranking,
  ) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

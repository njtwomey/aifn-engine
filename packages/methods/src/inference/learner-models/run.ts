/**
 * Streamed experiments with learner models on simulated students: one comparison of the models, and a sweep of the
 * zero-inflation rate.
 *
 * Following Twomey et al. (2022), §3.3, `learnerModelRun` fits IRT, a linear knowledge-tracing machine and IRT-ZILM to
 * the same training responses and compares ability recovery, bias by group, held-out prediction and, for IRT-ZILM, the
 * posterior that each zero is structural; `learnerEquitySweep` repeats the comparison on one dataset per
 * zero-inflation rate of the conditions, scoring each model's equity. Both are generators for streamed runs: the
 * comparison yields after every fit, the sweep after every dataset.
 */

import { child, stream, units } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  abilityEquity,
  parameterRecovery,
  responseScores,
  structuralZeroAuroc,
  type AbilityEquity,
  type ParameterRecovery,
  type ResponseScores,
} from './evaluation'
import {
  fitLearnerModel,
  predictLearner,
  structuralZeroPosterior,
  type LearnerModelFit,
  type LearnerModelKind,
} from './zilm'

/** Simulated learners with their truth (the shape of `aifn-methods/data/synthetic` `learnerResponses`). */
export interface LearnerData {
  /** The responses, $P \times I$: 0, 1, or NaN for not attempted. */
  responses: Tensor
  /** The student conditions $\Zmat$, $P \times K$ (0 or 1). */
  conditions: Tensor
  /** The item features $\Xmat$, $I \times F$. */
  itemFeatures: Tensor
  /** The true abilities $\theta_p$ (length $P$). */
  ability: Float64Array
  /** The true difficulties $b_i$ (length $I$). */
  difficulty: Float64Array
  /** The true discriminations $a_i$ (length $I$). */
  discrimination: Float64Array
  /** The true $\pi$ ($P \times I$, row-major); optional (real data has none). */
  pi?: Float64Array
  /** Which responses were structural zeros (1), $P \times I$ row-major; optional (real data has none). */
  structural?: Uint8Array
  /** Each student's group (0: no condition). */
  group: Int32Array
}

/** Options of `learnerModelRun` and `learnerEquitySweep`. */
export interface LearnerRunOptions {
  /** The models to fit, in order (default `irt`, `ktm`, `zilm`). */
  models?: readonly LearnerModelKind[]
  /** Share of the observed responses held out for the predictive scores (default 0.2). */
  testShare?: number
  /** Most L-BFGS steps per fit (default 1000). */
  maxSteps?: number
  /** The seed of the train/test split (default 0). */
  seed?: number | string
}

/** One fitted model with its evaluation. */
export interface LearnerModelResult {
  /** Which model. */
  model: LearnerModelKind
  /** The fit on the training responses. */
  fit: LearnerModelFit
  /** Its ability bias by group and equity gap (`abilityEquity`, group 0 the reference). */
  equity: AbilityEquity
  /** The correlations of its abilities, difficulties and discriminations with the truth. */
  recovery: { ability: ParameterRecovery; difficulty: ParameterRecovery; discrimination: ParameterRecovery }
  /** Its scores on the held-out responses. */
  test: ResponseScores
  /**
   * IRT-ZILM only: for every zero among the responses (training and held out), its student and item, the posterior
   * that it is structural, its fitted $\pi$, whether it was structural (1; empty unless the data carry the truth) and
   * the AUROC of the posterior for that truth (NaN without it).
   */
  zeros?: {
    student: Int32Array
    item: Int32Array
    posterior: Float64Array
    pi: Float64Array
    structural: Uint8Array
    auroc: number
  }
}

/** A learner-model comparison so far. */
export interface LearnerModelRunResult {
  /** The models fitted so far, in order. */
  results: LearnerModelResult[]
  /** The number of models the run fits. */
  total: number
  /** Whether every model has been fitted. */
  finished: boolean
  /** The held-out mask (1 test) over the response matrix. */
  test: Uint8Array
  /** The truth the estimates are scored against, and each student's number of conditions. */
  truth: { ability: Float64Array; difficulty: Float64Array; group: Int32Array; conditionCount: Int32Array }
}

const DEFAULT_MODELS: readonly LearnerModelKind[] = ['irt', 'ktm', 'zilm']

/**
 * The training and test masks over the observed responses: each observed response is held out with probability
 * `testShare`, by uniform draws from the child `split` of the stream `seed`.
 *
 * @param responses The $P \times I$ responses (NaN: not attempted, in neither mask).
 * @param testShare The probability that an observed response is held out.
 * @param seed The seed of the root stream.
 * @returns The flat responses `y` and the row-major masks `train` and `test` (1: in the set).
 */
function split(responses: Tensor, testShare: number, seed: number | string) {
  const y = toFlat(responses)
  const u = units(child(stream(seed), 'split'), y.length)
  const train = new Uint8Array(y.length)
  const test = new Uint8Array(y.length)
  for (let k = 0; k < y.length; k++) if (Number.isFinite(y[k])) (u[k] < testShare ? test : train)[k] = 1
  return { y, train, test }
}

/**
 * Fit one model on the training responses and evaluate it: ability equity, parameter recovery and held-out scores,
 * and for IRT-ZILM the structural-zero posterior of every zero.
 *
 * @param data The learners, with their truth.
 * @param model The model to fit.
 * @param y The flat responses (row-major), as `split` returns them.
 * @param train The training mask over the responses.
 * @param test The held-out mask over the responses.
 * @param maxSteps The most L-BFGS steps.
 * @param init A fit to start from (the IRT fit, for the later models), or none.
 * @returns The fit and its evaluation.
 */
function evaluate(
  data: LearnerData,
  model: LearnerModelKind,
  y: ArrayLike<number>,
  train: Uint8Array,
  test: Uint8Array,
  maxSteps: number,
  init?: LearnerModelFit,
): LearnerModelResult {
  const fit = fitLearnerModel(
    data.responses,
    { model, conditions: data.conditions, itemFeatures: data.itemFeatures, maxSteps, init },
    train,
  )
  const pred = predictLearner(fit, data.conditions, data.itemFeatures)
  const result: LearnerModelResult = {
    model,
    fit,
    equity: abilityEquity(fit.ability, data.ability, data.group),
    recovery: {
      ability: parameterRecovery(fit.ability, data.ability),
      difficulty: parameterRecovery(fit.difficulty, data.difficulty),
      discrimination: parameterRecovery(fit.discrimination, data.discrimination),
    },
    test: responseScores(y, pred.correct, test),
  }
  if (model === 'zilm') {
    const I = data.responses.shape[1]
    const student: number[] = []
    const item: number[] = []
    const posterior: number[] = []
    const pi: number[] = []
    const truth: number[] = []
    for (let k = 0; k < y.length; k++)
      if (y[k] === 0) {
        student.push(Math.floor(k / I))
        item.push(k % I)
        posterior.push(structuralZeroPosterior(pred.pi[k], pred.base[k]))
        pi.push(pred.pi[k])
        if (data.structural) truth.push(data.structural[k])
      }
    result.zeros = {
      student: Int32Array.from(student),
      item: Int32Array.from(item),
      posterior: Float64Array.from(posterior),
      pi: Float64Array.from(pi),
      structural: Uint8Array.from(truth),
      auroc: data.structural ? structuralZeroAuroc(posterior, truth) : NaN,
    }
  }
  return result
}

/**
 * Fit the learner models to the same training responses (a random share $1 - \text{testShare}$ of the observed ones)
 * and evaluate each: ability bias by group and the equity gap, parameter recovery, held-out scores and (IRT-ZILM)
 * structural-zero posteriors. Models after an IRT fit start from it. Yields once before the first fit and after every
 * fit. Deterministic in `seed`.
 *
 * @param data The learners, with their truth.
 * @param options The models, the held-out share, the L-BFGS limit and the seed of the split.
 * @returns A generator of the comparison so far; its return value is the finished comparison.
 *
 * @example IRT and IRT-ZILM on simulated students
 * // 45 students, a third with a condition; the odd items are timed, and a timed item gives a student with the
 * // condition a zero with probability 0.9 whatever their ability. On data this small IRT-ZILM finds no zero
 * // inflation here (its intercept runs far negative) and matches IRT.
 * const s = stream(1)
 * const ability = toFlat(normal(s, 0, 1, { shape: [45] }))
 * const timed = [0, 1, 0, 1, 0, 1, 0, 1]
 * const difficulty = Float64Array.from(timed, (_, i) => -1 + (2 * i) / 7)
 * const group = Int32Array.from(ability, (_, p) => (p % 3 === 0 ? 1 : 0))
 * const y = (p, b) => bernoulli(s, zilmProbability(ability[p], 1, b))
 * const answer = (g, p, b, i) => (g && timed[i] && uniform(s) < 0.9 ? 0 : y(p, b))
 * const rows = Array.from(group, (g, p) => Array.from(difficulty, (b, i) => answer(g, p, b, i)))
 * const data = {
 *   responses: tensor(rows),
 *   conditions: tensor(Array.from(group, (g) => [g])),
 *   itemFeatures: tensor(timed.map((t) => [t])),
 *   ability,
 *   difficulty,
 *   discrimination: new Float64Array(8).fill(1),
 *   group,
 * }
 * let last
 * for (const r of learnerModelRun(data, { models: ['irt', 'zilm'] })) last = r
 * for (const r of last.results) {
 *   print(r.model, 'gap:', r.equity.gap, 'ability r:', r.recovery.ability.pearson, 'test accuracy:', r.test.accuracy)
 * }
 * print('IRT-ZILM intercept:', last.results[1].fit.intercept, 'zeros scored:', last.results[1].zeros.posterior.length)
 */
export function* learnerModelRun(
  data: LearnerData,
  options: LearnerRunOptions = {},
): Generator<LearnerModelRunResult, LearnerModelRunResult> {
  const { models = DEFAULT_MODELS, testShare = 0.2, maxSteps = 1000, seed = 0 } = options
  const { y, train, test } = split(data.responses, testShare, seed)
  const results: LearnerModelResult[] = []
  const [P, K] = data.conditions.shape
  const z = toFlat(data.conditions)
  const conditionCount = Int32Array.from({ length: P }, (_, p) => {
    let c = 0
    for (let k = 0; k < K; k++) c += z[p * K + k]
    return c
  })
  const truth = { ability: data.ability, difficulty: data.difficulty, group: data.group, conditionCount }
  const snapshot = (finished: boolean) => ({ results: [...results], total: models.length, finished, test, truth })
  yield snapshot(false)
  for (const model of models) {
    // Later models start from the IRT fit when there is one.
    const init = results.find((r) => r.model === 'irt')?.fit
    results.push(evaluate(data, model, y, train, test, maxSteps, init))
    yield snapshot(results.length === models.length)
  }
  return snapshot(true)
}

/** One point of the equity sweep: one dataset, every model's equity and scores. */
export interface LearnerSweepPoint {
  /** The zero-inflation rate $r$ of the conditions in this dataset. */
  rate: number
  /**
   * The realised share of structural zeros among the observed responses of students with and without a condition (0
   * when the data carry no truth).
   */
  structuralShare: { with: number; without: number }
  /** Each model's ability bias and gap, ability recovery and held-out scores. */
  models: {
    model: LearnerModelKind
    equity: AbilityEquity
    ability: ParameterRecovery
    test: ResponseScores
  }[]
}

/** An equity sweep so far. */
export interface LearnerSweepResult {
  /** The datasets done so far, in order. */
  points: LearnerSweepPoint[]
  /** The number of datasets. */
  total: number
  /** Whether every dataset is done. */
  finished: boolean
}

/**
 * The equity sweep: for each dataset (simulated with zero-inflation rate `rates[j]` for every condition), fit the
 * models and record each one's ability bias by group, equity gap, ability recovery and held-out scores. Every dataset
 * is split with the same `seed`, and models after `irt` start from its fit. Yields once before the first dataset and
 * after every dataset.
 *
 * @param datasets The datasets, one per point.
 * @param rates The zero-inflation rate each dataset was simulated with, in the same order; only recorded.
 * @param options The models, the held-out share, the L-BFGS limit and the seed of the split.
 * @returns A generator of the sweep so far; its return value is the finished sweep.
 *
 * @example Two datasets, without and with zero inflation
 * // 45 students, a third with a condition; on the odd (timed) items a student with the condition gets a structural
 * // zero with probability `rate`. Datasets this small give noisy gaps, and here IRT-ZILM finds no zero inflation
 * // and matches IRT; the paper's datasets have hundreds of students.
 * const simulate = (rate) => {
 *   const s = stream(1)
 *   const ability = toFlat(normal(s, 0, 1, { shape: [45] }))
 *   const timed = [0, 1, 0, 1, 0, 1, 0, 1]
 *   const difficulty = Float64Array.from(timed, (_, i) => -1 + (2 * i) / 7)
 *   const group = Int32Array.from(ability, (_, p) => (p % 3 === 0 ? 1 : 0))
 *   const structural = new Uint8Array(45 * 8)
 *   const y = (p, b) => bernoulli(s, zilmProbability(ability[p], 1, b))
 *   const zero = (p, i) => (structural[p * 8 + i] = group[p] && timed[i] && uniform(s) < rate ? 1 : 0)
 *   const rows = Array.from(group, (_, p) => Array.from(difficulty, (b, i) => (zero(p, i) ? 0 : y(p, b))))
 *   const conditions = tensor(Array.from(group, (g) => [g]))
 *   const itemFeatures = tensor(timed.map((t) => [t]))
 *   const discrimination = new Float64Array(8).fill(1)
 *   const data = { responses: tensor(rows), conditions, itemFeatures, ability, difficulty, discrimination }
 *   return { ...data, structural, group }
 * }
 * const rates = [0, 0.9]
 * let last
 * for (const r of learnerEquitySweep(rates.map(simulate), rates, { models: ['irt', 'zilm'] })) last = r
 * for (const pt of last.points) {
 *   print(`rate ${pt.rate}: structural share with the condition`, pt.structuralShare.with)
 *   for (const m of pt.models) print(`  ${m.model} gap:`, m.equity.gap, 'ability r:', m.ability.pearson)
 * }
 */
export function* learnerEquitySweep(
  datasets: readonly LearnerData[],
  rates: readonly number[],
  options: LearnerRunOptions = {},
): Generator<LearnerSweepResult, LearnerSweepResult> {
  const { models = DEFAULT_MODELS, testShare = 0.2, maxSteps = 1000, seed = 0 } = options
  const points: LearnerSweepPoint[] = []
  const snapshot = (finished: boolean) => ({ points: [...points], total: datasets.length, finished })
  yield snapshot(false)
  for (let j = 0; j < datasets.length; j++) {
    const data = datasets[j]
    const { y, train, test } = split(data.responses, testShare, seed)
    const I = data.responses.shape[1]
    let sw = 0
    let nw = 0
    let so = 0
    let no = 0
    for (let k = 0; k < y.length; k++)
      if (Number.isFinite(y[k])) {
        const s = data.structural ? data.structural[k] : 0
        if (data.group[Math.floor(k / I)] === 0) {
          so += s
          no++
        } else {
          sw += s
          nw++
        }
      }
    let init: LearnerModelFit | undefined
    const evaluated = models.map((model) => {
      const r = evaluate(data, model, y, train, test, maxSteps, init)
      if (model === 'irt') init = r.fit
      return { model, equity: r.equity, ability: r.recovery.ability, test: r.test }
    })
    points.push({
      rate: rates[j],
      structuralShare: { with: nw ? sw / nw : NaN, without: no ? so / no : NaN },
      models: evaluated,
    })
    yield snapshot(j === datasets.length - 1)
  }
  return snapshot(true)
}

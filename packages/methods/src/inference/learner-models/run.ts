/**
 * Experiments with learner models on simulated students (Twomey et al., 2022, §3.3): fit IRT, a linear knowledge-tracing
 * machine and IRT-ZILM to the same training responses and compare ability recovery, bias by group, held-out prediction
 * and, for IRT-ZILM, the posterior that each zero is structural; and a sweep of the zero-inflation gap between groups,
 * one dataset per gap, scoring each model's equity. Both are generators for streamed runs: each yields after every fit.
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
  responses: Tensor
  conditions: Tensor
  itemFeatures: Tensor
  ability: Float64Array
  difficulty: Float64Array
  discrimination: Float64Array
  /** True π [students × items] and which zeros were structural; optional (real data has neither). */
  pi?: Float64Array
  structural?: Uint8Array
  /** Each student's group (0: no condition). */
  group: Int32Array
}

/** Options of `learnerModelRun` and `learnerEquitySweep`. */
export interface LearnerRunOptions {
  /** The models to fit, in order (default irt, ktm, zilm). */
  models?: readonly LearnerModelKind[]
  /** Share of the observed responses held out for the predictive scores (default 0.2). */
  testShare?: number
  /** Most L-BFGS steps per fit (default 1000). */
  maxSteps?: number
  seed?: number | string
}

/** One fitted model with its evaluation. */
export interface LearnerModelResult {
  model: LearnerModelKind
  fit: LearnerModelFit
  equity: AbilityEquity
  recovery: { ability: ParameterRecovery; difficulty: ParameterRecovery; discrimination: ParameterRecovery }
  test: ResponseScores
  /**
   * IRT-ZILM only: for every observed zero, the posterior that it is structural, its fitted π, whether it was
   * structural (1; when the data carry the truth) and the AUROC of the posterior for that truth.
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
  results: LearnerModelResult[]
  total: number
  finished: boolean
  /** The held-out mask (1 test) over the response matrix. */
  test: Uint8Array
  /** The truth the estimates are scored against, and each student's number of conditions. */
  truth: { ability: Float64Array; difficulty: Float64Array; group: Int32Array; conditionCount: Int32Array }
}

const DEFAULT_MODELS: readonly LearnerModelKind[] = ['irt', 'ktm', 'zilm']

/** The training and test masks over the observed responses. */
function split(responses: Tensor, testShare: number, seed: number | string) {
  const y = toFlat(responses)
  const u = units(child(stream(seed), 'split'), y.length)
  const train = new Uint8Array(y.length)
  const test = new Uint8Array(y.length)
  for (let k = 0; k < y.length; k++) if (Number.isFinite(y[k])) (u[k] < testShare ? test : train)[k] = 1
  return { y, train, test }
}

/** Fit one model on the training responses and evaluate it. */
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
 * Fit the learner models to the same training responses (a random `1 − testShare` of the observed ones) and evaluate
 * each: ability bias by group and the equity gap, parameter recovery, held-out scores and (IRT-ZILM) structural-zero
 * posteriors. Yields after every fit. Deterministic in `seed`.
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
  /** The zero-inflation rate r of the conditions in this dataset. */
  rate: number
  /** The realised share of structural zeros among the responses of students with and without a condition. */
  structuralShare: { with: number; without: number }
  models: {
    model: LearnerModelKind
    equity: AbilityEquity
    ability: ParameterRecovery
    test: ResponseScores
  }[]
}

/** An equity sweep so far. */
export interface LearnerSweepResult {
  points: LearnerSweepPoint[]
  total: number
  finished: boolean
}

/**
 * The equity sweep: for each dataset (simulated with zero-inflation rate `rates[j]` for every condition), fit the
 * models and record each one's ability bias by group, equity gap, ability recovery and held-out scores. Yields after
 * every dataset.
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

/**
 * The forward-evaluation study of stochastic vector field mixtures (Twomey, Kozłowski & Santos-Rodríguez, 2020, §4.3,
 * fig. 11): train several models on one dataset, then solve every instance's realised path alone with Dormand–Prince
 * and count its function evaluations (NFE), at several tolerances. The NFE usually reported for a model is that of
 * solving a whole batch as one system, with one step size for all; the per-instance distribution shows how many
 * instances needed fewer. The variance of each instance's VF along its path (VLoss, eq. 9, per instance) is recorded
 * beside its NFE: aligned VFs (eq. 16) leave the embedded error estimate near zero, so large steps pass.
 */

import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import type { SvfmLossSettings } from './losses'
import type { ComponentSelection, Svfm, SvfmParams } from './model'
import { svfm } from './model'
import { svfmRun, type SvfmRunData, type SvfmRunOptions } from './run'
import { instanceWork, realisation, realisedVariance, samplePaths } from './sampling'

/** One model of a study: its label, configuration and losses. */
export type NfeStudyModel = {
  /** Its name in the results. */
  label: string
  /** Components $K$. Default 1. */
  components?: number
  /** SVF units. Default false. */
  stochastic?: boolean
  /** The selection method. Default `'pick-and-stick'`. */
  selection?: ComponentSelection
  /** The losses it trains with. Default the predictive loss alone. */
  losses?: SvfmLossSettings
}

/** The models of fig. 11: VF, VF with TVLoss, and SVFM (here with and without TVLoss). */
export const NFE_STUDY_MODELS: readonly NfeStudyModel[] = [
  { label: 'VF' },
  { label: 'VF + TVLoss', losses: { transport: true, variance: true } },
  { label: 'SVFM', components: 4, stochastic: true },
  { label: 'SVFM + TVLoss', components: 4, stochastic: true, losses: { transport: true, variance: true } },
]

/** Options of {@link nfeStudy}; plain data. */
export type NfeStudyOptions = {
  /** The models, trained in turn. Default `NFE_STUDY_MODELS`. */
  models?: readonly NfeStudyModel[]
  /** Training of every model (task, grid, steps, …; the model fields are set per model). */
  training?: Omit<SvfmRunOptions, 'components' | 'stochastic' | 'selection' | 'losses'>
  /**
   * Relative tolerances of the sweep (the absolute tolerance is a hundredth of each). Default $10^{-2}, 10^{-3}, \dots,
   * 10^{-6}$.
   */
  tolerances?: readonly number[]
  /** Instances measured (spread evenly over the data). Default all, at most 1000. */
  instances?: number
  /** Seed of the runs (model $k$ trains with `seed + k`) and of the realisations. Default 0. */
  seed?: number
}

/** A trained model's measurements. */
export type NfeStudyResult = {
  /** The model's label. */
  label: string
  /** Accuracy on the whole set at the end of training (classification), or NaN. */
  accuracy: number
  /** The unweighted TLoss on the whole set at the end of training. */
  transport: number
  /** The unweighted VLoss on the whole set at the end of training. */
  variance: number
  /** Wall milliseconds of training, checkpoints included. */
  trainMs: number
  /** NFE of each instance alone, one array per tolerance. */
  perInstance: Int32Array[]
  /** NFE of all the instances solved as one system, per tolerance. */
  batch: number[]
  /** The variance of each instance's realised VF along its path. */
  fieldVariance: Float64Array
}

/** A snapshot of a study. */
export type NfeStudy = {
  /** What the study is doing. */
  phase: 'training' | 'measuring' | 'done'
  /** The index of the model being trained or measured. */
  current: number
  /** Its training progress: the steps taken. */
  step: number
  /** The steps of each model's training. */
  steps: number
  /** The models' labels. */
  labels: string[]
  /** The relative tolerances of the sweep. */
  tolerances: number[]
  /** Indices of the measured instances. */
  instances: Int32Array
  /** Their colour groups (filled once the first model is trained). */
  groups: Int32Array
  /** The results of the models measured so far. */
  results: NfeStudyResult[]
  /** The last training error, prefixed by its model's label, or null. */
  error: string | null
}

/**
 * Train each model in turn on `data` with `svfmRun`, then measure per-instance and batch NFE over the tolerances
 * (`instanceWork`) and the realised VF variance (`realisedVariance`, on mean paths for VF units and sampled ones for
 * SVF units). Yields progress while training and after each measurement. A model that ends with no parameters stops the
 * study, which then returns without yielding again.
 *
 * @param data The run data, as for `svfmRun`.
 * @param options The models, their shared training options, the tolerances, the instances and the seed; see
 *   `NfeStudyOptions`.
 * @returns A generator of `NfeStudy` snapshots; the last has `phase` `'done'`.
 *
 * @example TVLoss lowers the variance of each instance's VF along its path
 * const data = endpointTask({ x: tensor([[-1], [-0.5], [0.5], [1]]), y: tensor([-2, -1, 1, 2]) })
 * const models = [{ label: 'VF' }, { label: 'VF + TVLoss', losses: { transport: true, variance: true } }]
 * const options = { models, training: { steps: 10, hidden: 8, grid: 2 }, tolerances: [1e-3] }
 * let study
 * for (const s of nfeStudy(data, options)) study = s
 * for (const r of study.results) {
 *   print(r.label, 'NFE per instance:', r.perInstance[0], ' batch:', r.batch[0], ' VF variance:', r.fieldVariance)
 * }
 */
export function* nfeStudy(data: SvfmRunData, options: NfeStudyOptions = {}): Generator<NfeStudy, NfeStudy> {
  const { models = NFE_STUDY_MODELS, training = {}, tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], seed = 0 } = options
  const n = data.x.shape[0]
  const m = Math.min(options.instances ?? 1000, n)
  const instances = Int32Array.from({ length: m }, (_, i) => Math.round((i * (n - 1)) / Math.max(1, m - 1)))
  const steps = training.steps ?? 300
  const results: NfeStudyResult[] = []
  const state: NfeStudy = {
    phase: 'training',
    current: 0,
    step: 0,
    steps,
    labels: models.map((x) => x.label),
    tolerances: [...tolerances],
    instances,
    groups: new Int32Array(0),
    results,
    error: null,
  }
  const snap = (): NfeStudy => ({ ...state, results: results.slice() })
  for (let k = 0; k < models.length; k++) {
    const spec = models[k]
    state.current = k
    state.phase = 'training'
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now()
    let last = null
    for (const r of svfmRun(data, {
      ...training,
      components: spec.components ?? 1,
      stochastic: spec.stochastic ?? false,
      selection: spec.selection ?? 'pick-and-stick',
      losses: spec.losses ?? {},
      seed: seed + k,
      shown: 8,
      checkpoints: 1,
    })) {
      last = r
      state.step = r.done
      if (r.error) state.error = `${spec.label}: ${r.error}`
      if (!r.finished) yield snap()
    }
    if (!last || !last.params) {
      state.phase = 'done'
      return snap()
    }
    const trainMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0
    state.phase = 'measuring'
    yield snap()
    if (state.groups.length === 0) state.groups = Int32Array.from(instances, (i) => last!.groups[i])
    const model: Svfm = svfm({
      dim: last.dim,
      components: last.components,
      stochastic: last.stochastic,
      selection: last.selection,
      augment: last.stateDim - last.dim,
      context: data.context ? data.context.shape[1] : 0,
      hidden: training.hidden ?? 32,
      layers: training.layers ?? 1,
      activation: training.activation ?? 'relu',
      classes: last.classes,
      grid: last.gridTimes.length - 1,
      maxVariance: training.maxVariance ?? 0.5,
      ...training.architecture,
    })
    const params = last.params as SvfmParams
    const D = last.dim
    const xs = Float64Array.from({ length: m * D }, (_, q) => last!.data[instances[Math.floor(q / D)] * D + (q % D)])
    const C = data.context ? data.context.shape[1] : 0
    const ctxAll = data.context ? Float64Array.from(toFlat(data.context)) : null
    const cs = ctxAll
      ? Float64Array.from({ length: m * C }, (_, q) => ctxAll[instances[Math.floor(q / C)] * C + (q % C)])
      : null
    const real = realisation(child(stream(seed), 'study', k), m, model.stateDim)
    const mode = model.options.stochastic ? 'sample' : 'mean'
    const perInstance: Int32Array[] = []
    const batch: number[] = []
    for (const rtol of tolerances) {
      const w = instanceWork(model, params, xs, cs, real, { mode, rtol, atol: rtol * 1e-2 })
      perInstance.push(w.perInstance)
      batch.push(w.batch)
    }
    const paths = samplePaths(model, params, xs, cs, real, { mode, rtol: 1e-4, atol: 1e-6, framesPerInterval: 1 })
    const fieldVariance = realisedVariance(model, params, paths, cs, real, 1, mode)
    const shot = last.checkpoints[last.checkpoints.length - 1]
    results.push({
      label: spec.label,
      accuracy: shot?.accuracy ?? NaN,
      transport: shot?.transport ?? NaN,
      variance: shot?.variance ?? NaN,
      trainMs,
      perInstance,
      batch,
      fieldVariance,
    })
    yield snap()
  }
  state.phase = 'done'
  const done = snap()
  yield done
  return done
}

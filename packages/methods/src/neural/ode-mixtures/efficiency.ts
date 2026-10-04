/**
 * The forward-evaluation study of stochastic vector field mixtures (Twomey, Kozłowski & Santos-Rodríguez, 2020, §4.3,
 * fig. 11): train several models on one dataset, then solve every instance's realised path alone with Dormand–Prince
 * and count its function evaluations (NFE), at several tolerances. The NFE usually reported for a model is that of
 * solving a whole batch as one system, which the hardest instances set; the per-instance distribution shows how many
 * instances needed far fewer. The variance of each instance's VF along its path (VLoss, eq. 9, per instance) is
 * recorded beside its NFE: aligned VFs (eq. 16) leave the embedded error estimate near zero, so large steps pass.
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
  label: string
  components?: number
  stochastic?: boolean
  selection?: ComponentSelection
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
  models?: readonly NfeStudyModel[]
  /** Training of every model (task, grid, steps, …; the model fields are set per model). */
  training?: Omit<SvfmRunOptions, 'components' | 'stochastic' | 'selection' | 'losses'>
  /** Relative tolerances of the sweep (atol = rtol / 100). Default 1e-2 … 1e-6. */
  tolerances?: readonly number[]
  /** Instances measured (spread over the data). Default all, at most 1000. */
  instances?: number
  seed?: number
}

/** A trained model's measurements. */
export type NfeStudyResult = {
  label: string
  /** Accuracy (classification) or NaN, and the unweighted TLoss and VLoss on the whole set. */
  accuracy: number
  transport: number
  variance: number
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
  phase: 'training' | 'measuring' | 'done'
  /** The model being trained or measured, and its training progress. */
  current: number
  step: number
  steps: number
  labels: string[]
  tolerances: number[]
  /** Indices of the measured instances and their labels (colours). */
  instances: Int32Array
  groups: Int32Array
  results: NfeStudyResult[]
  error: string | null
}

/**
 * Train each model in turn on `data`, then measure per-instance and batch NFE over the tolerances and the realised
 * VF variance. Yields progress while training and after each measurement.
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

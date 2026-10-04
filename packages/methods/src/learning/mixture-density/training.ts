/**
 * Training mixture density networks: `mdnTraining`, Adam on the network's loss as a traceable `trainingLoop`;
 * `mixtureDensityRun`, a generator that trains an MDN and the squared-error network of the same body side by side on
 * one dataset and yields snapshots (curves and parameters at checkpoints) for a page to plot and play; and
 * `mixtureDensityNetwork`, the MDN as a registered estimator whose predictive is the mixture.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import {
  defineModel,
  withExpectation,
  withSampling,
  type FitOptions,
  type Supervised,
} from 'aifn-compute/learning/estimators'
import type { ScaleLink } from 'aifn-compute/learning/losses'
import { methodTraining, trainingLoop, type TrainingMethod, type TrainingState } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import {
  inputMatrix,
  inputStandardisation,
  mdnLogLikelihood,
  standardisedInputs,
  mdnLoss,
  mdnMeanSquaredError,
  mdnModel,
  mdnPredict,
  type MdnConfig,
  type MdnModel,
  type MdnSpec,
} from './model'

/** Inputs [n, d] and targets [n] or [n, D]. */
export type MdnData = { x: Tensor; y: Tensor }

/** Options of `mdnTraining`. */
export type MdnTrainingOptions = {
  /** Adam's step size (default 0.01). */
  stepSize?: number
  /** Rows per step (default all: full-batch). */
  batchSize?: Size
  /** Clip the gradient's global norm (default 10). */
  clipNorm?: number
}

/** Adam on the network's loss (mixture NLL or squared error), as a traceable `trainingLoop`. */
export function mdnTraining(
  model: MdnModel,
  data: MdnData,
  options: MdnTrainingOptions = {},
): Algorithm<{ params: Params[] }, TrainingState<Params[]>> {
  const { stepSize = 0.01, batchSize, clipNorm = 10 } = options
  return trainingLoop<Params[], { x: Tensor; y: Tensor }>({
    loss: (p, b) => mdnLoss(model, p, b.x, b.y),
    data: { x: inputMatrix(data.x), y: data.y },
    batchSize,
    optimizer: adamRule({ stepSize }) as never,
    clipNorm,
  })
}

// ── The side-by-side run ─────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `mixtureDensityRun`. */
export type MdnRunOptions = Omit<MdnConfig, 'inputs' | 'outputs' | 'objective'> & {
  data: MdnData
  /** Adam updates (default 2000). */
  steps?: Size
  /** Keep both networks' parameters every this many steps for the player (default steps/50, at least 1). */
  every?: Size
  stepSize?: number
  batchSize?: Size
  /**
   * How both networks train (default Adam with `stepSize` and `batchSize`); `{ method: 'lbfgs' }` trains them by
   * full-batch L-BFGS, one line-searched step per step.
   */
  method?: TrainingMethod
  /** The root stream's seed (default 'mdn'); both networks start from its child `init`. */
  seed?: string | number
}

/** Curves of a run, one entry per recorded step, on the training data. */
export type MdnHistory = {
  step: number[]
  /** The MDN's negative log-likelihood per row. */
  nll: number[]
  /** The squared-error network's, as a Gaussian with its residual variance. */
  meanNll: number[]
  /** The squared error of the MDN's mean E[y | x]. */
  mse: number[]
  /** The squared-error network's. */
  meanMse: number[]
}

/** Both networks' parameters at one step. */
export type MdnCheckpoint = { readonly step: Size; readonly mixture: Params[]; readonly mean: Params[] }

/** A snapshot of `mixtureDensityRun`. */
export type MdnSnapshot = {
  readonly step: Size
  readonly steps: Size
  readonly done: boolean
  /** The MDN's and the squared-error network's specs (rebuild them with `mdnModel`). */
  readonly spec: MdnSpec
  readonly meanSpec: MdnSpec
  readonly history: MdnHistory
  /** Parameters at step 0, every `every` steps and at the last step. */
  readonly checkpoints: readonly MdnCheckpoint[]
}

/**
 * Train an MDN with K components and the squared-error network with the same hidden layers on `data`, by Adam from
 * the same seed, yielding a snapshot every `every` steps: a generator, so a worker can stream the run to a page.
 */
export function* mixtureDensityRun(options: MdnRunOptions): Generator<MdnSnapshot> {
  const {
    data,
    steps = 2000,
    every: everyOption,
    stepSize = 0.01,
    batchSize,
    method = { method: 'adam', stepSize, batchSize, clipNorm: 10 },
    seed = 'mdn',
    ...structure
  } = options
  const x = inputMatrix(data.x)
  const outputs = data.y.shape.length === 1 ? 1 : data.y.shape[1]
  const scaling = inputStandardisation(x)
  const mdn = mdnModel({ ...structure, ...scaling, inputs: x.shape[1], outputs, objective: 'mixture' })
  const meanNet = mdnModel({ ...structure, ...scaling, inputs: x.shape[1], outputs, objective: 'squared-error' })
  const every = Math.max(1, everyOption ?? Math.round(steps / 50))
  const root = stream(seed)
  const history: MdnHistory = { step: [], nll: [], meanNll: [], mse: [], meanMse: [] }
  const checkpoints: MdnCheckpoint[] = []
  const record = (t: Size, pm: Params[], pe: Params[]) => {
    const a = mdnPredict(mdn, pm, x)
    const b = mdnPredict(meanNet, pe, x)
    history.step.push(t)
    history.nll.push(-mdnLogLikelihood(mdn, a, data.y))
    history.meanNll.push(-mdnLogLikelihood(meanNet, b, data.y))
    history.mse.push(mdnMeanSquaredError(a, data.y))
    history.meanMse.push(mdnMeanSquaredError(b, data.y))
  }
  // A run that stops early (converged L-BFGS) reports its last step as the total.
  let total = steps
  const snapshot = (t: Size, done: boolean): MdnSnapshot => ({
    step: t,
    steps: total,
    done,
    spec: mdn.spec,
    meanSpec: meanNet.spec,
    history: {
      step: [...history.step],
      nll: [...history.nll],
      meanNll: [...history.meanNll],
      mse: [...history.mse],
      meanMse: [...history.meanMse],
    },
    checkpoints: [...checkpoints],
  })
  const train = { x, y: data.y }
  const a = methodTraining((p: Params[], d: MdnData) => mdnLoss(mdn, p, d.x, d.y), train, method)
  const b = methodTraining((p: Params[], d: MdnData) => mdnLoss(meanNet, p, d.x, d.y), train, method)
  let sa = a.init({ params: mdn.init(child(root, 'init', 'mixture')) }, child(root, 'init'))
  let sb = b.init({ params: meanNet.init(child(root, 'init', 'mean')) }, child(root, 'init'))
  record(0, sa.params, sb.params)
  checkpoints.push({ step: 0, mixture: sa.params, mean: sb.params })
  yield snapshot(0, false)
  // Record about 200 points of the curves.
  const recordEvery = Math.max(1, Math.floor(steps / 200))
  for (let t = 0; t < steps; t++) {
    sa = a.step(sa, { t, stream: child(root, 'step', t) })
    sb = b.step(sb, { t, stream: child(root, 'step', t) })
    const k = t + 1
    if (k % recordEvery === 0 || k === steps) record(k, sa.params, sb.params)
    if (k % every === 0 || k === steps) {
      checkpoints.push({ step: k, mixture: sa.params, mean: sb.params })
      yield snapshot(k, k === steps)
    }
    if (sa.diverged || sb.diverged || (sa.stopped && sb.stopped)) {
      total = k
      // Converged L-BFGS (or divergence) ends the run early; its last state is a checkpoint.
      if (k % every !== 0 && k !== steps) {
        if (k % recordEvery !== 0) record(k, sa.params, sb.params)
        checkpoints.push({ step: k, mixture: sa.params, mean: sb.params })
      }
      yield snapshot(k, true)
      return
    }
  }
}

// ── The estimator ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Hyperparameters of `mixtureDensityNetwork`. */
export type MixtureDensityNetworkParams = {
  /** Mixture components K (default 3). */
  components?: Size
  /** Units of the single tanh hidden layer (default 20). */
  hidden?: Size
  /** Adam updates (default 1000). */
  steps?: Size
  /** Adam's step size (default 0.01). */
  stepSize?: number
  /** σ = floor + exp(s) (default) or floor + softplus(s). */
  scale?: ScaleLink
}

/**
 * The mixture density network as an estimator: an MLP with one tanh hidden layer and a K-component Gaussian mixture
 * head, fitted by full-batch Adam on the mixture NLL. Capabilities: `forward` (the head), `decide` (the most probable
 * mode, the answer on an inverse problem where the mean solves nothing), `predictive` (the mixture), `expect` and
 * `sample` (from the predictive).
 */
export function mixtureDensityNetwork(params: MixtureDensityNetworkParams = {}) {
  const { components = 3, hidden = 20, steps = 1000, stepSize = 0.01, scale = 'exp' } = params
  return {
    name: 'mixture-density-network' as const,
    params,
    fit({ x, y }: Supervised<Tensor, Tensor>, options: FitOptions = {}) {
      const xs = inputMatrix(x)
      const outputs = y.shape.length === 1 ? 1 : y.shape[1]
      const model = mdnModel({
        ...inputStandardisation(xs),
        inputs: xs.shape[1],
        outputs,
        components,
        hidden: [hidden],
        scale,
      })
      const s = options.stream ?? stream('mixture-density-network')
      const alg = mdnTraining(model, { x: xs, y }, { stepSize })
      let state = alg.init({ params: model.init(child(s, 'init')) }, child(s, 'init'))
      for (let t = 0; t < steps && !state.diverged; t++) state = alg.step(state, { t, stream: child(s, 'step', t) })
      const fitted = state.params
      const mixture = (input: Tensor) => mdnPredict(model, fitted, inputMatrix(input)).mixture!
      const base = {
        kind: 'model' as const,
        name: 'mixture-density-network' as const,
        spec: model.spec,
        params: fitted,
        mixture,
        forward: (input: Tensor) =>
          model.net.apply(fitted, standardisedInputs(model.spec, inputMatrix(input))) as Tensor,
        decide: (input: Tensor) => {
          const m = mixture(input)
          const out = new Float64Array(m.rows * outputs)
          for (let i = 0; i < m.rows; i++) out.set(m.modes(i)[0].value, i * outputs)
          return fromData(out, outputs === 1 ? [m.rows] : [m.rows, outputs])
        },
        predictive: (input: Tensor) => mixture(input).distribution(),
      }
      return withSampling(withExpectation(base))
    },
  }
}

defineModel(
  {
    key: 'mixtureDensityNetwork',
    module: 'learning/mixture-density',
    name: 'Mixture density network',
    summary:
      'An MLP whose outputs are the weights, means and scales of a Gaussian mixture over the target, fitted by maximum likelihood: a multimodal p(y | x).',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'sample'],
    hyper: space({
      components: int(1, 12, { default: 3 }),
      hidden: int(2, 128, { default: 20 }),
      steps: int(1, 20000, { default: 1000 }),
      stepSize: real(1e-5, 1, { default: 0.01 }),
      scale: oneOf(['exp', 'softplus']),
    }),
    notes: ['gaussian-mixture-model', 'multilayer-perceptron'],
    cite: ['bishop2006'],
  },
  mixtureDensityNetwork,
)

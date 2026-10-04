/**
 * An autoencoder training run as a generator of plain-data snapshots, for a worker to stream: Adam on minibatches by
 * `aifn-compute/nn/training`'s `trainingLoop`, with the loss parts on the whole training set every few steps and, at
 * checkpoints, the codes of the data, reconstructions, samples from the prior (per class for the conditional VAE,
 * from the codebook for the VQ-VAE) and, for a 2-d code, the decoder over a grid of codes.
 */

import type { Params } from 'aifn-compute/foundation/pytree'
import { child, integers, standardNormals, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { oneHot } from 'aifn-compute/learning/losses'
import { assignNearest } from 'aifn-compute/numerics/neighbours'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import {
  autoencoder,
  autoencoderLoss,
  autoencoderDecode,
  autoencoderEncode,
  initAutoencoder,
  type AutoencoderKind,
  type AutoencoderParams,
} from './models'

/** Options of `autoencoderRun`; plain data, so a worker task can carry them. */
export interface AutoencoderRunOptions {
  kind?: AutoencoderKind
  latent?: number
  hidden?: readonly number[]
  codes?: number
  beta?: number
  likelihood?: 'gaussian' | 'bernoulli'
  observationSd?: number
  /** Adam updates (default 2000), its step size (default 2e-3) and rows per step (default 64). */
  steps?: number
  stepSize?: number
  batchSize?: number
  seed?: number | string
  /** Checkpoints besides step 0 (default 30). */
  checkpoints?: number
  /** Rows reconstructed at each checkpoint (default 200), prior samples (default 400), grid cells per side (default 9). */
  shown?: number
  samples?: number
  grid?: number
}

/** One checkpoint. */
export interface AutoencoderCheckpoint {
  step: number
  /** The code (VAE: the mean; VQ-VAE: z_e) of every training row, row-major [n × latent]. */
  codes: Float64Array
  /** Reconstructions of the first `shown` rows, [shown × D]. */
  reconstructions: Float64Array
  /** Samples decoded from the prior, [samples × D], and their labels (CVAE) or codes (VQ-VAE); −1 otherwise. */
  samples: Float64Array
  sampleLabels: Int32Array
  /** The decoder over a grid of codes on [−3, 3]² (latent 2 only), [grid² × D], row-major in (z₂, z₁). */
  grid: Float64Array | null
  /** The codebook, [codes × latent] (VQ-VAE). */
  codebook: Float64Array | null
  /** The codebook entries used by the data (VQ-VAE). */
  used: number
}

/** A run so far. */
export interface AutoencoderRun {
  kind: AutoencoderKind
  steps: number
  done: number
  finished: boolean
  dimension: number
  latent: number
  gridSize: number
  /** The training rows [n × D] and their labels (0 when the data has none). */
  data: Float64Array
  labels: Int32Array
  /** Loss parts on the whole training set: step, reconstruction, regulariser (KL or VQ terms), total. */
  history: { step: number[]; reconstruction: number[]; regulariser: number[]; total: number[] }
  checkpoints: AutoencoderCheckpoint[]
}

const scalar = (v: Value) => {
  const u = unwrap(v)
  return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
}

/** Train an autoencoder on data { x [n, D], y? } and yield snapshots (module docs). Deterministic in `seed`. */
export function* autoencoderRun(
  data: { x: Tensor; y?: Tensor },
  options: AutoencoderRunOptions = {},
): Generator<AutoencoderRun, AutoencoderRun> {
  const { kind = 'vae', steps = 2000, stepSize = 2e-3, batchSize = 64, seed = 0, checkpoints = 30 } = options
  const { shown = 200, samples = 400, grid: gridSize = 9 } = options
  const x = data.x
  const [n, D] = x.shape
  const yInt = data.y ? Int32Array.from(toFlat(data.y)) : new Int32Array(n)
  const classes = data.y ? Math.max(...yInt) + 1 : 0
  const model = autoencoder({
    kind,
    inputs: D,
    latent: options.latent,
    hidden: options.hidden,
    codes: options.codes,
    beta: options.beta,
    likelihood: options.likelihood,
    observationSd: options.observationSd,
    classes,
  })
  const L = model.spec.latent
  const dataRows = Float64Array.from(toFlat(x))
  const labels = kind === 'cvae' ? oneHot(yInt, classes) : undefined
  const root = stream(seed)
  const train: Record<string, Tensor> = labels ? { x, labels } : { x }
  const alg = trainingLoop<AutoencoderParams & Params, Record<string, Tensor>>({
    loss: (p, b, ctx) => autoencoderLoss(model, p, b.x, ctx.stream ?? child(root, 'eps'), b.labels).loss,
    data: train,
    batchSize: Math.min(batchSize, n),
    optimizer: adamRule({ stepSize }) as UpdateRule<unknown>,
    clipNorm: 10,
  })
  const history = {
    step: [] as number[],
    reconstruction: [] as number[],
    regulariser: [] as number[],
    total: [] as number[],
  }
  const shots: AutoencoderCheckpoint[] = []
  const record = (t: number, p: AutoencoderParams) => {
    const parts = autoencoderLoss(model, p, x, child(root, 'record'), labels)
    history.step.push(t)
    history.reconstruction.push(scalar(parts.reconstruction))
    history.regulariser.push(scalar(parts.regulariser))
    history.total.push(scalar(parts.loss))
  }
  const prior = fromData(standardNormals(child(root, 'prior'), samples * L), [samples, L])
  const priorLabels = Int32Array.from({ length: samples }, (_, i) => (classes > 0 ? i % classes : -1))
  const priorCodes = Int32Array.from(toFlat(integers(child(root, 'codes'), model.spec.codes, { shape: [samples] })))
  const g = gridSize
  const gridCodes =
    L === 2 && g > 0
      ? fromData(
          Float64Array.from({ length: g * g * 2 }, (_, k) => {
            const cell = k >> 1
            const v = (i: number) => -3 + (6 * i) / (g - 1)
            return k % 2 === 0 ? v(cell % g) : v(Math.floor(cell / g))
          }),
          [g * g, 2],
        )
      : null
  const checkpoint = (t: number, p: AutoencoderParams): AutoencoderCheckpoint => {
    const enc = autoencoderEncode(model, p, x, labels)
    const mu = unwrap(enc.mean) as Tensor
    const codes = Float64Array.from(toFlat(mu))
    const m = Math.min(shown, n)
    let used = 0
    let zRecon: Tensor = fromData(codes.slice(0, m * L), [m, L])
    if (kind === 'vqvae') {
      const k = toFlat(assignNearest(mu, p.codebook).labels)
      used = new Set(k).size
      zRecon = fromData(
        Float64Array.from({ length: m * L }, (_, q) => toFlat(p.codebook)[k[Math.floor(q / L)] * L + (q % L)]),
        [m, L],
      )
    }
    const labelRows = (rows: ArrayLike<number>) => (labels ? oneHot(rows, classes) : undefined)
    const reconstructions = Float64Array.from(toFlat(autoencoderDecode(model, p, zRecon, labelRows(yInt.slice(0, m)))))
    let sampleZ: Tensor = prior
    if (kind === 'vqvae') {
      const C = toFlat(p.codebook)
      sampleZ = fromData(
        Float64Array.from({ length: samples * L }, (_, q) => C[priorCodes[Math.floor(q / L)] * L + (q % L)]),
        [samples, L],
      )
    } else if (kind === 'autoencoder') {
      // An autoencoder has no prior: draw from a Gaussian fitted to the data's codes.
      const mean = new Float64Array(L)
      const sd = new Float64Array(L)
      for (let i = 0; i < n; i++) for (let j = 0; j < L; j++) mean[j] += codes[i * L + j] / n
      for (let i = 0; i < n; i++) for (let j = 0; j < L; j++) sd[j] += (codes[i * L + j] - mean[j]) ** 2 / n
      const P = toFlat(prior)
      sampleZ = fromData(
        Float64Array.from(P, (v, q) => mean[q % L] + Math.sqrt(sd[q % L]) * v),
        [samples, L],
      )
    }
    return {
      step: t,
      codes,
      reconstructions,
      samples: Float64Array.from(toFlat(autoencoderDecode(model, p, sampleZ, labelRows(priorLabels)))),
      sampleLabels: kind === 'vqvae' ? priorCodes : priorLabels,
      grid: gridCodes && kind !== 'cvae' ? Float64Array.from(toFlat(autoencoderDecode(model, p, gridCodes))) : null,
      codebook: kind === 'vqvae' ? Float64Array.from(toFlat(p.codebook)) : null,
      used,
    }
  }
  const every = Math.max(1, Math.round(steps / checkpoints))
  const recordEvery = Math.max(1, Math.floor(steps / 200))
  const chunk = Math.max(1, Math.ceil(steps / 20))
  const snapshot = (done: number, finished: boolean): AutoencoderRun => ({
    kind,
    steps,
    done,
    finished,
    dimension: D,
    latent: L,
    gridSize: gridCodes ? g : 0,
    data: dataRows,
    labels: yInt,
    history: {
      step: [...history.step],
      reconstruction: [...history.reconstruction],
      regulariser: [...history.regulariser],
      total: [...history.total],
    },
    checkpoints: shots.slice(),
  })
  let state = alg.init(
    { params: initAutoencoder(model, child(root, 'init')) as AutoencoderParams & Params },
    child(root, 'init'),
  )
  record(0, state.params)
  shots.push(checkpoint(0, state.params))
  yield snapshot(0, false)
  for (let t = 0; t < steps; t++) {
    state = alg.step(state, { t, stream: child(root, 'step', t) })
    const k = t + 1
    if (k % recordEvery === 0 || k === steps) record(k, state.params)
    if (k % every === 0 || k === steps) shots.push(checkpoint(k, state.params))
    if (k === steps || state.diverged) {
      const last = snapshot(k, true)
      yield last
      return last
    }
    if (k % chunk === 0) yield snapshot(k, false)
  }
  return snapshot(steps, true)
}

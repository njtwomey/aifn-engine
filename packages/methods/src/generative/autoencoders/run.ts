/**
 * An autoencoder training run as a generator of plain-data snapshots, for a worker to stream: Adam on minibatches by
 * `aifn-compute/nn/training`'s `trainingLoop`, with the loss parts on the whole training set every few steps and, at
 * checkpoints, the codes of the data, reconstructions, samples from the prior (per class for the conditional VAE,
 * from the codebook for the VQ-VAE and the RQ-VAE) and, for a 2-d code, the decoder over a grid of codes.
 *
 * An RQ-VAE run also records what its hierarchy needs: each stage's latent error over training (how much of the code
 * each stage has learnt to carry), and at checkpoints every row's code stage by stage, the reconstructions from the
 * first $d$ stages for each $d$ (coarse to fine), the codewords used per stage, the number of distinct codes, and the
 * tree of code prefixes (`codePrefixTree`), ready for an icicle or sunburst plot.
 */

import type { Params } from 'aifn-compute/foundation/pytree'
import { child, integers, standardNormals, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { oneHot } from 'aifn-compute/learning/losses'
import { assignNearest, codePrefixTree, type CodeTreeNode } from 'aifn-compute/numerics/neighbours'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import {
  autoencoder,
  autoencoderLoss,
  autoencoderDecode,
  autoencoderEncode,
  autoencoderQuantise,
  initAutoencoder,
  initQuantiserCodebook,
  latentOfCodes,
  type AutoencoderKind,
  type AutoencoderParams,
} from './models'

/** Options of `autoencoderRun`; plain data, so a worker task can carry them. */
export interface AutoencoderRunOptions {
  kind?: AutoencoderKind
  latent?: number
  hidden?: readonly number[]
  codes?: number
  /** The RQ-VAE's stages (default 4) and whether they share one codebook (default false: one per stage). */
  depth?: number
  sharedCodebook?: boolean
  /**
   * How the codebooks start: drawn N(0, 1) (`'normal'`, the VQ-VAE's default) or from $k$-means of the initial encoder's
   * outputs (`'data'`, the RQ-VAE's default; `initQuantiserCodebook`).
   */
  codebookInit?: 'normal' | 'data'
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
  /** The code (VAE: the mean; VQ-VAE and RQ-VAE: $\zvec_e$) of every training row, row-major $[n \times L]$. */
  codes: Float64Array
  /** Reconstructions of the first `shown` rows, $[\mathrm{shown} \times p]$ for inputs of width $p$. */
  reconstructions: Float64Array
  /**
   * Samples decoded from the prior, $[\mathrm{samples} \times p]$, and their labels (CVAE), codes (VQ-VAE) or first-stage
   * codes (RQ-VAE); $-1$ otherwise.
   */
  samples: Float64Array
  sampleLabels: Int32Array
  /** The decoder over a grid of codes on $[-3, 3]^2$ (latent 2 only), $[g^2 \times p]$, row-major in $(z_2, z_1)$. */
  grid: Float64Array | null
  /** The codebooks, $[B K \times L]$ (VQ-VAE and RQ-VAE; `AutoencoderParams` gives the layout). */
  codebook: Float64Array | null
  /** The codebook entries used by the data (VQ-VAE; RQ-VAE: summed over its codebooks). */
  used: number
  /** RQ-VAE: every row's code, stage by stage, $[n \times D]$; null otherwise. */
  stageCodes: Int32Array | null
  /** RQ-VAE: reconstructions of the first `shown` rows from the first $d$ stages, for $d = 1, \dots, D$; null otherwise. */
  partialReconstructions: Float64Array[] | null
  /** RQ-VAE: the codewords each stage uses; null otherwise. */
  usedByStage: number[] | null
  /** RQ-VAE: the number of distinct whole codes among the rows (at most $K^D$); 0 otherwise. */
  uniqueCodes: number
  /** RQ-VAE: the tree of the rows' code prefixes (`codePrefixTree`); null otherwise. */
  codeTree: CodeTreeNode[] | null
}

/** A run so far. */
export interface AutoencoderRun {
  kind: AutoencoderKind
  steps: number
  done: number
  finished: boolean
  dimension: number
  latent: number
  /** Quantisation stages $D$ (1 but for an RQ-VAE) and codewords per codebook $K$. */
  depth: number
  codewords: number
  gridSize: number
  /** The training rows $[n \times p]$ and their labels (0 when the data has none). */
  data: Float64Array
  labels: Int32Array
  /**
   * Loss parts on the whole training set: step, reconstruction, regulariser (KL or VQ terms), total; for an RQ-VAE
   * also `stageError`, the mean $\lVert \zvec_e - \hat\zvec^{(d)} \rVert^2$ after each stage $d$ at every record.
   */
  history: { step: number[]; reconstruction: number[]; regulariser: number[]; total: number[]; stageError: number[][] }
  checkpoints: AutoencoderCheckpoint[]
}

/**
 * A rank-0 value as a number.
 *
 * @param v The value.
 * @returns Its number.
 */
const scalar = (v: Value) => {
  const u = unwrap(v)
  return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
}

/**
 * Train an autoencoder on data and yield snapshots of the run so far, about twenty over the run and the finished one
 * last: Adam on minibatches by `trainingLoop`, with the loss parts on the whole training set every few steps and
 * checkpoints (codes, reconstructions, prior samples, the decoded code grid, and for a VQ-VAE or RQ-VAE the codebooks
 * and codes) about `checkpoints` times. Deterministic in `seed`.
 *
 * @param data The training rows `x` $[n, p]$ and, for a conditional VAE, their integer labels `y`.
 * @param options The model (`kind`, widths, codebook, $\beta$, likelihood), the optimisation (steps, step size, batch
 *   size, seed) and what checkpoints record (how many, and how many rows, samples and grid cells).
 * @returns A generator of `AutoencoderRun` snapshots; the last has `finished` set.
 *
 * @example An RQ-VAE learns its stages: each one's error falls, and the code tree fills out
 * // Points of unit scale: a likelihood of the same scale (observationSd 1) keeps the commitment term in play.
 * const s = stream(5)
 * const x = normal(s, 0, 1, { shape: [200, 2] })
 * const options = { kind: 'rqvae', latent: 2, codes: 4, depth: 3, observationSd: 1, steps: 150, seed: 5 }
 * let run
 * for (const r of autoencoderRun({ x }, options)) run = r
 * print('latent error after stages 1, 2, 3, at the start:', run.history.stageError[0])
 * print('… and at the end:', run.history.stageError.at(-1))
 * const tree = run.checkpoints.at(-1).codeTree
 * print('distinct prefixes at depths 1, 2, 3:', [1, 2, 3].map((d) => tree.filter((node) => node.depth === d).length))
 */
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
    depth: options.depth,
    sharedCodebook: options.sharedCodebook,
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
    stageError: [] as number[][],
  }
  const { depth: depthOf, codes: K } = model.spec
  const shots: AutoencoderCheckpoint[] = []
  const record = (t: number, p: AutoencoderParams) => {
    const parts = autoencoderLoss(model, p, x, child(root, 'record'), labels)
    history.step.push(t)
    history.reconstruction.push(scalar(parts.reconstruction))
    history.regulariser.push(scalar(parts.regulariser))
    history.total.push(scalar(parts.loss))
    if (kind === 'rqvae') {
      const ze = toFlat(unwrap(autoencoderEncode(model, p, x).mean) as Tensor)
      const { partials } = autoencoderQuantise(model, p, x)
      history.stageError.push(partials.map((part) => toFlat(part).reduce((a, v, q) => a + (ze[q] - v) ** 2, 0) / n))
    }
  }
  const prior = fromData(standardNormals(child(root, 'prior'), samples * L), [samples, L])
  const priorLabels = Int32Array.from({ length: samples }, (_, i) => (classes > 0 ? i % classes : -1))
  const priorCodes = Int32Array.from(toFlat(integers(child(root, 'codes'), model.spec.codes, { shape: [samples] })))
  // RQ-VAE samples: a uniform code at every stage.
  const priorStageCodes =
    kind === 'rqvae'
      ? fromData(Int32Array.from(toFlat(integers(child(root, 'stage-codes'), K, { shape: [samples * depthOf] }))), [
          samples,
          depthOf,
        ])
      : null
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
    let rq: Pick<
      AutoencoderCheckpoint,
      'stageCodes' | 'partialReconstructions' | 'usedByStage' | 'uniqueCodes' | 'codeTree'
    > = {
      stageCodes: null,
      partialReconstructions: null,
      usedByStage: null,
      uniqueCodes: 0,
      codeTree: null,
    }
    if (kind === 'rqvae') {
      const q = autoencoderQuantise(model, p, x)
      const stageCodes = Int32Array.from(toFlat(q.codes))
      const firstRows = (t: Tensor) => fromData(Float64Array.from(toFlat(t)).slice(0, m * L), [m, L])
      zRecon = firstRows(q.quantised)
      const usedByStage = Array.from(
        { length: depthOf },
        (_, d) => new Set(Array.from({ length: n }, (_, i) => stageCodes[i * depthOf + d])).size,
      )
      used = model.spec.sharedCodebook ? new Set(stageCodes).size : usedByStage.reduce((a, b) => a + b, 0)
      rq = {
        stageCodes,
        partialReconstructions: q.partials.map((part) =>
          Float64Array.from(toFlat(autoencoderDecode(model, p, firstRows(part)))),
        ),
        usedByStage,
        uniqueCodes: new Set(
          Array.from({ length: n }, (_, i) => stageCodes.slice(i * depthOf, (i + 1) * depthOf).join(',')),
        ).size,
        codeTree: codePrefixTree(q.codes),
      }
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
    } else if (priorStageCodes) {
      sampleZ = latentOfCodes(model, p, priorStageCodes)
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
      sampleLabels:
        kind === 'vqvae'
          ? priorCodes
          : priorStageCodes
            ? Int32Array.from({ length: samples }, (_, i) => toFlat(priorStageCodes)[i * depthOf])
            : priorLabels,
      grid: gridCodes && kind !== 'cvae' ? Float64Array.from(toFlat(autoencoderDecode(model, p, gridCodes))) : null,
      codebook: kind === 'vqvae' || kind === 'rqvae' ? Float64Array.from(toFlat(p.codebook)) : null,
      used,
      ...rq,
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
    depth: depthOf,
    codewords: K,
    gridSize: gridCodes ? g : 0,
    data: dataRows,
    labels: yInt,
    history: {
      step: [...history.step],
      reconstruction: [...history.reconstruction],
      regulariser: [...history.regulariser],
      total: [...history.total],
      stageError: history.stageError.map((e) => [...e]),
    },
    checkpoints: shots.slice(),
  })
  const fresh = initAutoencoder(model, child(root, 'init'))
  const codebookInit = options.codebookInit ?? (kind === 'rqvae' ? 'data' : 'normal')
  const start = codebookInit === 'data' ? initQuantiserCodebook(model, fresh, x, child(root, 'codebook')) : fresh
  let state = alg.init({ params: start as AutoencoderParams & Params }, child(root, 'init'))
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

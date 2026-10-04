/**
 * A tiny CLIP (Radford et al., 2021, "Learning Transferable Visual Models From Natural Language Supervision"): two
 * encoders map two views of the same objects (an image and its caption, say) into one shared space, where each is
 * normalised onto the unit sphere; training by the symmetric InfoNCE loss over in-batch negatives, with a learnable
 * temperature, pulls each object's two embeddings together and pushes the batch's other pairs apart. A class is then
 * recognised "zero-shot" by encoding its description with the second encoder and picking the nearest.
 *
 * The encoders are small MLPs (`aifn-compute/nn`), the loss and temperature are `aifn-compute/learning/losses`' `infoNce` and
 * `learnedTemperature`, training is `aifn-compute/nn`'s `trainingLoop` with Adam, and the embedding is scored by
 * `aifn-compute/learning/metrics`' `alignment` and `uniformity` (Wang & Isola, 2020).
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, stream, type Stream } from 'aifn-compute/foundation/random'
import { div, fromData, norm, take, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { infoNce, learnedTemperature } from 'aifn-compute/learning/losses'
import { alignment, uniformity } from 'aifn-compute/learning/metrics'
import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { trainingLoop, type TrainingState } from 'aifn-compute/nn/training'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { adamRule } from 'aifn-compute/optim/first-order'

// ── Data ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Two views of the same objects, row by row: `a` [n, dA] and `b` [n, dB]. For zero-shot evaluation, `prototypes`
 * [K, dB] describes each class in the B view, and `truth.combination` gives each row's class and `truth.heldOut` marks
 * rows of classes left out of training (the shape of `aifn-methods/data`'s paired views).
 */
export type ContrastivePairs = {
  readonly a: Tensor
  readonly b: Tensor
  readonly prototypes?: Tensor
  readonly truth?: { readonly combination: Tensor; readonly heldOut: Tensor }
}

// ── The model ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The architecture of a two-tower model. */
export type TwoTowerConfig = {
  /** Input sizes of the two views. */
  readonly inA: Size
  readonly inB: Size
  /** Hidden units of each encoder's one hidden layer (default 32). */
  readonly hidden?: Size
  /** The shared embedding's dimension (default 2, so the embedding lies on the unit circle). */
  readonly dim?: Size
}

/** A two-tower model's parameters: the two encoders and the log logit scale log(1/τ). */
export type TwoTowerParams = {
  readonly a: Params[]
  readonly b: Params[]
  /** log(1/τ), shape [1]: trained when the temperature is learned, otherwise left at its initial value. */
  readonly logScale: Tensor
}

/** A two-tower model: two MLP encoders into a shared `dim`-dimensional space. */
export type TwoTower = {
  readonly config: Required<TwoTowerConfig>
  readonly encoderA: Layer<Params[]>
  readonly encoderB: Layer<Params[]>
  /** Fresh parameters, with τ at `temperature` (default 0.1). */
  init(s: Stream, temperature?: number): TwoTowerParams
}

/** Two MLP encoders (input → hidden → dim, ReLU) for the two views (CLIP's image and text towers, in miniature). */
export function TwoTower(config: TwoTowerConfig): TwoTower {
  const full = { inA: config.inA, inB: config.inB, hidden: config.hidden ?? 32, dim: config.dim ?? 2 }
  const encoderA = Mlp([full.inA, full.hidden, full.dim])
  const encoderB = Mlp([full.inB, full.hidden, full.dim])
  return {
    config: full,
    encoderA,
    encoderB,
    init: (s, temperature = 0.1) => ({
      a: encoderA.init(child(s, 'a')),
      b: encoderB.init(child(s, 'b')),
      logScale: fromData(Float64Array.of(Math.log(1 / temperature)), [1]),
    }),
  }
}

/** The temperature setting: a fixed τ, or `learned` from an initial τ (CLIP's log(1/τ) parameter). */
export type TemperatureSetting = number | 'learned'

/** Options of `contrastiveLoss`. */
export type ContrastiveLossOptions = {
  /** A fixed τ, or `learned` (default) to use the model's trained log(1/τ). */
  temperature?: TemperatureSetting
  /** The cap on the learned logit scale 1/τ (default 100, CLIP's). */
  maxScale?: number
}

/** The temperature of parameters `p` under a setting, as a value (traced when `p` is). */
export function temperatureOf(p: TwoTowerParams, setting: TemperatureSetting = 'learned', maxScale = 100): Value {
  return setting === 'learned' ? learnedTemperature(p.logScale, { maxScale }) : setting
}

/**
 * CLIP's loss on a batch of pairs: the symmetric InfoNCE of the two towers' embeddings, the rest of the batch serving
 * as negatives in both directions.
 */
export function contrastiveLoss(
  model: TwoTower,
  p: TwoTowerParams,
  batch: { a: Value; b: Value },
  { temperature = 'learned', maxScale = 100 }: ContrastiveLossOptions = {},
): Value {
  const za = model.encoderA.apply(p.a, batch.a)
  const zb = model.encoderB.apply(p.b, batch.b)
  return infoNce(za, zb, { temperature: temperatureOf(p, temperature, maxScale), symmetric: true })
}

/** A number or one-element value as a number. */
function scalarOf(v: Value): number {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

/** Rows divided by their norms, as a plain tensor. */
function unit(z: Value): Tensor {
  return unwrap(div(z, norm(z, -1, true))) as Tensor
}

/** The unit-norm embeddings of rows of view A [n, dA] (`view: 'a'`) or of view B [n, dB], [n, dim]. */
export function embed(model: TwoTower, p: TwoTowerParams, x: Tensor, view: 'a' | 'b'): Tensor {
  return unit(view === 'a' ? model.encoderA.apply(p.a, x) : model.encoderB.apply(p.b, x))
}

/** Cosine similarities [n, m] between unit-norm embeddings `za` [n, d] and `zb` [m, d]. */
export function similarities(za: Tensor, zb: Tensor): Float64Array {
  const [n, d] = za.shape
  const m = zb.shape[0]
  const a = toFlat(za)
  const b = toFlat(zb)
  const out = new Float64Array(n * m)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let s = 0
      for (let k = 0; k < d; k++) s += a[i * d + k] * b[j * d + k]
      out[i * m + j] = s
    }
  return out
}

/** The `k` rows of `keys` [m, d] most similar to `query` [d] (unit-norm), best first, with their cosine similarities. */
export function retrieve(query: ArrayLike<number>, keys: Tensor, k: Size): { index: number[]; similarity: number[] } {
  const [m, d] = keys.shape
  const v = toFlat(keys)
  const sims = Array.from({ length: m }, (_, j) => {
    let s = 0
    for (let c = 0; c < d; c++) s += query[c] * v[j * d + c]
    return s
  })
  const index = sims
    .map((_, j) => j)
    .sort((x, y) => sims[y] - sims[x])
    .slice(0, k)
  return { index, similarity: index.map((j) => sims[j]) }
}

/**
 * Zero-shot classification: each row of `za` [n, d] (A-view embeddings) is given the class whose B-view prototype
 * embedding (`zp` [K, d]) is most similar, among `candidates` (default every class). Returns the class per row (int32).
 */
export function zeroShot(za: Tensor, zp: Tensor, candidates?: readonly number[]): Int32Array {
  const K = zp.shape[0]
  const classes = candidates ?? Array.from({ length: K }, (_, k) => k)
  const s = similarities(za, zp)
  const n = za.shape[0]
  const out = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    let best = classes[0]
    for (const k of classes) if (s[i * K + k] > s[i * K + best]) best = k
    out[i] = best
  }
  return out
}

// ── Evaluation ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** How good an embedding is on held-back pairs. */
export type ContrastiveScores = {
  /** The symmetric InfoNCE of the whole evaluation set as one batch, at the current temperature. */
  readonly loss: number
  /** Wang & Isola's alignment E‖za − zb‖² over pairs (0 … 4). */
  readonly alignment: number
  /** Wang & Isola's uniformity, the mean over the two views of log E exp(−2‖z − z′‖²) (−8 … 0 on the circle). */
  readonly uniformity: number
  /** Zero-shot accuracy among every class, on rows of classes seen in training (NaN without prototypes). */
  readonly zeroShotSeen: number
  /** The same on rows of held-out classes (generalised zero-shot: every class is a candidate; NaN when none). */
  readonly zeroShotHeldOut: number
  /** Accuracy on rows of held-out classes when only the held-out classes are candidates (classic zero-shot). */
  readonly zeroShotHeldOutOnly: number
  /**
   * Top-1 retrieval: the share of A rows whose most similar B row describes the same object (with a truth: an object
   * of the same class, since several objects share a description).
   */
  readonly retrievalTop1: number
}

/** The share of rows in `rows` where `pred` equals `cls` (NaN for none). */
function accuracy(pred: ArrayLike<number>, cls: ArrayLike<number>, rows: readonly number[]): number {
  if (rows.length === 0) return NaN
  return rows.filter((i) => pred[i] === cls[i]).length / rows.length
}

/** Score parameters on evaluation pairs (embeddings, InfoNCE, alignment, uniformity, zero-shot and retrieval). */
export function scoreEmbedding(
  model: TwoTower,
  p: TwoTowerParams,
  data: ContrastivePairs,
  options: ContrastiveLossOptions = {},
): ContrastiveScores {
  const za = embed(model, p, data.a, 'a')
  const zb = embed(model, p, data.b, 'b')
  const loss = scalarOf(contrastiveLoss(model, p, { a: data.a, b: data.b }, options))
  const n = za.shape[0]
  const cls = data.truth ? toFlat(data.truth.combination) : Array.from({ length: n }, (_, i) => i)
  const s = similarities(za, zb)
  let hits = 0
  for (let i = 0; i < n; i++) {
    let best = 0
    for (let j = 1; j < n; j++) if (s[i * n + j] > s[i * n + best]) best = j
    if (cls[best] === cls[i]) hits++
  }
  let seen = NaN
  let heldOut = NaN
  let heldOutOnly = NaN
  if (data.prototypes && data.truth) {
    const zp = embed(model, p, data.prototypes, 'b')
    const held = toFlat(data.truth.heldOut)
    const all = Array.from({ length: n }, (_, i) => i)
    const heldRows = all.filter((i) => held[i] === 1)
    const pred = zeroShot(za, zp)
    seen = accuracy(
      pred,
      cls,
      all.filter((i) => held[i] === 0),
    )
    heldOut = accuracy(pred, cls, heldRows)
    const heldClasses = [...new Set(heldRows.map((i) => cls[i]))].sort((x, y) => x - y)
    if (heldClasses.length > 0) heldOutOnly = accuracy(zeroShot(za, zp, heldClasses), cls, heldRows)
  }
  return {
    loss,
    alignment: alignment(za, zb),
    uniformity: 0.5 * (uniformity(za) + uniformity(zb)),
    zeroShotSeen: seen,
    zeroShotHeldOut: heldOut,
    zeroShotHeldOutOnly: heldOutOnly,
    retrievalTop1: hits / n,
  }
}

// ── Training ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `contrastiveTraining` and `contrastiveTrainingRun`. */
export type ContrastiveTrainingOptions = ContrastiveLossOptions & {
  /** Hidden units per encoder (default 32). */
  hidden?: Size
  /** Embedding dimension (default 2). */
  dim?: Size
  /** Pairs per step (default 32): each pair's negatives are the batch's other B − 1 pairs. */
  batchSize?: Size
  /** Adam's step size (default 0.01). */
  stepSize?: number
  /** The initial τ (default 0.1; the fixed τ when `temperature` is a number). */
  initialTemperature?: number
}

/**
 * CLIP training as a traceable `trainingLoop`: minibatch Adam on the symmetric InfoNCE of shuffled batches of pairs.
 * With a fixed temperature, log(1/τ) gets a gradient too but its value is not read, so it has no effect.
 */
export function contrastiveTraining(
  model: TwoTower,
  data: ContrastivePairs,
  options: ContrastiveTrainingOptions = {},
): Algorithm<{ params: TwoTowerParams }, TrainingState<TwoTowerParams>> {
  return trainingLoop<TwoTowerParams, { a: Tensor; b: Tensor }>({
    loss: (p, batch) => contrastiveLoss(model, p, batch, options),
    data: { a: data.a, b: data.b },
    batchSize: Math.min(options.batchSize ?? 32, data.a.shape[0]),
    optimizer: adamRule({ stepSize: options.stepSize ?? 0.01 }) as never,
  })
}

/** One saved point of a training run: the parameters after `step` updates and how they score. */
export type ContrastiveCheckpoint = ContrastiveScores & {
  readonly step: Size
  readonly temperature: number
  readonly params: TwoTowerParams
}

/** A snapshot of `contrastiveTrainingRun`: the run so far. */
export type ContrastiveSnapshot = {
  readonly step: Size
  readonly steps: Size
  readonly config: Required<TwoTowerConfig>
  /** The minibatch loss and the temperature at every step so far (step 0 first). */
  readonly losses: readonly number[]
  readonly temperatures: readonly number[]
  /** Checkpoints from step 0, every `every` steps and at the end. */
  readonly checkpoints: readonly ContrastiveCheckpoint[]
}

/** Options of `contrastiveTrainingRun`. */
export type ContrastiveRunOptions = ContrastiveTrainingOptions & {
  /** Adam steps (default 600). */
  steps?: Size
  /** Steps between checkpoints (default 20). */
  every?: Size
  /** The root stream's seed (default 'tiny-clip'). */
  seed?: number | string
  /** Evaluate on at most this many evaluation pairs (default 300), so that checkpoints stay cheap. */
  evaluate?: Size
}

/** The first `m` rows of evaluation pairs. */
function head(data: ContrastivePairs, m: number): ContrastivePairs {
  const n = data.a.shape[0]
  if (m >= n) return data
  const ids = Array.from({ length: m }, (_, i) => i)
  const rows = (t: Tensor) => unwrap(take(t, ids)) as Tensor
  return {
    a: rows(data.a),
    b: rows(data.b),
    prototypes: data.prototypes,
    truth: data.truth && { combination: rows(data.truth.combination), heldOut: rows(data.truth.heldOut) },
  }
}

/**
 * Train a tiny CLIP on `train` pairs, yielding a snapshot every `every` steps (a generator, so a worker can stream the
 * run to a page): the loss and temperature of every step, and checkpoints of the parameters scored on `test` pairs
 * (alignment, uniformity, zero-shot accuracy on seen and held-out classes, top-1 retrieval). Step 0 is the untrained
 * model.
 */
export function* contrastiveTrainingRun(
  train: ContrastivePairs,
  test: ContrastivePairs,
  options: ContrastiveRunOptions = {},
): Generator<ContrastiveSnapshot> {
  const { steps = 600, every = 20, seed = 'tiny-clip', evaluate = 300, hidden, dim } = options
  const model = TwoTower({ inA: train.a.shape[1], inB: train.b.shape[1], hidden, dim })
  const root = stream(seed)
  const alg = contrastiveTraining(model, train, options)
  const evalSet = head(test, evaluate)
  const tauOf = (p: TwoTowerParams) => scalarOf(temperatureOf(p, options.temperature, options.maxScale))
  let state: TrainingState<TwoTowerParams> = alg.init(
    { params: model.init(child(root, 'init'), options.initialTemperature ?? 0.1) },
    child(root, 'init'),
  )
  const losses = [state.loss]
  const temperatures = [tauOf(state.params)]
  const checkpoint = (): ContrastiveCheckpoint => ({
    step: state.t,
    temperature: tauOf(state.params),
    params: state.params,
    ...scoreEmbedding(model, state.params, evalSet, options),
  })
  const checkpoints = [checkpoint()]
  const snapshot = (): ContrastiveSnapshot => ({
    step: state.t,
    steps,
    config: model.config,
    losses: [...losses],
    temperatures: [...temperatures],
    checkpoints: [...checkpoints],
  })
  yield snapshot()
  for (let t = 0; t < steps; t++) {
    state = alg.step(state, { t, stream: child(root, 'step', t) })
    losses.push(state.loss)
    temperatures.push(tauOf(state.params))
    if ((t + 1) % every === 0 || t + 1 === steps) {
      checkpoints.push(checkpoint())
      yield snapshot()
    }
  }
}

// ── Ablation ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** One finished run of an ablation: its setting and its final scores. */
export type ContrastiveAblationRun = ContrastiveScores & {
  readonly batchSize: Size
  readonly temperature: TemperatureSetting
  /** τ at the end (the fixed τ, or where the learned one ended). */
  readonly finalTemperature: number
}

/** Options of `contrastiveAblation`. */
export type ContrastiveAblationOptions = Omit<ContrastiveRunOptions, 'batchSize' | 'temperature' | 'every'> & {
  /** Batch sizes to try (default [4, 16, 64]). */
  batchSizes?: readonly Size[]
  /** Temperatures to try: fixed values and `learned` (default [0.02, 0.1, 0.5, 'learned']). */
  temperatures?: readonly TemperatureSetting[]
}

/**
 * Train one tiny CLIP per batch size and temperature, for the same number of steps from the same initialisation, and
 * score each on `test`: how batch size (the number of negatives) and temperature trade alignment against uniformity.
 * A generator: it yields the runs finished so far after each one.
 */
export function* contrastiveAblation(
  train: ContrastivePairs,
  test: ContrastivePairs,
  options: ContrastiveAblationOptions = {},
): Generator<readonly ContrastiveAblationRun[]> {
  const { batchSizes = [4, 16, 64], temperatures = [0.02, 0.1, 0.5, 'learned'], steps = 300, ...rest } = options
  const runs: ContrastiveAblationRun[] = []
  for (const batchSize of batchSizes)
    for (const temperature of temperatures) {
      let last: ContrastiveSnapshot | undefined
      const initialTemperature = temperature === 'learned' ? (rest.initialTemperature ?? 0.1) : temperature
      for (const s of contrastiveTrainingRun(train, test, {
        ...rest,
        steps,
        every: steps,
        batchSize,
        temperature,
        initialTemperature,
      }))
        last = s
      const { params: _, step: __, temperature: finalTemperature, ...scores } = last!.checkpoints.at(-1)!
      runs.push({ ...scores, batchSize, temperature, finalTemperature })
      yield [...runs]
    }
}

/**
 * A tiny CLIP (Radford et al., 2021, "Learning Transferable Visual Models From Natural Language Supervision"): two
 * encoders map two views of the same objects (an image and its caption, say) into one shared space, where each is
 * normalised onto the unit sphere; training by the symmetric InfoNCE loss over in-batch negatives, with a learnable
 * temperature, pulls each object's two embeddings together and pushes the batch's other pairs apart. A class is then
 * recognised "zero-shot" by encoding its description with the second encoder and picking the nearest.
 *
 * With $\avec_i$ and $\bvec_i$ the unit embeddings of pair $i$ of a batch of $B$, $\tau$ the temperature and
 * $s_{ij} = \avec_i^\top \bvec_j / \tau$, the loss is the mean of the two directions' cross-entropies,
 *
 * $$\mathcal{L} = -\frac{1}{2B} \sum_i \Bigl(\log \frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}} + \log \frac{e^{s_{ii}}}{\sum_j e^{s_{ji}}}\Bigr),$$
 *
 * and a learned temperature is trained as CLIP's log logit scale $\log(1/\tau)$, its scale $1/\tau$ capped at
 * `maxScale`.
 *
 * The encoders are small MLPs (`aifn-compute/nn`), the loss and temperature are `aifn-compute/learning/losses`'
 * `infoNce` and `learnedTemperature`, training is `aifn-compute/nn`'s `trainingLoop` with Adam, and the embedding is
 * scored by `aifn-compute/learning/metrics`' `alignment` and `uniformity` (Wang & Isola, 2020). An encoder whose
 * output for a row is exactly zero (every hidden ReLU off) gives that row a NaN embedding.
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
 * Two views of the same objects, row by row, and what zero-shot evaluation needs (the shape of `aifn-methods/data`'s
 * paired views).
 */
export type ContrastivePairs = {
  /** The A view, $n \times d_A$: one object per row. */
  readonly a: Tensor
  /** The B view, $n \times d_B$: row $i$ is the same object as row $i$ of `a`. */
  readonly b: Tensor
  /** Each of $K$ classes described in the B view, $K \times d_B$ (for zero-shot evaluation). */
  readonly prototypes?: Tensor
  /**
   * `combination`, each row's class (an index into `prototypes`), and `heldOut`, 1 for rows of classes left out of
   * training and 0 otherwise.
   */
  readonly truth?: { readonly combination: Tensor; readonly heldOut: Tensor }
}

// ── The model ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The architecture of a two-tower model. */
export type TwoTowerConfig = {
  /** Input size $d_A$ of the A view. */
  readonly inA: Size
  /** Input size $d_B$ of the B view. */
  readonly inB: Size
  /** Hidden units of each encoder's one hidden layer (default 32). */
  readonly hidden?: Size
  /** The shared embedding's dimension (default 2, so the embedding lies on the unit circle). */
  readonly dim?: Size
}

/** A two-tower model's parameters: the two encoders and the log logit scale $\log(1/\tau)$. */
export type TwoTowerParams = {
  /** The A view's encoder. */
  readonly a: Params[]
  /** The B view's encoder. */
  readonly b: Params[]
  /**
   * $\log(1/\tau)$, shape `[1]`: trained when the temperature is learned; with a fixed temperature it is not read, so
   * its gradient is zero and it stays at its initial value.
   */
  readonly logScale: Tensor
}

/** A two-tower model: two MLP encoders into a shared `dim`-dimensional space. */
export type TwoTower = {
  /** The architecture, with the defaults filled in. */
  readonly config: Required<TwoTowerConfig>
  /** The A view's encoder, $d_A$ inputs to `dim` outputs (not normalised). */
  readonly encoderA: Layer<Params[]>
  /** The B view's encoder, $d_B$ inputs to `dim` outputs (not normalised). */
  readonly encoderB: Layer<Params[]>
  /** Fresh parameters from stream `s`, with $\tau$ at `temperature` (default 0.1). */
  init(s: Stream, temperature?: number): TwoTowerParams
}

/**
 * Two MLP encoders (input, one ReLU hidden layer, `dim` outputs) for the two views (CLIP's image and text towers, in
 * miniature). Their outputs are normalised onto the unit sphere by the loss and by `embed`, not by the encoders.
 *
 * @param config The input sizes of the two views, the hidden units and the embedding's dimension.
 * @returns The model: its full configuration, the two encoders and `init`.
 *
 * @example A model of a 3-feature view and a 2-feature view
 * const model = TwoTower({ inA: 3, inB: 2, hidden: 8 })
 * const p = model.init(stream(0))
 * print('config:', model.config)
 * print('initial temperature:', temperatureOf(p))
 */
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

/** The temperature setting: a fixed $\tau$, or `learned` from an initial $\tau$ (CLIP's $\log(1/\tau)$ parameter). */
export type TemperatureSetting = number | 'learned'

/** Options of `contrastiveLoss`. */
export type ContrastiveLossOptions = {
  /** A fixed $\tau$, or `learned` (default) to use the model's trained $\log(1/\tau)$. */
  temperature?: TemperatureSetting
  /** The cap on the learned logit scale $1/\tau$ (default 100, CLIP's). */
  maxScale?: number
}

/**
 * The temperature of parameters `p` under a setting, as a value (traced when `p` is): the fixed $\tau$ itself, or
 * $\exp(-\min(s, \log \text{maxScale}))$ from the learned log logit scale $s$.
 *
 * @param p The model's parameters; only `logScale` is read, and only for a learned temperature.
 * @param setting A fixed $\tau$, or `learned`.
 * @param maxScale The cap on the learned logit scale $1/\tau$, so the smallest learned $\tau$ is `1 / maxScale`.
 * @returns $\tau$: the number given, or a one-element value.
 *
 * @example Learned, fixed and capped
 * const p = TwoTower({ inA: 2, inB: 2 }).init(stream(0), 0.07)
 * print('learned, from 0.07:', temperatureOf(p))
 * print('fixed at 0.5:', temperatureOf(p, 0.5))
 * print('learned, scale capped at 10:', temperatureOf(p, 'learned', 10))
 */
export function temperatureOf(p: TwoTowerParams, setting: TemperatureSetting = 'learned', maxScale = 100): Value {
  return setting === 'learned' ? learnedTemperature(p.logScale, { maxScale }) : setting
}

/**
 * CLIP's loss on a batch of pairs: the symmetric InfoNCE of the two towers' embeddings (cosine similarities over
 * $\tau$), the rest of the batch serving as negatives in both directions. Differentiable in the parameters.
 *
 * @param model The two-tower model.
 * @param p Its parameters.
 * @param batch The A view `a` and the B view `b` of the same $B$ objects, row by row.
 * @param options The temperature setting and the cap on a learned logit scale.
 * @param options.temperature A fixed $\tau$, or `learned` (default) for the parameters' own.
 * @param options.maxScale The cap on the learned logit scale $1/\tau$ (default 100).
 * @returns The loss, a scalar: near 0 when every pair is told apart from the rest, and about $\log B$ by chance.
 *
 * @example InfoNCE of aligned pairs against shuffled ones, after training
 * const a = normals(stream(0), [16, 2])
 * const b = matmul(a, tensor([[0, 1], [-1, 0]])) // the B view is the A view turned a quarter
 * const model = TwoTower({ inA: 2, inB: 2, hidden: 8 })
 * const alg = contrastiveTraining(model, { a, b }, { batchSize: 16, stepSize: 0.05 })
 * const { params } = run(alg, { params: model.init(stream(1)) }, 100, { stream: stream(2) })
 * const shuffled = take(b, Array.from(toFlat(permutation(stream(3), 16))))
 * print('aligned:', contrastiveLoss(model, params, { a, b }))
 * print('shuffled:', contrastiveLoss(model, params, { a, b: shuffled }))
 * print('log 16 =', Math.log(16))
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

/**
 * A number or one-element value as a number.
 *
 * @param v The value, traced or not.
 * @returns Its (first) number.
 */
function scalarOf(v: Value): number {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

/**
 * Rows divided by their norms, as a plain tensor. A zero row gives NaN.
 *
 * @param z The rows, $n \times d$.
 * @returns The unit rows, $n \times d$.
 */
function unit(z: Value): Tensor {
  return unwrap(div(z, norm(z, -1, true))) as Tensor
}

/**
 * The unit-norm embeddings of rows of one view, by that view's encoder.
 *
 * @param model The two-tower model.
 * @param p Its parameters.
 * @param x Rows of the view: $n \times d_A$ for `'a'`, $n \times d_B$ for `'b'`.
 * @param view Which view, and so which encoder.
 * @returns The embeddings, $n \times$ `dim`, each row of norm 1.
 *
 * @example Three rows of the A view on the unit circle
 * const model = TwoTower({ inA: 3, inB: 2, hidden: 8 })
 * const z = embed(model, model.init(stream(0)), tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 'a')
 * print('embeddings:', z)
 * print('norms:', norm(z, -1))
 */
export function embed(model: TwoTower, p: TwoTowerParams, x: Tensor, view: 'a' | 'b'): Tensor {
  return unit(view === 'a' ? model.encoderA.apply(p.a, x) : model.encoderB.apply(p.b, x))
}

/**
 * Cosine similarities between unit-norm embeddings: the inner products $\avec_i^\top \bvec_j$ (cosines only when
 * the rows have norm 1).
 *
 * @param za The first embeddings, $n \times d$.
 * @param zb The second embeddings, $m \times d$.
 * @returns The $n \times m$ similarities, row-major: entry `i * m + j` compares row `i` of `za` with row `j` of `zb`.
 *
 * @example Two directions against three
 * const za = tensor([[1, 0], [0, 1]])
 * const zb = tensor([[1, 0], [Math.SQRT1_2, Math.SQRT1_2], [-1, 0]])
 * print(similarities(za, zb))
 */
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

/**
 * The `k` rows of `keys` most similar to `query`, best first, by inner product (the cosine similarity for unit-norm
 * rows).
 *
 * @param query The query embedding, $d$ values.
 * @param keys The embeddings searched, $m \times d$.
 * @param k How many to return (all $m$ when `k` is larger).
 * @returns The rows' indices, best first, and their similarities.
 *
 * @example The two nearest of four directions
 * const keys = tensor([[1, 0], [0, 1], [-1, 0], [Math.SQRT1_2, Math.SQRT1_2]])
 * print(retrieve([1, 0], keys, 2))
 */
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
 * Zero-shot classification: each row of `za` (A-view embeddings) is given the class whose B-view prototype embedding
 * is most similar, among `candidates` (default every class). Ties go to the first candidate.
 *
 * @param za The A-view embeddings, $n \times d$.
 * @param zp The prototypes' B-view embeddings, $K \times d$, one per class.
 * @param candidates The classes allowed (default all $K$).
 * @returns The class of each row.
 *
 * @example Among every class, and among two
 * const za = tensor([[1, 0], [0.6, 0.8], [-1, 0]])
 * const zp = tensor([[1, 0], [0, 1], [-1, 0]])
 * print('among every class:', zeroShot(za, zp))
 * print('among classes 1 and 2:', zeroShot(za, zp, [1, 2]))
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
  /** Wang & Isola's alignment $\expect \norm{\avec - \bvec}^2$ over pairs, in $[0, 4]$. */
  readonly alignment: number
  /**
   * Wang & Isola's uniformity, the mean over the two views of $\log \expect \exp(-2 \norm{\zvec - \zvec'}^2)$, in
   * $[-8, 0]$.
   */
  readonly uniformity: number
  /** Zero-shot accuracy among every class, on rows of classes seen in training (NaN without prototypes or rows). */
  readonly zeroShotSeen: number
  /** The same on rows of held-out classes (generalised zero-shot: every class is a candidate; NaN when none). */
  readonly zeroShotHeldOut: number
  /**
   * Accuracy on rows of held-out classes when only the held-out classes are candidates (classic zero-shot; NaN when
   * none).
   */
  readonly zeroShotHeldOutOnly: number
  /**
   * Top-1 retrieval: the share of A rows whose most similar B row describes the same object (with a truth: an object
   * of the same class, since several objects share a description).
   */
  readonly retrievalTop1: number
}

/**
 * The share of rows in `rows` where `pred` equals `cls` (NaN for none).
 *
 * @param pred The predicted class of every row.
 * @param cls The true class of every row.
 * @param rows The rows to score.
 * @returns The accuracy over `rows`.
 */
function accuracy(pred: ArrayLike<number>, cls: ArrayLike<number>, rows: readonly number[]): number {
  if (rows.length === 0) return NaN
  return rows.filter((i) => pred[i] === cls[i]).length / rows.length
}

/**
 * Score parameters on evaluation pairs: the InfoNCE of the whole set as one batch, alignment, uniformity, zero-shot
 * accuracy (with prototypes and a truth) and top-1 retrieval. Retrieval counts a hit when the B row most similar to an
 * A row is its own pair, or with a truth any row of the same class.
 *
 * @param model The two-tower model.
 * @param p Its parameters.
 * @param data The evaluation pairs, with prototypes and a truth for the zero-shot scores.
 * @param options The temperature setting and the cap on a learned logit scale, for the loss.
 * @returns The scores; the zero-shot ones are NaN without prototypes and a truth.
 *
 * @example Before and after training on a quarter turn
 * const a = normals(stream(0), [16, 2])
 * const b = matmul(a, tensor([[0, 1], [-1, 0]]))
 * const model = TwoTower({ inA: 2, inB: 2, hidden: 8 })
 * const start = model.init(stream(1))
 * const alg = contrastiveTraining(model, { a, b }, { batchSize: 16, stepSize: 0.05 })
 * const { params } = run(alg, { params: start }, 100, { stream: stream(2) })
 * const pick = ({ loss, alignment, uniformity, retrievalTop1 }) => ({ loss, alignment, uniformity, retrievalTop1 })
 * print('untrained:', pick(scoreEmbedding(model, start, { a, b })))
 * print('trained:', pick(scoreEmbedding(model, params, { a, b })))
 */
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
  /** Pairs per step $B$ (default 32): each pair's negatives are the batch's other $B - 1$ pairs. */
  batchSize?: Size
  /** Adam's step size (default 0.01). */
  stepSize?: number
  /**
   * The initial $\tau$ (default 0.1), from which a learned temperature starts. `contrastiveTraining` does not read it;
   * `contrastiveTrainingRun` passes it to `init`.
   */
  initialTemperature?: number
}

/**
 * CLIP training as a traceable `trainingLoop`: minibatch Adam on the symmetric InfoNCE of shuffled batches of pairs.
 * With a fixed temperature, $\log(1/\tau)$ is not read, so its gradient is zero and it has no effect.
 *
 * @param model The two-tower model.
 * @param data The training pairs (only `a` and `b` are read).
 * @param options The temperature setting, its cap, the batch size and the step size; `hidden`, `dim` and
 *   `initialTemperature` are not read here.
 * @returns The algorithm, to run with `run` or `trace` from `{ params }`.
 *
 * @example The loss falls and the learned temperature sharpens
 * const a = normals(stream(0), [16, 2])
 * const b = matmul(a, tensor([[0, 1], [-1, 0]]))
 * const model = TwoTower({ inA: 2, inB: 2, hidden: 8 })
 * const alg = contrastiveTraining(model, { a, b }, { batchSize: 8, stepSize: 0.05 })
 * const record = { loss: (s) => s.loss, tau: (s) => Math.exp(-toFlat(s.params.logScale)[0]) }
 * const tr = trace(alg, { params: model.init(stream(1)) }, 100, { stream: stream(2), every: 25, record })
 * print('step:', tr.index)
 * print('loss:', tr.series.loss)
 * print('temperature:', tr.series.tau)
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
  /** Updates taken. */
  readonly step: Size
  /** $\tau$ under the run's temperature setting. */
  readonly temperature: number
  /** The parameters after `step` updates. */
  readonly params: TwoTowerParams
}

/** A snapshot of `contrastiveTrainingRun`: the run so far. */
export type ContrastiveSnapshot = {
  /** Steps taken. */
  readonly step: Size
  /** Steps in the whole run. */
  readonly steps: Size
  /** The model's architecture. */
  readonly config: Required<TwoTowerConfig>
  /** The minibatch loss at every step so far (step 0 first). */
  readonly losses: readonly number[]
  /** The temperature at every step so far (step 0 first). */
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

/**
 * The first `m` rows of evaluation pairs, with the truth cut to match and the prototypes kept whole.
 *
 * @param data The evaluation pairs.
 * @param m How many rows to keep; all of them when `m` is at least their number.
 * @returns The first `m` pairs (`data` itself when nothing is cut).
 */
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
 * model. Deterministic from the seed.
 *
 * @param train The training pairs.
 * @param test The evaluation pairs; the first `evaluate` rows are scored at each checkpoint.
 * @param options The architecture, the temperature, the training, the number of steps, the checkpoint interval, the
 *   seed and the evaluation size.
 * @returns A generator of snapshots: step 0, every `every` steps, and the last step.
 *
 * @example Checkpoints of a short run
 * const a = normals(stream(0), [32, 2])
 * const b = matmul(a, tensor([[0, 1], [-1, 0]]))
 * const options = { hidden: 16, batchSize: 16, stepSize: 0.05, steps: 60, every: 20, seed: 0 }
 * for (const s of contrastiveTrainingRun({ a, b }, { a, b }, options)) {
 *   const c = s.checkpoints.at(-1)
 *   print('step', s.step, ' tau', c.temperature, ' alignment', c.alignment, ' top-1 retrieval', c.retrievalTop1)
 * }
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
  /** The run's batch size. */
  readonly batchSize: Size
  /** The run's temperature setting. */
  readonly temperature: TemperatureSetting
  /** $\tau$ at the end (the fixed $\tau$, or where the learned one ended). */
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
 * Train one tiny CLIP per batch size and temperature, for the same number of steps (default 300) from the same
 * initialisation, and score each on `test`: how batch size (the number of negatives) and temperature trade alignment
 * against uniformity. A generator: it yields the runs finished so far after each one. A learned temperature starts at
 * `initialTemperature` (default 0.1).
 *
 * @param train The training pairs.
 * @param test The evaluation pairs.
 * @param options The batch sizes and temperatures to try, and the options shared by every run (as
 *   `contrastiveTrainingRun`'s, without the batch size, temperature and checkpoint interval).
 * @returns A generator of the finished runs, batch size by batch size, temperature by temperature.
 *
 * @example Two batch sizes and two temperatures
 * const a = normals(stream(0), [32, 2])
 * const b = matmul(a, tensor([[0, 1], [-1, 0]]))
 * const options = { hidden: 16, steps: 40, stepSize: 0.05, batchSizes: [4, 16], temperatures: [0.05, 'learned'] }
 * let runs
 * for (const r of contrastiveAblation({ a, b }, { a, b }, { ...options, seed: 0 })) runs = r
 * for (const r of runs)
 *   print(`B = ${r.batchSize}, tau = ${r.temperature}:`, 'alignment', r.alignment, ' uniformity', r.uniformity)
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

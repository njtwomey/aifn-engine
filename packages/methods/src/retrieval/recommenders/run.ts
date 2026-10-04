/**
 * `recommenderRun`: train one recommender of the module on implicit feedback and stream its progress, epoch by epoch,
 * for a page to plot and play: the training loss, recall@k and NDCG@k on held-out items, and checkpoints holding every
 * user's scores and a two-dimensional map of the learned embeddings.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { svd } from 'aifn-compute/numerics/linalg'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { factorScorer, alsFactors, implicitAls, type AlsState } from './factorisation'
import { evaluateRanking, interactionsFromRows, type Interactions, type Scorer } from './interactions'
import { neuralRecommender, type NeuralKind, type NeuralOptions } from './models'
import { itemKnn, popularity, userKnn } from './neighbourhood'

/** Implicit-feedback data as the generators of `aifn-methods/data` produce it. */
export type RecommenderData = {
  users: Size
  items: Size
  /** Training interactions as rows (user, item), and held-out rows. */
  train: Tensor
  test: Tensor
  /** Side features: a group per user and a category per item (int32). */
  userGroup?: Tensor
  itemCategory?: Tensor
  /** Each user's training items in time order. */
  sequences?: readonly (readonly number[])[]
}

/** Every recommender `recommenderRun` can train. */
export type RecommenderKind = 'popularity' | 'user-knn' | 'item-knn' | 'implicit-als' | NeuralKind

/** The recommenders in the order a page lists them, with a display name. */
export const RECOMMENDERS: readonly { kind: RecommenderKind; name: string; trained: boolean }[] = [
  { kind: 'popularity', name: 'Popularity', trained: false },
  { kind: 'user-knn', name: 'User-kNN', trained: false },
  { kind: 'item-knn', name: 'Item-kNN', trained: false },
  { kind: 'implicit-als', name: 'Implicit ALS', trained: true },
  { kind: 'logistic-mf', name: 'MF (SGD, logistic)', trained: true },
  { kind: 'bpr', name: 'BPR-MF', trained: true },
  { kind: 'factorisation-machine', name: 'Factorisation machine', trained: true },
  { kind: 'field-aware-factorisation-machine', name: 'Field-aware FM', trained: true },
  { kind: 'wide-and-deep', name: 'Wide & Deep', trained: true },
  { kind: 'deepfm', name: 'DeepFM', trained: true },
  { kind: 'neural-collaborative-filtering', name: 'NCF (NeuMF)', trained: true },
  { kind: 'two-tower', name: 'Two-tower', trained: true },
  { kind: 'sasrec', name: 'SASRec', trained: true },
]

/** Options of `recommenderRun`. */
export type RecommenderRunOptions = NeuralOptions & {
  data: RecommenderData
  model: RecommenderKind
  /** Passes over the training rows (ALS sweeps; ignored by the untrained models). Default 20. */
  epochs?: Size
  /** Adam's step size (default 0.01). */
  stepSize?: number
  /** Rows per gradient step (default 128; SASRec: 16 users). */
  batchSize?: Size
  /** The cut-off of recall@k and NDCG@k (default 10). */
  k?: Size
  /** Neighbours of the kNN models (default 20). */
  neighbours?: Size
  /** Implicit ALS's confidence slope α and penalty λ (default 10 and 0.1). */
  alpha?: number
  regularisation?: number
  /** The root seed (default 'recommender'). */
  seed?: string | number
}

/** Curves of a run, one entry per evaluated epoch (epoch 0 is the untrained model). */
export type RecommenderHistory = {
  epoch: number[]
  /** Mean training loss over the epoch's steps (NaN for the untrained models and at epoch 0). */
  loss: number[]
  recall: number[]
  ndcg: number[]
  hitRate: number[]
  coverage: number[]
}

/** The state shown at one epoch: every user's scores and the embedding map. */
export type RecommenderCheckpoint = {
  readonly epoch: Size
  /** Scores [users, items], row-major. */
  readonly scores: Float64Array
  /** Two-dimensional maps of the item (and user) embeddings by their first two principal components; null when the model has none. */
  readonly itemMap: Float64Array | null
  readonly userMap: Float64Array | null
}

/** A snapshot of `recommenderRun`. */
export type RecommenderSnapshot = {
  readonly epoch: Size
  readonly epochs: Size
  readonly done: boolean
  readonly model: RecommenderKind
  readonly history: RecommenderHistory
  readonly checkpoints: readonly RecommenderCheckpoint[]
}

/** Rows [n, d] projected on their first two principal axes, as [n, 2]. */
export function principalMap(rows: Tensor): Float64Array {
  const [n, d] = rows.shape
  const x = Float64Array.from(toFlat(rows))
  const mean = new Float64Array(d)
  for (let r = 0; r < n; r++) for (let c = 0; c < d; c++) mean[c] += x[r * d + c] / n
  for (let r = 0; r < n; r++) for (let c = 0; c < d; c++) x[r * d + c] -= mean[c]
  const out = new Float64Array(n * 2)
  if (d === 1) {
    for (let r = 0; r < n; r++) out[r * 2] = x[r]
    return out
  }
  const V = toFlat(svd(fromData(x, [n, d])).V)
  const kk = Math.min(n, d)
  for (let r = 0; r < n; r++)
    for (let a = 0; a < 2; a++) {
      let s = 0
      for (let c = 0; c < d; c++) s += x[r * d + c] * V[c * kk + a]
      out[r * 2 + a] = s
    }
  return out
}

const range = (n: Size) => Array.from({ length: n }, (_, i) => i)

/**
 * Train a recommender on `data.train` and evaluate it on `data.test` after every epoch (and before the first), yielding
 * a snapshot each time: a generator, so a worker can stream the run. Popularity and the kNN models have nothing to
 * train and yield once.
 */
export function* recommenderRun(options: RecommenderRunOptions): Generator<RecommenderSnapshot> {
  const {
    data,
    model,
    epochs: epochOption = 20,
    stepSize = 0.01,
    k = 10,
    neighbours = 20,
    seed = 'recommender',
  } = options
  const { users, items } = data
  const train: Interactions = interactionsFromRows(data.train, users, items)
  const test: Interactions = interactionsFromRows(data.test, users, items)
  const root = stream(seed)
  const history: RecommenderHistory = { epoch: [], loss: [], recall: [], ndcg: [], hitRate: [], coverage: [] }
  const checkpoints: RecommenderCheckpoint[] = []
  const allUsers = range(users)
  const record = (epoch: Size, loss: number, score: Scorer, maps: { items: Tensor | null; users: Tensor | null }) => {
    const r = evaluateRanking(score, train, test, k)
    history.epoch.push(epoch)
    history.loss.push(loss)
    history.recall.push(r.recall)
    history.ndcg.push(r.ndcg)
    history.hitRate.push(r.hitRate)
    history.coverage.push(r.coverage)
    checkpoints.push({
      epoch,
      scores: score(allUsers),
      itemMap: maps.items ? principalMap(maps.items) : null,
      userMap: maps.users ? principalMap(maps.users) : null,
    })
  }
  const snapshot = (epoch: Size, epochs: Size, done: boolean): RecommenderSnapshot => ({
    epoch,
    epochs,
    done,
    model,
    history: {
      epoch: [...history.epoch],
      loss: [...history.loss],
      recall: [...history.recall],
      ndcg: [...history.ndcg],
      hitRate: [...history.hitRate],
      coverage: [...history.coverage],
    },
    checkpoints: [...checkpoints],
  })

  if (model === 'popularity' || model === 'user-knn' || model === 'item-knn') {
    const score =
      model === 'popularity' ? popularity(train) : (model === 'user-knn' ? userKnn : itemKnn)(train, { k: neighbours })
    record(0, NaN, score, { items: null, users: null })
    yield snapshot(0, 0, true)
    return
  }

  const epochs = epochOption
  if (model === 'implicit-als') {
    const alg = implicitAls(train, {
      factors: options.dimension ?? 16,
      alpha: options.alpha ?? 10,
      regularisation: options.regularisation ?? 0.1,
    })
    let s: AlsState = alg.init(undefined, child(root, 'init'))
    const show = (st: AlsState, e: Size) => {
      const f = alsFactors(st)
      record(e, st.objective / (users * items), factorScorer(f), { items: st.Q, users: st.P })
    }
    show(s, 0)
    yield snapshot(0, epochs, false)
    for (let e = 1; e <= epochs; e++) {
      s = alg.step(s, { t: e - 1, stream: child(root, 'step', e) })
      show(s, e)
      yield snapshot(e, epochs, e === epochs || s.diverged === true)
      if (s.diverged) return
    }
    return
  }

  const ctx = {
    train,
    userGroup: data.userGroup ? toFlat(data.userGroup) : undefined,
    itemCategory: data.itemCategory ? toFlat(data.itemCategory) : undefined,
    sequences: data.sequences,
  }
  const rec = neuralRecommender(model, ctx, options)
  const rows = rec.data(child(root, 'negatives'))
  const n = Object.values(rows)[0].shape[0]
  const batchSize = Math.min(n, options.batchSize ?? (model === 'sasrec' ? 16 : 128))
  const perEpoch = Math.max(1, Math.floor(n / batchSize))
  const alg = trainingLoop<Params, Record<string, Tensor>>({
    loss: (p, b) => rec.loss(p, b),
    data: rows,
    batchSize,
    optimizer: adamRule({ stepSize }) as never,
    clipNorm: 10,
  })
  let s = alg.init({ params: rec.init(child(root, 'init')) }, child(root, 'train'))
  const show = (e: Size, loss: number) => {
    const emb = rec.embeddings(s.params)
    record(e, loss, rec.scorer(s.params), { items: emb.items, users: emb.users })
  }
  show(0, NaN)
  yield snapshot(0, epochs, false)
  let t = 0
  for (let e = 1; e <= epochs; e++) {
    let total = 0
    for (let b = 0; b < perEpoch; b++) {
      total += s.loss
      s = alg.step(s, { t, stream: child(root, 'step', t) })
      t++
      if (s.diverged) break
    }
    show(e, total / perEpoch)
    const done = e === epochs || s.diverged
    yield snapshot(e, epochs, done)
    if (s.diverged) return
  }
}

/**
 * Recommenders trained by gradient descent on implicit feedback, each a small parameter tree with a loss and a scorer,
 * trained through `aifn-compute/nn/training` with gradients from `aifn-compute/foundation/autodiff`:
 *
 * - **Logistic MF**: s_ui = p_uᵀq_i + b_i with binary cross-entropy on positives and sampled negatives.
 * - **BPR** (Rendle et al., 2009): the same scores, trained to rank a positive above a sampled negative,
 *   −log σ(s_ui − s_uj).
 * - **Factorisation machines** (Rendle, 2010) and **field-aware FM** (Juan et al., 2016) over four one-hot fields:
 *   user, item, user group and item category.
 * - **Wide & Deep** (Cheng et al., 2016): a linear model on the fields and the group × category cross, plus an MLP on
 *   their embeddings. **DeepFM** (Guo et al., 2017): an FM and an MLP sharing one set of embeddings.
 * - **NCF / NeuMF** (He et al., 2017): generalised matrix factorisation and an MLP on user and item embeddings, joined.
 * - **Two-tower** (Covington et al., 2016; Yi et al., 2019): user and item towers (MLPs on id and side embeddings),
 *   scored by inner product and trained with the in-batch softmax.
 * - **SASRec** (Kang and McAuley, 2018): item and position embeddings, one causal transformer block of
 *   `aifn-compute/nn/attention`, and the next item predicted at every position by a softmax over the catalogue.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, integers, normal, type Stream } from 'aifn-compute/foundation/random'
import {
  add,
  concat,
  expandDims,
  fromData,
  matmul,
  mul,
  reshape,
  square,
  sub,
  sum,
  take,
  toFlat,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { binaryCrossEntropyWithLogits, infoNce, softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { TransformerBlock, type TransformerBlockParams } from 'aifn-compute/nn/attention'
import { Mlp } from 'aifn-compute/nn/layers'
import { softplus } from 'aifn-compute/numerics/special'
import { itemsByUser, type Interactions, type Scorer } from './interactions'

/** What the gradient recommenders read besides the interactions: side features and the order of each user's items. */
export type RecommenderContext = {
  readonly train: Interactions
  /** A group per user (e.g. a segment), int32; default all 0. */
  readonly userGroup?: ArrayLike<number>
  /** A category per item, int32; default all 0. */
  readonly itemCategory?: ArrayLike<number>
  /** Each user's training items in time order (SASRec); default the order of `train`. */
  readonly sequences?: readonly (readonly number[])[]
}

/** The neural and factorisation recommenders. */
export type NeuralKind =
  | 'logistic-mf'
  | 'bpr'
  | 'factorisation-machine'
  | 'field-aware-factorisation-machine'
  | 'wide-and-deep'
  | 'deepfm'
  | 'neural-collaborative-filtering'
  | 'two-tower'
  | 'sasrec'

/** Hyperparameters shared by the gradient recommenders. */
export type NeuralOptions = {
  /** Embedding (factor) dimension (default 16). */
  dimension?: Size
  /** Hidden widths of the MLPs (default [32]). */
  hidden?: Size[]
  /** Sampled negatives per positive for pointwise and pairwise losses (default 4). */
  negatives?: Size
  /** L2 penalty on the embeddings of each training row (default 1e-2). */
  regularisation?: number
  /** In-batch softmax temperature of the two-tower model (default 0.1). */
  temperature?: number
  /** SASRec's maximum sequence length (default 10). */
  maxLength?: Size
}

/** A gradient recommender: its parameters' initialiser, training rows, loss, scorer and (when it has them) embeddings. */
export type NeuralRecommender = {
  readonly kind: NeuralKind
  init(s: Stream): Params
  /** The training rows (a batch of named tensors with equal first axes), drawn once from `s` (negatives). */
  data(s: Stream): Record<string, Tensor>
  loss(params: Params, batch: Record<string, Tensor>): Value
  scorer(params: Params): Scorer
  /** User and item embeddings to draw (factors, tower outputs or item vectors). */
  embeddings(params: Params): { users: Tensor | null; items: Tensor }
}

// ── Shared pieces ────────────────────────────────────────────────────────────────────────────────────────────────────

const ids = (t: Tensor): number[] => Array.from(toFlat(t))
const vec = (xs: ArrayLike<number>) => fromData(Float64Array.from(xs), [xs.length])
const table = (s: Stream, rows: Size, cols: Size, sd = 0.1) => normal(s, 0, sd, { shape: [rows, cols] }) as Tensor

/** Positives (u, i) with `negatives` uniformly drawn unseen items per positive, labelled 1 and 0. */
function pointwiseRows(ctx: RecommenderContext, negatives: Size, s: Stream) {
  const { train } = ctx
  const seen = itemsByUser(train)
  const u: number[] = []
  const i: number[] = []
  const y: number[] = []
  for (let r = 0; r < train.user.length; r++) {
    u.push(train.user[r])
    i.push(train.item[r])
    y.push(1)
    for (let n = 0; n < negatives; n++) {
      u.push(train.user[r])
      i.push(sampleUnseen(child(s, r, n), seen[train.user[r]], train.items))
      y.push(0)
    }
  }
  return { user: vec(u), item: vec(i), label: vec(y) }
}

/** Triples (u, i, j): each positive with `negatives` sampled unseen items. */
function pairwiseRows(ctx: RecommenderContext, negatives: Size, s: Stream) {
  const { train } = ctx
  const seen = itemsByUser(train)
  const u: number[] = []
  const i: number[] = []
  const j: number[] = []
  for (let r = 0; r < train.user.length; r++)
    for (let n = 0; n < negatives; n++) {
      u.push(train.user[r])
      i.push(train.item[r])
      j.push(sampleUnseen(child(s, r, n), seen[train.user[r]], train.items))
    }
  return { user: vec(u), item: vec(i), negative: vec(j) }
}

function sampleUnseen(s: Stream, seen: ReadonlySet<number>, items: Size): number {
  if (seen.size >= items) return integers(s, items)
  for (let t = 0; ; t++) {
    const j = integers(child(s, t), items)
    if (!seen.has(j)) return j
  }
}

/** Every (user, item) pair of a list of users, as two index arrays (users vary slowest). */
function allPairs(users: readonly number[], items: Size) {
  const u: number[] = []
  const i: number[] = []
  for (const v of users) for (let j = 0; j < items; j++) (u.push(v), i.push(j))
  return { u, i }
}

/** Evaluate a pairwise logit function on every pair of the users, as their score rows. */
function pairScorer(items: Size, logit: (u: number[], i: number[]) => Value): Scorer {
  return (users) => {
    const { u, i } = allPairs(users, items)
    return Float64Array.from(toFlat(logit(u, i) as Tensor))
  }
}

const l2 = (...xs: Value[]) => xs.reduce<Value>((acc, x) => add(acc, sum(square(x))), 0)
const rowsOf = (n: Size) => Math.max(1, n)

/** The four one-hot fields of a (user, item) row: user, item, user group, item category, as global feature ids. */
function fieldLayout(ctx: RecommenderContext) {
  const { users, items } = ctx.train
  const groups = ctx.userGroup ? Math.max(...Array.from(ctx.userGroup)) + 1 : 1
  const categories = ctx.itemCategory ? Math.max(...Array.from(ctx.itemCategory)) + 1 : 1
  const offsets = [0, users, users + items, users + items + groups]
  const features = users + items + groups + categories
  const group = (u: number) => (ctx.userGroup ? ctx.userGroup[u] : 0)
  const category = (i: number) => (ctx.itemCategory ? ctx.itemCategory[i] : 0)
  /** The feature ids [n, 4] of rows (u, i). */
  const featuresOf = (u: number[], i: number[]) => {
    const out = new Float64Array(u.length * 4)
    u.forEach((v, r) => {
      out[r * 4] = v
      out[r * 4 + 1] = offsets[1] + i[r]
      out[r * 4 + 2] = offsets[2] + group(v)
      out[r * 4 + 3] = offsets[3] + category(i[r])
    })
    return fromData(out, [u.length, 4])
  }
  return { features, fields: 4, groups, categories, featuresOf, group, category }
}

/** The FM interaction ½ Σ_k [(Σ_f v_fk)² − Σ_f v_fk²] of embeddings [n, F, k], as [n]. */
function fmInteraction(v: Value): Value {
  const total = sum(v, 1)
  return mul(0.5, sum(sub(square(total), sum(square(v), 1)), -1))
}

// ── The models ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** Build a gradient recommender of a kind on the training context. */
export function neuralRecommender(
  kind: NeuralKind,
  ctx: RecommenderContext,
  options: NeuralOptions = {},
): NeuralRecommender {
  const { dimension: k = 16, hidden = [32], negatives = 4, regularisation: lambda = 1e-2, temperature = 0.1 } = options
  const { users, items } = ctx.train
  const layout = fieldLayout(ctx)
  const pointwise = (s: Stream) => pointwiseRows(ctx, negatives, s)
  const bce = (logits: Value, y: Tensor) => binaryCrossEntropyWithLogits(logits, y)

  switch (kind) {
    case 'logistic-mf':
    case 'bpr': {
      type P = { P: Tensor; Q: Tensor; b: Tensor }
      const score = (p: P, u: number[], i: number[]) => add(sum(mul(take(p.P, u), take(p.Q, i)), -1), take(p.b, i))
      return {
        kind,
        init: (s) => ({
          P: table(child(s, 'P'), users, k),
          Q: table(child(s, 'Q'), items, k),
          b: vec(new Float64Array(items)),
        }),
        data: (s) => (kind === 'bpr' ? pairwiseRows(ctx, negatives, s) : pointwise(s)),
        loss: (params, b) => {
          const p = params as P
          const u = ids(b.user)
          const i = ids(b.item)
          const reg = mul(lambda / rowsOf(u.length), l2(take(p.P, u), take(p.Q, i)))
          if (kind === 'logistic-mf') return add(bce(score(p, u, i), b.label), reg)
          const j = ids(b.negative)
          // −log σ(s_ui − s_uj) = softplus(s_uj − s_ui).
          const margin = sub(score(p, u, j), score(p, u, i))
          return add(sum(softplus(margin)), mul(lambda, l2(take(p.P, u), take(p.Q, i), take(p.Q, j))))
        },
        scorer: (params) => pairScorer(items, (u, i) => score(params as P, u, i)),
        embeddings: (params) => ({ users: (params as P).P, items: (params as P).Q }),
      }
    }

    case 'factorisation-machine':
    case 'field-aware-factorisation-machine': {
      type P = { w0: Tensor; w: Tensor; V: Tensor }
      const F = layout.fields
      const ffm = kind === 'field-aware-factorisation-machine'
      const logit = (p: P, u: number[], i: number[]) => {
        const f = layout.featuresOf(u, i)
        const linear = add(p.w0, sum(take(p.w, f), -1))
        if (!ffm) return add(linear, fmInteraction(take(p.V, f)))
        // FFM: feature a meets feature c through v_{a, field(c)} · v_{c, field(a)}. V is [features, F, k], so the rows'
        // vectors are [n, slot, field, k]; permuted to [slot·field, n, k], entry a·F + c is v_{a, field c}.
        const e = reshape(transpose(take(p.V, f), [1, 2, 0, 3]), [F * F, u.length, k])
        let pairs: Value = 0
        for (let a = 0; a < F; a++)
          for (let c = a + 1; c < F; c++)
            pairs = add(pairs, sum(mul(take(e, [a * F + c]), take(e, [c * F + a])), [0, 2]))
        return add(linear, pairs)
      }
      return {
        kind,
        init: (s) => ({
          w0: fromData(Float64Array.of(0), []),
          w: vec(new Float64Array(layout.features)),
          V: ffm
            ? (normal(child(s, 'V'), 0, 0.1, { shape: [layout.features, F, k] }) as Tensor)
            : table(child(s, 'V'), layout.features, k),
        }),
        data: pointwise,
        loss: (params, b) => {
          const p = params as P
          const u = ids(b.user)
          const i = ids(b.item)
          return add(
            bce(logit(p, u, i), b.label),
            mul(lambda / rowsOf(u.length), l2(take(p.V, layout.featuresOf(u, i)))),
          )
        },
        scorer: (params) => pairScorer(items, (u, i) => logit(params as P, u, i)),
        embeddings: (params) => {
          const V = (params as P).V
          const flat = ffm ? reshape(V, [layout.features, F * k]) : V
          return {
            users: take(
              flat,
              Array.from({ length: users }, (_, u) => u),
            ) as Tensor,
            items: take(
              flat,
              Array.from({ length: items }, (_, i) => users + i),
            ) as Tensor,
          }
        },
      }
    }

    case 'wide-and-deep':
    case 'deepfm': {
      type P = { w0: Tensor; w: Tensor; cross: Tensor; V: Tensor; mlp: Params[] }
      const mlp = Mlp([layout.fields * k, ...hidden, 1])
      const crossOf = (u: number[], i: number[]) =>
        u.map((v, r) => layout.group(v) * layout.categories + layout.category(i[r]))
      const logit = (p: P, u: number[], i: number[]) => {
        const f = layout.featuresOf(u, i)
        const e = take(p.V, f) // [n, F, k]
        const deep = reshape(mlp.apply(p.mlp, reshape(e, [u.length, layout.fields * k])), [u.length])
        const linear = add(p.w0, sum(take(p.w, f), -1))
        if (kind === 'deepfm') return add(add(linear, fmInteraction(e)), deep)
        return add(add(linear, take(p.cross, crossOf(u, i))), deep)
      }
      return {
        kind,
        init: (s) => ({
          w0: fromData(Float64Array.of(0), []),
          w: vec(new Float64Array(layout.features)),
          cross: vec(new Float64Array(layout.groups * layout.categories)),
          V: table(child(s, 'V'), layout.features, k),
          mlp: mlp.init(child(s, 'mlp')),
        }),
        data: pointwise,
        loss: (params, b) => {
          const p = params as P
          const u = ids(b.user)
          const i = ids(b.item)
          return add(
            bce(logit(p, u, i), b.label),
            mul(lambda / rowsOf(u.length), l2(take(p.V, layout.featuresOf(u, i)))),
          )
        },
        scorer: (params) => pairScorer(items, (u, i) => logit(params as P, u, i)),
        embeddings: (params) => {
          const V = (params as P).V
          return {
            users: take(
              V,
              Array.from({ length: users }, (_, u) => u),
            ) as Tensor,
            items: take(
              V,
              Array.from({ length: items }, (_, i) => users + i),
            ) as Tensor,
          }
        },
      }
    }

    case 'neural-collaborative-filtering': {
      type P = { gmfUser: Tensor; gmfItem: Tensor; mlpUser: Tensor; mlpItem: Tensor; mlp: Params[]; out: Tensor }
      const last = hidden[hidden.length - 1] ?? k
      const mlp = Mlp([2 * k, ...hidden], { outputActivation: 'relu' })
      const logit = (p: P, u: number[], i: number[]) => {
        const gmf = mul(take(p.gmfUser, u), take(p.gmfItem, i)) // [n, k]
        const deep = mlp.apply(p.mlp, concat([take(p.mlpUser, u), take(p.mlpItem, i)], -1)) // [n, last]
        return reshape(matmul(concat([gmf, deep], -1), p.out), [u.length])
      }
      return {
        kind,
        init: (s) => ({
          gmfUser: table(child(s, 'gu'), users, k),
          gmfItem: table(child(s, 'gi'), items, k),
          mlpUser: table(child(s, 'mu'), users, k),
          mlpItem: table(child(s, 'mi'), items, k),
          mlp: mlp.init(child(s, 'mlp')),
          out: table(child(s, 'out'), k + last, 1, 0.3),
        }),
        data: pointwise,
        loss: (params, b) => {
          const p = params as P
          const u = ids(b.user)
          const i = ids(b.item)
          const reg = l2(take(p.gmfUser, u), take(p.gmfItem, i), take(p.mlpUser, u), take(p.mlpItem, i))
          return add(bce(logit(p, u, i), b.label), mul(lambda / rowsOf(u.length), reg))
        },
        scorer: (params) => pairScorer(items, (u, i) => logit(params as P, u, i)),
        embeddings: (params) => ({ users: (params as P).gmfUser, items: (params as P).gmfItem }),
      }
    }

    case 'two-tower': {
      type P = { user: Tensor; group: Tensor; item: Tensor; category: Tensor; userTower: Params[]; itemTower: Params[] }
      const userTower = Mlp([2 * k, ...hidden, k])
      const itemTower = Mlp([2 * k, ...hidden, k])
      const userVectors = (p: P, u: number[]) =>
        userTower.apply(p.userTower, concat([take(p.user, u), take(p.group, u.map(layout.group))], -1))
      const itemVectors = (p: P, i: number[]) =>
        itemTower.apply(p.itemTower, concat([take(p.item, i), take(p.category, i.map(layout.category))], -1))
      const allItems = Array.from({ length: items }, (_, i) => i)
      return {
        kind,
        init: (s) => ({
          user: table(child(s, 'user'), users, k),
          group: table(child(s, 'group'), layout.groups, k),
          item: table(child(s, 'item'), items, k),
          category: table(child(s, 'category'), layout.categories, k),
          userTower: userTower.init(child(s, 'userTower')),
          itemTower: itemTower.init(child(s, 'itemTower')),
        }),
        data: () => ({ user: vec(ctx.train.user), item: vec(ctx.train.item) }),
        loss: (params, b) => {
          const p = params as P
          const u = ids(b.user)
          const i = ids(b.item)
          // Each user's positive is its own item; the batch's other items are its negatives.
          const loss = infoNce(userVectors(p, u), itemVectors(p, i), { similarity: 'dot', temperature })
          return add(loss, mul(lambda / rowsOf(u.length), l2(take(p.user, u), take(p.item, i))))
        },
        scorer: (params) => (us) => {
          const p = params as P
          return Float64Array.from(
            toFlat(matmul(userVectors(p, [...us]), transpose(itemVectors(p, allItems))) as Tensor),
          )
        },
        embeddings: (params) => ({
          users: userVectors(
            params as P,
            Array.from({ length: users }, (_, u) => u),
          ) as Tensor,
          items: itemVectors(params as P, allItems) as Tensor,
        }),
      }
    }

    case 'sasrec': {
      const L = options.maxLength ?? 10
      const pad = items // the padding id: one extra row of the item table
      type P = { item: Tensor; position: Tensor; block: TransformerBlockParams }
      const block = TransformerBlock(k, { heads: 1, causal: true, hidden: 2 * k, placement: 'pre' })
      const sequences = ctx.sequences ?? itemsByUser(ctx.train).map((set) => Array.from(set))
      /** The last L items of a sequence, left-padded. */
      const window = (seq: readonly number[]) => {
        const tail = seq.slice(-L)
        return [...new Array<number>(L - tail.length).fill(pad), ...tail]
      }
      /** Hidden states [B, L, k] of padded windows [B, L]. */
      const hiddenStates = (p: P, windows: number[][]) => {
        const x = take(p.item, fromData(Float64Array.from(windows.flat()), [windows.length, L]))
        const positions = expandDims(
          take(
            p.position,
            Array.from({ length: L }, (_, t) => t),
          ),
          0,
        )
        return block.apply(p.block, add(x, positions))
      }
      const itemTable = (p: P) =>
        take(
          p.item,
          Array.from({ length: items }, (_, i) => i),
        )
      return {
        kind,
        init: (s) => ({
          item: table(child(s, 'item'), items + 1, k),
          position: table(child(s, 'position'), L, k),
          block: block.init(child(s, 'block')),
        }),
        data: () => {
          // One row per user: the window of inputs and the next item at each position (pad where there is none).
          const inputs: number[] = []
          const targets: number[] = []
          for (const seq of sequences) {
            if (seq.length < 2) continue
            inputs.push(...window(seq.slice(0, -1)))
            targets.push(...window(seq.slice(1)))
          }
          const n = inputs.length / L
          return {
            input: fromData(Float64Array.from(inputs), [n, L]),
            target: fromData(Float64Array.from(targets), [n, L]),
          }
        },
        loss: (params, b) => {
          const p = params as P
          const B = b.input.shape[0]
          const windows = Array.from({ length: B }, (_, r) => ids(b.input).slice(r * L, (r + 1) * L))
          const h = reshape(hiddenStates(p, windows), [B * L, k])
          const logits = matmul(h, transpose(itemTable(p)))
          // One-hot targets with zero rows at padded positions, so they add nothing to the summed loss.
          const t = ids(b.target)
          const onehot = new Float64Array(B * L * items)
          let real = 0
          t.forEach((j, r) => {
            if (j !== pad) ((onehot[r * items + j] = 1), real++)
          })
          const ce = softmaxCrossEntropy(logits, fromData(onehot, [B * L, items]), { reduction: 'sum' })
          return add(mul(1 / rowsOf(real), ce), mul(lambda, l2(p.item)))
        },
        scorer: (params) => (us) => {
          const p = params as P
          const h = hiddenStates(
            p,
            us.map((u) => window(sequences[u] ?? [])),
          )
          const lastStates = reshape(take(transpose(h, [1, 0, 2]), [L - 1]), [us.length, k])
          return Float64Array.from(toFlat(matmul(lastStates, transpose(itemTable(p))) as Tensor))
        },
        embeddings: (params) => ({ users: null, items: itemTable(params as P) as Tensor }),
      }
    }
  }
}

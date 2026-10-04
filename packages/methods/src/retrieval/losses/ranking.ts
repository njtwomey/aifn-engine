/**
 * Ranking losses of one list, or a batch of lists, as functions of the item scores s and graded relevance labels rel
 * (0 = irrelevant): pointwise (binary cross-entropy on rel > 0, squared error on the grades), pairwise (RankNet, the
 * pairwise hinge, BPR, LambdaRank's ΔNDCG-weighted RankNet) and listwise (softmax cross-entropy, ListNet, ListMLE,
 * ApproxNDCG). They reproduce the site's `ranking-losses.ts`, now vectorised over pairs and lists and differentiated
 * by `aifn-compute/foundation/autodiff` rather than by hand.
 *
 * Shapes: scores and grades have shape [n] (one list) or [B, n] (B lists of n items). Each loss sums over the items or
 * pairs of a list; the reduction then combines lists (`mean` by default). Pairs are the ordered (i, j) with
 * rel_i > rel_j, where i should rank above j. Gains are 2^rel − 1 unless `gain: 'linear'`, and positions are
 * discounted by 1/log₂(2 + position) with position 0 at the top, as in `aifn-compute/learning/metrics`.
 */

import { logSoftmax, sigmoid, softmax, softplus } from 'aifn-compute/numerics/special'
import {
  add,
  div,
  expandDims,
  fromData,
  log,
  logsumexp,
  matmul,
  maximum,
  mul,
  neg,
  shapeOfValue,
  square,
  squeeze,
  sub,
  sum,
  unwrap,
  where,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { defineLoss, type ReductionOptions, type Target } from 'aifn-compute/learning/losses'
import { dcg, gainFunction, positionDiscount as discount, type Gain as MetricGain } from 'aifn-compute/learning/metrics'
import { expectRank, flatValues, reduce } from 'aifn-compute/learning/losses'
import { integers, type Stream } from 'aifn-compute/foundation/random'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * How a relevance grade becomes a gain: `exponential` 2^rel − 1 (the default) or `linear` rel; the two named gains
 * of `aifn-compute/learning/metrics`' `Gain`.
 */
export type Gain = Extract<MetricGain, string>

const gainOf = (rel: number, gain: Gain) => gainFunction(gain)(rel)

/** A list's scores and grades as rows: `lists` rows of `n` items, each a Float64Array. */
type Rows = { lists: number; n: number; scores: Float64Array[]; grades: Float64Array[]; batched: boolean }

function rowsOf(scores: Value, relevance: Target): Rows {
  const shape = expectRank(scores, [1, 2], 'ranking scores')
  const batched = shape.length === 2
  const lists = batched ? shape[0] : 1
  const n = shape[shape.length - 1]
  const s = flatValues(unwrap(scores))
  const r = flatValues(relevance)
  if (r.length !== s.length)
    throw new ShapeError('losses', `losses: relevance has ${r.length} values for ${s.length} scores`)
  const cut = (a: Float64Array) => Array.from({ length: lists }, (_, b) => a.subarray(b * n, (b + 1) * n))
  return { lists, n, scores: cut(s), grades: cut(r), batched }
}

/** A constant tensor of shape [n, n] (one list) or [B, n, n], filled per list by `fill(b, i, j)`. */
function pairTensor(rows: Rows, fill: (b: number, i: number, j: number) => number): Tensor {
  const { lists, n } = rows
  const out = new Float64Array(lists * n * n)
  for (let b = 0; b < lists; b++)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[(b * n + i) * n + j] = fill(b, i, j)
  return fromData(out, rows.batched ? [lists, n, n] : [n, n])
}

/** The pair mask P[i, j] = 1 when rel_i > rel_j. */
function pairMask(rows: Rows): Tensor {
  return pairTensor(rows, (b, i, j) => (rows.grades[b][i] > rows.grades[b][j] ? 1 : 0))
}

/** The score differences s_i − s_j, shape [..., n, n]. */
function pairDifferences(scores: Value): Value {
  return sub(expandDims(scores, -1), expandDims(scores, -2))
}

/** Sum over a list's pairs, giving one value per list. */
function sumPairs(v: Value): Value {
  return sum(v, [-2, -1])
}

/** Positions (0 = top) of each item when sorted by score, ties broken by index. */
function positions(s: ArrayLike<number>): Int32Array {
  const order = Array.from({ length: s.length }, (_, i) => i).sort((a, b) => s[b] - s[a] || a - b)
  const pos = new Int32Array(s.length)
  order.forEach((item, r) => (pos[item] = r))
  return pos
}

/** The ideal DCG of a list's grades: `aifn-compute/learning/metrics`' `dcg` of the grades sorted best first. */
function idealDcg(grades: ArrayLike<number>, gain: Gain): number {
  return dcg(
    Float64Array.from(grades).sort((a, b) => b - a),
    { gain },
  )
}

const rankingInfo = (key: string, name: string, note: string, target?: string) =>
  ({
    key,
    name,
    module: 'applied/retrieval/losses',
    family: 'ranking',
    inputs: 'scores',
    notes: [note],
    ...(target ? { target } : {}),
  }) as const

// ── Pointwise ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Binary cross-entropy of each item's score as a logit, with label 1 for rel > 0: Σᵢ softplus(sᵢ) − yᵢsᵢ per list. */
export const pointwiseBce = defineLoss(
  rankingInfo('pointwiseBce', 'Pointwise binary cross-entropy', 'pointwise-recommendation-losses', 'P(relevant)'),
  (scores: Value, relevance: Target, { reduction }: ReductionOptions = {}): Value => {
    const y = fromData(
      flatValues(relevance).map((r) => (r > 0 ? 1 : 0)),
      shapeOfValue(scores),
    )
    return reduce(sum(sub(softplus(scores), mul(y, scores)), -1), reduction)
  },
)

/** Squared error between each item's score and its grade, Σᵢ (sᵢ − relᵢ)² per list. */
export const pointwiseSquaredError = defineLoss(
  rankingInfo(
    'pointwiseSquaredError',
    'Pointwise squared error',
    'pointwise-recommendation-losses',
    'the expected grade',
  ),
  (scores: Value, relevance: Target, { reduction }: ReductionOptions = {}): Value => {
    const rel = fromData(flatValues(relevance), shapeOfValue(scores))
    return reduce(sum(square(sub(scores, rel)), -1), reduction)
  },
)

// ── Pairwise ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `rankNet`. */
export type RankNetOptions = ReductionOptions & {
  /** The slope σ of the pair probability P(i ▷ j) = σ(σ·(sᵢ − sⱼ)). Default 1. */
  sigma?: number
}

/**
 * RankNet (Burges et al., 2005): the cross-entropy of the pair probability σ(σ(sᵢ − sⱼ)) against "i above j", summed
 * over the pairs with relᵢ > relⱼ: Σ softplus(−σ(sᵢ − sⱼ)). Unweighted, it bounds the number of misordered pairs.
 */
export const rankNet = defineLoss(
  rankingInfo('rankNet', 'RankNet (pairwise logistic)', 'pairwise-ranking-losses'),
  (scores: Value, relevance: Target, { reduction, sigma = 1 }: RankNetOptions = {}): Value => {
    const P = pairMask(rowsOf(scores, relevance))
    return reduce(sumPairs(mul(P, softplus(mul(-sigma, pairDifferences(scores))))), reduction)
  },
)

/** Options of `pairwiseHinge`. */
export type PairwiseHingeOptions = ReductionOptions & {
  /** The margin by which sᵢ should exceed sⱼ. Default 1. */
  margin?: number
}

/**
 * The pairwise hinge (RankSVM; Herbrich, Graepel & Obermayer, 2000; Joachims, 2002): Σ max(0, Δ − (sᵢ − sⱼ)) over the
 * pairs with relᵢ > relⱼ.
 */
export const pairwiseHinge = defineLoss(
  rankingInfo('pairwiseHinge', 'Pairwise hinge (RankSVM)', 'pairwise-ranking-losses'),
  (scores: Value, relevance: Target, { reduction, margin = 1 }: PairwiseHingeOptions = {}): Value => {
    const P = pairMask(rowsOf(scores, relevance))
    return reduce(sumPairs(mul(P, maximum(sub(margin, pairDifferences(scores)), 0))), reduction)
  },
)

/**
 * Bayesian personalised ranking (Rendle et al., 2009): −log σ(s⁺ − s⁻) for each (positive, sampled negative) pair.
 * `positive` has shape [B] and `negative` shape [B] or [B, k] (k negatives per positive); the losses of all pairs are
 * reduced together.
 */
export const bpr = defineLoss(
  {
    module: 'applied/retrieval/losses',
    key: 'bpr',
    name: 'Bayesian personalised ranking',
    family: 'ranking',
    inputs: 'scores',
    notes: ['bayesian-personalised-ranking'],
  },
  (positive: Value, negative: Value, { reduction }: ReductionOptions = {}): Value => {
    const pos = shapeOfValue(negative).length > shapeOfValue(positive).length ? expandDims(positive, -1) : positive
    return reduce(softplus(sub(negative, pos)), reduction)
  },
)

/** Options of `lambdaWeights` and `lambdaRank`. */
export type LambdaOptions = {
  gain?: Gain
}

/**
 * LambdaRank's pair weights |ΔNDCG_ij| (Burges, Ragno & Le, 2006; Burges, 2010): the change in the list's NDCG from
 * swapping items i and j in the ranking the current scores induce, |gᵢ − gⱼ|·|D(πᵢ) − D(πⱼ)| / IDCG, for pairs with
 * relᵢ > relⱼ (0 elsewhere). A constant tensor of shape [n, n] or [B, n, n]; ties in the scores break by index. A list
 * with no relevant item has weights 0.
 */
export function lambdaWeights(scores: Value, relevance: Target, { gain = 'exponential' }: LambdaOptions = {}): Tensor {
  const rows = rowsOf(scores, relevance)
  const pos = rows.scores.map(positions)
  const ideal = rows.grades.map((g) => idealDcg(g, gain))
  return pairTensor(rows, (b, i, j) => {
    const g = rows.grades[b]
    if (!(g[i] > g[j]) || ideal[b] === 0) return 0
    return Math.abs((gainOf(g[i], gain) - gainOf(g[j], gain)) * (discount(pos[b][i]) - discount(pos[b][j]))) / ideal[b]
  })
}

/**
 * LambdaRank as a loss: the RankNet pair losses weighted by |ΔNDCG_ij| (`lambdaWeights`), Σ |ΔNDCG_ij|·softplus(−σ(sᵢ −
 * sⱼ)). The weights are held constant, so its gradient in the scores is exactly the lambda gradients λᵢ; there is no
 * smooth loss whose gradient they are everywhere (LambdaLoss, Wang et al. 2018, gives one bound).
 */
export const lambdaRank = defineLoss(
  rankingInfo('lambdaRank', 'LambdaRank (ΔNDCG-weighted RankNet)', 'lambda-gradients', 'NDCG'),
  (scores: Value, relevance: Target, { reduction, sigma = 1, gain }: RankNetOptions & LambdaOptions = {}): Value => {
    const W = lambdaWeights(scores, relevance, { gain })
    return reduce(sumPairs(mul(W, softplus(mul(-sigma, pairDifferences(scores))))), reduction)
  },
)

/** Options of `warp` and `warpWeights`. */
export type WarpOptions = ReductionOptions & {
  /** The margin by which the positive should beat each negative. Default 1. */
  margin?: number
  /** The rank weights αₖ of L(r) = Σ_{k ≤ r} αₖ: `harmonic` αₖ = 1/k (the default) or `constant` αₖ = 1 (L(r) = r). */
  weighting?: 'harmonic' | 'constant'
  /**
   * Given, each positive's rank is estimated by sampling, as WARP trains: negatives are drawn uniformly (with
   * replacement) until one violates the margin; after N draws the rank estimate is ⌊M/N⌋. Omitted, the exact
   * margin-violating rank is counted over all M negatives.
   */
  stream?: Stream
  /** Most draws per positive when sampling (default M); a positive with no violator found contributes 0. */
  maxDraws?: number
}

/** L(r) = Σ_{k=1}^{r} αₖ, WARP's weight of a positive with margin-violating rank r (L(0) = 0). */
export function warpRankWeight(r: number, weighting: 'harmonic' | 'constant' = 'harmonic'): number {
  if (weighting === 'constant') return r
  let h = 0
  for (let k = 1; k <= r; k++) h += 1 / k
  return h
}

/**
 * WARP's constant pair weights w [B, M] (or [M] for one positive), with the loss Σⱼ wᵢⱼ max(0, Δ − sᵢ + sⱼ) (Weston,
 * Bengio & Usunier, 2011). Exact: with rᵢ = #{j : Δ + sⱼ > sᵢ} the margin-violating rank, every violator gets
 * L(rᵢ)/rᵢ, spreading L(rᵢ) evenly over the rᵢ hinge terms. Sampled (`stream`): the first violator found, on draw N,
 * gets L(⌊M/N⌋) and every other negative 0, a one-term estimate of the exact sum. Ties in the scores count as
 * violations only within the margin, as the hinge does.
 */
export function warpWeights(positive: Value, negatives: Value, options: WarpOptions = {}): Tensor {
  const { margin = 1, weighting = 'harmonic', stream } = options
  const shape = expectRank(negatives, [1, 2], 'warp negatives')
  const M = shape[shape.length - 1]
  const B = shape.length === 2 ? shape[0] : 1
  const sPos = flatValues(unwrap(positive))
  const sNeg = flatValues(unwrap(negatives))
  if (sPos.length !== B) throw new ShapeError('warp', `warp: ${sPos.length} positives for ${B} rows of negatives`)
  const maxDraws = options.maxDraws ?? M
  const w = new Float64Array(B * M)
  for (let b = 0; b < B; b++) {
    const violates = (j: number) => margin + sNeg[b * M + j] > sPos[b]
    if (stream === undefined) {
      let r = 0
      for (let j = 0; j < M; j++) if (violates(j)) r++
      if (r === 0) continue
      const each = warpRankWeight(r, weighting) / r
      for (let j = 0; j < M; j++) if (violates(j)) w[b * M + j] = each
    } else {
      for (let draw = 1; draw <= maxDraws; draw++) {
        const j = integers(stream, M)
        if (violates(j)) {
          w[b * M + j] = warpRankWeight(Math.floor(M / draw), weighting)
          break
        }
      }
    }
  }
  return fromData(w, shape)
}

/**
 * WARP, the weighted approximate-rank pairwise loss (Weston, Bengio & Usunier, 2011): for positive scores [B] and
 * negative scores [B, M], Σⱼ wᵢⱼ max(0, Δ − sᵢ + sⱼ) with the rank weights of `warpWeights` held constant, so a
 * positive buried under many violators gets a large update and one near the top a small one. Exact by default;
 * pass a `stream` for WARP's sampled rank estimate.
 */
export const warp = defineLoss(
  {
    ...rankingInfo('warp', 'WARP (weighted approximate-rank pairwise)', 'pairwise-ranking-losses', 'precision at k'),
    glossary: 'warp',
    cite: ['weston2011wsabie'],
  },
  (positive: Value, negatives: Value, options: WarpOptions = {}): Value => {
    const W = warpWeights(positive, negatives, options)
    const pos = expectRank(negatives, [1, 2], 'warp negatives').length === 2 ? expandDims(positive, -1) : positive
    const hinge = maximum(sub(add(options.margin ?? 1, negatives), pos), 0)
    return reduce(sum(mul(W, hinge), -1), options.reduction)
  },
)

// ── Listwise ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Per-list cross-entropy −Σᵢ tᵢ log softmax(s)ᵢ against constant target rows. */
function listCrossEntropy(scores: Value, target: Value): Value {
  return neg(sum(mul(target, logSoftmax(scores)), -1))
}

/**
 * Softmax cross-entropy over the list: the target is the grades normalised to sum to one (uniform when no item is
 * relevant), −Σᵢ (relᵢ/Σrel) log softmax(s)ᵢ. It bounds −log NDCG (Bruch et al., 2019).
 */
export const listwiseSoftmax = defineLoss(
  rankingInfo('listwiseSoftmax', 'Listwise softmax cross-entropy', 'listwise-ranking-losses'),
  (scores: Value, relevance: Target, { reduction }: ReductionOptions = {}): Value => {
    const rows = rowsOf(scores, relevance)
    const target = new Float64Array(rows.lists * rows.n)
    rows.grades.forEach((g, b) => {
      const total = g.reduce((a, v) => a + v, 0)
      g.forEach((v, i) => (target[b * rows.n + i] = total > 0 ? v / total : 1 / rows.n))
    })
    return reduce(listCrossEntropy(scores, fromData(target, shapeOfValue(scores))), reduction)
  },
)

/**
 * ListNet with top-one probabilities (Cao et al., 2007): the cross-entropy between softmax(rel) and softmax(s), per
 * list.
 */
export const listNet = defineLoss(
  rankingInfo('listNet', 'ListNet (top-one)', 'listwise-ranking-losses'),
  (scores: Value, relevance: Target, { reduction }: ReductionOptions = {}): Value => {
    const rel = fromData(flatValues(relevance), shapeOfValue(scores))
    return reduce(listCrossEntropy(scores, softmax(rel)), reduction)
  },
)

/**
 * ListMLE (Xia et al., 2008): the negative log-likelihood, under the Plackett–Luce model, of the permutation π that
 * sorts the list by relevance (ties by index): Σₖ [log Σ_{m ≥ k} e^{s_{π(m)}} − s_{π(k)}].
 */
export const listMle = defineLoss(
  rankingInfo('listMle', 'ListMLE (Plackett–Luce)', 'listwise-ranking-losses'),
  (scores: Value, relevance: Target, { reduction }: ReductionOptions = {}): Value => {
    const rows = rowsOf(scores, relevance)
    const { n } = rows
    const order = rows.grades.map((g) => Array.from({ length: n }, (_, i) => i).sort((a, b) => g[b] - g[a] || a - b))
    // A permutation matrix per list, so the sorted scores are a matrix product (differentiable).
    const perm = pairTensor(rows, (b, k, i) => (order[b][k] === i ? 1 : 0))
    const sorted = squeeze(matmul(perm, expandDims(scores, -1)), -1)
    const suffix = fromData(
      Float64Array.from({ length: n * n }, (_, q) => (q % n >= Math.floor(q / n) ? 1 : 0)),
      [n, n],
    )
    const tails = logsumexp(where(suffix, expandDims(sorted, -2), -Infinity), -1)
    return reduce(sum(sub(tails, sorted), -1), reduction)
  },
)

/** Options of `approxNdcg`. */
export type ApproxNdcgOptions = ReductionOptions & {
  /** The temperature T of the sigmoid that smooths each rank. Default 1; smaller is closer to the true rank. */
  temperature?: number
  gain?: Gain
}

/**
 * ApproxNDCG (Qin, Liu & Li, 2010): NDCG with each position replaced by the smooth rank π̂ᵢ = ½ + Σⱼ σ((sⱼ − sᵢ)/T)
 * (the j = i term contributes ½), so the loss −Σᵢ gᵢ / log₂(1 + π̂ᵢ) / IDCG is differentiable. It tends to −NDCG as T →
 * 0. A list with no relevant item contributes 0.
 */
export const approxNdcg = defineLoss(
  rankingInfo('approxNdcg', 'ApproxNDCG', 'listwise-ranking-losses', 'NDCG'),
  (scores: Value, relevance: Target, { reduction, temperature = 1, gain = 'exponential' }: ApproxNdcgOptions = {}) => {
    const rows = rowsOf(scores, relevance)
    const shape = shapeOfValue(scores)
    const weights = new Float64Array(rows.lists * rows.n)
    rows.grades.forEach((g, b) => {
      const ideal = idealDcg(g, gain)
      g.forEach((r, i) => (weights[b * rows.n + i] = ideal > 0 ? gainOf(r, gain) / ideal : 0))
    })
    // σ((sⱼ − sᵢ)/T) at [i, j]; summing over j includes σ(0) = ½ for j = i.
    const beaten = sigmoid(div(neg(pairDifferences(scores)), temperature))
    const rank = add(0.5, sum(beaten, -1))
    const dcg = sum(div(fromData(weights, shape), div(log(add(1, rank)), Math.LN2)), -1)
    return reduce(neg(dcg), reduction)
  },
)

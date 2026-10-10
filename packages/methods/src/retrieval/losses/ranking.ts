/**
 * Ranking losses of one list, or a batch of lists, as functions of the item scores $s_i$ and graded relevance labels
 * $\mathrm{rel}_i$ (0 is irrelevant): pointwise (binary cross-entropy on $\mathrm{rel}_i > 0$, squared error on the
 * grades), pairwise (RankNet, the pairwise hinge, BPR, LambdaRank's $\lvert \Delta \mathrm{NDCG} \rvert$-weighted
 * RankNet, WARP) and listwise (softmax cross-entropy, ListNet, ListMLE, ApproxNDCG). They reproduce the site's
 * `ranking-losses.ts`, now vectorised over pairs and lists and differentiated by `aifn-compute/foundation/autodiff`
 * rather than by hand.
 *
 * Shapes: scores and grades have shape `[n]` (one list) or `[B, n]` ($B$ lists of $n$ items). Each loss sums over the
 * items or pairs of a list; the reduction then combines lists (`mean` by default). Pairs are the ordered $(i, j)$ with
 * $\mathrm{rel}_i > \mathrm{rel}_j$, where $i$ should rank above $j$. Gains are $2^{\mathrm{rel}} - 1$ unless
 * `gain: 'linear'`, and positions are discounted by $1/\log_2(2 + p)$ with position $p = 0$ at the top, as in
 * `aifn-compute/learning/metrics`. Grades, pair weights and masks are constants: only the scores are differentiated.
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
 * How a relevance grade becomes a gain: `exponential` $2^{\mathrm{rel}} - 1$ (the default) or `linear`
 * $\mathrm{rel}$; the two named gains of `aifn-compute/learning/metrics`' `Gain`.
 */
export type Gain = Extract<MetricGain, string>

/**
 * The gain of one grade.
 *
 * @param rel The relevance grade $\mathrm{rel}$.
 * @param gain Which gain: `exponential` or `linear`.
 * @returns $2^{\mathrm{rel}} - 1$ or $\mathrm{rel}$.
 */
const gainOf = (rel: number, gain: Gain) => gainFunction(gain)(rel)

/**
 * A batch's scores and grades as rows: `lists` rows of `n` items, each a `Float64Array` view of one list; `batched` is
 * true when the scores had shape `[B, n]`.
 */
type Rows = { lists: number; n: number; scores: Float64Array[]; grades: Float64Array[]; batched: boolean }

/**
 * Read the scores' values and the grades as one row per list. Throws `ShapeError` when the scores are not of rank 1 or
 * 2, or when there are not as many grades as scores.
 *
 * @param scores The scores, shape `[n]` or `[B, n]` (a traced value is read through `unwrap`).
 * @param relevance The grades, as many as the scores, in the same row-major order.
 * @returns The scores and grades cut into rows.
 */
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

/**
 * A constant tensor of shape `[n, n]` (one list) or `[B, n, n]`, filled per list by `fill(b, i, j)`.
 *
 * @param rows The lists, which fix $B$, $n$ and whether the result has a batch axis.
 * @param fill The entry at list `b`, row `i`, column `j`.
 * @returns The filled tensor.
 */
function pairTensor(rows: Rows, fill: (b: number, i: number, j: number) => number): Tensor {
  const { lists, n } = rows
  const out = new Float64Array(lists * n * n)
  for (let b = 0; b < lists; b++)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[(b * n + i) * n + j] = fill(b, i, j)
  return fromData(out, rows.batched ? [lists, n, n] : [n, n])
}

/**
 * The pair mask $P_{ij} = 1$ when $\mathrm{rel}_i > \mathrm{rel}_j$, and 0 otherwise.
 *
 * @param rows The lists, whose grades are compared.
 * @returns The mask, shape `[n, n]` or `[B, n, n]`.
 */
function pairMask(rows: Rows): Tensor {
  return pairTensor(rows, (b, i, j) => (rows.grades[b][i] > rows.grades[b][j] ? 1 : 0))
}

/**
 * The score differences $s_i - s_j$ at $[i, j]$ (differentiable).
 *
 * @param scores The scores, shape `[..., n]`.
 * @returns The differences, shape `[..., n, n]`.
 */
function pairDifferences(scores: Value): Value {
  return sub(expandDims(scores, -1), expandDims(scores, -2))
}

/**
 * Sum over a list's pairs, giving one value per list.
 *
 * @param v Per-pair values, shape `[..., n, n]`.
 * @returns The sums over the last two axes, shape `[...]`.
 */
function sumPairs(v: Value): Value {
  return sum(v, [-2, -1])
}

/**
 * Positions (0 at the top) of each item when sorted by score, highest first, ties broken by index.
 *
 * @param s The scores of one list.
 * @returns The position of each item, in item order.
 */
function positions(s: ArrayLike<number>): Int32Array {
  const order = Array.from({ length: s.length }, (_, i) => i).sort((a, b) => s[b] - s[a] || a - b)
  const pos = new Int32Array(s.length)
  order.forEach((item, r) => (pos[item] = r))
  return pos
}

/**
 * The ideal DCG of a list's grades: `aifn-compute/learning/metrics`' `dcg` of the grades sorted best first.
 *
 * @param grades The grades of one list (not modified).
 * @param gain Which gain turns a grade into a value.
 * @returns The largest DCG any ordering of the list reaches; 0 when no item is relevant.
 */
function idealDcg(grades: ArrayLike<number>, gain: Gain): number {
  return dcg(
    Float64Array.from(grades).sort((a, b) => b - a),
    { gain },
  )
}

/**
 * The registry metadata of a ranking loss of this module.
 *
 * @param key The loss's key in `retrievalLossRegistry`, its export name.
 * @param name The loss's display name.
 * @param note The slug of the site note that explains it.
 * @param target What the loss's minimiser estimates, when it has a named target.
 * @returns The `info` that `defineLoss` attaches.
 */
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

/**
 * Binary cross-entropy of each item's score as a logit, with label $y_i = 1$ for $\mathrm{rel}_i > 0$:
 * $\sum_i \operatorname{softplus}(s_i) - y_i s_i$ per list.
 *
 * @param scores The scores $s_i$ as logits, shape `[n]` or `[B, n]`.
 * @param relevance The grades, of the same size; only whether each is positive is read.
 * @param options The reduction over lists.
 * @returns The loss, reduced over lists (mean by default).
 *
 * @example Confident and right, against confident and wrong
 * const rel = [2, 0, 1]
 * print('right:', pointwiseBce(tensor([3, -3, 3]), rel))
 * print('wrong:', pointwiseBce(tensor([-3, 3, -3]), rel))
 * print('3 softplus(-3) =', 3 * Math.log(1 + Math.exp(-3)))
 */
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

/**
 * Squared error between each item's score and its grade, $\sum_i (s_i - \mathrm{rel}_i)^2$ per list.
 *
 * @param scores The scores $s_i$, shape `[n]` or `[B, n]`.
 * @param relevance The grades, of the same size, as regression targets.
 * @param options The reduction over lists.
 * @returns The loss, reduced over lists (mean by default).
 *
 * @example Two lists: one exact, one off by a half on each item
 * const rel = tensor([[2, 0, 1], [2, 0, 1]])
 * print('per list:', pointwiseSquaredError(tensor([[2, 0, 1], [2.5, 0.5, 1.5]]), rel, { reduction: 'none' }))
 */
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
  /**
   * The slope $\sigma$ of the pair probability $\pr(i \succ j) = \operatorname{sigmoid}(\sigma (s_i - s_j))$. Default
   * 1.
   */
  sigma?: number
}

/**
 * RankNet (Burges et al., 2005): the cross-entropy of the pair probability $\operatorname{sigmoid}(\sigma (s_i - s_j))$
 * against "$i$ above $j$", summed over the pairs with $\mathrm{rel}_i > \mathrm{rel}_j$:
 * $\sum_{ij} P_{ij} \operatorname{softplus}(-\sigma (s_i - s_j))$. With $\sigma = 1$ each misordered pair costs at
 * least $\log 2$, so the loss over $\log 2$ bounds the number of misordered pairs.
 *
 * @param scores The scores $s_i$, shape `[n]` or `[B, n]`.
 * @param relevance The grades, of the same size, which define the pairs.
 * @param options The slope $\sigma$ and the reduction over lists.
 * @returns The loss, reduced over lists (mean by default).
 *
 * @example The loss of a list in order and of the same list reversed
 * const rel = [2, 1, 0]
 * print('in order:', rankNet(tensor([2, 1, 0]), rel))
 * print('reversed:', rankNet(tensor([0, 1, 2]), rel))
 * const sp = (x) => Math.log(1 + Math.exp(x))
 * print('2 softplus(-1) + softplus(-2) =', 2 * sp(-1) + sp(-2))
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
  /** The margin $\Delta$ by which $s_i$ should exceed $s_j$. Default 1. */
  margin?: number
}

/**
 * The pairwise hinge (RankSVM; Herbrich, Graepel & Obermayer, 2000; Joachims, 2002):
 * $\sum_{ij} P_{ij} \max(0, \Delta - (s_i - s_j))$ over the pairs with $\mathrm{rel}_i > \mathrm{rel}_j$. It is zero
 * once every pair is ordered with room $\Delta$ to spare.
 *
 * @param scores The scores $s_i$, shape `[n]` or `[B, n]`.
 * @param relevance The grades, of the same size, which define the pairs.
 * @param options The margin $\Delta$ and the reduction over lists.
 * @returns The loss, reduced over lists (mean by default).
 *
 * @example Zero when every pair is ordered by the margin
 * const rel = [2, 1, 0]
 * print('ordered by 2 and 1:', pairwiseHinge(tensor([3, 1, 0]), rel))
 * print('ordered by 0.5:', pairwiseHinge(tensor([1, 0.5, 0]), rel))
 * print('reversed:', pairwiseHinge(tensor([0, 1, 3]), rel))
 */
export const pairwiseHinge = defineLoss(
  rankingInfo('pairwiseHinge', 'Pairwise hinge (RankSVM)', 'pairwise-ranking-losses'),
  (scores: Value, relevance: Target, { reduction, margin = 1 }: PairwiseHingeOptions = {}): Value => {
    const P = pairMask(rowsOf(scores, relevance))
    return reduce(sumPairs(mul(P, maximum(sub(margin, pairDifferences(scores)), 0))), reduction)
  },
)

/**
 * Bayesian personalised ranking (Rendle et al., 2009): $-\log \operatorname{sigmoid}(s^+ - s^-)$ for each (positive,
 * sampled negative) pair, computed as $\operatorname{softplus}(s^- - s^+)$. The losses of all pairs are reduced
 * together.
 *
 * @param positive The positive items' scores $s^+$, shape `[B]`.
 * @param negative The negatives' scores $s^-$, shape `[B]` (one per positive) or `[B, k]` ($k$ per positive).
 * @param options The reduction over pairs.
 * @returns The loss, reduced over every pair (mean by default).
 *
 * @example Each pair's loss as the positive's lead grows
 * print('per pair:', bpr(tensor([-2, 0, 2]), tensor([0, 0, 0]), { reduction: 'none' }))
 * print('-log sigmoid(2) =', Math.log(1 + Math.exp(-2)))
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

/** Options of `lambdaWeights` and `lambdaRank`: `gain`, how a grade becomes a gain (default `exponential`). */
export type LambdaOptions = {
  /** How a grade becomes a gain (default `exponential`). */
  gain?: Gain
}

/**
 * LambdaRank's pair weights $\lvert \Delta \mathrm{NDCG}_{ij} \rvert$ (Burges, Ragno & Le, 2006; Burges, 2010): the
 * change in the list's NDCG from swapping items $i$ and $j$ in the ranking the current scores induce,
 * $\lvert g_i - g_j \rvert \, \lvert D(\pi_i) - D(\pi_j) \rvert / \mathrm{IDCG}$, with $g$ the gains, $\pi_i$ item
 * $i$'s position and $D$ the discount, for pairs with $\mathrm{rel}_i > \mathrm{rel}_j$ (0 elsewhere). Ties in the
 * scores break by index. A list with no relevant item has weights 0.
 *
 * @param scores The current scores, shape `[n]` or `[B, n]`; read as values only, to rank the items.
 * @param relevance The grades, of the same size.
 * @param options The options.
 * @param options.gain How a grade becomes a gain (default `exponential`).
 * @returns The weights, a constant tensor of shape `[n, n]` or `[B, n, n]`, row $i$ and column $j$.
 *
 * @example Only the pairs with a higher grade first get weight
 * print(lambdaWeights(tensor([0, 1, 2]), [2, 0, 1]))
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
 * LambdaRank as a loss: the RankNet pair losses weighted by $\lvert \Delta \mathrm{NDCG}_{ij} \rvert$
 * (`lambdaWeights`), $\sum_{ij} \lvert \Delta \mathrm{NDCG}_{ij} \rvert \operatorname{softplus}(-\sigma (s_i - s_j))$.
 * The weights are held constant, so its gradient in the scores is exactly the lambda gradients $\lambda_i$; there is
 * no smooth loss whose gradient they are everywhere (LambdaLoss, Wang et al. 2018, gives one bound).
 *
 * @param scores The scores $s_i$, shape `[n]` or `[B, n]`.
 * @param relevance The grades, of the same size.
 * @param options The slope $\sigma$ (default 1), the gain (default `exponential`) and the reduction over lists.
 * @returns The loss, reduced over lists (mean by default).
 *
 * @example A misordered top costs more than a misordered bottom
 * const rel = [3, 2, 1, 0]
 * print('in order:', lambdaRank(tensor([4, 3, 2, 1]), rel))
 * print('bottom two swapped:', lambdaRank(tensor([4, 3, 1, 2]), rel))
 * print('top two swapped:', lambdaRank(tensor([3, 4, 2, 1]), rel))
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
  /** The margin $\Delta$ by which the positive should beat each negative. Default 1. */
  margin?: number
  /**
   * The rank weights $\alpha_k$ of $L(r) = \sum_{k \le r} \alpha_k$: `harmonic` $\alpha_k = 1/k$ (the default) or
   * `constant` $\alpha_k = 1$ ($L(r) = r$).
   */
  weighting?: 'harmonic' | 'constant'
  /**
   * Given, each positive's rank is estimated by sampling, as WARP trains: negatives are drawn uniformly (with
   * replacement) until one violates the margin; after $N$ draws the rank estimate is $\lfloor M/N \rfloor$. Omitted,
   * the exact margin-violating rank is counted over all $M$ negatives.
   */
  stream?: Stream
  /** Most draws per positive when sampling (default $M$); a positive with no violator found contributes 0. */
  maxDraws?: number
}

/**
 * $L(r) = \sum_{k=1}^{r} \alpha_k$, WARP's weight of a positive with margin-violating rank $r$ ($L(0) = 0$).
 *
 * @param r The rank $r$: how many negatives violate the margin (a whole number).
 * @param weighting `harmonic` ($\alpha_k = 1/k$, so $L(r)$ is the $r$-th harmonic number) or `constant` ($L(r) = r$).
 * @returns $L(r)$.
 *
 * @example Harmonic weights grow like log r, constant ones like r
 * print('harmonic:', [0, 1, 2, 3, 10].map((r) => warpRankWeight(r)))
 * print('constant:', [0, 1, 2, 3, 10].map((r) => warpRankWeight(r, 'constant')))
 */
export function warpRankWeight(r: number, weighting: 'harmonic' | 'constant' = 'harmonic'): number {
  if (weighting === 'constant') return r
  let h = 0
  for (let k = 1; k <= r; k++) h += 1 / k
  return h
}

/**
 * WARP's constant pair weights $w_{ij}$, with the loss $\sum_j w_{ij} \max(0, \Delta - s_i + s_j)$ (Weston, Bengio &
 * Usunier, 2011). Exact: with $r_i = \lvert \{ j : \Delta + s_j > s_i \} \rvert$ the margin-violating rank, every
 * violator gets $L(r_i)/r_i$, spreading $L(r_i)$ evenly over the $r_i$ hinge terms. Sampled (`stream`): the first
 * violator found, on draw $N$, gets $L(\lfloor M/N \rfloor)$ and every other negative 0, a one-term estimate of the
 * exact sum. A negative counts as a violator only when $\Delta + s_j > s_i$ strictly, as the hinge is then positive.
 * Throws `ShapeError` when there is not one positive per row of negatives.
 *
 * @param positive The positives' scores $s_i$, shape `[B]` (or one score for a single row of negatives).
 * @param negatives The negatives' scores $s_j$, shape `[B, M]` or `[M]`; read as values only.
 * @param options The margin $\Delta$, the rank weighting, and `stream` and `maxDraws` for the sampled estimate.
 * @returns The weights, a constant tensor of the shape of `negatives`.
 *
 * @example Three of four negatives violate the margin, so each gets L(3)/3
 * print('exact:', warpWeights(tensor([1]), tensor([[0.5, 2, -3, 0.8]])))
 * print('L(3)/3 =', warpRankWeight(3) / 3)
 * print('sampled:', warpWeights(tensor([1]), tensor([[0.5, 2, -3, 0.8]]), { stream: stream(0) }))
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
 * WARP, the weighted approximate-rank pairwise loss (Weston, Bengio & Usunier, 2011):
 * $\sum_j w_{ij} \max(0, \Delta - s_i + s_j)$ with the rank weights of `warpWeights` held constant, so a positive
 * buried under many violators gets a large update and one near the top a small one. Exact by default; pass a
 * `stream` for WARP's sampled rank estimate. As in LightFM's `warp` loss.
 *
 * @param positive The positives' scores $s_i$, shape `[B]` (or one score for a single row of negatives).
 * @param negatives The negatives' scores $s_j$, shape `[B, M]` or `[M]`.
 * @param options The margin $\Delta$, the rank weighting, the sampling `stream` and `maxDraws`, and the reduction.
 * @returns The loss, reduced over positives (mean by default).
 *
 * @example A positive near the top against one buried under the negatives
 * const negatives = tensor([[0, 1, 2.5]])
 * print('near the top:', warp(tensor([3]), negatives))
 * print('buried:', warp(tensor([0]), negatives))
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

/**
 * Per-list cross-entropy $-\sum_i t_i \log \operatorname{softmax}(\svec)_i$ against constant target rows.
 *
 * @param scores The scores $\svec$, shape `[n]` or `[B, n]`.
 * @param target The target distribution $\tvec$ of each list, of the same shape.
 * @returns The cross-entropy of each list, shape `[]` or `[B]`.
 */
function listCrossEntropy(scores: Value, target: Value): Value {
  return neg(sum(mul(target, logSoftmax(scores)), -1))
}

/**
 * Softmax cross-entropy over the list: the target is the grades normalised to sum to one (uniform when no item is
 * relevant), $-\sum_i (\mathrm{rel}_i / \sum_j \mathrm{rel}_j) \log \operatorname{softmax}(\svec)_i$. It bounds
 * $-\log \mathrm{NDCG}$ (Bruch et al., 2019).
 *
 * @param scores The scores $\svec$, shape `[n]` or `[B, n]`.
 * @param relevance The grades, of the same size (non-negative).
 * @param options The reduction over lists.
 * @returns The loss, reduced over lists (mean by default).
 *
 * @example The loss of a list in order and of the same list reversed
 * const rel = [2, 1, 0]
 * print('in order:', listwiseSoftmax(tensor([2, 1, 0]), rel))
 * print('reversed:', listwiseSoftmax(tensor([0, 1, 2]), rel))
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
 * ListNet with top-one probabilities (Cao et al., 2007): the cross-entropy between
 * $\operatorname{softmax}(\mathrm{rel})$ and $\operatorname{softmax}(\svec)$, per list. It is smallest when the scores
 * equal the grades up to a constant, where it is the entropy of $\operatorname{softmax}(\mathrm{rel})$.
 *
 * @param scores The scores $\svec$, shape `[n]` or `[B, n]`.
 * @param relevance The grades, of the same size.
 * @param options The reduction over lists.
 * @returns The loss, reduced over lists (mean by default).
 *
 * @example Scores equal to the grades reach the minimum, a shift leaves it unchanged
 * const rel = [2, 1, 0]
 * print('scores = grades:', listNet(tensor([2, 1, 0]), rel))
 * print('scores = grades + 5:', listNet(tensor([7, 6, 5]), rel))
 * print('reversed:', listNet(tensor([0, 1, 2]), rel))
 */
export const listNet = defineLoss(
  rankingInfo('listNet', 'ListNet (top-one)', 'listwise-ranking-losses'),
  (scores: Value, relevance: Target, { reduction }: ReductionOptions = {}): Value => {
    const rel = fromData(flatValues(relevance), shapeOfValue(scores))
    return reduce(listCrossEntropy(scores, softmax(rel)), reduction)
  },
)

/**
 * ListMLE (Xia et al., 2008): the negative log-likelihood, under the Plackett–Luce model, of the permutation $\pi$
 * that sorts the list by relevance (ties by index): $\sum_k [\log \sum_{m \ge k} e^{s_{\pi(m)}} - s_{\pi(k)}]$.
 *
 * @param scores The scores $s_i$, shape `[n]` or `[B, n]`.
 * @param relevance The grades, of the same size; only their order is used.
 * @param options The reduction over lists.
 * @returns The loss, reduced over lists (mean by default).
 *
 * @example Two items: the loss is softplus of the wrong one's lead
 * print('in order:', listMle(tensor([1, 0]), [1, 0]), ' softplus(-1) =', Math.log(1 + Math.exp(-1)))
 * print('reversed:', listMle(tensor([0, 1]), [1, 0]), ' softplus(1) =', Math.log(1 + Math.exp(1)))
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
  /** The temperature $T$ of the sigmoid that smooths each rank. Default 1; smaller is closer to the true rank. */
  temperature?: number
  /** How a grade becomes a gain (default `exponential`). */
  gain?: Gain
}

/**
 * ApproxNDCG (Qin, Liu & Li, 2010): NDCG with each 1-based rank replaced by the smooth rank
 * $\hat\pi_i = \tfrac{1}{2} + \sum_j \operatorname{sigmoid}((s_j - s_i)/T)$ (the $j = i$ term contributes
 * $\tfrac{1}{2}$), so the loss $-\sum_i g_i / \log_2(1 + \hat\pi_i) / \mathrm{IDCG}$ is differentiable. It tends to
 * $-\mathrm{NDCG}$ as $T \to 0$. A list with no relevant item contributes 0.
 *
 * @param scores The scores $s_i$, shape `[n]` or `[B, n]`.
 * @param relevance The grades, of the same size.
 * @param options The temperature $T$, the gain and the reduction over lists.
 * @returns The loss, reduced over lists (mean by default): between $-1$ and 0 per list.
 *
 * @example A list in ideal order tends to minus one as the temperature falls
 * const rel = [2, 1, 0]
 * for (const temperature of [1, 0.1, 0.01])
 *   print('T =', temperature, ':', approxNdcg(tensor([2, 1, 0]), rel, { temperature }))
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

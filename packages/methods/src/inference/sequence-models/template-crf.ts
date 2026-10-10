/**
 * The linear-chain CRF over CRF++-style feature templates: its potentials and decodings, its likelihood and gradient,
 * and its training by L-BFGS, OWL-QN, SGD or Adam.
 *
 * The model is the linear-chain CRF of Lafferty, McCallum & Pereira (2001) with the features of CRF++ (Kudo 2005) and
 * CRFsuite (Okazaki 2007), in the notation of Twomey, Diethe & Flach (2016):
 *
 * $P(\yvec \mid \xvec) = \frac{1}{Z} \prod_n \psi_n(y_n) \prod_{n \ge 1} \Psi_{n-1}(y_{n-1}, y_n)$, with
 * $\psi_n(k) = \exp \sum_{u \text{ fires at } n} \lambda_{u,k}$ and
 * $\Psi_{n-1}(i, k) = \exp \sum_{b \text{ fires at } n} \lambda_{b,i,k}$.
 *
 * Every unigram string $u$ of the feature index (`aifn-compute/text/features`) is $K$ feature functions
 * $f_{u,k}(y_n, \xvec, n) = [u \text{ fires at } n][y_n = k]$, and every bigram string $b$ is $K^2$ functions
 * $f_{b,i,k}(y_{n-1}, y_n, \xvec, n) = [b \text{ fires at } n][y_{n-1} = i][y_n = k]$. The weights $\lambdavec$ are
 * one vector: the unigram block (string $u$, label $k$ at $uK + k$), then the bigram block (string $b$, labels $i, k$
 * at $UK + bK^2 + iK + k$), with $U$ and $B$ the numbers of unigram and bigram strings. A sequence is its token rows
 * (one row of string columns per position), which the index encodes; strings the index does not hold fire nothing.
 *
 * Inference runs the chain engines of `aifn-compute/inference/exact` on the log-potentials $\log \psi_n$
 * ($N \times K$) and $\log \Psi_n$ ($(N - 1) \times K \times K$): forward messages
 * $\alphavec_n = \Psimat_{n-1}^\top \gammavec_{n-1}$ with $\gammavec_n = \alphavec_n \odot \psivec_n$, backward
 * messages $\betavec_n = \Psimat_n \deltavec_{n+1}$ with $\deltavec_{n+1} = \betavec_{n+1} \odot \psivec_{n+1}$,
 * marginals $\propto \alphavec_n \odot \psivec_n \odot \betavec_n$, and Viterbi.
 *
 * Training minimises the regularised negative conditional log-likelihood
 *
 * $L(\lambdavec) = -\sum_m \log P(\yvec_m \mid \xvec_m) + c_1 \norm{\lambdavec}_1 + c_2 \norm{\lambdavec}_2^2$
 *
 * (CRFsuite's `c1` and `c2`; CRF++'s `-c C` is $c_2 = 1/(2C)$ with L2, $c_1 = 1/C$ with L1). Its gradient is expected
 * minus observed feature counts (Sutton & McCallum 2012, eq. 5.6), read from the node and pairwise marginals. L-BFGS
 * ($c_1 = 0$) and OWL-QN ($c_1 > 0$) come from `aifn-compute/optim/second-order`; SGD and Adam take minibatches of
 * sequences.
 *
 * The examples write out the parse of a template file by hand, abridged to the fields the CRF reads, because
 * `parseTemplates` of `aifn-compute/text/features` is not in their scope; in code, parse the file with it.
 */

import type { Size, Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, permutation, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import {
  chainForwardBackward,
  chainViterbi,
  posteriorDecode,
  type ChainMarginals,
  type PosteriorDecoding,
  type ViterbiResult,
} from 'aifn-compute/inference/exact'
import { adamRule, sgdRule, type UpdateRule } from 'aifn-compute/optim/first-order'
import { owlqn, type OwlqnState } from 'aifn-compute/optim/second-order'
import {
  encodeTemplateRows,
  featureIndex,
  type EncodedSequence,
  type FeatureIndex,
  type FeatureTemplates,
  type TemplateMacro,
  type TokenRows,
} from 'aifn-compute/text/features'

/** A training sequence: token rows and a label per position. */
export interface LabelledSequence {
  /** The token rows: one row of string columns per position. */
  readonly rows: TokenRows
  /** The label of each position, as a string (one per row). */
  readonly labels: readonly string[]
}

/** A linear-chain CRF over the strings of a feature index (plain data). */
export interface TemplateCrf {
  /** The tag `template-crf`. */
  readonly kind: 'template-crf'
  /** The feature index: the templates and the unigram and bigram strings the weights belong to. */
  readonly index: FeatureIndex
  /** The label set, in weight order. */
  readonly labels: readonly string[]
  /** $\lambdavec$: the unigram block ($U \times K$), then the bigram block ($B \times K \times K$), row-major. */
  readonly weights: Float64Array
}

/**
 * The number of weights of a CRF over `index` with `K` labels: $UK + BK^2$.
 *
 * @param index The feature index, with $U$ unigram and $B$ bigram strings.
 * @param K The number of labels.
 * @returns The length of the weight vector $\lambdavec$.
 *
 * @example The word templates on three sentences
 * // `parseTemplates('U00:%x[0,0]\nB')` of aifn-compute/text/features, abridged: the word, and label transitions
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus().slice(0, 3))
 * print('U =', p.U, 'B =', p.index.bigram.length, 'K =', p.K)
 * print('weights:', crfWeightCount(p.index, p.K))
 */
export const crfWeightCount = (index: FeatureIndex, K: Size): Size =>
  index.unigram.length * K + index.bigram.length * K * K

/**
 * A CRF over an index and labels, with zero weights unless given. Throws `DomainError` for fewer than two labels, or
 * for `weights` of the wrong length.
 *
 * @param index The feature index the weights belong to.
 * @param labels The label set, in weight order; at least two.
 * @param weights The weight vector $\lambdavec$, of length `crfWeightCount(index, labels.length)`; copied. Zeros when
 *   left out.
 * @returns The CRF.
 *
 * @example An untrained CRF labels every position alike
 * // `parseTemplates('U00:%x[0,0]\nB')` of aifn-compute/text/features, abridged: the word, and label transitions
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const crf = templateCrf(p.index, p.labels)
 * print('labels:', crf.labels, 'weights:', crf.weights.length)
 * print(templateCrfMarginals(crf, posRows(['The', 'dog', 'runs'])).marginals)
 *
 * @example With trained weights
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const fit = run(crfTraining(p), undefined, 30)
 * const crf = templateCrf(p.index, p.labels, toFlat(fit.weights))
 * print(templateCrfViterbi(crf, posRows(['They', 'watch', 'the', 'light', '.'])).labels)
 */
export function templateCrf(index: FeatureIndex, labels: readonly string[], weights?: ArrayLike<number>): TemplateCrf {
  const D = crfWeightCount(index, labels.length)
  if (labels.length < 2) throw new DomainError('templateCrf', 'templateCrf: needs at least two labels')
  if (weights && weights.length !== D)
    throw new DomainError('templateCrf', `templateCrf: expected ${D} weights, got ${weights.length}`)
  return { kind: 'template-crf', index, labels, weights: weights ? Float64Array.from(weights) : new Float64Array(D) }
}

/**
 * The labels of the data in order of first appearance.
 *
 * @param data The labelled sequences.
 * @returns Each distinct label once.
 *
 * @example The tags of two sentences
 * print(labelSet(toyPosCorpus().slice(0, 2)))
 */
export function labelSet(data: readonly LabelledSequence[]): string[] {
  const seen = new Set<string>()
  for (const s of data) for (const y of s.labels) seen.add(y)
  return [...seen]
}

/**
 * The log-potentials of an encoded sequence: $\log \psi_n(k)$ ($N \times K$) and $\log \Psi_n(i, k)$ between positions
 * $n$ and $n + 1$ ($(N - 1) \times K \times K$), each the sum of the weights of the strings that fire.
 *
 * @param weights The weight vector $\lambdavec$ ($UK + BK^2$ values; see the file comment for its layout).
 * @param e The sequence encoded against the index the weights belong to (`encodeTemplateRows`, or the `sequence` of
 *   `encodeLabelled`).
 * @param U The number of unigram strings of the index, which places the bigram block at $UK$.
 * @param K The number of labels.
 * @returns `logUnary` ($N \times K$) and `logPairwise` ($(N - 1) \times K \times K$, slice $n$ between positions $n$
 *   and $n + 1$).
 *
 * @example With every weight 0.1, one word string and one `B` string fire at each position
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus().slice(0, 1))
 * const w = new Float64Array(p.dimension).fill(0.1)
 * const { logUnary, logPairwise } = crfLogPotentials(w, p.encoded[0].sequence, p.U, p.K)
 * print('log ψ =', logUnary)
 * print('log Ψ shape:', logPairwise.shape, 'first entry:', logPairwise.data[0])
 */
export function crfLogPotentials(
  weights: ArrayLike<number>,
  e: EncodedSequence,
  U: Size,
  K: Size,
): { logUnary: Matrix; logPairwise: Tensor } {
  const N = e.length
  const unary = new Float64Array(N * K)
  for (let n = 0; n < N; n++)
    for (let j = e.unigramStart[n]; j < e.unigramStart[n + 1]; j++) {
      const at = e.unigramIds[j] * K
      for (let k = 0; k < K; k++) unary[n * K + k] += weights[at + k]
    }
  const KK = K * K
  const pair = new Float64Array(Math.max(N - 1, 0) * KK)
  const off = U * K
  for (let n = 1; n < N; n++)
    for (let j = e.bigramStart[n]; j < e.bigramStart[n + 1]; j++) {
      const at = off + e.bigramIds[j] * KK
      for (let q = 0; q < KK; q++) pair[(n - 1) * KK + q] += weights[at + q]
    }
  return { logUnary: fromData(unary, [N, K]), logPairwise: fromData(pair, [Math.max(N - 1, 0), K, K]) }
}

/**
 * The log-potentials of token rows under a CRF: the rows encoded against its index, then `crfLogPotentials`.
 *
 * @param crf The CRF.
 * @param rows The token rows of the sequence.
 * @returns `logUnary` ($N \times K$) and `logPairwise` ($(N - 1) \times K \times K$).
 */
const potentialsOf = (crf: TemplateCrf, rows: TokenRows) =>
  crfLogPotentials(crf.weights, encodeTemplateRows(crf.index, rows), crf.index.unigram.length, crf.labels.length)

/**
 * The marginals of a sequence by log-space forward–backward (`chainForwardBackward`): $\log \alphavec_n$,
 * $\log \betavec_n$, $P(y_n \mid \xvec)$, the pairwise marginals and $\log Z$.
 *
 * @param crf The CRF.
 * @param rows The token rows of the sequence, with the columns the templates read.
 * @returns The log messages, the marginals ($N \times K$, in the order of `crf.labels`), the pairwise marginals and
 *   $\log Z$.
 *
 * @example How sure the tagger is of each tag of "they watch the light"
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const crf = templateCrf(p.index, p.labels, toFlat(run(crfTraining(p), undefined, 30).weights))
 * const m = templateCrfMarginals(crf, posRows(['They', 'watch', 'the', 'light', '.']))
 * print('labels:', crf.labels)
 * toRows(m.marginals).forEach((row, n) => print(`position ${n}:`, row))
 */
export function templateCrfMarginals(crf: TemplateCrf, rows: TokenRows): ChainMarginals {
  const p = potentialsOf(crf, rows)
  return chainForwardBackward(p.logUnary, p.logPairwise)
}

/**
 * The Viterbi labelling of a sequence: the most probable labelling as a whole, by max-product in log space
 * (`chainViterbi`).
 *
 * @param crf The CRF.
 * @param rows The token rows of the sequence.
 * @returns The Viterbi result (`path` as label ids, `logProbability` its unnormalised log score, and the tables), with
 *   `labels`, the path as strings.
 *
 * @example Tagging a sentence the CRF has not seen
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const crf = templateCrf(p.index, p.labels, toFlat(run(crfTraining(p), undefined, 30).weights))
 * const v = templateCrfViterbi(crf, posRows(['They', 'watch', 'the', 'light', '.']))
 * print('tags:', v.labels)
 * print('log score:', v.logProbability)
 */
export function templateCrfViterbi(crf: TemplateCrf, rows: TokenRows): ViterbiResult & { labels: string[] } {
  const p = potentialsOf(crf, rows)
  const v = chainViterbi(p.logUnary, p.logPairwise)
  return { ...v, labels: Array.from(toFlat(v.path), (k) => crf.labels[k]) }
}

/**
 * Posterior (max-marginal) decoding of a sequence, $\hat{y}_n = \argmax_k P(y_n = k \mid \xvec)$ (`posteriorDecode`
 * on the forward–backward marginals), with the labels as strings and the path's unnormalised log score. It maximises
 * the expected number of correct labels; Viterbi maximises the probability of the whole labelling.
 *
 * @param crf The CRF.
 * @param rows The token rows of the sequence.
 * @returns The decoding (`path`, `confidence` per position and `expectedCorrect`), with `labels`, the path as strings,
 *   and `logScore`, its unnormalised log score.
 *
 * @example Each tag with its marginal probability
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const crf = templateCrf(p.index, p.labels, toFlat(run(crfTraining(p), undefined, 30).weights))
 * const d = templateCrfPosterior(crf, posRows(['They', 'watch', 'the', 'light', '.']))
 * print('tags:', d.labels)
 * print('confidence:', d.confidence)
 * print('expected correct:', d.expectedCorrect, 'of 5')
 */
export function templateCrfPosterior(
  crf: TemplateCrf,
  rows: TokenRows,
): PosteriorDecoding & { labels: string[]; logScore: number } {
  const p = potentialsOf(crf, rows)
  const d = posteriorDecode(chainForwardBackward(p.logUnary, p.logPairwise).marginals)
  const path = Array.from(toFlat(d.path))
  return { ...d, labels: path.map((k) => crf.labels[k]), logScore: pathScore(p, path, crf.labels.length) }
}

/**
 * The unnormalised log score $\sum_n \log \psi_n(y_n) + \sum_{n \ge 1} \log \Psi_{n-1}(y_{n-1}, y_n)$ of a labelling.
 *
 * @param crf The CRF.
 * @param rows The token rows of the sequence.
 * @param path The labelling as label ids (indices into `crf.labels`), one per row.
 * @returns The log score; subtract $\log Z$ (`templateCrfMarginals`) for $\log P(\yvec \mid \xvec)$.
 *
 * @example The Viterbi path scores highest
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const crf = templateCrf(p.index, p.labels, toFlat(run(crfTraining(p), undefined, 30).weights))
 * const rows = posRows(['They', 'watch', 'the', 'light', '.'])
 * const v = templateCrfViterbi(crf, rows)
 * print(v.labels, templateCrfScore(crf, rows, toFlat(v.path)))
 * const other = ['PRON', 'NOUN', 'DET', 'NOUN', '.']
 * print(other, templateCrfScore(crf, rows, other.map((y) => crf.labels.indexOf(y))))
 * print('log P(Viterbi path) =', templateCrfScore(crf, rows, toFlat(v.path)) - templateCrfMarginals(crf, rows).logZ)
 */
export function templateCrfScore(crf: TemplateCrf, rows: TokenRows, path: ArrayLike<number>): number {
  return pathScore(potentialsOf(crf, rows), path, crf.labels.length)
}

/**
 * The unnormalised log score of a labelling from its log-potentials.
 *
 * @param p The log-potentials: `logUnary` ($N \times K$) and `logPairwise` ($(N - 1) \times K \times K$).
 * @param path The labelling as label ids, one per position.
 * @param K The number of labels.
 * @returns $\sum_n \log \psi_n(y_n) + \sum_{n \ge 1} \log \Psi_{n-1}(y_{n-1}, y_n)$.
 */
function pathScore(p: { logUnary: Matrix; logPairwise: Tensor }, path: ArrayLike<number>, K: Size): number {
  let s = 0
  for (let n = 0; n < path.length; n++) {
    s += p.logUnary.data[n * K + path[n]]
    if (n > 0) s += p.logPairwise.data[(n - 1) * K * K + path[n - 1] * K + path[n]]
  }
  return s
}

/**
 * The log-potentials of a sequence under a CRF, $\log \psi_n$ and $\log \Psi_n$: its rows encoded against the CRF's
 * index, then `crfLogPotentials`.
 *
 * @param crf The CRF.
 * @param rows The token rows of the sequence.
 * @returns `logUnary` ($N \times K$) and `logPairwise` ($(N - 1) \times K \times K$).
 *
 * @example The potentials of a three-word sentence
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const crf = templateCrf(p.index, p.labels, toFlat(run(crfTraining(p), undefined, 30).weights))
 * const { logUnary, logPairwise } = templateCrfPotentials(crf, posRows(['The', 'dog', 'runs']))
 * print('labels:', crf.labels)
 * print('log ψ of "dog":', toRows(logUnary)[1])
 * print('log Ψ shape:', logPairwise.shape)
 */
export function templateCrfPotentials(crf: TemplateCrf, rows: TokenRows): { logUnary: Matrix; logPairwise: Tensor } {
  return potentialsOf(crf, rows)
}

/** A feature string that fires at a position, with the template it came from, its id and its weights. */
export interface FiringFeature {
  /** Whether the template is a unigram (`U`) or a bigram (`B`) one. */
  readonly kind: 'unigram' | 'bigram'
  /** The expanded string, e.g. `U01:o`. */
  readonly string: string
  /** The template line it expands. */
  readonly template: string
  /** The template's macros: the cells %x[r,c] it reads. */
  readonly macros: readonly TemplateMacro[]
  /** Its id in the index, or $-1$ when the string was not seen in training (it then fires nothing). */
  readonly id: number
  /**
   * $\lambda_{u,k}$ per label $k$ (unigram, length $K$), or $\lambda_{b,i,k}$ row-major (bigram, $K \times K$). Zeros
   * when unknown.
   */
  readonly weights: readonly number[]
}

/**
 * Every template's string at position $n$ (bigram templates only for $n \ge 1$), with its weights: what the CRF sees
 * at that position. A cell outside the sequence reads `_B-1`, `_B-2`, ... before it and `_B+1`, ... after it, as in
 * CRF++.
 *
 * @param crf The CRF.
 * @param rows The token rows of the sequence.
 * @param n The position, from 0.
 * @returns One entry per template that applies at `n`, in the order of the templates.
 *
 * @example What fires at "zebra" and at "runs"
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const crf = templateCrf(p.index, p.labels, toFlat(run(crfTraining(p), undefined, 30).weights))
 * const rows = posRows(['The', 'zebra', 'runs'])
 * firingFeatures(crf, rows, 1).forEach((f) => print(f.kind, f.string, 'id', f.id))
 * const runs = firingFeatures(crf, rows, 2)[0]
 * print('labels:', crf.labels)
 * print(runs.string, 'weights:', runs.weights)
 */
export function firingFeatures(crf: TemplateCrf, rows: TokenRows, n: Size): FiringFeature[] {
  const K = crf.labels.length
  const U = crf.index.unigram.length
  const out: FiringFeature[] = []
  for (const t of crf.index.templates.templates) {
    if (t.kind === 'bigram' && n === 0) continue
    let s = t.pieces[0]
    t.macros.forEach((m, k) => {
      const at = n + m.row
      s +=
        (at < 0 ? `_B${at}` : at >= rows.length ? `_B+${at - rows.length + 1}` : rows[at][m.column]) + t.pieces[k + 1]
    })
    const id = (t.kind === 'unigram' ? crf.index.unigramIds : crf.index.bigramIds).get(s) ?? -1
    const width = t.kind === 'unigram' ? K : K * K
    const base = t.kind === 'unigram' ? id * K : U * K + id * K * K
    out.push({
      kind: t.kind,
      string: s,
      template: t.text,
      macros: t.macros,
      id,
      weights: Array.from({ length: width }, (_, q) => (id < 0 ? 0 : crf.weights[base + q])),
    })
  }
  return out
}

/** A unigram string's weights per label, for ranking. */
export interface WeightedString {
  /** The expanded string, e.g. `U00:runs`. */
  readonly string: string
  /** Its id among the index's unigram strings. */
  readonly id: number
  /** $\lambda_{u,k}$ per label $k$ (length $K$). */
  readonly weights: readonly number[]
  /** How often it fired in training. */
  readonly count: number
}

/**
 * The unigram strings with the largest $\lvert \lambda_{u,k} \rvert$ for label $k$ (`label` given), or with the largest
 * $\max_k \lvert \lambda_{u,k} \rvert$ (no label), at most `count`, largest first; strings whose score is zero are left
 * out. With `relative` and a label, strings are ranked instead by how much they favour label $k$ over the others,
 * $\lambda_{u,k} - \frac{1}{K} \sum_j \lambda_{u,j}$ (with two labels, half the log-odds they add), and only those
 * that favour it (a positive score) are kept.
 *
 * @param crf The CRF.
 * @param count The most strings to return.
 * @param label The label id $k$ (an index into `crf.labels`) to rank for; every label when left out.
 * @param options `relative`: rank by the weight relative to the mean over labels (only with `label`; default false).
 * @returns The strings with their ids, their weights per label and how often each fired in training.
 *
 * @example The words that most favour VERB
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const crf = templateCrf(p.index, p.labels, toFlat(run(crfTraining(p), undefined, 30).weights))
 * const verb = crf.labels.indexOf('VERB')
 * topFeatures(crf, 5, verb, { relative: true }).forEach((f) => print(f.string, f.weights[verb], 'seen', f.count))
 */
export function topFeatures(
  crf: TemplateCrf,
  count: Size,
  label?: Size,
  options: { relative?: boolean } = {},
): WeightedString[] {
  const K = crf.labels.length
  const scoreOf = (u: number) => {
    if (label !== undefined && options.relative) {
      let mean = 0
      for (let k = 0; k < K; k++) mean += crf.weights[u * K + k] / K
      return crf.weights[u * K + label] - mean
    }
    if (label !== undefined) return Math.abs(crf.weights[u * K + label])
    let m = 0
    for (let k = 0; k < K; k++) m = Math.max(m, Math.abs(crf.weights[u * K + k]))
    return m
  }
  return crf.index.unigram
    .map((_, u) => u)
    .filter((u) => scoreOf(u) > 0)
    .sort((a, b) => scoreOf(b) - scoreOf(a))
    .slice(0, count)
    .map((u) => ({
      string: crf.index.unigram[u],
      id: u,
      weights: Array.from({ length: K }, (_, k) => crf.weights[u * K + k]),
      count: crf.index.unigramCounts[u],
    }))
}

/**
 * The $K \times K$ weights $\lambda_{b,i,k}$ of the plain transition string `B` (or of another bigram string), row $i$
 * the previous label and column $k$ the current one, or null if the string is not indexed.
 *
 * @param crf The CRF.
 * @param string The bigram string, as the index holds it.
 * @returns The weights as rows, or null.
 *
 * @example What follows a determiner
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const crf = templateCrf(p.index, p.labels, toFlat(run(crfTraining(p), undefined, 30).weights))
 * const T = transitionWeights(crf)
 * print('labels:', crf.labels)
 * print('after DET:', T[crf.labels.indexOf('DET')])
 * print('B01:x is not indexed:', transitionWeights(crf, 'B01:x'))
 */
export function transitionWeights(crf: TemplateCrf, string = 'B'): number[][] | null {
  const id = crf.index.bigramIds.get(string)
  if (id === undefined) return null
  const K = crf.labels.length
  const base = crf.index.unigram.length * K + id * K * K
  return Array.from({ length: K }, (_, i) => Array.from({ length: K }, (_, k) => crf.weights[base + i * K + k]))
}

/**
 * The number of non-zero weights.
 *
 * @param weights The weights.
 * @returns How many are not zero.
 *
 * @example Two of four weights are active
 * print(activeWeights([0, 1.5, 0, -2]))
 */
export function activeWeights(weights: ArrayLike<number>): Size {
  let a = 0
  for (let i = 0; i < weights.length; i++) if (weights[i] !== 0) a++
  return a
}

// ── Likelihood and gradient ──────────────────────────────────────────────────────────────────────────────────────────

/** A sequence encoded once for training: feature ids and label ids. */
export interface EncodedLabelled {
  /** The ids of the strings that fire at each position (`encodeTemplateRows`). */
  readonly sequence: EncodedSequence
  /** The label id of each position. */
  readonly labels: Int32Array
}

/**
 * Encode training data against an index and a label set. Throws `DomainError` for a label not in `labels`, or a
 * sequence whose numbers of rows and labels differ.
 *
 * @param index The feature index; strings it does not hold are left out.
 * @param labels The label set, in weight order: a label's id is its index here.
 * @param data The labelled sequences.
 * @returns One encoded sequence per input, in order.
 *
 * @example The first sentence of the toy corpus
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const data = toyPosCorpus().slice(0, 1)
 * const p = crfProblem(templates, data)
 * const [e] = encodeLabelled(p.index, p.labels, data)
 * print('labels:', p.labels, 'ids:', e.labels)
 * print('unigram ids:', e.sequence.unigramIds, 'bigram ids:', e.sequence.bigramIds)
 *
 * @example A label outside the set is refused
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus().slice(0, 1))
 * try {
 *   encodeLabelled(p.index, p.labels, [{ rows: [['dog']], labels: ['X'] }])
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function encodeLabelled(
  index: FeatureIndex,
  labels: readonly string[],
  data: readonly LabelledSequence[],
): EncodedLabelled[] {
  const ids = new Map(labels.map((y, k) => [y, k]))
  return data.map((s, m) => {
    if (s.labels.length !== s.rows.length)
      throw new DomainError(
        'encodeLabelled',
        `encodeLabelled: sequence ${m} has ${s.rows.length} rows but ${s.labels.length} labels`,
      )
    return {
      sequence: encodeTemplateRows(index, s.rows),
      labels: Int32Array.from(s.labels, (y) => {
        const k = ids.get(y)
        if (k === undefined) throw new DomainError('encodeLabelled', `encodeLabelled: unknown label '${y}'`)
        return k
      }),
    }
  })
}

/**
 * The negative log-likelihood $-\sum_m \log P(\yvec_m \mid \xvec_m)$ over `data`, and its gradient (added into
 * `grad`): for each unigram string $u$ firing at $n$, $\partial / \partial \lambda_{u,k}$ gains
 * $P(y_n = k \mid \xvec) - [y_n = k]$; for each bigram string $b$ firing at $n \ge 1$,
 * $\partial / \partial \lambda_{b,i,k}$ gains $P(y_{n-1} = i, y_n = k \mid \xvec) - [y_{n-1} = i, y_n = k]$. Empty
 * sequences add nothing.
 *
 * @param weights The weight vector $\lambdavec$ ($UK + BK^2$ values).
 * @param data The encoded training sequences.
 * @param U The number of unigram strings of the index, which places the bigram block at $UK$.
 * @param K The number of labels.
 * @param grad Where the gradient is accumulated, of the length of `weights`: added to, not overwritten, so pass zeros
 *   for the gradient alone. Left out, only the value is computed.
 * @returns The negative log-likelihood (without regularisation).
 *
 * @example At zero weights every labelling is equally likely
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const tokens = p.encoded.reduce((a, e) => a + e.labels.length, 0)
 * print('NLL:', crfNegLogLikelihood(new Float64Array(p.dimension), p.encoded, p.U, p.K))
 * print('tokens × log K:', tokens * Math.log(p.K))
 *
 * @example The gradient is expected minus observed counts
 * // "the" is always DET: at zero weights each label expects count / K of its firings, and DET observes all of them.
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const g = new Float64Array(p.dimension)
 * const nll = crfNegLogLikelihood(new Float64Array(p.dimension), p.encoded, p.U, p.K, g)
 * const u = p.index.unigramIds.get('U00:the')
 * print('labels:', p.labels, 'count of U00:the:', p.index.unigramCounts[u])
 * print('gradient:', g.slice(u * p.K, u * p.K + p.K))
 */
export function crfNegLogLikelihood(
  weights: ArrayLike<number>,
  data: readonly EncodedLabelled[],
  U: Size,
  K: Size,
  grad?: Float64Array,
): number {
  const KK = K * K
  const off = U * K
  let nll = 0
  for (const { sequence: e, labels: y } of data) {
    const N = e.length
    if (N === 0) continue
    const p = crfLogPotentials(weights, e, U, K)
    const fb = chainForwardBackward(p.logUnary, p.logPairwise)
    let score = 0
    for (let n = 0; n < N; n++) {
      score += p.logUnary.data[n * K + y[n]]
      if (n > 0) score += p.logPairwise.data[(n - 1) * KK + y[n - 1] * K + y[n]]
    }
    nll += fb.logZ - score
    if (!grad) continue
    const P = fb.marginals.data
    const Q = fb.pairwise.data
    for (let n = 0; n < N; n++)
      for (let j = e.unigramStart[n]; j < e.unigramStart[n + 1]; j++) {
        const at = e.unigramIds[j] * K
        for (let k = 0; k < K; k++) grad[at + k] += P[n * K + k]
        grad[at + y[n]] -= 1
      }
    for (let n = 1; n < N; n++)
      for (let j = e.bigramStart[n]; j < e.bigramStart[n + 1]; j++) {
        const at = off + e.bigramIds[j] * KK
        for (let q = 0; q < KK; q++) grad[at + q] += Q[(n - 1) * KK + q]
        grad[at + y[n - 1] * K + y[n]] -= 1
      }
  }
  return nll
}

/**
 * The smooth training objective $-\sum_m \log P(\yvec_m \mid \xvec_m) + c_2 \lVert \lambdavec \rVert_2^2$ as an
 * `ObjectiveFn` of the weight vector (the L1 term is OWL-QN's `l1`, not part of it).
 *
 * @param data The encoded training sequences.
 * @param U The number of unigram strings of the index.
 * @param K The number of labels.
 * @param c2 The L2 strength $c_2$ (0 for none).
 * @returns A function of the weight vector (length $UK + BK^2$) returning the objective's `value` and its `grad`.
 *
 * @example The objective and its gradient at zero weights
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus().slice(0, 5))
 * const f = crfObjective(p.encoded, p.U, p.K, 0.01)
 * const { value, grad } = f(zeros([p.dimension]))
 * print('value:', value, 'gradient norm:', Math.hypot(...toFlat(grad)))
 */
export function crfObjective(data: readonly EncodedLabelled[], U: Size, K: Size, c2: number) {
  return (w: Vector) => {
    const x = toFlat(w)
    const g = new Float64Array(x.length)
    let value = crfNegLogLikelihood(x, data, U, K, g)
    if (c2 > 0)
      for (let i = 0; i < x.length; i++) {
        value += c2 * x[i] * x[i]
        g[i] += 2 * c2 * x[i]
      }
    return { value, grad: fromData(g, [x.length]) }
  }
}

// ── Training ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The optimisers of `crfTraining`. */
export type CrfOptimizer = 'lbfgs' | 'owlqn' | 'sgd' | 'adam'

/** Options of `crfTraining`. */
export interface CrfTrainingOptions {
  /** L-BFGS ($c_2$ only), OWL-QN ($c_1$ and $c_2$: elastic net), SGD or Adam ($c_2$ only). Default `'lbfgs'`. */
  optimizer?: CrfOptimizer
  /** L1 strength $c_1$ (OWL-QN only; ignored otherwise). Default 0. */
  c1?: number
  /** L2 strength $c_2$. Default 0.01. */
  c2?: number
  /** Curvature pairs kept by L-BFGS / OWL-QN (default 10). */
  memory?: Size
  /**
   * L-BFGS / OWL-QN: stop when the (pseudo-)gradient norm falls below this times the number of sequences (default
   * $10^{-5}$). SGD and Adam ignore it and never report convergence.
   */
  tolerance?: number
  /** SGD / Adam: the step size (defaults 0.1 for SGD and 0.05 for Adam). */
  stepSize?: number
  /** SGD / Adam: sequences per minibatch (default 16; 0 for all of them, a full batch). */
  batchSize?: Size
}

/** The state of `crfTraining`: one quasi-Newton iteration, or one epoch of SGD / Adam, per step. */
export interface CrfTrainingState extends Status {
  /** The weights $\lambdavec$. */
  readonly weights: Vector
  /** The full objective $L(\lambdavec) = \mathrm{NLL} + c_1 \norm{\lambdavec}_1 + c_2 \norm{\lambdavec}_2^2$. */
  readonly objective: number
  /** $-\sum_m \log P(\yvec_m \mid \xvec_m)$ over the training data. */
  readonly nll: number
  /** The norm of the (pseudo-)gradient of $L$. */
  readonly gradNorm: number
  /** The number of non-zero weights. */
  readonly active: Size
  /**
   * Objective and gradient evaluations so far: the quasi-Newton count (line-search trials included), or for SGD /
   * Adam one per epoch plus the first.
   */
  readonly evaluations: number
  /** The quasi-Newton state (L-BFGS / OWL-QN), or null. */
  readonly inner: OwlqnState | null
  /** The first-order rule's state (SGD / Adam), or null. */
  readonly rule: unknown
  /** True when the quasi-Newton line search made no progress (always false for SGD / Adam). */
  readonly stalled: boolean
}

/**
 * The L1 norm.
 *
 * @param x The values.
 * @returns $\sum_i \lvert x_i \rvert$.
 */
const l1Norm = (x: ArrayLike<number>) => {
  let a = 0
  for (let i = 0; i < x.length; i++) a += Math.abs(x[i])
  return a
}

/**
 * Train a template CRF's weights on encoded data (see the file comment), as a step-through algorithm: each step is
 * one L-BFGS or OWL-QN iteration (the line search included), or one epoch of minibatch SGD / Adam over a shuffled
 * order drawn from `ctx.stream`. The state carries the objective, the NLL, the gradient norm and the number of non-zero
 * weights. `init` takes the starting weights (zeros when undefined) and throws `DomainError` when their length is not
 * the problem's `dimension`. The run is done when the quasi-Newton method converges or stalls, or the objective is
 * not finite; SGD and Adam run until stopped. For SGD and Adam each minibatch's gradient is that of $L / M$ ($M$
 * sequences) restricted to the minibatch, so a step size means the same whatever the size of the data.
 *
 * @param problem The training problem, from `crfProblem`.
 * @param options The optimiser and its settings (see `CrfTrainingOptions`).
 * @returns The algorithm, named `crf-` and the optimiser; its start is the initial weight vector or undefined.
 *
 * @example L-BFGS on the toy tagging corpus
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const s = run(crfTraining(p), undefined, 50)
 * print('steps:', s.t, 'converged:', s.converged)
 * print('objective:', s.objective, 'NLL:', s.nll, 'active weights:', s.active, 'of', p.dimension)
 *
 * @example OWL-QN: an L1 penalty sets weights to exactly zero
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * for (const c1 of [0, 0.1, 1]) {
 *   const s = run(crfTraining(p, { optimizer: 'owlqn', c1 }), undefined, 50)
 *   print(`c1 = ${c1}: active weights ${s.active} of ${p.dimension}, NLL ${s.nll.toFixed(2)}`)
 * }
 *
 * @example Five epochs of Adam
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * const s = run(crfTraining(p, { optimizer: 'adam', batchSize: 8 }), undefined, 5, { stream: stream(1) })
 * print('epochs:', s.t, 'objective:', s.objective, 'NLL:', s.nll)
 */
export function crfTraining(
  problem: CrfProblem,
  options: CrfTrainingOptions = {},
): Algorithm<ArrayLike<number> | undefined, CrfTrainingState> {
  const { optimizer = 'lbfgs', c2 = 0.01, memory = 10, tolerance = 1e-5 } = options
  const { encoded: data, U, K, dimension } = problem
  const c1 = optimizer === 'owlqn' ? (options.c1 ?? 0) : 0
  const batchSize = options.batchSize === 0 ? Math.max(1, data.length) : (options.batchSize ?? 16)
  const M = Math.max(1, data.length)
  const f = crfObjective(data, U, K, c2)
  const quasiNewton = optimizer === 'lbfgs' || optimizer === 'owlqn'
  // The tolerance is per sequence, so it does not tighten as the data grow.
  const qn = owlqn(f, { l1: c1, memory, tolerance: tolerance * M })
  const rule: UpdateRule<unknown> | null = quasiNewton
    ? null
    : ((optimizer === 'sgd'
        ? sgdRule({ stepSize: options.stepSize ?? 0.1 })
        : adamRule({ stepSize: options.stepSize ?? 0.05 })) as UpdateRule<unknown>)
  const full = (w: Float64Array) => {
    const out = f(fromData(w, [w.length]))
    return { value: out.value, nll: out.value - c2 * w.reduce((a, v) => a + v * v, 0), grad: toFlat(out.grad) }
  }
  const stateOf = (w: Float64Array, extra: Partial<CrfTrainingState> & { t: number; evaluations: number }) => {
    const ev = full(w)
    return {
      t: extra.t,
      weights: fromData(w, [w.length]),
      objective: ev.value + c1 * l1Norm(w),
      nll: ev.nll,
      gradNorm: Math.hypot(...ev.grad),
      active: activeWeights(w),
      evaluations: extra.evaluations + 1,
      inner: null,
      rule: extra.rule ?? null,
      stalled: false,
      converged: false,
      diverged: !Number.isFinite(ev.value),
    } satisfies CrfTrainingState
  }
  return {
    name: `crf-${optimizer}`,
    init: (w0, s) => {
      if (w0 && w0.length !== dimension)
        throw new DomainError('crfTraining', `crfTraining: expected ${dimension} starting weights, got ${w0.length}`)
      const w = w0 ? Float64Array.from(w0) : new Float64Array(dimension)
      if (quasiNewton) {
        const inner = qn.init({ x0: w }, s)
        return {
          t: 0,
          weights: inner.x,
          objective: inner.value,
          nll: inner.smoothValue - c2 * l2(toFlat(inner.x)),
          gradNorm: inner.gradNorm,
          active: inner.nonzero,
          evaluations: 1,
          inner,
          rule: null,
          stalled: false,
          converged: inner.converged,
          diverged: inner.diverged,
        }
      }
      return stateOf(w, { t: 0, evaluations: 0, rule: rule!.init(fromData(w, [w.length])) })
    },
    step: (st, ctx) => {
      if (quasiNewton) {
        const inner = qn.step(st.inner!, ctx)
        return {
          t: st.t + 1,
          weights: inner.x,
          objective: inner.value,
          nll: inner.smoothValue - c2 * l2(toFlat(inner.x)),
          gradNorm: inner.gradNorm,
          active: inner.nonzero,
          evaluations: inner.evaluations,
          inner,
          rule: null,
          stalled: inner.stalled,
          converged: inner.converged,
          diverged: inner.diverged,
        }
      }
      // One epoch: minibatches of a shuffled order; each minibatch's gradient is of L/M restricted to it.
      let w = Float64Array.from(toFlat(st.weights))
      let rs = st.rule
      const order = Array.from(toFlat(permutation(child(ctx.stream, 'order'), data.length)))
      for (let b = 0; b < order.length; b += batchSize) {
        const batch = order.slice(b, b + batchSize).map((i) => data[i])
        const g = new Float64Array(w.length)
        crfNegLogLikelihood(w, batch, U, K, g)
        const scale = 1 / batch.length
        for (let i = 0; i < w.length; i++) g[i] = g[i] * scale + (2 * c2 * w[i]) / M
        const out = rule!.update(fromData(g, [w.length]), rs, fromData(w, [w.length]))
        rs = out.state
        const u = toFlat(out.updates as Tensor)
        w = w.map((v, i) => v + u[i])
      }
      return stateOf(w, { t: st.t + 1, evaluations: st.evaluations, rule: rs })
    },
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}

/**
 * The squared L2 norm.
 *
 * @param x The values.
 * @returns $\sum_i x_i^2$.
 */
const l2 = (x: ArrayLike<number>) => {
  let a = 0
  for (let i = 0; i < x.length; i++) a += x[i] * x[i]
  return a
}

/** Options of `crfTrainingRun`: those of `crfProblem` and `crfTraining`, and the length of the run. */
export interface FitTemplateCrfOptions extends CrfTrainingOptions {
  /** CRF++'s `-f`: keep strings seen at least this often (default 1). */
  minFrequency?: Size
  /** The label set (default: the data's labels in order of first appearance). */
  labels?: readonly string[]
  /** The most steps (default 200). */
  maxSteps?: Size
}

/** A training problem, as `crfProblem` builds it and `crfTraining` takes it. */
export interface CrfProblem {
  /** The index of the data's feature strings. */
  readonly index: FeatureIndex
  /** The label set, in weight order. */
  readonly labels: readonly string[]
  /** The data, encoded against the index and the labels. */
  readonly encoded: readonly EncodedLabelled[]
  /** $U$, the number of unigram strings. */
  readonly U: Size
  /** $K$, the number of labels. */
  readonly K: Size
  /** The number of weights, $UK + BK^2$. */
  readonly dimension: Size
}

/**
 * Index the data's strings (keeping those seen at least `minFrequency` times) and encode the data: what `crfTraining`
 * takes. Throws `DomainError` as `encodeLabelled` does.
 *
 * @param templates The parsed templates (`parseTemplates` of `aifn-compute/text/features`).
 * @param data The labelled training sequences.
 * @param options `minFrequency`, CRF++'s `-f` (default 1), and `labels`, the label set in weight order (default: the
 *   data's labels in order of first appearance).
 * @returns The problem: index, labels, encoded data and sizes.
 *
 * @example The toy corpus with word templates
 * // `parseTemplates('U00:%x[0,0]\nB')` of aifn-compute/text/features, abridged: the word, and label transitions
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * const p = crfProblem(templates, toyPosCorpus())
 * print('labels:', p.labels)
 * print('U =', p.U, 'K =', p.K, 'weights =', p.dimension)
 * const rare = crfProblem(templates, toyPosCorpus(), { minFrequency: 2 })
 * print('seen twice or more: U =', rare.U, 'dropped:', rare.index.dropped)
 */
export function crfProblem(
  templates: FeatureTemplates,
  data: readonly LabelledSequence[],
  options: { minFrequency?: Size; labels?: readonly string[] } = {},
): CrfProblem {
  const index = featureIndex(
    templates,
    data.map((s) => s.rows),
    { minFrequency: options.minFrequency ?? 1 },
  )
  const labels = options.labels ?? labelSet(data)
  const encoded = encodeLabelled(index, labels, data)
  const U = index.unigram.length
  const K = labels.length
  return { index, labels, encoded, U, K, dimension: crfWeightCount(index, K) }
}

/**
 * CRF++'s `-c C` as CRFsuite strengths: $c_2 = 1/(2C)$ for L2, $c_1 = 1/C$ for L1. Throws `DomainError` unless
 * $C > 0$.
 *
 * @param C CRF++'s hyperparameter $C$: larger fits the data more closely.
 * @param kind The regulariser CRF++ was run with (`-a CRF-L2` or `CRF-L1`).
 * @returns `c1` and `c2` for `crfTraining`, the other one 0.
 *
 * @example CRF++'s default C = 1, and C = 10
 * print(crfppRegularisation(1))
 * print(crfppRegularisation(10, 'L1'))
 */
export function crfppRegularisation(C: number, kind: 'L1' | 'L2' = 'L2'): { c1: number; c2: number } {
  if (!(C > 0)) throw new DomainError('crfppRegularisation', 'crfppRegularisation: C must be > 0')
  return kind === 'L2' ? { c1: 0, c2: 1 / (2 * C) } : { c1: 1 / C, c2: 0 }
}

/** A snapshot of `crfTrainingRun`. */
export interface CrfSnapshot {
  /** The steps taken (0 before the first). */
  readonly step: Size
  /** The most steps the run takes. */
  readonly maxSteps: Size
  /** The CRF with the weights so far (a copy). */
  readonly crf: TemplateCrf
  /** Per step so far, from step 0: objective, NLL, gradient norm and non-zero weights. */
  readonly history: {
    readonly objective: readonly number[]
    readonly nll: readonly number[]
    readonly gradNorm: readonly number[]
    readonly active: readonly number[]
  }
  /** Whether the optimiser has converged. */
  readonly converged: boolean
  /** Whether this is the last snapshot: converged, stalled, diverged or at `maxSteps`. */
  readonly done: boolean
  /** Milliseconds since the run started. */
  readonly ms: number
}

/**
 * Train a template CRF from token rows and labels, yielding a snapshot after every step (step 0 first): the model so
 * far and the objective, NLL, gradient norm and active-weight history. A generator, so the lab's worker streams it.
 * It builds the problem with `crfProblem`, then runs `crfTraining` from zero weights with the root stream `seed`.
 *
 * @param templateSource The parsed templates (`parseTemplates` of `aifn-compute/text/features`).
 * @param data The labelled training sequences.
 * @param options Those of `crfProblem` and `crfTraining`, `maxSteps` (default 200), and `seed`, the root stream of
 *   the run (default `'crf'`; it orders the SGD and Adam minibatches).
 * @returns A generator of snapshots, the last with `done` set.
 *
 * @example Twenty L-BFGS steps, snapshot by snapshot
 * // `parseTemplates('U00:%x[0,0]\nB')` of aifn-compute/text/features, abridged: the word, and label transitions
 * const word = { kind: 'unigram', text: 'U00:%x[0,0]', pieces: ['U00:', ''], macros: [{ row: 0, column: 0 }] }
 * const templates = { templates: [word, { kind: 'bigram', text: 'B', pieces: ['B'], macros: [] }] }
 * let last
 * for (const snap of crfTrainingRun(templates, toyPosCorpus(), { maxSteps: 20 })) last = snap
 * print('step', last.step, 'of', last.maxSteps, 'done:', last.done, 'converged:', last.converged)
 * print('objective:', last.history.objective.map((v) => Number(v.toFixed(1))))
 */
export function* crfTrainingRun(
  templateSource: FeatureTemplates,
  data: readonly LabelledSequence[],
  options: FitTemplateCrfOptions & { seed?: string | number } = {},
): Generator<CrfSnapshot> {
  const { maxSteps = 200, seed = 'crf' } = options
  const t0 = Date.now()
  const p = crfProblem(templateSource, data, options)
  const alg = crfTraining(p, options)
  const s0 = stream(seed)
  let s = alg.init(new Float64Array(p.dimension), child(s0, 'init'))
  const history = { objective: [s.objective], nll: [s.nll], gradNorm: [s.gradNorm], active: [s.active] }
  const snap = (done: boolean): CrfSnapshot => ({
    step: s.t,
    maxSteps,
    crf: { kind: 'template-crf', index: p.index, labels: p.labels, weights: Float64Array.from(toFlat(s.weights)) },
    history: {
      objective: [...history.objective],
      nll: [...history.nll],
      gradNorm: [...history.gradNorm],
      active: [...history.active],
    },
    converged: s.converged === true,
    done,
    ms: Date.now() - t0,
  })
  const finished = () => s.t >= maxSteps || (alg.done?.(s) ?? false)
  yield snap(finished())
  while (!finished()) {
    s = alg.step(s, { t: s.t, stream: child(s0, 'step', s.t) })
    history.objective.push(s.objective)
    history.nll.push(s.nll)
    history.gradNorm.push(s.gradNorm)
    history.active.push(s.active)
    yield snap(finished())
  }
}

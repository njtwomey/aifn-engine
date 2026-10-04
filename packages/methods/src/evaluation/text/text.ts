/**
 * Text metrics: BLEU (corpus and sentence, with smoothing), chrF, ROUGE-N and ROUGE-L, edit-distance alignment with
 * WER, CER, MER, WIP and WIL, translation edit rate with greedy block shifts, SQuAD exact match and token F₁, and
 * BERTScore from given token embeddings. Every metric takes the reference first and the candidate second; corpus
 * metrics take arrays (one candidate per entry, each with one reference or an array of references).
 */

import { defineMetric, type Rows } from 'aifn-compute/learning/metrics'
import { editDistance } from 'aifn-compute/optim/programming'
import { denseMatrix as dense, divide } from 'aifn-compute/learning/metrics'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** A tokeniser: text to tokens. */
export type Tokeniser = (text: string) => string[]

/** The default tokeniser: split on whitespace, no other change. Normalise case and punctuation before, and say so. */
export const whitespaceTokens: Tokeniser = (s) => s.split(/\s+/).filter(Boolean)

/** A normalising tokeniser: lowercase, punctuation (other than apostrophes) to spaces, split on whitespace. */
export const words: Tokeniser = (s) =>
  s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, ' ')
    .split(/\s+/)
    .filter(Boolean)

/** Text: a string, tokenised by the metric's tokeniser (pass a custom `tokenise` for pre-split input). */
export type Text = string

const tokensOf = (t: Text, tokenise: Tokeniser) => tokenise(t)

/** The n-grams of a token list, as joined strings (tokens separated by U+0001). */
export function ngrams(tokens: readonly string[], n: number): string[] {
  return Array.from({ length: Math.max(0, tokens.length - n + 1) }, (_, i) => tokens.slice(i, i + n).join('\u0001'))
}

function counts(items: readonly string[]): Map<string, number> {
  const m = new Map<string, number>()
  for (const it of items) m.set(it, (m.get(it) ?? 0) + 1)
  return m
}

const total = (m: Map<string, number>) => [...m.values()].reduce((a, b) => a + b, 0)

/** Matches clipped at the reference count: Σ_g min(count_cand(g), cap(g)). */
function clippedMatches(candidate: Map<string, number>, cap: Map<string, number>): number {
  let m = 0
  for (const [k, c] of candidate) m += Math.min(c, cap.get(k) ?? 0)
  return m
}

/** A corpus as parallel lists of references (each one or several) and candidates. */
type Corpus = { refs: (readonly string[])[]; cands: string[] }

/** References for a corpus: one entry per candidate, each a string or an array of alternative references. */
export type References = string | readonly (string | readonly string[])[]

/**
 * A single candidate (a string, whose reference is a string or an array of alternative references), or a corpus
 * (an array of candidates with one references entry each).
 */
function corpusOf(references: References, candidates: string | readonly string[]): Corpus {
  if (typeof candidates === 'string')
    return {
      refs: [typeof references === 'string' ? [references] : (references as readonly string[])],
      cands: [candidates],
    }
  if (typeof references === 'string' || references.length !== candidates.length)
    throw new ShapeError('metrics', `metrics: a corpus needs one references entry per candidate (${candidates.length})`)
  return { refs: references.map((r) => (typeof r === 'string' ? [r] : [...r])), cands: [...candidates] }
}

// ── BLEU ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Smoothing of zero n-gram counts for sentence-level BLEU (Chen and Cherry 2014): `none`; `add-one` (their method 2,
 * add 1 to matched and total counts for n ≥ 2); `epsilon` (method 1, ε = 0.1 matches when an order has none);
 * `exponential` (method 3, NIST: the k-th order without matches gets precision 1/(2^k · total)).
 */
export type BleuSmoothing = 'none' | 'add-one' | 'epsilon' | 'exponential'

/** Options of BLEU. */
export type BleuOptions = {
  /** Maximum n-gram order N (default 4), with uniform weights 1/N. */
  maxOrder?: number
  smoothing?: BleuSmoothing
  tokenise?: Tokeniser
}

/** BLEU with its parts. */
export type BleuResult = {
  score: number
  /** The (smoothed) n-gram precisions p₁ … p_N. */
  precisions: number[]
  /** Clipped matches and candidate n-gram totals per order, pooled over the corpus. */
  matched: number[]
  totals: number[]
  brevityPenalty: number
  candidateLength: number
  referenceLength: number
}

/**
 * Corpus BLEU (Papineni et al. 2002; bleu): BP · exp(Σₙ (1/N) log pₙ), with pₙ the clipped n-gram precision pooled over
 * the corpus (each candidate n-gram count capped at its largest count in any one reference), c the total candidate
 * length, r the total effective reference length (per sentence the reference length closest to the candidate's,
 * shorter on ties), and BP = 1 if c > r else e^{1 − r/c}. In [0, 1].
 */
export function bleuScore(
  references: References,
  candidates: string | readonly string[],
  options: BleuOptions = {},
): BleuResult {
  const { refs, cands } = corpusOf(references, candidates)
  const tok = options.tokenise ?? whitespaceTokens
  const N = options.maxOrder ?? 4
  const matched = new Array<number>(N).fill(0)
  const totals = new Array<number>(N).fill(0)
  let c = 0
  let r = 0
  cands.forEach((cand, s) => {
    const h = tokensOf(cand, tok)
    const rs = refs[s].map((x) => tokensOf(x, tok))
    for (let n = 1; n <= N; n++) {
      const cap = new Map<string, number>()
      for (const ref of rs) for (const [k, v] of counts(ngrams(ref, n))) cap.set(k, Math.max(cap.get(k) ?? 0, v))
      const hc = counts(ngrams(h, n))
      matched[n - 1] += clippedMatches(hc, cap)
      totals[n - 1] += total(hc)
    }
    c += h.length
    r += rs
      .map((x) => x.length)
      .reduce((best, len) => {
        const d = Math.abs(len - h.length)
        const bd = Math.abs(best - h.length)
        return d < bd || (d === bd && len < best) ? len : best
      })
  })
  const smoothing = options.smoothing ?? 'none'
  let zeros = 0
  const precisions = matched.map((m, i) => {
    if (smoothing === 'add-one' && i > 0) return (m + 1) / (totals[i] + 1)
    if (totals[i] === 0) return 0
    if (m === 0 && smoothing === 'epsilon') return 0.1 / totals[i]
    if (m === 0 && smoothing === 'exponential') return 1 / (2 ** ++zeros * totals[i])
    return m / totals[i]
  })
  const bp = c === 0 ? 0 : c > r ? 1 : Math.exp(1 - r / c)
  const score = precisions.some((p) => p === 0) ? 0 : bp * Math.exp(precisions.reduce((s, p) => s + Math.log(p), 0) / N)
  return { score, precisions, matched, totals, brevityPenalty: bp, candidateLength: c, referenceLength: r }
}

/** Corpus BLEU as a metric (see `bleuScore`); for one sentence, pass strings. */
export const bleu = defineMetric(
  {
    module: 'applied/evaluation/text',
    key: 'bleu',
    name: 'BLEU',
    inputs: 'sequences',
    direction: 'higher',
    range: [0, 1],
    notes: ['bleu'],
    capability: 'decide',
  },
  (references: References, candidates: string | readonly string[], options: BleuOptions = {}): number =>
    bleuScore(references, candidates, options).score,
)

// ── chrF ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * chrF_β (Popović 2015; chrf): character n-gram precision and recall (spaces removed, counts clipped), each averaged
 * over n = 1 … N (default 6), combined as (1 + β²)PR/(β²P + R) with β = 2 by default. Over a corpus, n-gram statistics
 * are pooled before the averages. Returns the score with its average precision and recall.
 */
export function chrFScore(
  references: References,
  candidates: string | readonly string[],
  options: { maxOrder?: number; beta?: number } = {},
): { score: number; precision: number; recall: number } {
  const { refs, cands } = corpusOf(references, candidates)
  const N = options.maxOrder ?? 6
  const beta = options.beta ?? 2
  const matched = new Float64Array(N)
  const candTotal = new Float64Array(N)
  const refTotal = new Float64Array(N)
  const strip = (t: string) => t.replace(/\s+/g, '')
  cands.forEach((cand, s) => {
    const h = strip(cand)
    const r = strip(refs[s][0])
    for (let n = 1; n <= N; n++) {
      const grams = (x: string) =>
        counts(Array.from({ length: Math.max(0, x.length - n + 1) }, (_, i) => x.slice(i, i + n)))
      const hc = grams(h)
      const rc = grams(r)
      matched[n - 1] += clippedMatches(hc, rc)
      candTotal[n - 1] += total(hc)
      refTotal[n - 1] += total(rc)
    }
  })
  let P = 0
  let R = 0
  for (let n = 0; n < N; n++) {
    P += divide(matched[n], candTotal[n], 0) / N
    R += divide(matched[n], refTotal[n], 0) / N
  }
  const b2 = beta * beta
  return { score: P + R === 0 ? 0 : ((1 + b2) * P * R) / (b2 * P + R), precision: P, recall: R }
}

/** chrF as a metric (see `chrFScore`). Uses the first reference of each candidate. */
export const chrF = defineMetric(
  {
    module: 'applied/evaluation/text',
    key: 'chrF',
    name: 'chrF',
    inputs: 'sequences',
    direction: 'higher',
    range: [0, 1],
    notes: ['chrf'],
    capability: 'decide',
  },
  (
    references: References,
    candidates: string | readonly string[],
    options: { maxOrder?: number; beta?: number } = {},
  ): number => chrFScore(references, candidates, options).score,
)

// ── ROUGE ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Precision, recall and F of an overlap score. */
export type Prf = { precision: number; recall: number; f: number }

const fOf = (p: number, r: number, beta = 1) =>
  p === 0 && r === 0 ? 0 : ((1 + beta * beta) * p * r) / (r + beta * beta * p)

/**
 * ROUGE-N (Lin 2004; rouge) of one candidate against one or more references: recall is the clipped n-gram matches over
 * the references' n-grams, precision over the candidate's; with several references both sum over them.
 */
export function rougeNScores(
  references: Text | readonly Text[],
  candidate: Text,
  options: { n?: number; tokenise?: Tokeniser } = {},
): Prf {
  const tok = options.tokenise ?? whitespaceTokens
  const n = options.n ?? 1
  const refs = typeof references === 'string' ? [references] : references
  const hc = counts(ngrams(tokensOf(candidate, tok), n))
  let m = 0
  let refTotal = 0
  let candTotal = 0
  for (const ref of refs) {
    const rc = counts(ngrams(tokensOf(ref, tok), n))
    m += clippedMatches(hc, rc)
    refTotal += total(rc)
    candTotal += total(hc)
  }
  const precision = divide(m, candTotal, 0)
  const recall = divide(m, refTotal, 0)
  return { precision, recall, f: fOf(precision, recall) }
}

/** Length of the longest common subsequence of two token lists, by dynamic programming in O(|a||b|). */
export function longestCommonSubsequence(a: readonly string[], b: readonly string[]): number {
  let prev = new Int32Array(b.length + 1)
  for (let i = 0; i < a.length; i++) {
    const row = new Int32Array(b.length + 1)
    for (let j = 0; j < b.length; j++) row[j + 1] = a[i] === b[j] ? prev[j] + 1 : Math.max(prev[j + 1], row[j])
    prev = row
  }
  return prev[b.length]
}

/** ROUGE-L: R = LCS/|reference|, P = LCS/|candidate|, F_β (default β = 1). */
export function rougeLScores(
  reference: Text,
  candidate: Text,
  options: { beta?: number; tokenise?: Tokeniser } = {},
): Prf {
  const tok = options.tokenise ?? whitespaceTokens
  const r = tokensOf(reference, tok)
  const h = tokensOf(candidate, tok)
  const l = longestCommonSubsequence(r, h)
  const precision = divide(l, h.length, 0)
  const recall = divide(l, r.length, 0)
  return { precision, recall, f: fOf(precision, recall, options.beta ?? 1) }
}

/** ROUGE-N F-score (or `measure: 'recall' | 'precision'`) as a metric. Default n = 1. */
export const rougeN = defineMetric(
  {
    module: 'applied/evaluation/text',
    key: 'rougeN',
    name: 'ROUGE-N',
    inputs: 'sequences',
    direction: 'higher',
    range: [0, 1],
    notes: ['rouge'],
    capability: 'decide',
  },
  (
    references: Text | readonly Text[],
    candidate: Text,
    options: { n?: number; measure?: keyof Prf; tokenise?: Tokeniser } = {},
  ): number => rougeNScores(references, candidate, options)[options.measure ?? 'f'],
)

/** ROUGE-L F-score (or `measure`) as a metric. */
export const rougeL = defineMetric(
  {
    module: 'applied/evaluation/text',
    key: 'rougeL',
    name: 'ROUGE-L',
    inputs: 'sequences',
    direction: 'higher',
    range: [0, 1],
    notes: ['rouge'],
    capability: 'decide',
  },
  (
    reference: Text,
    candidate: Text,
    options: { measure?: keyof Prf; beta?: number; tokenise?: Tokeniser } = {},
  ): number => rougeLScores(reference, candidate, options)[options.measure ?? 'f'],
)

// ── Edit distance ────────────────────────────────────────────────────────────────────────────────────────────────────

/** One step of an alignment. */
export type EditOperation = {
  op: 'hit' | 'substitution' | 'deletion' | 'insertion'
  reference?: string
  hypothesis?: string
}

/** A minimum-edit alignment and its counts: N = H + S + D reference tokens, P = H + S + I hypothesis tokens. */
export type Alignment = {
  distance: number
  operations: EditOperation[]
  hits: number
  substitutions: number
  deletions: number
  insertions: number
}

/**
 * The Levenshtein alignment of a reference and a hypothesis token list (word-and-character-error-rates), by
 * `aifn-compute/optim/programming`'s `editDistance` with unit costs: its operations named in WER terms (hit, substitution,
 * deletion, insertion) and counted. On ties the traceback prefers a hit or substitution, then a deletion, then an
 * insertion.
 */
export function editAlignment(reference: readonly string[], hypothesis: readonly string[]): Alignment {
  const r = editDistance(reference, hypothesis)
  const operations: EditOperation[] = r.operations.map((o) =>
    o.op === 'match' || o.op === 'substitute'
      ? { op: o.op === 'match' ? 'hit' : 'substitution', reference: reference[o.i], hypothesis: hypothesis[o.j] }
      : o.op === 'delete'
        ? { op: 'deletion', reference: reference[o.i] }
        : { op: 'insertion', hypothesis: hypothesis[o.j] },
  )
  return {
    distance: r.distance,
    operations,
    hits: r.counts.match,
    substitutions: r.counts.substitute,
    deletions: r.counts.delete,
    insertions: r.counts.insert,
  }
}

/** Pooled alignment counts over a corpus of (reference, hypothesis) pairs. */
function pooledCounts(
  references: string | readonly string[],
  hypotheses: string | readonly string[],
  split: Tokeniser,
) {
  const refs = typeof references === 'string' ? [references] : references
  const hyps = typeof hypotheses === 'string' ? [hypotheses] : hypotheses
  if (refs.length !== hyps.length)
    throw new ShapeError('metrics', 'metrics: references and hypotheses differ in number')
  const out = { H: 0, S: 0, D: 0, I: 0 }
  refs.forEach((r, k) => {
    const a = editAlignment(split(r), split(hyps[k]))
    out.H += a.hits
    out.S += a.substitutions
    out.D += a.deletions
    out.I += a.insertions
  })
  return out
}

const characters: Tokeniser = (s) => Array.from(s)

const errorRateInfo = (key: string, name: string, range: readonly [number, number] = [0, Infinity]) =>
  ({
    module: 'applied/evaluation/text',
    key,
    name,
    inputs: 'sequences',
    direction: 'lower',
    range,
    notes: ['word-and-character-error-rates'],
    capability: 'decide',
  }) as const

/**
 * Word error rate (S + D + I)/N over a corpus (total edits over total reference words); unbounded above. Strings are
 * split on whitespace unless `tokenise` is given.
 */
export const wordErrorRate = defineMetric(
  errorRateInfo('wordErrorRate', 'Word error rate'),
  (
    references: string | readonly string[],
    hypotheses: string | readonly string[],
    options: { tokenise?: Tokeniser } = {},
  ): number => {
    const c = pooledCounts(references, hypotheses, options.tokenise ?? whitespaceTokens)
    return divide(c.S + c.D + c.I, c.H + c.S + c.D)
  },
)

/** Character error rate: WER over characters (spaces included). */
export const characterErrorRate = defineMetric(
  errorRateInfo('characterErrorRate', 'Character error rate'),
  (references: string | readonly string[], hypotheses: string | readonly string[]): number => {
    const c = pooledCounts(references, hypotheses, characters)
    return divide(c.S + c.D + c.I, c.H + c.S + c.D)
  },
)

/** Match error rate (S + D + I)/(H + S + D + I), in [0, 1] (Morris et al. 2004). */
export const matchErrorRate = defineMetric(
  errorRateInfo('matchErrorRate', 'Match error rate', [0, 1]),
  (
    references: string | readonly string[],
    hypotheses: string | readonly string[],
    options: { tokenise?: Tokeniser } = {},
  ): number => {
    const c = pooledCounts(references, hypotheses, options.tokenise ?? whitespaceTokens)
    return divide(c.S + c.D + c.I, c.H + c.S + c.D + c.I)
  },
)

/** Word information lost 1 − (H/N)(H/P) (Morris et al. 2004); word information preserved is 1 − WIL. */
export const wordInformationLost = defineMetric(
  errorRateInfo('wordInformationLost', 'Word information lost', [0, 1]),
  (
    references: string | readonly string[],
    hypotheses: string | readonly string[],
    options: { tokenise?: Tokeniser } = {},
  ): number => {
    const c = pooledCounts(references, hypotheses, options.tokenise ?? whitespaceTokens)
    return 1 - divide(c.H, c.H + c.S + c.D) * divide(c.H, c.H + c.S + c.I)
  },
)

// ── Translation edit rate ────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The fewest edits (insertions, deletions, substitutions and block shifts, each costing 1) turning a hypothesis into a
 * reference, by the greedy shift search of Snover et al. (2006): repeatedly apply the shift of a hypothesis block (up
 * to `maxShiftSize` words, default 10) that occurs in the reference and most reduces the Levenshtein distance, while
 * some shift reduces it; then add the remaining Levenshtein distance. An upper bound on the true minimum.
 */
export function translationEdits(
  reference: readonly string[],
  hypothesis: readonly string[],
  maxShiftSize = 10,
): { edits: number; shifts: number } {
  let h = [...hypothesis]
  const refText = `\u0001${reference.join('\u0001')}\u0001`
  let shifts = 0
  let cost = editAlignment(reference, h).distance
  for (;;) {
    let best: string[] | null = null
    let bestCost = cost
    for (let len = 1; len <= Math.min(maxShiftSize, h.length); len++)
      for (let i = 0; i + len <= h.length; i++) {
        const block = h.slice(i, i + len)
        if (!refText.includes(`\u0001${block.join('\u0001')}\u0001`)) continue
        const rest = [...h.slice(0, i), ...h.slice(i + len)]
        for (let j = 0; j <= rest.length; j++) {
          if (j === i) continue
          const moved = [...rest.slice(0, j), ...block, ...rest.slice(j)]
          const c = editAlignment(reference, moved).distance
          if (c < bestCost) {
            bestCost = c
            best = moved
          }
        }
      }
    if (!best) break
    h = best
    cost = bestCost
    shifts++
  }
  return { edits: shifts + cost, shifts }
}

/**
 * Translation edit rate (Snover et al. 2006; translation-edit-rate): the fewest edits, shifts included, to the closest
 * reference, divided by the average reference length, pooled over a corpus. 0 for an exact match; can exceed 1.
 */
export const translationEditRate = defineMetric(
  {
    module: 'applied/evaluation/text',
    key: 'translationEditRate',
    name: 'Translation edit rate',
    inputs: 'sequences',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['translation-edit-rate'],
    capability: 'decide',
  },
  (
    references: References,
    candidates: string | readonly string[],
    options: { tokenise?: Tokeniser; maxShiftSize?: number } = {},
  ): number => {
    const { refs, cands } = corpusOf(references, candidates)
    const tok = options.tokenise ?? whitespaceTokens
    let edits = 0
    let length = 0
    cands.forEach((cand, s) => {
      const h = tokensOf(cand, tok)
      const rs = refs[s].map((r) => tokensOf(r, tok))
      edits += Math.min(...rs.map((r) => translationEdits(r, h, options.maxShiftSize).edits))
      length += rs.reduce((a, r) => a + r.length, 0) / rs.length
    })
    return divide(edits, length)
  },
)

// ── SQuAD ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** SQuAD answer normalisation: lowercase, remove ASCII punctuation, remove the articles a, an, the, collapse spaces. */
export function normaliseAnswer(s: string): string {
  return s
    .toLowerCase()
    .replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g, '')
    .replace(/\b(a|an|the)\b/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .join(' ')
}

function squadF1Of(gold: string, prediction: string): number {
  const g = normaliseAnswer(gold).split(' ').filter(Boolean)
  const p = normaliseAnswer(prediction).split(' ').filter(Boolean)
  if (g.length === 0 || p.length === 0) return g.length === p.length ? 1 : 0
  const common = clippedMatches(counts(p), counts(g))
  if (common === 0) return 0
  const precision = common / p.length
  const recall = common / g.length
  return (2 * precision * recall) / (precision + recall)
}

/** Gold answers per question (one string or several) and one predicted answer per question. */
type QaInput = { golds: (string | readonly string[])[]; predictions: string[] }

function qaOf(
  golds: string | readonly (string | readonly string[])[],
  predictions: string | readonly string[],
): QaInput {
  if (typeof predictions === 'string')
    return { golds: [typeof golds === 'string' ? golds : (golds as readonly string[])], predictions: [predictions] }
  if (typeof golds === 'string' || golds.length !== predictions.length)
    throw new ShapeError('metrics', 'metrics: one gold entry per prediction')
  return { golds: [...golds], predictions: [...predictions] }
}

const asList = (g: string | readonly string[]) => (typeof g === 'string' ? [g] : g)

/**
 * SQuAD exact match (Rajpurkar et al. 2016; squad-exact-match-and-f1): 1 when the normalised prediction equals a
 * normalised gold answer, averaged over questions.
 */
export const squadExactMatch = defineMetric(
  {
    module: 'applied/evaluation/text',
    key: 'squadExactMatch',
    name: 'SQuAD exact match',
    inputs: 'sequences',
    direction: 'higher',
    range: [0, 1],
    notes: ['squad-exact-match-and-f1'],
    capability: 'decide',
  },
  (golds: string | readonly (string | readonly string[])[], predictions: string | readonly string[]): number => {
    const q = qaOf(golds, predictions)
    return (
      q.predictions.reduce(
        (s, p, i) => s + (asList(q.golds[i]).some((g) => normaliseAnswer(g) === normaliseAnswer(p)) ? 1 : 0),
        0,
      ) / q.predictions.length
    )
  },
)

/** SQuAD token F₁: the best multiset-overlap F₁ against any gold answer, averaged over questions. */
export const squadF1 = defineMetric(
  {
    module: 'applied/evaluation/text',
    key: 'squadF1',
    name: 'SQuAD token F₁',
    inputs: 'sequences',
    direction: 'higher',
    range: [0, 1],
    notes: ['squad-exact-match-and-f1'],
    capability: 'decide',
  },
  (golds: string | readonly (string | readonly string[])[], predictions: string | readonly string[]): number => {
    const q = qaOf(golds, predictions)
    return (
      q.predictions.reduce((s, p, i) => s + Math.max(...asList(q.golds[i]).map((g) => squadF1Of(g, p))), 0) /
      q.predictions.length
    )
  },
)

// ── BERTScore ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * BERTScore from contextual token embeddings (Zhang et al. 2020; bertscore): rows are normalised to unit length, each
 * reference token takes its most similar candidate token for recall and each candidate token its most similar
 * reference token for precision; F is their harmonic mean. Optional importance weights (e.g. idf) weight each token's
 * best similarity. The embedding model is the caller's.
 */
export function bertScore(
  referenceEmbeddings: Rows,
  candidateEmbeddings: Rows,
  options: { referenceWeights?: ArrayLike<number>; candidateWeights?: ArrayLike<number> } = {},
): Prf {
  const unit = (x: Rows) => {
    const D = dense(x, 'bertScore')
    for (let i = 0; i < D.rows; i++) {
      let s = 0
      for (let c = 0; c < D.cols; c++) s += D.data[i * D.cols + c] ** 2
      const norm = Math.sqrt(s)
      for (let c = 0; c < D.cols; c++) D.data[i * D.cols + c] /= norm
    }
    return D
  }
  const R = unit(referenceEmbeddings)
  const C = unit(candidateEmbeddings)
  if (R.cols !== C.cols) throw new ShapeError('metrics', 'metrics: bertScore: embeddings differ in dimension')
  const sim = (i: number, j: number) => {
    let s = 0
    for (let c = 0; c < R.cols; c++) s += R.data[i * R.cols + c] * C.data[j * C.cols + c]
    return s
  }
  const side = (rows: number, cols: number, f: (a: number, b: number) => number, w?: ArrayLike<number>) => {
    let s = 0
    let t = 0
    for (let a = 0; a < rows; a++) {
      let best = -Infinity
      for (let b = 0; b < cols; b++) best = Math.max(best, f(a, b))
      const wa = w ? w[a] : 1
      s += wa * best
      t += wa
    }
    return s / t
  }
  const recall = side(R.rows, C.rows, (i, j) => sim(i, j), options.referenceWeights)
  const precision = side(C.rows, R.rows, (j, i) => sim(i, j), options.candidateWeights)
  return { precision, recall, f: (2 * precision * recall) / (precision + recall) }
}

// ── Backretrieval ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Backretrieval BkR@K (Fain, Twomey and Bollegala 2021; backretrieval): each source query q retrieves the target text
 * j*(q) with the highest text similarity (`textSimilarity`, N × M, source queries by target texts), then ranks every
 * source image by its similarity to j*'s image (`imageSimilarity`, M × N, target images by source images); the score
 * is the fraction of queries whose own image ranks in the top K, with rank 1 + #{images strictly more similar} (ties
 * favour the query). Default K = 10. Evaluates the text embedding; the image similarity is a fixed instrument.
 */
export const backretrieval = defineMetric(
  {
    module: 'applied/evaluation/text',
    key: 'backretrieval',
    name: 'Backretrieval at K',
    inputs: 'vectors',
    direction: 'higher',
    range: [0, 1],
    notes: ['backretrieval'],
  },
  (textSimilarity: Rows, imageSimilarity: Rows, options: { k?: number } = {}): number => {
    const T = dense(textSimilarity, 'backretrieval')
    const G = dense(imageSimilarity, 'backretrieval')
    const N = T.rows
    const M = T.cols
    if (G.rows !== M || G.cols !== N)
      throw new ShapeError('metrics', `metrics: backretrieval: image similarity must be ${M} × ${N}`)
    const K = options.k ?? 10
    let hits = 0
    for (let q = 0; q < N; q++) {
      let best = 0
      for (let j = 1; j < M; j++) if (T.data[q * M + j] > T.data[q * M + best]) best = j
      const own = G.data[best * N + q]
      let rank = 1
      for (let i = 0; i < N; i++) if (G.data[best * N + i] > own) rank++
      if (rank <= K) hits++
    }
    return hits / N
  },
)

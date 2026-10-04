/**
 * Information measures of discrete distributions: entropy, joint and conditional entropy, cross-entropy, KL and
 * Jensen–Shannon divergences, the f-divergence family, total variation and Hellinger distances, mutual information and
 * pointwise mutual information from a joint table, and the differential entropy of a distribution object. Definitions
 * follow Cover and Thomas (2006), "Elements of Information Theory", 2nd ed., ch. 2 and 8.
 *
 * Inputs are probability vectors or tables (tensors, possibly traced, or number arrays). They are normalised here,
 * as scipy.stats.entropy does, so counts may be passed directly. Results are in nats unless `base` is given (2 for
 * bits). Everything except `fDivergence` is a composition of primitives, so it is differentiable in the inputs.
 */

import { xlogy } from 'aifn-compute/numerics/special'
import type { Distribution } from 'aifn-compute/probability/distributions'
import {
  abs,
  add,
  div,
  fromRows,
  isTensor,
  isTraced,
  log,
  mul,
  neg,
  reshape,
  shapeOfValue,
  sqrt,
  square,
  sub,
  sum,
  tensor,
  toFlat,
  unwrap,
  where,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A probability vector or table: a tensor (possibly traced) or plain numbers. */
export type Probabilities = Value | ArrayLike<number> | readonly (readonly number[])[]

/** Options shared by the measures. */
export type BaseOption = {
  /** Logarithm base: e (default, nats), 2 (bits) or any base > 1. */
  base?: number
}

/** Convert plain arrays to a tensor; tensors and traced values pass through. */
export function asValue(p: Probabilities): Value {
  if (typeof p === 'number' || isTensor(p) || isTraced(p)) return p as Value
  const first = (p as ArrayLike<unknown>)[0]
  if (Array.isArray(first)) return fromRows(p as number[][])
  return tensor(Array.from(p as ArrayLike<number>))
}

/** Normalise along the last axis (a vector sums to 1; each row of a batch sums to 1). */
function normalise(p: Value): Value {
  return div(p, sum(p, -1, true))
}

/** Normalise a whole table to sum to 1. */
function normaliseAll(p: Value): Value {
  return div(p, sum(p))
}

/** Sum over the last axis, giving a number for a single vector. */
function sumLast(v: Value): Value {
  return shapeOfValue(v).length === 1 ? sum(v) : sum(v, -1)
}

/** Change of base: divide nats by log(base). */
function inBase(v: Value, base: number | undefined): Value {
  if (base === undefined || base === Math.E) return v
  if (!(base > 0 && base !== 1)) throw new DomainError('inBase', `base must be positive and not 1, got ${base}`)
  return div(v, Math.log(base))
}

/**
 * Shannon entropy H(p) = −Σ pₖ log pₖ (with 0 log 0 = 0) of a probability vector, or of each row of a batch
 * `[..., K]` (like scipy.stats.entropy with axis = −1). The input is normalised first.
 */
export function entropy(p: Probabilities, { base }: BaseOption = {}): Value {
  const q = normalise(asValue(p))
  return inBase(neg(sumLast(xlogy(q, q))), base)
}

/** The joint entropy H(X, Y) = −Σ p(x, y) log p(x, y) of a joint table (any shape; normalised as a whole). */
export function jointEntropy(joint: Probabilities, { base }: BaseOption = {}): Value {
  const q = normaliseAll(asValue(joint))
  return inBase(neg(sum(xlogy(q, q))), base)
}

/** The marginals of a joint table [|X|, |Y|]: p(x) (row sums) and p(y) (column sums), after normalisation. */
function marginals(joint: Value): { pxy: Value; px: Value; py: Value } {
  const shape = shapeOfValue(joint)
  if (shape.length !== 2) throw new ShapeError('marginals', 'expected a joint table of shape [|X|, |Y|]')
  const pxy = normaliseAll(joint)
  return { pxy, px: sum(pxy, 1), py: sum(pxy, 0) }
}

/**
 * The conditional entropy H(Y | X) = H(X, Y) − H(X) of a joint table whose rows index X and columns Y. With
 * `given: 'y'`, H(X | Y).
 */
export function conditionalEntropy(
  joint: Probabilities,
  { base, given = 'x' }: BaseOption & { given?: 'x' | 'y' } = {},
): Value {
  const { pxy, px, py } = marginals(asValue(joint))
  const m = given === 'x' ? px : py
  return inBase(sub(neg(sum(xlogy(pxy, pxy))), neg(sum(xlogy(m, m)))), base)
}

/**
 * The cross-entropy H(p, q) = −Σ pₖ log qₖ (∞ when q puts no mass where p does), per row of a batch. Both inputs are
 * normalised.
 */
export function crossEntropy(p: Probabilities, q: Probabilities, { base }: BaseOption = {}): Value {
  return inBase(neg(sumLast(xlogy(normalise(asValue(p)), normalise(asValue(q))))), base)
}

/**
 * The Kullback–Leibler divergence KL(p ‖ q) = Σ pₖ log(pₖ/qₖ) ≥ 0, ∞ when q puts no mass where p does (scipy's
 * `entropy(p, q)`), per row of a batch. For distribution objects see `aifn-compute/probability/distributions`' `kl`.
 */
export function klDivergence(p: Probabilities, q: Probabilities, { base }: BaseOption = {}): Value {
  const a = normalise(asValue(p))
  const b = normalise(asValue(q))
  return inBase(sumLast(sub(xlogy(a, a), xlogy(a, b))), base)
}

/**
 * The Jensen–Shannon divergence JS(p, q) = ½ KL(p ‖ m) + ½ KL(q ‖ m) with m = (p + q)/2: symmetric, finite, and at
 * most log 2 (1 bit). Lin (1991), "Divergence measures based on the Shannon entropy", IEEE Trans. Inf. Theory 37(1).
 */
export function jensenShannonDivergence(p: Probabilities, q: Probabilities, { base }: BaseOption = {}): Value {
  const a = normalise(asValue(p))
  const b = normalise(asValue(q))
  const m = mul(0.5, add(a, b))
  const half = (x: Value) => sumLast(sub(xlogy(x, x), xlogy(x, m)))
  return inBase(mul(0.5, add(half(a), half(b))), base)
}

/** The Jensen–Shannon distance √JS(p, q), a metric (scipy.spatial.distance.jensenshannon). */
export function jensenShannonDistance(p: Probabilities, q: Probabilities, options: BaseOption = {}): Value {
  return sqrt(jensenShannonDivergence(p, q, options))
}

/** Total variation distance ½ Σ |pₖ − qₖ| ∈ [0, 1] (the f-divergence of f(t) = ½|t − 1|). */
export function totalVariation(p: Probabilities, q: Probabilities): Value {
  return mul(0.5, sumLast(abs(sub(normalise(asValue(p)), normalise(asValue(q))))))
}

/**
 * Hellinger distance H(p, q) = √(½ Σ (√pₖ − √qₖ)²) ∈ [0, 1]. Computed from the squared differences (not as
 * √(1 − Σ √(pq))), so rounding cannot make it NaN.
 */
export function hellingerDistance(p: Probabilities, q: Probabilities): Value {
  const d = sub(sqrt(normalise(asValue(p))), sqrt(normalise(asValue(q))))
  return sqrt(mul(0.5, sumLast(square(d))))
}

// ── f-divergences ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A convex generator f with f(1) = 0, defining D_f(p ‖ q) = Σ qₖ f(pₖ/qₖ) (Csiszár, 1967; Ali and Silvey, 1966).
 * `f(0)` must be defined (possibly ∞). `slope` is lim_{t→∞} f(t)/t, which gives the terms where qₖ = 0 < pₖ.
 */
export type FGenerator = {
  name: string
  f: (t: number) => number
  slope: number
}

/** The standard generators. */
export const fGenerators = {
  /** f(t) = t log t: KL(p ‖ q). */
  kl: { name: 'KL', f: (t: number) => (t === 0 ? 0 : t * Math.log(t)), slope: Infinity },
  /** f(t) = −log t: KL(q ‖ p). */
  reverseKl: { name: 'reverse KL', f: (t: number) => -Math.log(t), slope: 0 },
  /** f(t) = ½|t − 1|: total variation. */
  totalVariation: { name: 'total variation', f: (t: number) => 0.5 * Math.abs(t - 1), slope: 0.5 },
  /** f(t) = ½(√t − 1)²: the squared Hellinger distance. */
  squaredHellinger: { name: 'squared Hellinger', f: (t: number) => 0.5 * (Math.sqrt(t) - 1) ** 2, slope: 0.5 },
  /** f(t) = (t − 1)²: Pearson's χ². */
  pearsonChiSquare: { name: 'Pearson χ²', f: (t: number) => (t - 1) ** 2, slope: Infinity },
  /** f(t) = (t − 1)²/t: Neyman's χ². */
  neymanChiSquare: { name: 'Neyman χ²', f: (t: number) => (t - 1) ** 2 / t, slope: 1 },
  /** f(t) = ½[t log t − (t + 1) log((t + 1)/2)]: Jensen–Shannon. */
  jensenShannon: {
    name: 'Jensen–Shannon',
    f: (t: number) => 0.5 * ((t === 0 ? 0 : t * Math.log(t)) - (t + 1) * Math.log((t + 1) / 2)),
    slope: 0.5 * Math.LN2,
  },
} satisfies Record<string, FGenerator>

/**
 * The f-divergence D_f(p ‖ q) = Σ_{qₖ>0} qₖ f(pₖ/qₖ) + slope · Σ_{qₖ=0} pₖ for probability vectors p and q (normalised
 * here), with a generator from `fGenerators` or your own. Not differentiable (f is a plain function).
 */
export function fDivergence(p: Probabilities, q: Probabilities, generator: FGenerator): number {
  const a = flatProbabilities(p, 'fDivergence')
  const b = flatProbabilities(q, 'fDivergence')
  if (a.length !== b.length) throw new ShapeError('fDivergence', 'fDivergence: p and q have different lengths')
  let total = 0
  for (let k = 0; k < a.length; k++) {
    if (b[k] > 0) total += b[k] * generator.f(a[k] / b[k])
    else if (a[k] > 0) total += a[k] * generator.slope
  }
  return total
}

/**
 * A probability vector (or table, read row-major) normalised to sum to 1, as plain numbers, for algorithms that walk
 * probabilities one by one (prefix codes, arithmetic coding). Negative entries are an error, and so is traced input
 * (such algorithms have no derivative).
 */
export function flatProbabilities(p: Probabilities, where: string): number[] {
  const v = asValue(p)
  if (isTraced(v)) throw new Error(`${where}: not differentiable (a traced value was passed)`)
  const r = unwrap(v)
  const values = typeof r === 'number' ? [r] : toFlat(r)
  const total = values.reduce((s, x) => s + x, 0)
  for (const x of values)
    if (!(x >= 0)) throw new DomainError(where, `${where}: probabilities must be non-negative, got ${x}`)
  return values.map((x) => x / total)
}

// ── Mutual information from a joint table ────────────────────────────────────────────────────────────────────────────

/**
 * Mutual information I(X; Y) = Σ p(x, y) log[p(x, y) / (p(x) p(y))] = KL(p_XY ‖ p_X ⊗ p_Y) of a joint table
 * [|X|, |Y|] (normalised here). From counts, this is scikit-learn's `mutual_info_score` on the contingency table.
 */
export function mutualInformation(joint: Probabilities, { base }: BaseOption = {}): Value {
  const { pxy, px, py } = marginals(asValue(joint))
  const product = mul(column(px), py)
  return inBase(sum(sub(xlogy(pxy, pxy), xlogy(pxy, product))), base)
}

/** A vector [n] as a column [n, 1]. */
function column(v: Value): Value {
  return reshape(v, [shapeOfValue(v)[0], 1])
}

/**
 * Pointwise mutual information pmi(x, y) = log[p(x, y) / (p(x) p(y))] for every cell of a joint table (−∞ where
 * p(x, y) = 0). With `normalised`, npmi = pmi / (−log p(x, y)) ∈ [−1, 1] (Bouma, 2009), −1 where p(x, y) = 0.
 */
export function pointwiseMutualInformation(
  joint: Probabilities,
  { base, normalised = false }: BaseOption & { normalised?: boolean } = {},
): Value {
  const { pxy, px, py } = marginals(asValue(joint))
  const pmi = sub(log(pxy), log(mul(column(px), py)))
  if (!normalised) return inBase(pmi, base)
  const zero = unwrap(pxy)
  const empty = typeof zero === 'number' ? (zero === 0 ? 1 : 0) : where(unwrap(pxy), 0, 1)
  return where(empty, -1, div(pmi, neg(log(pxy))))
}

/** The differential entropy (continuous) or entropy (discrete) of a distribution object, in the given base. */
export function differentialEntropy(d: Distribution, { base }: BaseOption = {}): Value {
  return inBase(d.entropy() as Value, base)
}

/**
 * Information measures of discrete distributions: entropy, joint and conditional entropy, cross-entropy,
 * Kullback–Leibler and Jensen–Shannon divergences, the $f$-divergence family, total variation and Hellinger distances,
 * mutual information and pointwise mutual information from a joint table, and the differential entropy of a
 * distribution object. Definitions follow Cover and Thomas (2006), "Elements of Information Theory", 2nd ed., ch. 2 and
 * 8.
 *
 * Inputs are probability vectors or tables (tensors, possibly traced, or number arrays). They are normalised here, as
 * scipy.stats.entropy does, so counts may be passed directly. Results are in nats unless `base` is given (2 for bits).
 * Terms with $p_k = 0$ count as $0 \log 0 = 0$. Everything except `fDivergence` and `flatProbabilities` is a
 * composition of primitives, so it is differentiable in the inputs.
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

/**
 * A probability vector or table: a tensor (possibly traced), a number array, or an array of rows for a table. It need
 * not sum to 1 (counts are fine): the measures normalise it.
 */
export type Probabilities = Value | ArrayLike<number> | readonly (readonly number[])[]

/** Options shared by the measures. */
export type BaseOption = {
  /** Logarithm base: $e$ (default, nats), 2 (bits), or any positive base other than 1 (others throw). */
  base?: number
}

/**
 * Convert plain arrays to a tensor; numbers, tensors and traced values pass through.
 *
 * @param p The probabilities: an array of rows becomes a matrix, any other array-like a vector.
 * @returns The probabilities as a value the tensor operations accept (not normalised).
 *
 * @example A vector and a table
 * print('vector:', asValue([1, 2, 3]))
 * print('table:', asValue([[0.25, 0.25], [0.5, 0]]))
 */
export function asValue(p: Probabilities): Value {
  if (typeof p === 'number' || isTensor(p) || isTraced(p)) return p as Value
  const first = (p as ArrayLike<unknown>)[0]
  if (Array.isArray(first)) return fromRows(p as number[][])
  return tensor(Array.from(p as ArrayLike<number>))
}

/**
 * Normalise along the last axis (a vector sums to 1; each row of a batch sums to 1).
 *
 * @param p Non-negative weights, a vector or a batch of rows.
 * @returns The weights divided by their sum along the last axis.
 */
function normalise(p: Value): Value {
  return div(p, sum(p, -1, true))
}

/**
 * Normalise a whole table to sum to 1.
 *
 * @param p Non-negative weights of any shape.
 * @returns The weights divided by their total.
 */
function normaliseAll(p: Value): Value {
  return div(p, sum(p))
}

/**
 * Sum over the last axis, giving a number for a single vector.
 *
 * @param v A vector, or a batch whose last axis holds the terms.
 * @returns The sum: a scalar for a vector, one value per row of a batch.
 */
function sumLast(v: Value): Value {
  return shapeOfValue(v).length === 1 ? sum(v) : sum(v, -1)
}

/**
 * Change of base: divide nats by $\log b$. Throws a `DomainError` for a base that is not positive or is 1.
 *
 * @param v A value in nats.
 * @param base The base $b$ of the logarithm wanted; `undefined` or $e$ leaves `v` in nats.
 * @returns The value in the units of base $b$.
 */
function inBase(v: Value, base: number | undefined): Value {
  if (base === undefined || base === Math.E) return v
  if (!(base > 0 && base !== 1)) throw new DomainError('inBase', `base must be positive and not 1, got ${base}`)
  return div(v, Math.log(base))
}

/**
 * The Shannon entropy $\entropy(p) = -\sum_k p_k \log p_k$ (with $0 \log 0 = 0$) of a probability vector, or of each
 * row of a batch of shape $[\dots, K]$ (like scipy.stats.entropy with `axis=-1`). The input is normalised first.
 *
 * @param p The probabilities or counts, along the last axis.
 * @param options The units.
 * @param options.base The base of the logarithm (default $e$, nats; 2 for bits).
 * @returns The entropy: a number for a vector, one value per row of a batch.
 *
 * @example A fair coin, a fair die and counts
 * print('fair coin, bits:', entropy([0.5, 0.5], { base: 2 }))
 * print('fair coin, nats (log 2):', entropy([0.5, 0.5]))
 * print('fair die, bits (log2 6):', entropy([1, 1, 1, 1, 1, 1], { base: 2 }))
 * print('counts 1, 1, 2 (1.5 bits):', entropy([1, 1, 2], { base: 2 }))
 *
 * @example One entropy per row
 * print(entropy(tensor([[0.5, 0.5], [1, 0]]), { base: 2 }))
 */
export function entropy(p: Probabilities, { base }: BaseOption = {}): Value {
  const q = normalise(asValue(p))
  return inBase(neg(sumLast(xlogy(q, q))), base)
}

/**
 * The joint entropy $\entropy(X, Y) = -\sum_{x, y} p(x, y) \log p(x, y)$ of a joint table (any shape, normalised as a
 * whole).
 *
 * @param joint The joint probabilities or counts; every cell is one outcome.
 * @param options The units.
 * @param options.base The base of the logarithm (default $e$, nats; 2 for bits).
 * @returns The joint entropy, a number.
 *
 * @example Two independent fair bits carry two bits
 * print(jointEntropy([[1, 1], [1, 1]], { base: 2 }))
 */
export function jointEntropy(joint: Probabilities, { base }: BaseOption = {}): Value {
  const q = normaliseAll(asValue(joint))
  return inBase(neg(sum(xlogy(q, q))), base)
}

/**
 * The joint table normalised, with its marginals $p(x)$ (row sums) and $p(y)$ (column sums). Throws a `ShapeError`
 * unless the table is a matrix.
 *
 * @param joint The joint table, of shape $[\lvert X \rvert, \lvert Y \rvert]$, rows indexing $X$.
 * @returns `pxy`, the normalised table, and its marginals `px` and `py`.
 */
function marginals(joint: Value): { pxy: Value; px: Value; py: Value } {
  const shape = shapeOfValue(joint)
  if (shape.length !== 2) throw new ShapeError('marginals', 'expected a joint table of shape [|X|, |Y|]')
  const pxy = normaliseAll(joint)
  return { pxy, px: sum(pxy, 1), py: sum(pxy, 0) }
}

/**
 * The conditional entropy $\entropy(Y \mid X) = \entropy(X, Y) - \entropy(X)$ of a joint table whose rows index $X$
 * and columns $Y$. With `given: 'y'`, $\entropy(X \mid Y)$.
 *
 * @param joint The joint probabilities or counts, a matrix of shape $[\lvert X \rvert, \lvert Y \rvert]$.
 * @param options The units and the conditioning variable.
 * @param options.base The base of the logarithm (default $e$, nats; 2 for bits).
 * @param options.given The variable conditioned on: `'x'` (the rows, default) or `'y'` (the columns).
 * @returns The conditional entropy, a number.
 *
 * @example Y copies X, so knowing X leaves nothing
 * const joint = [[0.5, 0], [0, 0.5]]
 * print('H(Y | X):', conditionalEntropy(joint, { base: 2 }))
 * print('H(Y | X), independent:', conditionalEntropy([[1, 1], [1, 1]], { base: 2 }))
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
 * The cross-entropy $\entropy(p, q) = -\sum_k p_k \log q_k$ ($\infty$ when $q$ puts no mass where $p$ does), per row
 * of a batch. Both inputs are normalised.
 *
 * @param p The true distribution $p$, along the last axis.
 * @param q The model distribution $q$, of the same shape (or broadcasting to it).
 * @param options The units.
 * @param options.base The base of the logarithm (default $e$, nats; 2 for bits).
 * @returns The cross-entropy: a number for vectors, one value per row of a batch.
 *
 * @example Coding a fair coin with a biased code
 * print('H(p, p) (1 bit):', crossEntropy([0.5, 0.5], [0.5, 0.5], { base: 2 }))
 * print('H(p, q):', crossEntropy([0.5, 0.5], [0.75, 0.25], { base: 2 }))
 */
export function crossEntropy(p: Probabilities, q: Probabilities, { base }: BaseOption = {}): Value {
  return inBase(neg(sumLast(xlogy(normalise(asValue(p)), normalise(asValue(q))))), base)
}

/**
 * The Kullback–Leibler divergence $\KL(p \,\Vert\, q) = \sum_k p_k \log(p_k / q_k) \ge 0$, $\infty$ when $q$ puts no
 * mass where $p$ does (scipy's `entropy(p, q)`), per row of a batch. Both inputs are normalised. For distribution
 * objects see `aifn-compute/probability/distributions`' `kl`.
 *
 * @param p The distribution $p$, along the last axis.
 * @param q The reference distribution $q$, of the same shape (or broadcasting to it).
 * @param options The units.
 * @param options.base The base of the logarithm (default $e$, nats; 2 for bits).
 * @returns The divergence: a number for vectors, one value per row of a batch.
 *
 * @example Asymmetric, and infinite where q has no mass
 * print('KL(p || q):', klDivergence([0.5, 0.5], [0.75, 0.25], { base: 2 }))
 * print('KL(q || p):', klDivergence([0.75, 0.25], [0.5, 0.5], { base: 2 }))
 * print('KL(p || (1, 0)):', klDivergence([0.5, 0.5], [1, 0]))
 */
export function klDivergence(p: Probabilities, q: Probabilities, { base }: BaseOption = {}): Value {
  const a = normalise(asValue(p))
  const b = normalise(asValue(q))
  return inBase(sumLast(sub(xlogy(a, a), xlogy(a, b))), base)
}

/**
 * The Jensen–Shannon divergence
 * $\operatorname{JS}(p, q) = \tfrac{1}{2}\KL(p \,\Vert\, m) + \tfrac{1}{2}\KL(q \,\Vert\, m)$ with $m = (p + q)/2$:
 * symmetric, finite, and at most $\log 2$ (1 bit). Lin (1991), "Divergence measures based on the Shannon entropy",
 * IEEE Trans. Inf. Theory 37(1). Per row of a batch; both inputs are normalised.
 *
 * @param p One distribution, along the last axis.
 * @param q The other, of the same shape (or broadcasting to it).
 * @param options The units.
 * @param options.base The base of the logarithm (default $e$, nats; 2 for bits).
 * @returns The divergence: a number for vectors, one value per row of a batch.
 *
 * @example Disjoint supports reach the bound of 1 bit
 * print('disjoint:', jensenShannonDivergence([1, 0], [0, 1], { base: 2 }))
 * print('equal:', jensenShannonDivergence([0.3, 0.7], [0.3, 0.7], { base: 2 }))
 */
export function jensenShannonDivergence(p: Probabilities, q: Probabilities, { base }: BaseOption = {}): Value {
  const a = normalise(asValue(p))
  const b = normalise(asValue(q))
  const m = mul(0.5, add(a, b))
  const half = (x: Value) => sumLast(sub(xlogy(x, x), xlogy(x, m)))
  return inBase(mul(0.5, add(half(a), half(b))), base)
}

/**
 * The Jensen–Shannon distance $\sqrt{\operatorname{JS}(p, q)}$, a metric (scipy.spatial.distance.jensenshannon).
 *
 * @param p One distribution, along the last axis.
 * @param q The other, of the same shape (or broadcasting to it).
 * @param options The units of the divergence under the root (`base`, as `jensenShannonDivergence`).
 * @returns The distance: a number for vectors, one value per row of a batch.
 *
 * @example Disjoint supports, in bits
 * print(jensenShannonDistance([1, 0], [0, 1], { base: 2 }))
 */
export function jensenShannonDistance(p: Probabilities, q: Probabilities, options: BaseOption = {}): Value {
  return sqrt(jensenShannonDivergence(p, q, options))
}

/**
 * The total variation distance $\tfrac{1}{2}\sum_k \lvert p_k - q_k \rvert \in [0, 1]$ (the $f$-divergence of
 * $f(t) = \tfrac{1}{2}\lvert t - 1 \rvert$), per row of a batch. Both inputs are normalised.
 *
 * @param p One distribution, along the last axis.
 * @param q The other, of the same shape (or broadcasting to it).
 * @returns The distance: a number for vectors, one value per row of a batch.
 *
 * @example Two coins
 * print(totalVariation([0.5, 0.5], [0.75, 0.25]))
 */
export function totalVariation(p: Probabilities, q: Probabilities): Value {
  return mul(0.5, sumLast(abs(sub(normalise(asValue(p)), normalise(asValue(q))))))
}

/**
 * The Hellinger distance $H(p, q) = \sqrt{\tfrac{1}{2}\sum_k (\sqrt{p_k} - \sqrt{q_k})^2} \in [0, 1]$, per row of a
 * batch. Computed from the squared differences (not as $\sqrt{1 - \sum_k \sqrt{p_k q_k}}$), so rounding cannot make
 * it NaN. Both inputs are normalised.
 *
 * @param p One distribution, along the last axis.
 * @param q The other, of the same shape (or broadcasting to it).
 * @returns The distance: a number for vectors, one value per row of a batch.
 *
 * @example Disjoint and equal
 * print('disjoint:', hellingerDistance([1, 0], [0, 1]))
 * print('equal:', hellingerDistance([0.3, 0.7], [0.3, 0.7]))
 */
export function hellingerDistance(p: Probabilities, q: Probabilities): Value {
  const d = sub(sqrt(normalise(asValue(p))), sqrt(normalise(asValue(q))))
  return sqrt(mul(0.5, sumLast(square(d))))
}

// ── f-divergences ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A convex generator $f$ with $f(1) = 0$, defining $D_f(p \,\Vert\, q) = \sum_k q_k f(p_k / q_k)$ (Csiszár, 1967; Ali
 * and Silvey, 1966).
 */
export type FGenerator = {
  /** A readable name, e.g. `KL`. */
  name: string
  /** The generator $f(t)$ for $t \ge 0$; $f(0)$ must be defined (possibly $\infty$). */
  f: (t: number) => number
  /** $\lim_{t \to \infty} f(t)/t$, which gives the terms where $q_k = 0 < p_k$. */
  slope: number
}

/** The standard generators, by name. */
export const fGenerators = {
  /** $f(t) = t \log t$: $\KL(p \,\Vert\, q)$. */
  kl: { name: 'KL', f: (t: number) => (t === 0 ? 0 : t * Math.log(t)), slope: Infinity },
  /** $f(t) = -\log t$: $\KL(q \,\Vert\, p)$. */
  reverseKl: { name: 'reverse KL', f: (t: number) => -Math.log(t), slope: 0 },
  /** $f(t) = \tfrac{1}{2}\lvert t - 1 \rvert$: total variation. */
  totalVariation: { name: 'total variation', f: (t: number) => 0.5 * Math.abs(t - 1), slope: 0.5 },
  /** $f(t) = \tfrac{1}{2}(\sqrt{t} - 1)^2$: the squared Hellinger distance. */
  squaredHellinger: { name: 'squared Hellinger', f: (t: number) => 0.5 * (Math.sqrt(t) - 1) ** 2, slope: 0.5 },
  /** $f(t) = (t - 1)^2$: Pearson's $\chi^2$. */
  pearsonChiSquare: { name: 'Pearson χ²', f: (t: number) => (t - 1) ** 2, slope: Infinity },
  /** $f(t) = (t - 1)^2 / t$: Neyman's $\chi^2$. */
  neymanChiSquare: { name: 'Neyman χ²', f: (t: number) => (t - 1) ** 2 / t, slope: 1 },
  /** $f(t) = \tfrac{1}{2}[t \log t - (t + 1) \log((t + 1)/2)]$: Jensen–Shannon. */
  jensenShannon: {
    name: 'Jensen–Shannon',
    f: (t: number) => 0.5 * ((t === 0 ? 0 : t * Math.log(t)) - (t + 1) * Math.log((t + 1) / 2)),
    slope: 0.5 * Math.LN2,
  },
} satisfies Record<string, FGenerator>

/**
 * The $f$-divergence $D_f(p \,\Vert\, q) = \sum_{q_k > 0} q_k f(p_k / q_k) + s \sum_{q_k = 0} p_k$ ($s$ the
 * generator's `slope`) for probability vectors $p$ and $q$ (normalised here), with a generator from `fGenerators` or
 * your own. Not differentiable ($f$ is a plain function): traced input throws, as do negative entries (a
 * `DomainError`) and inputs of different lengths (a `ShapeError`).
 *
 * @param p The distribution $p$, read flat (row-major for a table).
 * @param q The reference distribution $q$, with as many entries as $p$.
 * @param generator The generator $f$ with its slope at infinity.
 * @returns The divergence, a number (in nats for the logarithmic generators).
 *
 * @example The same pair of coins under four generators
 * const p = [0.5, 0.5]
 * const q = [0.75, 0.25]
 * print('KL:', fDivergence(p, q, fGenerators.kl), 'klDivergence:', klDivergence(p, q))
 * print('total variation:', fDivergence(p, q, fGenerators.totalVariation))
 * print('Pearson chi-square:', fDivergence(p, q, fGenerators.pearsonChiSquare))
 * print('KL where q has no mass:', fDivergence(p, [1, 0], fGenerators.kl))
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
 * probabilities one by one (prefix codes, arithmetic coding). Negative entries throw a `DomainError`, and traced input
 * throws too (such algorithms have no derivative).
 *
 * @param p The probabilities or counts.
 * @param where The caller's name, for error messages.
 * @returns The entries divided by their total, as a plain array (NaN entries when they sum to 0).
 *
 * @example Counts to probabilities
 * print(flatProbabilities([[1, 1], [2, 0]], 'example'))
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
 * The mutual information
 * $I(X; Y) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)} = \KL(p_{XY} \,\Vert\, p_X \otimes p_Y)$ of a joint
 * table (normalised here). From counts, this is scikit-learn's `mutual_info_score` on the contingency table. Throws a
 * `ShapeError` unless the table is a matrix.
 *
 * @param joint The joint probabilities or counts, of shape $[\lvert X \rvert, \lvert Y \rvert]$, rows indexing $X$.
 * @param options The units.
 * @param options.base The base of the logarithm (default $e$, nats; 2 for bits).
 * @returns The mutual information, a number.
 *
 * @example A copied bit shares one bit; independent bits share none
 * print('copy:', mutualInformation([[0.5, 0], [0, 0.5]], { base: 2 }))
 * print('independent:', mutualInformation([[1, 1], [1, 1]], { base: 2 }))
 * print('from counts:', mutualInformation([[30, 10], [10, 30]], { base: 2 }))
 */
export function mutualInformation(joint: Probabilities, { base }: BaseOption = {}): Value {
  const { pxy, px, py } = marginals(asValue(joint))
  const product = mul(column(px), py)
  return inBase(sum(sub(xlogy(pxy, pxy), xlogy(pxy, product))), base)
}

/**
 * A vector as a column.
 *
 * @param v A vector of length $n$.
 * @returns The same values with shape $[n, 1]$.
 */
function column(v: Value): Value {
  return reshape(v, [shapeOfValue(v)[0], 1])
}

/**
 * The pointwise mutual information $\operatorname{pmi}(x, y) = \log \frac{p(x, y)}{p(x) p(y)}$ for every cell of a
 * joint table ($-\infty$ where $p(x, y) = 0$). With `normalised`, the normalised form
 * $\operatorname{npmi} = \operatorname{pmi} / (-\log p(x, y)) \in [-1, 1]$ (Bouma, 2009), $-1$ where $p(x, y) = 0$.
 * Throws a `ShapeError` unless the table is a matrix.
 *
 * @param joint The joint probabilities or counts, of shape $[\lvert X \rvert, \lvert Y \rvert]$, rows indexing $X$.
 * @param options The units and the normalisation.
 * @param options.base The base of the logarithm (default $e$, nats; 2 for bits). The normalised form has no units, and
 *   ignores it.
 * @param options.normalised Whether to divide by $-\log p(x, y)$, giving values in $[-1, 1]$.
 * @returns A table of the joint's shape.
 *
 * @example A copied bit, plain and normalised
 * const joint = [[0.5, 0], [0, 0.5]]
 * print('pmi, bits:', pointwiseMutualInformation(joint, { base: 2 }))
 * print('npmi:', pointwiseMutualInformation(joint, { normalised: true }))
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

/**
 * The differential entropy (continuous) or entropy (discrete) of a distribution object, in the given base: its
 * `entropy()`, converted from nats.
 *
 * @param d The distribution.
 * @param options The units.
 * @param options.base The base of the logarithm (default $e$, nats; 2 for bits).
 * @returns The entropy: a number, or one per member of a batch of distributions.
 *
 * @example A distribution's entropy in bits
 * // Only `entropy()` is read: a stand-in with the standard normal's entropy, 0.5 log(2 pi e) nats.
 * const standardNormal = { entropy: () => 0.5 * Math.log(2 * Math.PI * Math.E) }
 * print('nats:', differentialEntropy(standardNormal))
 * print('bits:', differentialEntropy(standardNormal, { base: 2 }))
 */
export function differentialEntropy(d: Distribution, { base }: BaseOption = {}): Value {
  return inBase(d.entropy() as Value, base)
}

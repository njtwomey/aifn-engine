/**
 * Resampling for imbalanced classes, as in imbalanced-learn (Lemaître, Nogueira and Aridas, 2017): random over- and
 * under-sampling; SMOTE (Chawla et al., 2002), which places synthetic minority points on segments between a minority
 * point and one of its k nearest minority neighbours; borderline-SMOTE (Han, Wang and Mao, 2005), which synthesises
 * only from minority points "in danger" (at least half, but not all, of their m nearest neighbours in the data belong
 * to other classes); ADASYN (He et al., 2008), which synthesises from each minority point in proportion to the share of
 * other classes among its k nearest neighbours; and Tomek links (Tomek, 1976), pairs of points of different classes
 * that are each other's nearest neighbour, whose majority member is removed.
 *
 * Every class except the largest is over-sampled to the largest class's count (imbalanced-learn's `'auto'`), and
 * under-sampling cuts every class to the smallest's. Each sampler returns the new rows with their origin: the index of
 * the source row, or $-1$ for a synthetic one, and for synthetic rows the two parents and the interpolation weight. A
 * synthetic row is $\xvec = \xvec_b + u(\xvec_{b'} - \xvec_b)$ for a base row $b$, one of its nearest neighbours $b'$
 * in the same class, and $u \sim \Unif(0, 1)$. Labels are read as integers; neighbours are by Euclidean distance.
 */

import { child, units, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type MatrixLike, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'
import { roundHalfEven } from 'aifn-compute/nn/quantise'
import { bruteForceNeighbours } from 'aifn-compute/numerics/neighbours'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** A resampled dataset. */
export interface Resampled {
  /** The resampled rows, $m \times d$: the kept (or copied) source rows, then the synthetic rows. */
  readonly x: Tensor
  /** The integer class label of each output row, $m$ values. */
  readonly y: Tensor
  /** The source row of each output row, or $-1$ for a synthetic row. */
  readonly origin: Int32Array
  /**
   * For each synthetic row (in output order): its base row $b$, its neighbour row $b'$ and the gap $u$ in
   * $\xvec = \xvec_b + u(\xvec_{b'} - \xvec_b)$ (all empty for a sampler that makes no synthetic rows).
   */
  readonly synthetic: { readonly base: Int32Array; readonly neighbour: Int32Array; readonly gap: Float64Array }
}

/**
 * A dataset read for resampling: `n` rows of `d` features, `x` the row-major values ($n d$ of them), `y` the integer
 * labels, `classes` the distinct labels in ascending order, and `counts` the number of rows of each.
 */
type Data = { n: number; d: number; x: Float64Array; y: Int32Array; classes: number[]; counts: Map<number, number> }

/**
 * Read a feature matrix and its labels for resampling. Throws `ShapeError` when they have different numbers of rows.
 *
 * @param x The features, $n \times d$.
 * @param y The class labels, $n$ values, truncated to integers.
 * @param where The caller's name, for error messages.
 * @returns The data as row-major values with integer labels, its classes and their counts.
 */
function prepare(x: MatrixLike, y: VectorLike, where: string): Data {
  const X = dense.toMatrixF64(x, where)
  const labels = dense.toF64(y, where)
  if (labels.length !== X.m) throw new ShapeError(where, `${where}: x and y have different rows`)
  const yi = Int32Array.from(labels)
  const counts = new Map<number, number>()
  for (const c of yi) counts.set(c, (counts.get(c) ?? 0) + 1)
  return { n: X.m, d: X.n, x: X.data, y: yi, classes: [...counts.keys()].sort((a, b) => a - b), counts }
}

/**
 * The number of rows of the largest class.
 *
 * @param d The data.
 * @returns The largest class count.
 */
const majorityCount = (d: Data) => Math.max(...d.counts.values())
/**
 * The indices of the rows of one class, ascending.
 *
 * @param d The data.
 * @param c The class label.
 * @returns The row indices whose label is `c`.
 */
const rowsOfClass = (d: Data, c: number) => Array.from({ length: d.n }, (_, i) => i).filter((i) => d.y[i] === c)

/**
 * Assemble the output: the kept original rows, then the synthetic rows.
 *
 * @param d The source data; not modified.
 * @param kept The source rows to output first, in order (a row may repeat, as in over-sampling).
 * @param synth The synthetic rows, as parallel lists: each one's base row, neighbour row, gap $u$ and label.
 * @returns The resampled dataset, with each row's origin and the synthetic rows' parents and gaps.
 */
function assemble(
  d: Data,
  kept: readonly number[],
  synth: { base: number[]; neighbour: number[]; gap: number[]; label: number[] },
): Resampled {
  const m = kept.length + synth.base.length
  const x = new Float64Array(m * d.d)
  const y = new Int32Array(m)
  const origin = new Int32Array(m)
  kept.forEach((i, r) => {
    x.set(d.x.subarray(i * d.d, (i + 1) * d.d), r * d.d)
    y[r] = d.y[i]
    origin[r] = i
  })
  synth.base.forEach((b, k) => {
    const r = kept.length + k
    const nb = synth.neighbour[k]
    const u = synth.gap[k]
    for (let j = 0; j < d.d; j++) x[r * d.d + j] = d.x[b * d.d + j] + u * (d.x[nb * d.d + j] - d.x[b * d.d + j])
    y[r] = synth.label[k]
    origin[r] = -1
  })
  return {
    x: fromData(x, [m, d.d]),
    y: fromData(y, [m]),
    origin,
    synthetic: {
      base: Int32Array.from(synth.base),
      neighbour: Int32Array.from(synth.neighbour),
      gap: Float64Array.from(synth.gap),
    },
  }
}

/**
 * Every row index of the data, $0, \dots, n - 1$.
 *
 * @param d The data.
 * @returns The indices in a new array.
 */
const all = (d: Data) => Array.from({ length: d.n }, (_, i) => i)

/**
 * Random over-sampling, as imbalanced-learn's `RandomOverSampler`: every row is kept, then copies of each smaller
 * class's rows, drawn uniformly with replacement, are appended until every class has the majority's count. The copies
 * have their source row as `origin`; nothing is synthetic.
 *
 * @param s The random stream the copies are drawn from (one child stream per class).
 * @param x The features, $n \times d$.
 * @param y The integer class labels, $n$ values.
 * @returns The original rows followed by the copies, class by class in ascending label order.
 *
 * @example Five rows of class 0 and two of class 1 become five and five
 * const x = tensor([[0], [1], [2], [3], [4], [10], [11]])
 * const y = tensor([0, 0, 0, 0, 0, 1, 1])
 * const r = randomOverSample(stream(1), x, y)
 * print('y =', r.y)
 * print('origin:', r.origin)
 */
export function randomOverSample(s: Stream, x: MatrixLike, y: VectorLike): Resampled {
  const d = prepare(x, y, 'randomOverSample')
  const target = majorityCount(d)
  const kept = all(d)
  for (const c of d.classes) {
    const rows = rowsOfClass(d, c)
    const need = target - rows.length
    const u = units(child(s, 'class', c), need)
    for (let k = 0; k < need; k++) kept.push(rows[Math.floor(u[k] * rows.length)])
  }
  return assemble(d, kept, { base: [], neighbour: [], gap: [], label: [] })
}

/**
 * Random under-sampling, as imbalanced-learn's `RandomUnderSampler`: every class is cut to the smallest class's count
 * by sampling its rows without replacement.
 *
 * @param s The random stream the kept rows are drawn from (one child stream per class).
 * @param x The features, $n \times d$.
 * @param y The integer class labels, $n$ values.
 * @returns The kept rows, class by class in ascending label order and in their original order within a class.
 *
 * @example Five rows of class 0 cut to two
 * const x = tensor([[0], [1], [2], [3], [4], [10], [11]])
 * const y = tensor([0, 0, 0, 0, 0, 1, 1])
 * const r = randomUnderSample(stream(1), x, y)
 * print('y =', r.y)
 * print('kept rows:', r.origin)
 */
export function randomUnderSample(s: Stream, x: MatrixLike, y: VectorLike): Resampled {
  const d = prepare(x, y, 'randomUnderSample')
  const target = Math.min(...d.counts.values())
  const kept: number[] = []
  for (const c of d.classes) {
    const rows = rowsOfClass(d, c)
    const u = units(child(s, 'class', c), rows.length)
    // A partial Fisher–Yates shuffle picks `target` rows without replacement.
    for (let k = 0; k < target; k++) {
      const j = k + Math.floor(u[k] * (rows.length - k))
      ;[rows[k], rows[j]] = [rows[j], rows[k]]
    }
    kept.push(...rows.slice(0, target).sort((a, b) => a - b))
  }
  return assemble(d, kept, { base: [], neighbour: [], gap: [], label: [] })
}

/**
 * The $k$ nearest neighbours (global row indices) of each of `rows` among `pool`, excluding itself, nearest first.
 *
 * @param d The data.
 * @param rows The rows whose neighbours are wanted.
 * @param pool The rows the neighbours are chosen from (the row itself is skipped when it is in the pool).
 * @param k The number of neighbours wanted; fewer are returned when the pool is too small.
 * @returns For each of `rows`, the indices of its nearest rows of `pool`.
 */
function neighboursWithin(d: Data, rows: readonly number[], pool: readonly number[], k: number): number[][] {
  const P = new Float64Array(pool.length * d.d)
  pool.forEach((i, r) => P.set(d.x.subarray(i * d.d, (i + 1) * d.d), r * d.d))
  const Q = new Float64Array(rows.length * d.d)
  rows.forEach((i, r) => Q.set(d.x.subarray(i * d.d, (i + 1) * d.d), r * d.d))
  const kk = Math.min(k + 1, pool.length)
  const nb = bruteForceNeighbours(fromData(P, [pool.length, d.d]), fromData(Q, [rows.length, d.d]), kk)
  const idx = toFlat(nb.indices)
  return rows.map((i, r) => {
    const out: number[] = []
    for (let j = 0; j < kk && out.length < k; j++) {
      const g = pool[idx[r * kk + j]]
      if (g !== i) out.push(g)
    }
    return out
  })
}

/** Options of the SMOTE family. */
export interface SmoteOptions {
  /** The number of nearest neighbours of the same class to interpolate towards (default 5). */
  k?: number
}

/**
 * Synthesise `counts[r]` points from each `bases[r]` towards a random one of its $k$ nearest neighbours in class `c`,
 * each at a uniform gap along the segment. Nothing is made when the class has fewer than 2 rows.
 *
 * @param s The random stream (its child for class `c` draws the neighbours and gaps).
 * @param d The data.
 * @param c The class being grown; the neighbours are chosen among its rows.
 * @param bases The rows to synthesise from, all of class `c`.
 * @param counts How many points to synthesise from each of `bases`.
 * @param k The number of nearest neighbours in the class to choose among.
 * @param synth The synthetic rows so far, appended to in place.
 */
function synthesise(
  s: Stream,
  d: Data,
  c: number,
  bases: readonly number[],
  counts: readonly number[],
  k: number,
  synth: { base: number[]; neighbour: number[]; gap: number[]; label: number[] },
) {
  const minority = rowsOfClass(d, c)
  if (minority.length < 2) return
  const nbs = neighboursWithin(d, bases, minority, k)
  const total = counts.reduce((a, b) => a + b, 0)
  const u = units(child(s, 'class', c), 2 * total)
  let at = 0
  bases.forEach((b, r) => {
    for (let q = 0; q < counts[r]; q++) {
      const nb = nbs[r][Math.floor(u[2 * at] * nbs[r].length)]
      synth.base.push(b)
      synth.neighbour.push(nb)
      synth.gap.push(u[2 * at + 1])
      synth.label.push(c)
      at++
    }
  })
}

/**
 * SMOTE (Chawla et al., 2002), as imbalanced-learn's `SMOTE`: every non-majority class is grown to the majority count
 * by synthetic points on segments between its rows and their $k$ nearest neighbours in the class. The base of each
 * synthetic point is drawn uniformly with replacement from the class. A class of fewer than 2 rows is left as it is.
 *
 * @param s The random stream the bases, neighbours and gaps are drawn from.
 * @param x The features, $n \times d$.
 * @param y The integer class labels, $n$ values.
 * @param options The number $k$ of neighbours.
 * @returns Every original row, then the synthetic rows.
 *
 * @example Two minority points give synthetic points on the segment between them
 * const x = tensor([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [0, 5], [1, 5]])
 * const y = tensor([0, 0, 0, 0, 0, 0, 1, 1])
 * const r = smote(stream(1), x, y, { k: 1 })
 * print('y =', r.y)
 * print('synthetic rows =', slice(r.x, [8, 12]))
 * print('gaps:', r.synthetic.gap)
 */
export function smote(s: Stream, x: MatrixLike, y: VectorLike, options: SmoteOptions = {}): Resampled {
  const { k = 5 } = options
  const d = prepare(x, y, 'smote')
  const target = majorityCount(d)
  const synth = { base: [] as number[], neighbour: [] as number[], gap: [] as number[], label: [] as number[] }
  for (const c of d.classes) {
    const rows = rowsOfClass(d, c)
    const need = target - rows.length
    if (need <= 0) continue
    // Bases drawn uniformly with replacement, as imbalanced-learn's `_make_samples`.
    const u = units(child(s, 'bases', c), need)
    const counts = new Array<number>(rows.length).fill(0)
    for (let q = 0; q < need; q++) counts[Math.floor(u[q] * rows.length)]++
    synthesise(s, d, c, rows, counts, k, synth)
  }
  return assemble(d, all(d), synth)
}

/** Where a row of class $c$ stands among its $m$ nearest neighbours in the whole data. */
export type BorderStatus = 'safe' | 'danger' | 'noise'

/**
 * Borderline-SMOTE's classification of the rows of class `c` (Han, Wang and Mao, 2005): with $m'$ of a row's $m$
 * nearest neighbours (in all the data) from other classes, `noise` when $m' = m$, `danger` when $m/2 \le m' < m$, else
 * `safe`.
 *
 * @param x The features, $n \times d$.
 * @param y The integer class labels, $n$ values.
 * @param c The class whose rows are classified.
 * @param m The number $m$ of nearest neighbours looked at (fewer when the data has at most $m$ rows).
 * @returns The rows of class `c` (ascending) and the status of each.
 *
 * @example A minority point among the majority is noise, one at the edge is in danger
 * const x = tensor([[0], [1], [2], [3], [4], [5], [2.5], [6], [7], [8], [9]])
 * const y = tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
 * const { rows, status } = borderStatus(x, y, 1, 4)
 * print('rows:', rows)
 * print('status:', status)
 */
export function borderStatus(
  x: MatrixLike,
  y: VectorLike,
  c: number,
  m = 10,
): { rows: Int32Array; status: BorderStatus[] } {
  const d = prepare(x, y, 'borderStatus')
  const rows = rowsOfClass(d, c)
  const nbs = neighboursWithin(d, rows, all(d), m)
  const status = nbs.map((nb): BorderStatus => {
    const other = nb.filter((j) => d.y[j] !== c).length
    return other === nb.length ? 'noise' : other >= nb.length / 2 ? 'danger' : 'safe'
  })
  return { rows: Int32Array.from(rows), status }
}

/**
 * Borderline-SMOTE (variant 1; Han, Wang and Mao, 2005), as imbalanced-learn's `BorderlineSMOTE`: SMOTE from the
 * `danger` rows of each non-majority class only (see `borderStatus`), towards their $k$ nearest neighbours in the
 * class. A class with no row in danger is left as it is.
 *
 * @param s The random stream the bases, neighbours and gaps are drawn from.
 * @param x The features, $n \times d$.
 * @param y The integer class labels, $n$ values.
 * @param options The number `k` of neighbours to interpolate towards, and the number `m` of neighbours (in all the
 *   data, default 10) that decide whether a row is in danger.
 * @returns Every original row, then the synthetic rows.
 *
 * @example Only the minority point at the border is used as a base
 * const x = tensor([[-3], [-2], [-1], [0], [1], [2], [3], [4], [5], [2.5], [6], [7], [8], [9]])
 * const y = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
 * const r = borderlineSmote(stream(1), x, y, { k: 2, m: 4 })
 * print('bases:', r.synthetic.base, ' neighbours:', r.synthetic.neighbour)
 * print('synthetic x =', slice(r.x, [14, 18]))
 */
export function borderlineSmote(
  s: Stream,
  x: MatrixLike,
  y: VectorLike,
  options: SmoteOptions & { m?: number } = {},
): Resampled {
  const { k = 5, m = 10 } = options
  const d = prepare(x, y, 'borderlineSmote')
  const target = majorityCount(d)
  const synth = { base: [] as number[], neighbour: [] as number[], gap: [] as number[], label: [] as number[] }
  for (const c of d.classes) {
    const need = target - (d.counts.get(c) ?? 0)
    if (need <= 0) continue
    const b = borderStatus(x, y, c, m)
    const danger = Array.from(b.rows).filter((_, r) => b.status[r] === 'danger')
    if (danger.length === 0) continue
    const u = units(child(s, 'bases', c), need)
    const counts = new Array<number>(danger.length).fill(0)
    for (let q = 0; q < need; q++) counts[Math.floor(u[q] * danger.length)]++
    synthesise(s, d, c, danger, counts, k, synth)
  }
  return assemble(d, all(d), synth)
}

/**
 * ADASYN's allocation for class $c$ (He et al., 2008): $r_i$, the share of other classes among each row's $k$ nearest
 * neighbours (in all the data), normalised to sum to 1, and $g_i = \operatorname{round}(r_i G)$ synthetic points per
 * row, with $G$ the shortfall to the majority count and ties rounded to even (numpy's `rint`). The $g_i$ need not sum
 * to exactly $G$. When no row has a neighbour of another class, every $r_i$ and $g_i$ is 0.
 *
 * @param x The features, $n \times d$.
 * @param y The integer class labels, $n$ values.
 * @param c The class whose rows are weighted.
 * @param k The number $k$ of nearest neighbours looked at.
 * @returns The rows of class `c` (ascending), their normalised ratios $r_i$ and their counts $g_i$.
 *
 * @example Points nearer the other class get more synthetic points
 * const x = tensor([[-3], [-2], [-1], [0], [1], [2], [3], [4], [5], [2.5], [6], [7], [8], [9]])
 * const y = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
 * const w = adasynWeights(x, y, 1, 4)
 * print('rows:', w.rows)
 * print('ratio:', w.ratio)
 * print('counts:', w.counts)
 */
export function adasynWeights(
  x: MatrixLike,
  y: VectorLike,
  c: number,
  k = 5,
): { rows: Int32Array; ratio: Float64Array; counts: Int32Array } {
  const d = prepare(x, y, 'adasynWeights')
  const rows = rowsOfClass(d, c)
  const need = majorityCount(d) - rows.length
  const nbs = neighboursWithin(d, rows, all(d), k)
  const ratio = Float64Array.from(nbs, (nb) => nb.filter((j) => d.y[j] !== c).length / k)
  const total = ratio.reduce((a, b) => a + b, 0)
  if (total > 0) for (let r = 0; r < ratio.length; r++) ratio[r] /= total
  // Round half to even, as numpy's `rint`.
  const counts = Int32Array.from(ratio, (r) => roundHalfEven(r * need))
  return { rows: Int32Array.from(rows), ratio, counts }
}

/**
 * ADASYN (He et al., 2008), as imbalanced-learn's `ADASYN`: each non-majority class synthesises from each of its rows
 * the number of points `adasynWeights` allots it, so more from rows whose neighbourhoods are dominated by other
 * classes, towards their $k$ nearest neighbours in the class. Class counts end near, not always at, the majority's.
 *
 * @param s The random stream the neighbours and gaps are drawn from.
 * @param x The features, $n \times d$.
 * @param y The integer class labels, $n$ values.
 * @param options The number $k$ of neighbours, used both for the weights and for the interpolation.
 * @returns Every original row, then the synthetic rows.
 *
 * @example Most synthetic points come from the minority rows near the majority
 * const x = tensor([[-3], [-2], [-1], [0], [1], [2], [3], [4], [5], [2.5], [6], [7], [8], [9]])
 * const y = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
 * const r = adasyn(stream(1), x, y, { k: 4 })
 * print('y =', r.y)
 * print('bases:', r.synthetic.base)
 */
export function adasyn(s: Stream, x: MatrixLike, y: VectorLike, options: SmoteOptions = {}): Resampled {
  const { k = 5 } = options
  const d = prepare(x, y, 'adasyn')
  const target = majorityCount(d)
  const synth = { base: [] as number[], neighbour: [] as number[], gap: [] as number[], label: [] as number[] }
  for (const c of d.classes) {
    if (target - (d.counts.get(c) ?? 0) <= 0) continue
    const w = adasynWeights(x, y, c, k)
    synthesise(s, d, c, Array.from(w.rows), Array.from(w.counts), k, synth)
  }
  return assemble(d, all(d), synth)
}

/**
 * Tomek links (Tomek, 1976): pairs $(i, j)$, $i < j$, of rows of different classes that are each other's nearest
 * neighbour.
 *
 * @param x The features, $n \times d$.
 * @param y The integer class labels, $n$ values.
 * @returns The links as pairs of row indices, in increasing order of $i$.
 *
 * @example One pair of mutual nearest neighbours of different classes
 * const x = tensor([[0], [1], [1.4], [3], [5]])
 * const y = tensor([0, 0, 1, 0, 1])
 * print('links:', tomekLinks(x, y))
 */
export function tomekLinks(x: MatrixLike, y: VectorLike): [number, number][] {
  const d = prepare(x, y, 'tomekLinks')
  const nn = neighboursWithin(d, all(d), all(d), 1).map((v) => v[0])
  const links: [number, number][] = []
  for (let i = 0; i < d.n; i++) {
    const j = nn[i]
    if (j > i && nn[j] === i && d.y[i] !== d.y[j]) links.push([i, j])
  }
  return links
}

/**
 * Remove every Tomek-link member that is not of the minority class (imbalanced-learn's `'auto'`): with two classes, the
 * majority member; with more, both members of a link between two non-minority classes. The minority is the class with
 * the fewest rows (the lowest label among ties).
 *
 * @param x The features, $n \times d$.
 * @param y The integer class labels, $n$ values.
 * @returns The rows kept (in their original order) with the `links` that were found.
 *
 * @example The majority member of the link is removed
 * const x = tensor([[0], [1], [1.4], [3], [5]])
 * const y = tensor([0, 0, 1, 0, 1])
 * const r = removeTomekLinks(x, y)
 * print('links:', r.links)
 * print('kept rows:', r.origin)
 */
export function removeTomekLinks(x: MatrixLike, y: VectorLike): Resampled & { links: [number, number][] } {
  const d = prepare(x, y, 'removeTomekLinks')
  const links = tomekLinks(x, y)
  const minority = d.classes.reduce((a, c) => ((d.counts.get(c) ?? 0) < (d.counts.get(a) ?? 0) ? c : a), d.classes[0])
  const drop = new Set<number>()
  // imbalanced-learn's 'auto': every class but the minority loses its linked rows.
  for (const [i, j] of links) {
    if (d.y[i] !== minority) drop.add(i)
    if (d.y[j] !== minority) drop.add(j)
  }
  const kept = all(d).filter((i) => !drop.has(i))
  return { ...assemble(d, kept, { base: [], neighbour: [], gap: [], label: [] }), links }
}

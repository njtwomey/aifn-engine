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
 * the source row, or −1 for a synthetic one, and for synthetic rows the two parents and the interpolation weight.
 */

import { child, units, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type MatrixLike, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'
import { roundHalfEven } from 'aifn-compute/nn/quantise'
import { bruteForceNeighbours } from 'aifn-compute/numerics/neighbours'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** A resampled dataset. */
export interface Resampled {
  readonly x: Tensor
  readonly y: Tensor
  /** The source row of each output row, or −1 for a synthetic row. */
  readonly origin: Int32Array
  /** For each synthetic row (in output order): its base row, its neighbour row and the gap u in x = base + u (nb − base). */
  readonly synthetic: { readonly base: Int32Array; readonly neighbour: Int32Array; readonly gap: Float64Array }
}

type Data = { n: number; d: number; x: Float64Array; y: Int32Array; classes: number[]; counts: Map<number, number> }

function prepare(x: MatrixLike, y: VectorLike, where: string): Data {
  const X = dense.toMatrixF64(x, where)
  const labels = dense.toF64(y, where)
  if (labels.length !== X.m) throw new ShapeError(where, `${where}: x and y have different rows`)
  const yi = Int32Array.from(labels)
  const counts = new Map<number, number>()
  for (const c of yi) counts.set(c, (counts.get(c) ?? 0) + 1)
  return { n: X.m, d: X.n, x: X.data, y: yi, classes: [...counts.keys()].sort((a, b) => a - b), counts }
}

const majorityCount = (d: Data) => Math.max(...d.counts.values())
const rowsOfClass = (d: Data, c: number) => Array.from({ length: d.n }, (_, i) => i).filter((i) => d.y[i] === c)

/** Assemble the output: the kept original rows, then the synthetic rows. */
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

const all = (d: Data) => Array.from({ length: d.n }, (_, i) => i)

/** Copies of minority rows drawn uniformly with replacement until every class has the majority's count. */
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

/** Every class cut to the smallest class's count by sampling without replacement. */
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

/** The k nearest neighbours (global row indices) of each of `rows` among `pool`, excluding itself. */
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
  /** Minority neighbours to interpolate towards (default 5). */
  k?: number
}

/** Synthesise `counts[r]` points from each `bases[r]` towards a random one of its minority neighbours. */
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

/** SMOTE: every non-majority class grown to the majority count by interpolating between minority neighbours. */
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

/** Where each row of class c stands among its m nearest neighbours in the whole data. */
export type BorderStatus = 'safe' | 'danger' | 'noise'

/**
 * Borderline-SMOTE's classification of the rows of class `c`: with m′ of a row's m nearest neighbours (in all the data)
 * from other classes, `noise` when m′ = m, `danger` when m/2 ≤ m′ < m, else `safe`.
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

/** Borderline-SMOTE (variant 1): SMOTE from the danger rows of each non-majority class only. */
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
 * ADASYN's allocation for class c: rᵢ, the share of other classes among each row's k nearest neighbours, normalised to
 * sum to 1, and gᵢ = round(rᵢ · G) synthetic points per row with G the shortfall to the majority count.
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

/** ADASYN: synthesise more points from minority rows whose neighbourhoods are dominated by other classes. */
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

/** Tomek links: pairs (i, j), i < j, of different classes that are each other's nearest neighbour. */
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
 * majority member; with more, both members of a link between two non-minority classes.
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

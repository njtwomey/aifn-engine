/**
 * Internal helpers of the low-density separation methods: feature rows given as plain arrays, and their packing into
 * one row-major `Float64Array`.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'

/** Rows of features: one array of $d$ numbers per point, its feature vector $\phivec(\xvec)$. */
export type FeatureRows = readonly ArrayLike<number>[]

/**
 * The rows packed row-major, $n \times d$: row $i$ at entries $id$ to $id + d - 1$. Throws `ShapeError` for a row of
 * another length.
 *
 * @param rows The feature rows; not modified.
 * @param dim The number $d$ of features every row must have.
 * @param where The caller's name, for error messages.
 * @returns A new array of $nd$ values.
 */
export function packRows(rows: FeatureRows, dim: number, where: string): Float64Array {
  const out = new Float64Array(rows.length * dim)
  rows.forEach((r, i) => {
    if (r.length !== dim) throw new ShapeError(where, `${where}: row ${i} has ${r.length} features, expected ${dim}`)
    for (let c = 0; c < dim; c++) out[i * dim + c] = r[c]
  })
  return out
}

/**
 * Column-wise transforms of a table into one feature matrix, after scikit-learn's `ColumnTransformer` (Buitinck et al.,
 * 2013, "API design for machine learning software: experiences from the scikit-learn project").
 */

import { child } from 'aifn-compute/foundation/random'
import type { Column, Dataset, Estimator, FitOptions, Table } from 'aifn-compute/learning/estimators'
import type { Transforms } from 'aifn-compute/learning/estimators'
import { fromData, isTensor, reshape, type Tensor } from 'aifn-compute/foundation/tensor'
import type { FittedOf } from './pipeline'
import { DomainError } from 'aifn-compute/foundation/errors'

/** What to do with a named column: a transformer, pass it through (numeric columns only), or drop it. */
// oxlint-disable-next-line no-explicit-any -- each column's transformer has its own input type
export type ColumnSpec =
  { readonly name: string; fit(data: any, options?: FitOptions): Transforms<any, Tensor> } | 'passthrough' | 'drop'

/** The fitted column transform. */
export interface ColumnsModel<C extends Record<string, ColumnSpec>> extends Transforms<Table, Tensor> {
  readonly kind: 'model'
  /** Which composition made the model. */
  readonly composition: 'columns'
  /** Each transformed column's fitted transform, by name. */
  readonly steps: { readonly [K in keyof C as C[K] extends 'drop' | 'passthrough' ? never : K]: FittedOf<C[K]> }
  /** The output columns [start, end) of each input column, by name. */
  readonly slices: Readonly<Record<string, readonly [number, number]>>
  /** A name per output column, "column:feature". */
  readonly featureNames: readonly string[]
}

/** Numeric columns become [n, k] matrices; label lists are passed as they are. */
function asInput(c: Column): Column {
  if (isTensor(c) && c.shape.length === 1) return reshape(c, [c.shape[0], 1])
  return c
}

function numeric(name: string, c: Column): Tensor {
  const t = asInput(c)
  if (!isTensor(t)) throw new DomainError('columns', `columns: "${name}" is not numeric and cannot pass through`)
  return t
}

/** Row-major copy of each block side by side: [n, Σ kᵢ]. */
function concatColumns(blocks: Tensor[], n: number): Tensor {
  const widths = blocks.map((b) => b.shape[1])
  const width = widths.reduce((a, b) => a + b, 0)
  const out = new Float64Array(n * width)
  let offset = 0
  blocks.forEach((b, k) => {
    const w = widths[k]
    // Read through the tensor's strides: a block may be a view.
    const [s0, s1] = b.strides
    for (let i = 0; i < n; i++)
      for (let j = 0; j < w; j++) out[i * width + offset + j] = b.data[b.offset + i * s0 + j * s1]
    offset += w
  })
  return fromData(out, [n, width])
}

/**
 * Apply a transformer to each named column of a table and place the outputs side by side, in the order of `spec`'s
 * keys, then (with `remainder: 'passthrough'`) the remaining numeric columns. Numeric columns of shape [n] are given
 * to transformers as [n, 1]; label lists are given as they are (for encoders). Targets pass to every transformer's
 * fit, so target encoding works per column.
 *
 * @example columns({ age: standardScaler(), city: oneHotEncoder() })
 */
export function columns<const C extends Record<string, ColumnSpec>>(
  spec: C,
  { remainder = 'drop' }: { remainder?: 'drop' | 'passthrough' } = {},
): Estimator<Dataset<Table, Tensor>, ColumnsModel<C>> {
  const names = Object.keys(spec)
  return {
    name: `columns(${names.join(', ')})`,
    params: {
      spec: Object.fromEntries(
        names.map((k) => [k, typeof spec[k] === 'string' ? spec[k] : (spec[k] as { name: string }).name]),
      ),
      remainder,
    },
    fit(data, options = {}) {
      const table = data.x
      for (const name of names) if (!(name in table)) throw new DomainError('columns', `columns: no column "${name}"`)
      const rest = remainder === 'passthrough' ? Object.keys(table).filter((k) => !(k in spec)) : []
      const steps: Record<string, Transforms<unknown, Tensor>> = {}
      names.forEach((name, k) => {
        const s = spec[name]
        if (typeof s === 'string') return
        steps[name] = s.fit(
          { ...data, x: asInput(table[name]) },
          { ...options, stream: options.stream && child(options.stream, 'column', k) },
        )
      })
      const blocksOf = (input: Table): [string, Tensor][] => {
        const out: [string, Tensor][] = []
        for (const name of names) {
          const s = spec[name]
          if (s === 'drop') continue
          if (!(name in input)) throw new DomainError('columns', `columns: no column "${name}"`)
          out.push([
            name,
            s === 'passthrough' ? numeric(name, input[name]) : steps[name].transform(asInput(input[name])),
          ])
        }
        for (const name of rest) out.push([name, numeric(name, input[name])])
        return out
      }
      const first = blocksOf(table)
      const slices: Record<string, [number, number]> = {}
      const featureNames: string[] = []
      let offset = 0
      for (const [name, block] of first) {
        const w = block.shape[1]
        slices[name] = [offset, offset + w]
        const inner = (steps[name] as { featureNames?: readonly string[] } | undefined)?.featureNames
        for (let j = 0; j < w; j++)
          featureNames.push(inner?.[j] !== undefined ? `${name}:${inner[j]}` : w === 1 ? name : `${name}:${j}`)
        offset += w
      }
      return {
        kind: 'model',
        composition: 'columns',
        steps: steps as ColumnsModel<C>['steps'],
        slices,
        featureNames,
        transform(input: Table) {
          const blocks = blocksOf(input)
          const n = blocks[0]?.[1].shape[0] ?? 0
          return concatColumns(
            blocks.map(([, b]) => b),
            n,
          )
        },
      }
    },
  }
}

/**
 * Discrete factor graphs as plain data, and the factor algebra every discrete engine builds on: products,
 * marginalisation by sum or max, conditioning on evidence and normalisation (Koller & Friedman 2009, "Probabilistic
 * Graphical Models", §4.2 and §9.3; Kschischang, Frey & Loeliger 2001, "Factor graphs and the sum-product algorithm").
 *
 * A factor's table is a float64 tensor whose axes follow its scope: `table[a₀, a₁, …]` is the potential at
 * `x[scope[0]] = a₀, x[scope[1]] = a₁, …`. Potentials are non-negative (not logs).
 */

import { shape, structuredGraph, type StructuredGraph } from 'aifn-compute/graph/structured'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A potential over a few discrete variables; `table` has shape `scope.map((v) => cardinalities[v])`. */
export interface DiscreteFactor {
  scope: readonly number[]
  table: Tensor
  /** For display, e.g. `ψ₁₂`. */
  name?: string
}

/**
 * A discrete factor graph: variables 0 … V − 1 with the given cardinalities, and factors over them. The joint is
 * p(x) = (1/Z) Πₐ fₐ(x_{scope(a)}).
 */
export interface DiscreteFactorGraph {
  cardinalities: readonly number[]
  factors: readonly DiscreteFactor[]
  /** Variable names for display (default `x0`, `x1`, …). */
  names?: readonly string[]
}

/** One edge of a factor graph: between variable `variable` and factor `factor`, at `position` in its scope. */
export interface FactorGraphEdge {
  variable: number
  factor: number
  position: number
}

/** Row-major strides of a table with these axis sizes. */
export function stridesOf(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length)
  let s = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = s
    s *= shape[i]
  }
  return strides
}

/** The number of entries of a table with these axis sizes. */
export const tableSize = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

/** Contiguous float64 values of a table (a copy when the tensor is a strided view). */
export const valuesOf = (t: Tensor): Float64Array =>
  t.data instanceof Float64Array && t.offset === 0 && t.data.length === tableSize(t.shape)
    ? t.data
    : Float64Array.from(toFlat(t))

/** Visit every assignment of variables with these cardinalities in row-major order. */
export function forEachAssignment(shape: readonly number[], fn: (assignment: Int32Array, flat: number) => void): void {
  const n = tableSize(shape)
  const a = new Int32Array(shape.length)
  for (let flat = 0; flat < n; flat++) {
    fn(a, flat)
    for (let i = shape.length - 1; i >= 0; i--) {
      if (++a[i] < shape[i]) break
      a[i] = 0
    }
  }
}

/**
 * Build a factor from a scope, the graph's cardinalities and its values (a tensor, a flat array in row-major order, or
 * a function of the assignment). Throws on a negative or NaN potential.
 */
export function discreteFactor(
  scope: readonly number[],
  cardinalities: readonly number[],
  values: Tensor | ArrayLike<number> | ((assignment: Int32Array) => number),
  name?: string,
): DiscreteFactor {
  const shape = scope.map((v) => cardinalities[v])
  const n = tableSize(shape)
  let data: Float64Array
  if (typeof values === 'function') {
    data = new Float64Array(n)
    forEachAssignment(shape, (a, flat) => (data[flat] = values(a)))
  } else {
    data = 'shape' in values ? Float64Array.from(toFlat(values)) : Float64Array.from(values)
  }
  if (data.length !== n)
    throw new ShapeError('discreteFactor', `discreteFactor: ${data.length} values for a table of ${n}`)
  for (const v of data)
    if (!(v >= 0)) throw new DomainError('discreteFactor', `discreteFactor: potential ${v} is not non-negative`)
  return { scope: [...scope], table: fromData(data, shape), ...(name === undefined ? {} : { name }) }
}

/**
 * A gate (Minka and Winn, 2008): a selector variable c with K values switches between K factors, so the gated factor is
 * φ(c = k, x) = fₖ(x) over c and the union of the cases' scopes (a case's potential is constant along variables it does
 * not touch). A gate makes mixtures, model selection and context-specific independence explicit in the factor graph,
 * and message passing through it is the usual sum–product. With `selector` given as an evidence-free variable, the
 * marginal of c is the posterior probability of each case.
 */
export function gateFactor(
  selector: number,
  cases: readonly DiscreteFactor[],
  cardinalities: readonly number[],
  name?: string,
): DiscreteFactor {
  if (cases.length !== cardinalities[selector])
    throw new ShapeError(
      'gateFactor',
      `gateFactor: ${cases.length} cases for a selector with ${cardinalities[selector]} values`,
    )
  const others = [...new Set(cases.flatMap((f) => f.scope))].sort((a, b) => a - b)
  if (others.includes(selector))
    throw new DomainError('gateFactor', 'gateFactor: a case may not depend on its own selector')
  const scope = [selector, ...others]
  const tables = cases.map((f) => ({ f, data: valuesOf(f.table), strides: stridesOf(f.table.shape) }))
  return discreteFactor(
    scope,
    cardinalities,
    (a) => {
      const { f, data, strides } = tables[a[0]]
      let flat = 0
      f.scope.forEach((v, i) => (flat += a[1 + others.indexOf(v)] * strides[i]))
      return data[flat]
    },
    name ?? 'gate',
  )
}

/** Check a factor graph's scopes and table shapes; returns it unchanged. */
export function discreteFactorGraph(
  cardinalities: readonly number[],
  factors: readonly DiscreteFactor[],
  names?: readonly string[],
): DiscreteFactorGraph {
  factors.forEach((f, k) => {
    f.scope.forEach((v, i) => {
      if (!(v >= 0 && v < cardinalities.length))
        throw new DomainError(`factor ${k}`, `factor ${k}: variable ${v} is out of range`)
      if (f.table.shape[i] !== cardinalities[v])
        throw new ShapeError(
          `factor ${k}`,
          `factor ${k}: axis ${i} has size ${f.table.shape[i]}, variable ${v} has ${cardinalities[v]}`,
        )
    })
    if (new Set(f.scope).size !== f.scope.length)
      throw new DomainError(`factor ${k}`, `factor ${k}: repeated variable in scope`)
  })
  return { cardinalities: [...cardinalities], factors, ...(names ? { names: [...names] } : {}) }
}

/** The display name of variable v. */
export const variableName = (g: DiscreteFactorGraph, v: number): string => g.names?.[v] ?? `x${v}`

/** Every (variable, factor) edge, grouped by factor in scope order. */
export function factorGraphEdges(g: DiscreteFactorGraph): FactorGraphEdge[] {
  return g.factors.flatMap((f, factor) => f.scope.map((variable, position) => ({ variable, factor, position })))
}

/**
 * The factor graph as a structured graph: nodes 0 … V − 1 are the variables (role `latent`, named `x<v>`, labelled
 * with their display names) and V … V + F − 1 the factors (role `factor`, named `f<k>`, labelled with their names);
 * edge k, undirected, is `factorGraphEdges(g)[k]`.
 */
export function bipartiteGraph(g: DiscreteFactorGraph): StructuredGraph<{ variable?: number; factor?: number }> {
  return structuredGraph<{ variable?: number; factor?: number }>({
    nodes: [
      ...g.cardinalities.map((_, v) => ({
        name: `x${v}`,
        role: 'latent' as const,
        group: null,
        label: variableName(g, v),
        data: { variable: v },
      })),
      ...g.factors.map((f, k) => ({
        name: `f${k}`,
        role: 'factor' as const,
        group: null,
        label: f.name ?? `f_{${k}}`,
        data: { factor: k },
      })),
    ],
    edges: factorGraphEdges(g).map((e) => ({ from: `x${e.variable}`, to: `f${e.factor}`, directed: false })),
  })
}

/** True when the factor graph has no cycles (it is a tree or a forest), so sum-product is exact on it. */
export function isTree(g: DiscreteFactorGraph): boolean {
  const s = shape(bipartiteGraph(g))
  return s === 'chain' || s === 'tree'
}

/** The product of two factors, over the union of their scopes (a's variables first). */
export function factorProduct(a: DiscreteFactor, b: DiscreteFactor, cardinalities: readonly number[]): DiscreteFactor {
  const scope = [...a.scope, ...b.scope.filter((v) => !a.scope.includes(v))]
  const shape = scope.map((v) => cardinalities[v])
  const av = valuesOf(a.table)
  const bv = valuesOf(b.table)
  const as = stridesOf(a.table.shape)
  const bs = stridesOf(b.table.shape)
  const aAxis = scope.map((v) => a.scope.indexOf(v))
  const bAxis = scope.map((v) => b.scope.indexOf(v))
  const out = new Float64Array(tableSize(shape))
  forEachAssignment(shape, (x, flat) => {
    let ia = 0
    let ib = 0
    for (let i = 0; i < scope.length; i++) {
      if (aAxis[i] >= 0) ia += x[i] * as[aAxis[i]]
      if (bAxis[i] >= 0) ib += x[i] * bs[bAxis[i]]
    }
    out[flat] = av[ia] * bv[ib]
  })
  return { scope, table: fromData(out, shape) }
}

/** The product of several factors; an empty list gives the constant factor 1. */
export function factorProductAll(factors: readonly DiscreteFactor[], cardinalities: readonly number[]): DiscreteFactor {
  let out: DiscreteFactor = { scope: [], table: fromData(new Float64Array([1]), []) }
  for (const f of factors) out = factorProduct(out, f, cardinalities)
  return out
}

/**
 * Sum (or maximise) the given variables out of a factor. With `mode: 'max'` the result is a max-marginal
 * (max-product).
 */
export function factorMarginalise(
  f: DiscreteFactor,
  variables: readonly number[],
  mode: 'sum' | 'max' = 'sum',
): DiscreteFactor {
  const keep = f.scope.map((v, i) => (variables.includes(v) ? -1 : i)).filter((i) => i >= 0)
  const scope = keep.map((i) => f.scope[i])
  const shape = keep.map((i) => f.table.shape[i])
  const ks = stridesOf(shape)
  const out = new Float64Array(tableSize(shape)).fill(mode === 'sum' ? 0 : -Infinity)
  const values = valuesOf(f.table)
  forEachAssignment(f.table.shape, (x, flat) => {
    let j = 0
    for (let k = 0; k < keep.length; k++) j += x[keep[k]] * ks[k]
    out[j] = mode === 'sum' ? out[j] + values[flat] : Math.max(out[j], values[flat])
  })
  return { scope, table: fromData(out, shape) }
}

/** Condition a factor on evidence (variable → observed value): the observed variables leave its scope. */
export function factorReduce(f: DiscreteFactor, evidence: ReadonlyMap<number, number>): DiscreteFactor {
  const keep = f.scope.map((v, i) => (evidence.has(v) ? -1 : i)).filter((i) => i >= 0)
  if (keep.length === f.scope.length) return f
  const strides = stridesOf(f.table.shape)
  let base = 0
  f.scope.forEach((v, i) => {
    if (evidence.has(v)) base += evidence.get(v)! * strides[i]
  })
  const shape = keep.map((i) => f.table.shape[i])
  const values = valuesOf(f.table)
  const out = new Float64Array(tableSize(shape))
  forEachAssignment(shape, (x, flat) => {
    let j = base
    for (let k = 0; k < keep.length; k++) j += x[k] * strides[keep[k]]
    out[flat] = values[j]
  })
  return { scope: keep.map((i) => f.scope[i]), table: fromData(out, shape), ...(f.name ? { name: f.name } : {}) }
}

/** A factor scaled to sum to one, with the log of the sum it had (−∞ for an all-zero factor, which stays zero). */
export function normaliseFactor(f: DiscreteFactor): { factor: DiscreteFactor; logNormaliser: number } {
  const v = valuesOf(f.table)
  let z = 0
  for (const x of v) z += x
  if (!(z > 0)) return { factor: f, logNormaliser: -Infinity }
  return {
    factor: {
      ...f,
      table: fromData(
        v.map((x) => x / z),
        f.table.shape,
      ),
    },
    logNormaliser: Math.log(z),
  }
}

/** The variables sharing a factor with v (its Markov blanket in the factor graph), ascending. */
export function factorGraphNeighbours(g: DiscreteFactorGraph, v: number): number[] {
  const out = new Set<number>()
  for (const f of g.factors) if (f.scope.includes(v)) for (const u of f.scope) if (u !== v) out.add(u)
  return [...out].sort((a, b) => a - b)
}

/** The unnormalised log-probability Σₐ log fₐ(x) of a full assignment. */
export function logPotential(g: DiscreteFactorGraph, assignment: ArrayLike<number>): number {
  let total = 0
  for (const f of g.factors) {
    const strides = stridesOf(f.table.shape)
    let j = 0
    f.scope.forEach((v, i) => (j += assignment[v] * strides[i]))
    total += Math.log(valuesOf(f.table)[j])
  }
  return total
}

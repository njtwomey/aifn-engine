/**
 * Discrete factor graphs as plain data, and the factor algebra every discrete engine builds on: products,
 * marginalisation by sum or max, conditioning on evidence and normalisation (Koller & Friedman 2009, "Probabilistic
 * Graphical Models", §4.2 and §9.3; Kschischang, Frey & Loeliger 2001, "Factor graphs and the sum-product algorithm").
 *
 * Variables are numbered $0, \dots, V - 1$, and variable $v$ takes the values $0, \dots, K_v - 1$, with $K_v$ its
 * entry of the graph's `cardinalities`. A factor's table is a float64 tensor whose axes follow its scope: its entry at
 * $(a_0, a_1, \dots)$ is the potential at $x_{s_0} = a_0, x_{s_1} = a_1, \dots$, where $s_i$ is `scope[i]`. Potentials
 * are non-negative (not logs), and tables are read in row-major order.
 */

import { shape, structuredGraph, type StructuredGraph } from 'aifn-compute/graph/structured'
import { fromData, isContiguous, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A potential over a few discrete variables; `table` has shape `scope.map((v) => cardinalities[v])`. */
export interface DiscreteFactor {
  /** The variables the factor depends on, in the order of the table's axes, without repeats. */
  scope: readonly number[]
  /** The non-negative potentials, one axis per variable of `scope`. */
  table: Tensor
  /** A name for display, such as `p(x)` or `\psi_{12}`; `bipartiteGraph` uses it as the factor's label. */
  name?: string
}

/**
 * A discrete factor graph: variables $0, \dots, V - 1$ with the given cardinalities, and factors over them. The joint
 * is $p(\xvec) = \frac{1}{Z} \prod_a f_a(\xvec_{\text{scope}(a)})$.
 */
export interface DiscreteFactorGraph {
  /** The number of values of each variable, $K_v$ for variable $v$. */
  cardinalities: readonly number[]
  /** The factors, each over variables of the graph. */
  factors: readonly DiscreteFactor[]
  /** Variable names for display (default `x0`, `x1`, ...). */
  names?: readonly string[]
}

/** One edge of a factor graph: between variable `variable` and factor `factor`, at `position` in its scope. */
export interface FactorGraphEdge {
  /** The variable's index in the graph. */
  variable: number
  /** The factor's index in the graph's `factors`. */
  factor: number
  /** Where the variable sits in the factor's `scope`, which is the axis of its table. */
  position: number
}

/**
 * Row-major strides of a table with these axis sizes: the step in the flat index for one step along each axis.
 *
 * @param shape The size of each axis of the table.
 * @returns One stride per axis; the last is 1, and each other is the product of the sizes after it.
 *
 * @example The strides of a $2 \times 3 \times 4$ table
 * print('strides:', stridesOf([2, 3, 4]))
 */
export function stridesOf(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length)
  let s = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = s
    s *= shape[i]
  }
  return strides
}

/**
 * The number of entries of a table with these axis sizes.
 *
 * @param shape The size of each axis of the table.
 * @returns The product of the sizes: 1 for no axes (a constant factor).
 *
 * @example A $2 \times 3 \times 4$ table, and a table with no axes
 * print('2 × 3 × 4:', tableSize([2, 3, 4]))
 * print('no axes:', tableSize([]))
 */
export const tableSize = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

/**
 * Contiguous float64 values of a table, in row-major order. The tensor's own storage is returned (so it must not be
 * modified) when it is float64, row-major (`isContiguous`), starts at offset 0 and holds exactly one value per entry.
 * Otherwise, as for a transposed or strided view, the entries are copied out in row-major order.
 *
 * @param t The table: a tensor of any shape.
 * @returns Its entries in row-major order.
 *
 * @example The entries of a $2 \times 2$ table
 * print(valuesOf(tensor([[1, 2], [3, 4]])))
 */
export const valuesOf = (t: Tensor): Float64Array =>
  t.data instanceof Float64Array && t.offset === 0 && t.data.length === tableSize(t.shape) && isContiguous(t)
    ? t.data
    : Float64Array.from(toFlat(t))

/**
 * Visit every assignment of variables with these cardinalities in row-major order (the last variable changes
 * fastest).
 *
 * @param shape The number of values of each variable, as the axis sizes of the table being visited.
 * @param fn Called once per assignment with the assignment (one value per variable) and its flat row-major index. The
 *   assignment array is reused and changed after each call: copy it to keep it.
 *
 * @example The six assignments of two variables with 2 and 3 values
 * const seen = []
 * forEachAssignment([2, 3], (a, flat) => seen.push(`${flat}: (${a[0]}, ${a[1]})`))
 * print(seen)
 */
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
 * a function of the assignment). Throws `ShapeError` when the number of values does not match the table, and
 * `DomainError` on a negative or NaN potential.
 *
 * @param scope The variables the factor is over, in the order of the table's axes.
 * @param cardinalities The number of values of every variable of the graph, indexed by variable (not by position in
 *   `scope`).
 * @param values The potentials: a tensor or array whose entries, read in row-major order, fill the table, or a function
 *   called with each assignment of the scope's variables (in `scope` order) that returns its potential. The values
 *   are copied.
 * @param name A name for display; left out, the factor has none.
 * @returns The factor, its table of shape `scope.map((v) => cardinalities[v])`.
 *
 * @example A conditional probability table $p(x_1 \mid x_0)$
 * const f = discreteFactor([0, 1], [2, 2], [0.9, 0.1, 0.2, 0.8], 'p(x1 | x0)')
 * print('scope:', f.scope)
 * print('table:', f.table)
 *
 * @example Potentials from a function of the assignment
 * // An agreement potential over three values: 2 when the variables are equal, 1 otherwise.
 * const f = discreteFactor([0, 1], [3, 3], (a) => (a[0] === a[1] ? 2 : 1))
 * print('table:', f.table)
 *
 * @example A negative potential throws
 * try {
 *   discreteFactor([0], [2], [0.5, -0.5])
 * } catch (e) {
 *   print(e.message)
 * }
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
 * A gate (Minka and Winn, 2008, "Gates"): a selector variable $c$ with $K$ values switches between $K$ factors, so the
 * gated factor is $\phi(c = k, \xvec) = f_k(\xvec)$ over $c$ and the union of the cases' scopes (a case's potential is
 * constant along variables it does not touch). A gate makes mixtures, model selection and context-specific
 * independence explicit in the factor graph, and message passing through it is the usual sum-product. With `selector`
 * given as an evidence-free variable, the marginal of $c$ is the posterior probability of each case. Throws
 * `ShapeError` when the number of cases is not the selector's number of values, and `DomainError` when a case depends
 * on the selector.
 *
 * @param selector The variable $c$ that picks the case: case $k$ applies when $c = k$.
 * @param cases The factors $f_0, \dots, f_{K-1}$, one per value of the selector, none of them over the selector.
 * @param cardinalities The number of values of every variable of the graph, indexed by variable.
 * @param name A name for display (default `gate`).
 * @returns The gated factor, over the selector first and then the cases' variables in ascending order.
 *
 * @example Two cases of $p(x \mid c)$, and the posterior of the case given $x = 1$
 * const cards = [2, 2] // variable 0 is the selector c, variable 1 is x
 * const f0 = discreteFactor([1], cards, [0.9, 0.1])
 * const f1 = discreteFactor([1], cards, [0.2, 0.8])
 * const gate = gateFactor(0, [f0, f1], cards)
 * print('scope:', gate.scope)
 * print('table:', gate.table)
 * const prior = discreteFactor([0], cards, [0.5, 0.5])
 * const joint = factorReduce(factorProduct(prior, gate, cards), new Map([[1, 1]]))
 * print('p(c | x = 1):', normaliseFactor(joint).factor.table)
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

/**
 * A discrete factor graph from its cardinalities, factors and names, after checking every factor against the
 * cardinalities. Throws `DomainError` when a factor names a variable out of range or repeats one in its scope, and
 * `ShapeError` when a table's axis does not have its variable's number of values.
 *
 * @param cardinalities The number of values of each variable; its length is the number of variables. Copied.
 * @param factors The factors, each over variables $0, \dots, V - 1$. Kept as given, not copied.
 * @param names Display names, one per variable (copied); left out, variables are shown as `x0`, `x1`, ...
 * @returns The graph.
 *
 * @example Two binary variables, $p(\text{rain})\,p(\text{wet} \mid \text{rain})$
 * const cards = [2, 2]
 * const g = discreteFactorGraph(
 *   cards,
 *   [discreteFactor([0], cards, [0.6, 0.4]), discreteFactor([0, 1], cards, [0.9, 0.1, 0.2, 0.8])],
 *   ['rain', 'wet'],
 * )
 * print('cardinalities:', g.cardinalities)
 * print('scopes:', g.factors.map((f) => f.scope))
 *
 * @example A table that does not match the cardinalities throws
 * try {
 *   discreteFactorGraph([2, 3], [discreteFactor([0, 1], [2, 2], [1, 1, 1, 1])])
 * } catch (e) {
 *   print(e.message)
 * }
 */
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

/**
 * The display name of variable $v$: its entry of the graph's `names`, or `x<v>` when it has none.
 *
 * @param g The factor graph.
 * @param v The variable's index.
 * @returns The name.
 *
 * @example Named and unnamed variables
 * const named = discreteFactorGraph([2, 2], [], ['rain', 'wet'])
 * const plain = discreteFactorGraph([2, 2], [])
 * print('named:', variableName(named, 1))
 * print('plain:', variableName(plain, 1))
 */
export const variableName = (g: DiscreteFactorGraph, v: number): string => g.names?.[v] ?? `x${v}`

/**
 * Every (variable, factor) edge, grouped by factor in scope order.
 *
 * @param g The factor graph.
 * @returns One edge per variable of each factor's scope: factor 0's first, in its scope's order, then factor 1's, and
 *   so on.
 *
 * @example The edges of a chain $x_0 - f_0 - x_1 - f_1 - x_2$
 * const c = [2, 2, 2]
 * const g = discreteFactorGraph(c, [discreteFactor([0, 1], c, [1, 2, 2, 1]), discreteFactor([1, 2], c, [1, 2, 2, 1])])
 * print(factorGraphEdges(g).map((e) => `x${e.variable} - f${e.factor} (position ${e.position})`))
 */
export function factorGraphEdges(g: DiscreteFactorGraph): FactorGraphEdge[] {
  return g.factors.flatMap((f, factor) => f.scope.map((variable, position) => ({ variable, factor, position })))
}

/**
 * The factor graph as a structured graph: nodes $0, \dots, V - 1$ are the variables (role `latent`, named `x<v>`,
 * labelled with their display names) and $V, \dots, V + F - 1$ the factors (role `factor`, named `f<k>`, labelled with
 * their names, or $f_k$ for a factor with none); edge $k$, undirected, is `factorGraphEdges(g)[k]`. No node is in a
 * group.
 *
 * @param g The factor graph, with $V$ variables and $F$ factors.
 * @returns The bipartite structured graph, whose node data holds the `variable` or `factor` index it stands for.
 *
 * @example Two variables joined by one factor
 * const c = [2, 2]
 * const g = discreteFactorGraph(c, [discreteFactor([0, 1], c, [1, 2, 2, 1], 'psi')], ['a', 'b'])
 * const b = bipartiteGraph(g)
 * print('nodes:', b.attributes.map((n) => `${n.name} (${n.role}, ${n.label})`))
 * print('edges:', b.edges.map((e) => `${e.from} - ${e.to}`))
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

/**
 * True when the factor graph has no cycles (it is a tree or a forest), so sum-product is exact on it. The test is on
 * the bipartite graph of variables and factors (`bipartiteGraph`), so two factors sharing two variables make a cycle.
 *
 * @param g The factor graph.
 * @returns Whether the variable-factor graph is acyclic.
 *
 * @example A chain is a tree; a triangle of pairwise factors is not
 * const c = [2, 2, 2]
 * const pair = (u, v) => discreteFactor([u, v], c, [2, 1, 1, 2])
 * print('chain:', isTree(discreteFactorGraph(c, [pair(0, 1), pair(1, 2)])))
 * print('triangle:', isTree(discreteFactorGraph(c, [pair(0, 1), pair(1, 2), pair(0, 2)])))
 */
export function isTree(g: DiscreteFactorGraph): boolean {
  const s = shape(bipartiteGraph(g))
  return s === 'chain' || s === 'tree'
}

/**
 * The product of two factors, over the union of their scopes (a's variables first): its potential at an assignment is
 * the product of the two factors' potentials there. The result has no name.
 *
 * @param a The first factor; its variables come first in the result's scope.
 * @param b The second factor; its variables not in `a` follow, in its own order.
 * @param cardinalities The number of values of every variable of the graph, indexed by variable.
 * @returns The product factor.
 *
 * @example The joint $p(x_0, x_1) = p(x_0)\,p(x_1 \mid x_0)$
 * const cards = [2, 2]
 * const prior = discreteFactor([0], cards, [0.6, 0.4])
 * const likelihood = discreteFactor([0, 1], cards, [0.9, 0.1, 0.2, 0.8])
 * const joint = factorProduct(prior, likelihood, cards)
 * print('scope:', joint.scope)
 * print('table:', joint.table)
 */
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

/**
 * The product of several factors, multiplied in order with `factorProduct`; an empty list gives the constant factor 1
 * (empty scope).
 *
 * @param factors The factors to multiply; the result's scope lists their variables in order of first appearance.
 * @param cardinalities The number of values of every variable of the graph, indexed by variable.
 * @returns The product factor.
 *
 * @example Three factors of a chain, and the empty product
 * // x0 is a fair coin, and x1 and x2 copy it.
 * const c = [2, 2, 2]
 * const copy = (u, v) => discreteFactor([u, v], c, [1, 0, 0, 1])
 * const all = factorProductAll([discreteFactor([0], c, [0.5, 0.5]), copy(0, 1), copy(1, 2)], c)
 * print('scope:', all.scope)
 * print('table:', all.table)
 * print('empty product:', factorProductAll([], c).table)
 */
export function factorProductAll(factors: readonly DiscreteFactor[], cardinalities: readonly number[]): DiscreteFactor {
  let out: DiscreteFactor = { scope: [], table: fromData(new Float64Array([1]), []) }
  for (const f of factors) out = factorProduct(out, f, cardinalities)
  return out
}

/**
 * Sum (or maximise) the given variables out of a factor. With `mode: 'max'` the result is a max-marginal
 * (max-product). The result has no name.
 *
 * @param f The factor.
 * @param variables The variables to remove; any not in the factor's scope are ignored.
 * @param mode `'sum'` adds the potentials over the removed variables' values, `'max'` keeps the largest.
 * @returns A factor over the remaining variables, in their order in `f`.
 *
 * @example The marginal and the max-marginal of $x_1$
 * const cards = [2, 2]
 * const joint = discreteFactor([0, 1], cards, [0.54, 0.06, 0.08, 0.32])
 * print('sum over x0:', factorMarginalise(joint, [0]).table)
 * print('max over x0:', factorMarginalise(joint, [0], 'max').table)
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

/**
 * Condition a factor on evidence: the observed variables leave its scope, and the table keeps the slice at their
 * observed values. Not normalised (see `normaliseFactor`). The factor keeps its name.
 *
 * @param f The factor.
 * @param evidence The observed value of each observed variable, keyed by variable; variables not in the factor's scope
 *   are ignored. Values are not range-checked.
 * @returns The reduced factor, or `f` itself when no observed variable is in its scope.
 *
 * @example Observe $x_1 = 1$ in $p(x_0, x_1)$, then normalise for $p(x_0 \mid x_1 = 1)$
 * const joint = discreteFactor([0, 1], [2, 2], [0.54, 0.06, 0.08, 0.32])
 * const reduced = factorReduce(joint, new Map([[1, 1]]))
 * print('scope:', reduced.scope)
 * print('p(x0, x1 = 1):', reduced.table)
 * print('p(x0 | x1 = 1):', normaliseFactor(reduced).factor.table)
 */
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

/**
 * A factor scaled to sum to one, with the log of the sum it had ($-\infty$ for an all-zero factor, which is returned
 * as it is).
 *
 * @param f The factor; not modified.
 * @returns `factor`, the scaled factor (with `f`'s scope and name), and `logNormaliser`, $\log Z$ for $Z$ the sum of
 *   `f`'s potentials.
 *
 * @example Scale to sum to one, and an all-zero factor
 * const { factor, logNormaliser } = normaliseFactor(discreteFactor([0], [2], [1, 3]))
 * print('factor:', factor.table)
 * print('log Z:', logNormaliser, '= log 4:', Math.log(4))
 * print('all zero:', normaliseFactor(discreteFactor([0], [2], [0, 0])).logNormaliser)
 */
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

/**
 * The variables sharing a factor with $v$ (its Markov blanket in the factor graph), ascending.
 *
 * @param g The factor graph.
 * @param v The variable's index.
 * @returns The other variables of every factor whose scope holds $v$, without repeats.
 *
 * @example The middle of a chain $x_0 - x_1 - x_2$ and its end
 * const c = [2, 2, 2]
 * const g = discreteFactorGraph(c, [discreteFactor([0, 1], c, [1, 2, 2, 1]), discreteFactor([1, 2], c, [1, 2, 2, 1])])
 * print('neighbours of x1:', factorGraphNeighbours(g, 1))
 * print('neighbours of x0:', factorGraphNeighbours(g, 0))
 */
export function factorGraphNeighbours(g: DiscreteFactorGraph, v: number): number[] {
  const out = new Set<number>()
  for (const f of g.factors) if (f.scope.includes(v)) for (const u of f.scope) if (u !== v) out.add(u)
  return [...out].sort((a, b) => a - b)
}

/**
 * The unnormalised log-probability $\sum_a \log f_a(\xvec_{\text{scope}(a)})$ of a full assignment: $-\infty$ when a
 * factor is zero there. Values are not range-checked.
 *
 * @param g The factor graph.
 * @param assignment A value for every variable, indexed by variable.
 * @returns The sum of the logs of the factors' potentials at the assignment.
 *
 * @example $\log p(\text{rain} = 0, \text{wet} = 1) = \log(0.6 \times 0.1)$
 * const cards = [2, 2]
 * const g = discreteFactorGraph(cards, [
 *   discreteFactor([0], cards, [0.6, 0.4]),
 *   discreteFactor([0, 1], cards, [0.9, 0.1, 0.2, 0.8]),
 * ])
 * print('log potential:', logPotential(g, [0, 1]))
 * print('log 0.06:', Math.log(0.06))
 */
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

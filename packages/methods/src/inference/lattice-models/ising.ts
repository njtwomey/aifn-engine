/**
 * The Ising model (Ising, 1925; Koller and Friedman, 2009, "Probabilistic Graphical Models", §4.4.1) as a discrete
 * factor graph: pairwise spins on any graph, or on a lattice declared with `aifn-compute/graph/structured`'s
 * `latticeTemplate`, and marginal inference by the engine the graph's shape allows.
 *
 * Spins are $x_i \in \{-1, +1\}$, stored as the values 0 ($-1$) and 1 ($+1$) of a binary variable, and the joint is
 * $p(\xvec) \propto \exp\paren{\sum_{(i, j)} J_{ij} x_i x_j + \sum_i h_i x_i}$: one factor $e^{h_i x_i}$ per site
 * and one factor $e^{J_{ij} x_i x_j}$ per edge, so the energy of a configuration is minus its log potential. A model
 * built on a lattice keeps the declaration, so that `isingShape` reads `lattice` from it rather than `general` from the
 * factors.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { type Graph } from 'aifn-compute/graph'
import { latticeTemplate, shape, unroll, type GraphShape, type StructuredGraph } from 'aifn-compute/graph/structured'
import { chainSumProduct } from 'aifn-compute/inference/exact'
import { beliefPropagationSteps, type BeliefPropagationOptions } from 'aifn-compute/inference/message-passing'
import {
  bipartiteGraph,
  type DiscreteFactorGraph,
  type DiscreteFactor,
  discreteFactor,
  discreteFactorGraph,
} from 'aifn-compute/inference/model'

/**
 * An Ising model: a discrete factor graph that keeps the structured graph it was declared on, when there is one, so
 * that its shape is read from the declaration (a lattice template) rather than recovered from the factors.
 */
export interface IsingModel extends DiscreteFactorGraph {
  /** The structured interaction graph (`latticeTemplate`, unrolled), or absent for a plain graph. */
  readonly structure?: StructuredGraph
}

/**
 * A pairwise Ising model on a graph as a factor graph: spins $x_i \in \{-1, +1\}$ (value 0 is $-1$, value 1 is $+1$),
 * with $p(\xvec) \propto \exp\paren{\sum_{(i, j)} J_{ij} x_i x_j + \sum_i h_i x_i}$. The factors are one unary
 * $e^{h_i x_i}$ per node (in node order), then one pairwise $e^{J_{ij} x_i x_j}$ per edge (in edge order). The result
 * has no `structure`, so `isingShape` reads its shape from the factors.
 *
 * @param graph The interaction graph: one spin per node, one coupling per edge. Each edge is used once as listed (its
 *   direction is ignored), and the graph's `labels`, if any, name the variables.
 * @param coupling The coupling $J$: one number for every edge, or one per edge, indexed like `graph.edges`. Positive
 *   values favour aligned neighbours (ferromagnetic), negative ones opposed neighbours.
 * @param field The external field $h$: one number for every node, or one per node. Positive values favour $+1$.
 * @returns The factor graph over `graph.nodes` binary variables.
 *
 * @example Two coupled spins
 * // Value 0 is spin -1 and value 1 is spin +1; the second spin feels a field of 0.5.
 * const graph = { kind: 'graph', nodes: 2, edges: [{ from: 0, to: 1 }], directed: false }
 * const model = isingModel(graph, 1, [0, 0.5])
 * print('factors:', model.factors.map((f) => f.name))
 * print('field on spin 1:', model.factors[1].table)
 * print('coupling table:', model.factors[2].table)
 */
export function isingModel(
  graph: Graph,
  coupling: number | ArrayLike<number>,
  field: number | ArrayLike<number>,
): IsingModel {
  const J = (k: number) => (typeof coupling === 'number' ? coupling : coupling[k])
  const h = (i: number) => (typeof field === 'number' ? field : field[i])
  const cards = new Array<number>(graph.nodes).fill(2)
  const spin = (a: number) => (a === 0 ? -1 : 1)
  const factors: DiscreteFactor[] = []
  for (let i = 0; i < graph.nodes; i++)
    factors.push(discreteFactor([i], cards, (a) => Math.exp(h(i) * spin(a[0])), `φ${i}`))
  graph.edges.forEach((e, k) =>
    factors.push(
      discreteFactor([e.from, e.to], cards, (a) => Math.exp(J(k) * spin(a[0]) * spin(a[1])), `ψ${e.from},${e.to}`),
    ),
  )
  return discreteFactorGraph(cards, factors, graph.labels)
}

/**
 * The Ising model on a $\text{rows} \times \text{cols}$ lattice with 4 neighbours per site (a torus with
 * `periodic`): the structure is `latticeTemplate(rows, cols)`, unrolled so that site $(i, j)$ is node
 * $i \cdot \text{cols} + j$, named $x_{i,j}$. The model keeps that structure, so `isingShape` reads it as a lattice.
 *
 * @param rows The number of rows of sites.
 * @param cols The number of columns of sites.
 * @param coupling The coupling $J$: one number for every edge, or one per edge of the unrolled lattice, in its edge
 *   order.
 * @param field The external field $h$: one number for every site, or one per site, indexed $i \cdot \text{cols} + j$.
 * @param options The lattice's options: `periodic` wraps the edges round into a torus (default false).
 * @returns The factor graph, with the unrolled lattice as its `structure`.
 *
 * @example The energy of three configurations on a $3 \times 3$ lattice
 * // The energy is minus the log potential: minus the sum of the logs of the factors at the configuration.
 * const model = isingLattice(3, 3, 1, 0)
 * const at = (f, x) => toArray(f.table).flat()[f.scope.reduce((k, v) => 2 * k + x[v], 0)]
 * const energy = (x) => -model.factors.reduce((e, f) => e + Math.log(at(f, x)), 0)
 * print('sites:', model.names)
 * print('all up:', energy([1, 1, 1, 1, 1, 1, 1, 1, 1]))
 * print('centre flipped:', energy([1, 1, 1, 1, 0, 1, 1, 1, 1]))
 * print('checkerboard:', energy([1, 0, 1, 0, 1, 0, 1, 0, 1]))
 */
export function isingLattice(
  rows: number,
  cols: number,
  coupling: number | ArrayLike<number>,
  field: number | ArrayLike<number>,
  options: { periodic?: boolean } = {},
): IsingModel {
  const structure = unroll(latticeTemplate(rows, cols, { periodic: options.periodic }))
  return { ...isingModel(structure, coupling, field), structure }
}

/**
 * The shape of an Ising model's interaction graph: `chain`, `tree`, `lattice` or `general`. A model built by
 * `isingLattice` reads it from its lattice declaration (`lattice`, or `chain` for a $1 \times n$ strip, which the
 * acyclic test reaches first); any other model reads it from its factor graph, where a grid reads as `general`.
 *
 * @param model The model: an `IsingModel` with a `structure`, or any discrete factor graph.
 * @returns The shape, as `shape` of `aifn-compute/graph/structured` gives it.
 *
 * @example The same grid, declared and plain
 * const lattice = isingLattice(3, 3, 1, 0)
 * print('1 x 4 strip:', isingShape(isingLattice(1, 4, 1, 0)))
 * print('3 x 3 lattice:', isingShape(lattice))
 * print('3 x 3 torus:', isingShape(isingLattice(3, 3, 1, 0, { periodic: true })))
 * print('3 x 3 grid as a plain graph:', isingShape(isingModel(lattice.structure, 1, 0)))
 */
export function isingShape(model: DiscreteFactorGraph | IsingModel): GraphShape {
  return shape('structure' in model && model.structure ? model.structure : bipartiteGraph(model))
}

/**
 * Marginal inference on an Ising model with the engine its shape allows: forward–backward on a chain
 * (`chainSumProduct`, exact), belief propagation with the tree schedule on a tree (exact), and loopy belief propagation
 * (flooding, with `options`) otherwise (Murphy, Weiss and Jordan, 1999). A chain with `evidence` or with `mode: 'max'`
 * goes to belief propagation too, which is exact there. Run it with `run(alg, undefined, n)`: the chain engine is done
 * after $2N - 1$ steps for $N$ spins and reports `marginals`; belief propagation reports `beliefs` and sets `converged`.
 *
 * @param model The Ising model, from `isingModel` or `isingLattice` (any discrete factor graph is accepted).
 * @param options The options of belief propagation (schedule, damping, tolerance, evidence, sum or max mode); the
 *   chain engine ignores them.
 * @returns The shape that chose the engine, and the algorithm.
 *
 * @example A strip is solved exactly by forward-backward
 * // A field of 1 on the first spin only, passed down the strip by a coupling of 0.5.
 * const { shape, algorithm } = isingInference(isingLattice(1, 4, 0.5, [1, 0, 0, 0]))
 * const s = run(algorithm, undefined, 7)
 * print('shape:', shape)
 * print('p(spin = +1):', s.marginals.map((m) => toArray(m)[1]))
 *
 * @example Loopy belief propagation on a $3 \times 3$ lattice
 * const { shape, algorithm } = isingInference(isingLattice(3, 3, 0.3, 0.2))
 * const s = run(algorithm, undefined, 50)
 * print('shape:', shape)
 * print('converged after', s.sweep, 'sweeps')
 * print('p(spin = +1):', s.beliefs.map((b) => toArray(b)[1]))
 */
export function isingInference(
  model: DiscreteFactorGraph,
  options: BeliefPropagationOptions = {},
): { shape: GraphShape; algorithm: Algorithm<void, Status> } {
  const graphShape = isingShape(model)
  if (graphShape === 'chain' && options.evidence === undefined && options.mode !== 'max')
    return { shape: graphShape, algorithm: chainSumProduct(model) }
  return { shape: graphShape, algorithm: beliefPropagationSteps(model, options) }
}

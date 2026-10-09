/**
 * `infer`: pick an inference engine for a model and its data by the shape of its structured graph (Koller & Friedman
 * 2009, ch. 9–12): forward–backward on a chain, exact belief propagation on a tree, loopy belief propagation on any
 * other discrete model, and Gibbs sampling when a latent variable is not discrete. The engine table is static data;
 * applications add engines by passing their own table (`withEngines`), never by registering into a global.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { shape, type GraphShape } from 'aifn-compute/graph/structured'
import { beliefPropagationSteps } from 'aifn-compute/inference/message-passing'
import { chainSumProduct, enumerationSteps, variableEliminationSteps } from 'aifn-compute/inference/exact'
import { compileGaussianModel, modelExpectationPropagation } from 'aifn-compute/inference/expectation-propagation'
import { modelGibbs } from 'aifn-compute/inference/stochastic'
import {
  bipartiteGraph,
  toDiscreteFactorGraph,
  toFactorGraph,
  type Bindings,
  type Model,
  type ModelDiscreteGraph,
} from 'aifn-compute/inference/model'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * What an engine sees of a model: the model, its bindings, its discrete tabulation (null when a latent variable is not
 * discrete with finitely many values) and the shape of that factor graph; both computed once, on first use.
 */
export interface InferenceContext {
  /** The model `infer` was given. */
  model: Model
  /** Its data and sizes, as given to `infer`. */
  bindings: Bindings
  /** The model as a discrete factor graph with the data clamped, or null when a latent variable is not discrete. */
  discrete(): ModelDiscreteGraph | null
  /** The shape of the discrete factor graph (`'chain'`, `'tree'`, ...), or null when there is none. */
  shape(): GraphShape | null
}

/** An engine: a name, a test of whether it applies, and the algorithm it runs (factory form, no start). */
export interface EngineRegistration {
  /** The name `infer` reports, and that `InferOptions.engine` selects it by. */
  name: string
  /** Whether the engine applies to the model: `infer` takes the first engine of the table for which this is true. */
  matches(context: InferenceContext): boolean
  /** Build the engine's algorithm for the model (run with no start). */
  create(context: InferenceContext): Algorithm<void, Status>
}

/** An ordered table of engines: `infer` takes the first that matches. */
export type EngineTable = readonly EngineRegistration[]

/**
 * The discrete factor graph of the context's model, for the engines that need one. Throws `DomainError` when a latent
 * variable is not discrete.
 *
 * @param c The model, its bindings and the lazily computed tabulation, as `infer` builds it.
 * @returns The `graph` of the context's discrete tabulation.
 */
const discreteGraph = (c: InferenceContext) => {
  const d = c.discrete()
  if (!d) throw new DomainError('infer', `infer: ${c.model.name} has a latent variable that is not discrete`)
  return d.graph
}

/**
 * True when the model compiles for EP: linear-Gaussian latents with interval and Gaussian evidence. Any error of
 * `compileGaussianModel` counts as no.
 *
 * @param c The model and its bindings, tried with `compileGaussianModel`.
 * @returns Whether the model compiled.
 */
function linearGaussian(c: InferenceContext): boolean {
  try {
    compileGaussianModel(c.model, c.bindings)
    return true
  } catch {
    return false
  }
}

/**
 * The built-in engines, in the order `infer` tries them: `forward-backward` (chain-shaped discrete models:
 * `chainSumProduct`), `belief-propagation` (other discrete models: exact on trees with the tree schedule, loopy with
 * flooding otherwise), `expectation-propagation` (linear-Gaussian models with interval and Gaussian evidence:
 * `modelExpectationPropagation`), and `gibbs` (anything else: `modelGibbs`). `variable-elimination` and `enumeration`
 * run only when named.
 */
export const builtInEngines: EngineTable = [
  {
    name: 'forward-backward',
    matches: (c) => c.shape() === 'chain',
    create: (c) => chainSumProduct(discreteGraph(c)),
  },
  {
    name: 'belief-propagation',
    matches: (c) => c.discrete() !== null,
    create: (c) => beliefPropagationSteps(discreteGraph(c)),
  },
  {
    name: 'expectation-propagation',
    matches: (c) => linearGaussian(c),
    create: (c) => modelExpectationPropagation(c.model, c.bindings),
  },
  {
    name: 'gibbs',
    matches: () => true,
    create: (c) => modelGibbs(c.model, c.bindings),
  },
  {
    name: 'variable-elimination',
    matches: () => false,
    create: (c) => variableEliminationSteps(discreteGraph(c)),
  },
  {
    name: 'enumeration',
    matches: () => false,
    create: (c) => enumerationSteps(discreteGraph(c)),
  },
]

/**
 * A table with `extra` engines tried before those of `table`, so an application's engine wins over a built-in one
 * that also matches. Neither table is modified.
 *
 * @param table The engines to fall back on, in their order (default `builtInEngines`).
 * @param extra The engines to try first, in the order given.
 * @returns A new table: `extra`, then `table`.
 *
 * @example An extra engine is tried first
 * const idle = { name: 'idle', init: () => ({ t: 0 }), step: (s) => s }
 * const quiet = { name: 'quiet', matches: () => true, create: () => idle }
 * print('engines:', withEngines(builtInEngines, quiet).map((e) => e.name))
 */
export function withEngines(table: EngineTable = builtInEngines, ...extra: EngineRegistration[]): EngineTable {
  return [...extra, ...table]
}

/** An engine chosen for a model: its name, the shape it was chosen for, and the algorithm to run. */
export interface Inference {
  /** The name of the chosen engine. */
  engine: string
  /** The shape of the discrete model's factor graph, or null when the model is not discrete. */
  shape: GraphShape | null
  /**
   * The discrete tabulation (its `keys` name the variables; $\log Z$ plus `logConstant` is $\log p(\text{data})$), or
   * null.
   */
  discrete: ModelDiscreteGraph | null
  /** The engine's algorithm, to run with no start (`run(algorithm, undefined, steps)`). */
  algorithm: Algorithm<void, Status>
}

/** Options of {@link infer}. */
export interface InferOptions {
  /** Use this engine by name rather than the first that matches. */
  engine?: string
  /** The engine table (default `builtInEngines`; see `withEngines`). */
  engines?: EngineTable
}

/**
 * Pick an inference engine for a model and data and build its algorithm. The model is tabulated as a discrete factor
 * graph when it can be (data clamped), and the shape of that graph picks the path: a chain runs forward–backward, a
 * tree exact belief propagation, any other discrete model loopy belief propagation; a linear-Gaussian model with
 * interval or Gaussian evidence runs expectation propagation; any other model with a continuous latent variable runs
 * Gibbs sampling. Run the result with the runners of `aifn-compute/foundation/trace`
 * (`run(inference.algorithm, undefined, n, { stream })`). Throws `DomainError` when no engine matches or none has the
 * name asked for.
 *
 * @param model The model, as `model` of `aifn-compute/inference/model` builds it (plain data).
 * @param bindings The data clamped on observed nodes and the sizes of plates and chains (default none).
 * @param options An engine to use by name, and the table to choose from (default the first match in
 *   `builtInEngines`).
 * @returns The chosen engine's name, the shape and tabulation it was chosen by (null for a model that is not
 *   discrete), and its algorithm, ready to run with no start.
 *
 * @example A normal mean with a conjugate prior runs expectation propagation
 * // mu ~ Normal(0, sd 10), and x ~ Normal(mu, sd 1) in a plate: the plain data `model(...)` builds.
 * const mu = { kind: 'ref', node: 'mu' }
 * const normalMean = {
 *   kind: 'graph', name: 'normal mean', directed: true, nodes: 2, labels: ['mu', 'x'],
 *   edges: [{ from: 0, to: 1, directed: true }],
 *   attributes: [
 *     { name: 'mu', role: 'latent', group: null, data: { dist: { family: 'Normal', args: [0, 10] } } },
 *     { name: 'x', role: 'observed', group: 'points', data: { dist: { family: 'Normal', args: [mu, 1] } } },
 *   ],
 *   groups: [{ name: 'points', kind: 'plate', size: 'n', index: ['p'], parent: null }],
 *   sizes: ['n'],
 * }
 * const inference = infer(normalMean, { data: { x: [1.2, 0.8, 1.1] } })
 * print('engine:', inference.engine)
 * const s = run(inference.algorithm, undefined, 20)
 * print('posterior mean', s.means, 'exact', 3.1 / (3 + 1 / 100))
 * print('posterior variance', s.variances, 'exact', 1 / (3 + 1 / 100))
 *
 * @example A discrete model runs an exact message-passing engine
 * // A fair coin c, seen through a noisy sensor y with p(y = 1 | c) = 0.2 or 0.9.
 * const qc = { kind: 'ref', node: 'q', select: 'c' }
 * const coin = {
 *   kind: 'graph', name: 'coin', directed: true, nodes: 3, labels: ['c', 'q', 'y'],
 *   edges: [{ from: 1, to: 2, directed: true }, { from: 0, to: 2, directed: true }],
 *   attributes: [
 *     { name: 'c', role: 'latent', group: null, data: { dist: { family: 'Bernoulli', args: [0.5] } } },
 *     { name: 'q', role: 'parameter', group: null, data: { value: [0.2, 0.9] } },
 *     { name: 'y', role: 'observed', group: null, data: { dist: { family: 'Bernoulli', args: [qc] } } },
 *   ],
 *   groups: [],
 *   sizes: [],
 * }
 * const inference = infer(coin, { data: { y: 1 } })
 * print('engine:', inference.engine, 'shape:', inference.shape)
 * const s = run(inference.algorithm, undefined, 100)
 * print('p(c | y = 1) =', s.marginals[0], 'exact', [0.2 / 1.1, 0.9 / 1.1])
 * print('p(y = 1) =', Math.exp(s.logZ + inference.discrete.logConstant))
 */
export function infer(model: Model, bindings: Bindings = {}, options: InferOptions = {}): Inference {
  const table = options.engines ?? builtInEngines
  let discrete: ModelDiscreteGraph | null | undefined
  let graphShape: GraphShape | null | undefined
  const context: InferenceContext = {
    model,
    bindings,
    discrete: () => {
      if (discrete === undefined) {
        // Discrete when every latent variable has finitely many values; other failures are errors, not a fallback.
        const fg = toFactorGraph(model, bindings)
        const finite = fg.attributes
          .slice(0, fg.variables)
          .every((v) => v.role === 'observed' || v.data!.cardinality != null)
        discrete = finite ? toDiscreteFactorGraph(fg.expanded) : null
      }
      return discrete
    },
    shape: () => {
      if (graphShape === undefined) {
        const d = context.discrete()
        graphShape = d ? shape(bipartiteGraph(d.graph)) : null
      }
      return graphShape
    },
  }
  const chosen =
    options.engine === undefined ? table.find((e) => e.matches(context)) : table.find((e) => e.name === options.engine)
  if (!chosen)
    throw new DomainError(
      'infer',
      `infer: no engine ${options.engine === undefined ? 'matches' : `named ${options.engine}`}`,
    )
  return {
    engine: chosen.name,
    shape: context.shape(),
    discrete: context.discrete(),
    algorithm: chosen.create(context),
  }
}

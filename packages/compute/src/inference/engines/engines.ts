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
  model: Model
  bindings: Bindings
  discrete(): ModelDiscreteGraph | null
  shape(): GraphShape | null
}

/** An engine: a name, a test of whether it applies, and the algorithm it runs (factory form, no start). */
export interface EngineRegistration {
  name: string
  matches(context: InferenceContext): boolean
  create(context: InferenceContext): Algorithm<void, Status>
}

/** An ordered table of engines: `infer` takes the first that matches. */
export type EngineTable = readonly EngineRegistration[]

const discreteGraph = (c: InferenceContext) => {
  const d = c.discrete()
  if (!d) throw new DomainError('infer', `infer: ${c.model.name} has a latent variable that is not discrete`)
  return d.graph
}

/** True when the model compiles for EP: linear-Gaussian latents with interval and Gaussian evidence. */
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
 * `modelExpectationPropagation`), and `gibbs` (anything else: `modelGibbs`). `variable-elimination` and `enumeration` run only
 * when named.
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

/** A table with `extra` engines tried before those of `table` (default the built-ins). */
export function withEngines(table: EngineTable = builtInEngines, ...extra: EngineRegistration[]): EngineTable {
  return [...extra, ...table]
}

/** An engine chosen for a model: its name, the shape it was chosen for, and the algorithm to run. */
export interface Inference {
  engine: string
  /** The shape of the discrete model's factor graph, or null when the model is not discrete. */
  shape: GraphShape | null
  /** The discrete tabulation (its `keys` name the variables; log Z plus `logConstant` is log p(data)), or null. */
  discrete: ModelDiscreteGraph | null
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
 * (`run(inference.algorithm, undefined, n, { stream })`).
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

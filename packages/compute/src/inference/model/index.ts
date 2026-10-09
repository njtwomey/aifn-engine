/**
 * `aifn-compute/inference/model`: the model description language, and the discrete factor graphs that exact inference
 * and message passing run on.
 *
 * - Describing a model: `model` declares named parameters, latent, observed and deterministic nodes on a builder, with
 *   plates and chains as the groups of a structured graph of `aifn-compute/graph/structured`; `dist` gives the
 *   conditional distributions. A model is plain, serialisable data; `nodeArgs`, `argRefs` and `plateChain` read it.
 * - Expanding, scoring and sampling: `expandModel` unrolls the plates and chains against sizes, constants and data into
 *   instances (keyed by `instanceKey`). `logJoint` and `instanceLogDensity` score values, `conditionalOf` gives an
 *   instance's distribution given the rest, and `sampleModel` draws ancestrally (`nestedValues` turns the draws into
 *   data). Beneath them: `environment` (the values by key, deterministic nodes computed on demand), `argValue`,
 *   `resolveRef` (the instances an `at` reference may select), `distOf`, `realise` and `evaluateOp`.
 * - Dependence between instances, looking through deterministic nodes: `stochasticParents`, `dependencyMaps` and
 *   `modelMarkovBlanket`; `cardinalityOf` gives a discrete node's number of values.
 * - A model's structures: `toFactorGraph` (one factor per stochastic instance, listed by `factorsOf`),
 *   `toDiscreteFactorGraph` (tables over the latent variables with the data clamped, for the discrete engines), and
 *   diagram data for plate notation and factor graphs (`toPlateDiagram`, `toFactorDiagram`).
 * - Discrete factors: `discreteFactor`, `gateFactor` (a selector variable switching between factors, for mixtures and
 *   model selection) and `discreteFactorGraph`; their algebra `factorProduct`, `factorProductAll`, `factorMarginalise`
 *   (by sum or max), `factorReduce` (conditioning on evidence) and `normaliseFactor`; on a graph `logPotential`,
 *   `factorGraphEdges`, `factorGraphNeighbours`, `bipartiteGraph`, `isTree` and `variableName`; and the row-major table
 *   helpers `stridesOf`, `tableSize`, `valuesOf` and `forEachAssignment`.
 * - `modelFunctions`: the module's functions as registry entries, keyed by name.
 *
 * A malformed model or factor throws `DomainError`, and a table of the wrong size `ShapeError`. Potentials are
 * non-negative numbers, not logs, in row-major tables whose axes follow the factor's scope.
 */

export {
  bipartiteGraph,
  discreteFactor,
  discreteFactorGraph,
  factorGraphEdges,
  factorGraphNeighbours,
  factorMarginalise,
  factorProduct,
  factorProductAll,
  factorReduce,
  forEachAssignment,
  gateFactor,
  isTree,
  logPotential,
  normaliseFactor,
  stridesOf,
  tableSize,
  valuesOf,
  variableName,
  type DiscreteFactor,
  type DiscreteFactorGraph,
  type FactorGraphEdge,
} from './factors'
export {
  argRefs,
  argValue,
  cardinalityOf,
  conditionalOf,
  dependencyMaps,
  dist,
  distOf,
  environment,
  evaluateOp,
  expandModel,
  instanceKey,
  instanceLogDensity,
  logJoint,
  model,
  modelMarkovBlanket,
  nestedValues,
  nodeArgs,
  plateChain,
  realise,
  resolveRef,
  sampleModel,
  stochasticParents,
  type Arg,
  type Bindings,
  type DeterministicOp,
  type DistSpec,
  type Env,
  type ExpandedModel,
  type Family,
  type GroupOptions,
  type Instance,
  type Model,
  type ModelBuilder,
  type ModelNode,
  type ModelNodeData,
  type ModelScope,
  type Nested,
  type NodeHandle,
  type NodeOptions,
  type NodeRef,
  type NodeValue,
  type SizeRef,
} from './model'
export {
  factorsOf,
  toDiscreteFactorGraph,
  toFactorDiagram,
  toFactorGraph,
  toPlateDiagram,
  type FactorDiagramOptions,
  type FactorNodeData,
  type ModelDiscreteGraph,
  type ModelFactorGraph,
} from './structure'
export { modelFunctions } from './registry'

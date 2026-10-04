/**
 * The functions of `aifn-compute/inference/model`: the model language (declare, expand, sample, score), its structures
 * (factor graphs, plate and factor diagrams) and discrete factors.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as factors from './factors'
import * as model from './model'
import * as structure from './structure'

const fn = definer<FunctionInfo>('function', 'inference/model')
const PP = ['probabilistic-programming', 'model-based-machine-learning']
const FG = ['factor-graph']

fn(
  {
    key: 'model',
    name: 'Model',
    summary: 'A probabilistic model as named random and deterministic nodes, with plates.',
    role: 'construction',
    notes: [...PP, 'bayesian-network'],
  },
  model.model,
)
fn(
  { key: 'plateChain', name: 'Chain over a plate', role: 'construction', notes: [...PP, 'hidden-markov-model'] },
  model.plateChain,
)
fn({ key: 'expandModel', name: 'Expand plates', role: 'transform', notes: PP }, model.expandModel)
fn(
  {
    key: 'sampleModel',
    name: 'Ancestral sampling',
    role: 'simulation',
    random: true,
    notes: [...PP, 'bayesian-network'],
  },
  model.sampleModel,
)
fn({ key: 'logJoint', name: 'Log joint density', role: 'estimator', notes: PP }, model.logJoint)
fn(
  { key: 'instanceLogDensity', name: 'Log density of one node instance', role: 'estimator', notes: PP },
  model.instanceLogDensity,
)
fn(
  { key: 'conditionalOf', name: 'Full conditional of a node', role: 'inference', notes: ['gibbs-sampling', ...PP] },
  model.conditionalOf,
)
fn(
  {
    key: 'modelMarkovBlanket',
    name: 'Markov blanket',
    role: 'property',
    notes: ['bayesian-network', 'gibbs-sampling'],
  },
  model.modelMarkovBlanket,
)
fn(
  { key: 'stochasticParents', name: 'Stochastic parents', role: 'property', notes: ['bayesian-network'] },
  model.stochasticParents,
)
fn(
  { key: 'toFactorGraph', name: 'Model to factor graph', role: 'transform', returns: 'graph', notes: [...FG, ...PP] },
  structure.toFactorGraph,
)
fn(
  { key: 'toDiscreteFactorGraph', name: 'Model to discrete factor graph', role: 'transform', notes: FG },
  structure.toDiscreteFactorGraph,
)
fn(
  {
    key: 'toPlateDiagram',
    name: 'Plate diagram',
    role: 'transform',
    returns: 'graph',
    notes: ['bayesian-network', ...PP],
  },
  structure.toPlateDiagram,
)
fn(
  { key: 'toFactorDiagram', name: 'Factor diagram', role: 'transform', returns: 'graph', notes: FG },
  structure.toFactorDiagram,
)
fn({ key: 'factorsOf', name: 'Factors of a model', role: 'transform', notes: FG }, structure.factorsOf)
fn(
  { key: 'discreteFactor', name: 'Discrete factor', role: 'construction', notes: [...FG, 'markov-random-field'] },
  factors.discreteFactor,
)
fn(
  {
    key: 'discreteFactorGraph',
    name: 'Discrete factor graph',
    role: 'construction',
    notes: [...FG, 'markov-random-field'],
  },
  factors.discreteFactorGraph,
)
fn(
  {
    key: 'gateFactor',
    name: 'Gate factor',
    summary: 'A selector variable switches between factors: φ(c = k, x) = f_k(x), for mixtures and model selection.',
    role: 'construction',
    notes: ['gates-in-factor-graphs', ...FG],
    cite: ['minka2008gates'],
  },
  factors.gateFactor,
)
fn(
  { key: 'factorProduct', name: 'Factor product', role: 'transform', notes: ['variable-elimination', ...FG] },
  factors.factorProduct,
)
fn(
  { key: 'factorProductAll', name: 'Product of factors', role: 'transform', notes: ['variable-elimination'] },
  factors.factorProductAll,
)
fn(
  {
    key: 'factorMarginalise',
    name: 'Sum a variable out of a factor',
    role: 'transform',
    notes: ['variable-elimination'],
  },
  factors.factorMarginalise,
)
fn(
  { key: 'factorReduce', name: 'Condition a factor on evidence', role: 'transform', notes: ['variable-elimination'] },
  factors.factorReduce,
)
fn({ key: 'normaliseFactor', name: 'Normalise a factor', role: 'transform', notes: FG }, factors.normaliseFactor)
fn(
  { key: 'logPotential', name: 'Log potential of an assignment', role: 'property', notes: ['markov-random-field'] },
  factors.logPotential,
)
fn(
  {
    key: 'isTree',
    name: 'Is the factor graph a tree',
    role: 'property',
    notes: ['belief-propagation', 'junction-tree-algorithm'],
  },
  factors.isTree,
)
fn(
  { key: 'bipartiteGraph', name: 'Factor graph as a bipartite graph', role: 'transform', returns: 'graph', notes: FG },
  factors.bipartiteGraph,
)

/** The functions of the module, keyed by name. */
export const modelFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', model, structure, factors) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >

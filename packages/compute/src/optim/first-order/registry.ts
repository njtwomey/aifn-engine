/**
 * The algorithms of `aifn-compute/optim/first-order`, registered with what each factory takes (`problem`) and the roles
 * of its state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic
 * trace view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as rules from './rules'
import * as conjugateGradient from './conjugateGradient'
import * as coordinateDescent from './coordinateDescent'
import * as firstOrder from './firstOrder'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optim/first-order')

algorithm(
  {
    key: 'gradientDescent',
    name: 'Gradient descent',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['gradient-descent', 'convergence-of-gradient-descent'],
  },
  firstOrder.gradientDescent,
)
algorithm(
  {
    key: 'momentum',
    name: 'Momentum',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['momentum-and-nesterov'],
    cite: ['polyak1964'],
  },
  firstOrder.momentum,
)
algorithm(
  {
    key: 'nesterov',
    name: 'Nesterov accelerated gradient',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['momentum-and-nesterov', 'accelerated-gradient-methods'],
    cite: ['nesterov1983'],
  },
  firstOrder.nesterov,
)
algorithm(
  {
    key: 'adagrad',
    name: 'AdaGrad',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['adagrad-and-rmsprop'],
    cite: ['duchi2011'],
  },
  firstOrder.adagrad,
)
algorithm(
  {
    key: 'rmsprop',
    name: 'RMSProp',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['adagrad-and-rmsprop'],
    cite: ['tieleman2012'],
  },
  firstOrder.rmsprop,
)
algorithm(
  {
    key: 'adam',
    name: 'Adam',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    glossary: 'adam',
    notes: ['adam'],
    cite: ['kingma2014'],
  },
  firstOrder.adam,
)
algorithm(
  {
    key: 'adamw',
    name: 'AdamW',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    glossary: 'adamw',
    notes: ['decoupled-weight-decay', 'adam'],
    cite: ['loshchilov2019'],
  },
  firstOrder.adamw,
)
algorithm(
  {
    key: 'conjugateGradient',
    name: 'Nonlinear conjugate gradient',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    cite: ['nocedal2006'],
  },
  conjugateGradient.conjugateGradient,
)
algorithm(
  {
    key: 'linearConjugateGradient',
    name: 'Linear conjugate gradient',
    summary: 'Conjugate gradient for Ax = b with A symmetric positive definite.',
    problem: 'quadratic-program',
    state: { iterate: 'x', objective: 'residualNorm', stepSize: 'alpha', flags: ['converged', 'diverged', 'stalled'] },
    notes: ['iterative-linear-solvers', 'quadratic-programming'],
    cite: ['hestenes1952'],
  },
  conjugateGradient.linearConjugateGradient,
)
algorithm(
  {
    key: 'coordinateDescent',
    name: 'Coordinate descent',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', grad: 'grad', flags: ['converged', 'diverged'] },
    notes: ['coordinate-descent'],
    cite: ['wright2015'],
  },
  coordinateDescent.coordinateDescent,
)

/** Every algorithm of the module, keyed by factory name. */
export const firstOrderAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', conjugateGradient, coordinateDescent, firstOrder) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'optim/first-order')

fn(
  {
    key: 'sgdRule',
    name: 'SGD update rule (with momentum)',
    role: 'transform',
    notes: ['gradient-descent', 'momentum-and-nesterov'],
    cite: ['robbins1951', 'polyak1964'],
  },
  rules.sgdRule,
)
fn(
  {
    key: 'adagradRule',
    name: 'AdaGrad update rule',
    role: 'transform',
    notes: ['adagrad-and-rmsprop'],
    cite: ['duchi2011'],
  },
  rules.adagradRule,
)
fn(
  {
    key: 'rmspropRule',
    name: 'RMSProp update rule',
    role: 'transform',
    notes: ['adagrad-and-rmsprop'],
    cite: ['tieleman2012'],
  },
  rules.rmspropRule,
)
fn(
  { key: 'adamRule', name: 'Adam update rule', role: 'transform', notes: ['adam'], cite: ['kingma2015'] },
  rules.adamRule,
)
fn(
  {
    key: 'adamwRule',
    name: 'AdamW update rule',
    role: 'transform',
    notes: ['decoupled-weight-decay', 'adam'],
    cite: ['loshchilov2019'],
  },
  rules.adamwRule,
)
fn({ key: 'chainRules', name: 'Chain update rules', role: 'construction' }, rules.chainRules)
fn(
  { key: 'globalNorm', name: 'Global gradient norm', role: 'property', notes: ['vanishing-and-exploding-gradients'] },
  rules.globalNorm,
)
fn(
  {
    key: 'clipByGlobalNorm',
    name: 'Clip by global norm',
    role: 'transform',
    notes: ['vanishing-and-exploding-gradients'],
    cite: ['pascanu2013'],
  },
  rules.clipByGlobalNorm,
)

/** The update rules of the module, keyed by name. */
export const firstOrderFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', rules) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >

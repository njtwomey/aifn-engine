/**
 * The registry of `aifn-methods/neural/full-batch`: the streamed comparison of full-batch L-BFGS against first-order
 * training of a small MLP, and the model it trains, as functions.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as comparison from './comparison'

const fn = definer<FunctionInfo>('function', 'neural/full-batch')

fn(
  {
    key: 'comparisonModel',
    name: 'Small MLP for the optimiser comparison',
    summary:
      'An MLP of a given width, depth and activation with one output, and the map from a flat θ to its parameters.',
    role: 'construction',
    notes: ['multilayer-perceptron', 'activation-functions'],
  },
  comparison.comparisonModel,
)
fn(
  {
    key: 'fullBatchComparison',
    name: 'Streamed L-BFGS vs first-order training run',
    summary:
      'Train a small MLP from the same initial weights by full-batch L-BFGS, gradient descent, Adam and SGD, recording loss against iterations and gradient evaluations and the L-BFGS line-search internals.',
    role: 'simulation',
    random: true,
    notes: ['quasi-newton-methods', 'line-search', 'gradient-descent', 'adam', 'minibatching-and-batch-size'],
    cite: ['liu1989', 'nocedal2006', 'kingma2015'],
  },
  comparison.fullBatchComparison,
)

/** The functions of the module, keyed by name. */
export const fullBatchFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', comparison) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >

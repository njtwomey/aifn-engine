/**
 * Registry entries of `aifn-methods/optimisation/portfolio`: greedy portfolio construction on a candidate by task
 * table of losses, its leave-one-task-out assessment, and the average rank.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as portfolio from './portfolio'

const fn = definer<FunctionInfo>('function', 'optimisation/portfolio')

fn(
  {
    key: 'greedyPortfolio',
    name: 'Greedy portfolio',
    summary:
      'Add the configuration that most lowers the mean over tasks of the best member’s loss, k times; the optimum too.',
    role: 'construction',
    cite: ['wistuba2015', 'feurer2022autosklearn2', 'nemhauser1978'],
  },
  portfolio.greedyPortfolio,
)

fn(
  {
    key: 'portfolioLoss',
    name: 'Portfolio loss',
    summary: 'The best member’s loss on each task, capped at the loss of an uncovered task, and its mean.',
    role: 'property',
    cite: ['feurer2022autosklearn2'],
  },
  portfolio.portfolioLoss,
)

fn(
  {
    key: 'leaveOneTaskOut',
    name: 'Leave-one-task-out portfolio assessment',
    summary: 'Build the greedy portfolio without each task in turn and score its first members on the task left out.',
    role: 'estimator',
    cite: ['feurer2022autosklearn2'],
  },
  portfolio.leaveOneTaskOut,
)

fn(
  {
    key: 'averageRank',
    name: 'Average rank over tasks',
    summary: 'Rank the configurations on each task (ties averaged) and average the ranks: a meta-learned ordering.',
    role: 'transform',
    cite: ['feurer2015warmstart'],
  },
  portfolio.averageRank,
)

/** The functions of the module, keyed by name. */
export const portfolioFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', portfolio) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >

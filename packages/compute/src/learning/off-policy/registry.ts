/**
 * The registry of `aifn-compute/learning/off-policy`: the off-policy value estimators, the slate estimators and the propensity
 * models, all functions (`role: 'estimator'` or `'fit'`).
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as estimators from './estimators'
import * as propensity from './propensity'
import * as slate from './slate'

const fn = definer<FunctionInfo>('function', 'learning/off-policy')

const OPE = ['off-policy-evaluation', 'counterfactual-evaluation-of-recommenders', 'logging-policies-and-propensities']

fn(
  {
    key: 'importanceWeights',
    name: 'Importance weights',
    tex: 'w_i = \\pi(a_i \\mid x_i)/\\pi_0(a_i \\mid x_i)',
    summary: 'The ratio of the target to the logging probability of each logged action.',
    role: 'estimator',
    notes: OPE,
    cite: ['horvitz1952', 'bottou2013'],
  },
  estimators.importanceWeights,
)
fn(
  {
    key: 'ips',
    name: 'Inverse propensity scoring (IPS)',
    tex: '\\tfrac1n \\sum_i w_i r_i',
    summary: 'Reweight each logged reward by its importance weight; unbiased, with variance growing with the weights.',
    role: 'estimator',
    notes: [...OPE, 'off-policy-evaluation-in-reinforcement-learning', 'propensity-in-recommendation'],
    cite: ['horvitz1952', 'li2011', 'dudik2011'],
  },
  estimators.ips,
)
fn(
  {
    key: 'clippedIps',
    name: 'Clipped IPS',
    tex: '\\tfrac1n \\sum_i \\min(w_i, M) r_i',
    summary: 'Cap the importance weights at M: less variance, a downward bias that grows as M falls.',
    role: 'estimator',
    notes: OPE,
    cite: ['bottou2013'],
  },
  estimators.clippedIps,
)
fn(
  {
    key: 'snips',
    name: 'Self-normalised IPS (SNIPS)',
    tex: '\\sum_i w_i r_i / \\sum_i w_i',
    summary: 'Divide by the sum of the weights: biased but consistent, bounded by the reward range.',
    role: 'estimator',
    notes: OPE,
    cite: ['swaminathan2015b'],
  },
  estimators.snips,
)
fn(
  {
    key: 'directMethod',
    name: 'Direct method (DM)',
    tex: '\\tfrac1n \\sum_i \\sum_a \\pi(a \\mid x_i) \\hat q(x_i, a)',
    summary: 'Average a reward model over the target’s actions: low variance, biased as the model is.',
    role: 'estimator',
    notes: OPE,
    cite: ['dudik2011'],
  },
  estimators.directMethod,
)
fn(
  {
    key: 'doublyRobust',
    name: 'Doubly robust (DR)',
    summary: 'The direct method corrected by IPS on the model’s residuals; unbiased if either part is right.',
    role: 'estimator',
    notes: [...OPE, 'off-policy-evaluation-in-reinforcement-learning'],
    cite: ['dudik2011'],
  },
  estimators.doublyRobust,
)
fn(
  {
    key: 'switchDoublyRobust',
    name: 'Switch doubly robust',
    summary: 'Doubly robust where the importance weight is at most τ, the reward model alone above it.',
    role: 'estimator',
    notes: OPE,
    cite: ['dudik2011'],
  },
  estimators.switchDoublyRobust,
)
fn(
  {
    key: 'slatePseudoInverse',
    name: 'Slate pseudo-inverse estimator',
    tex: '\\tfrac1n \\sum_i r_i \\theta_i^\\top \\Gamma^+ 1_{s_i}',
    summary: 'Off-policy value of a slate policy under additive rewards, from the logging policy’s pairwise marginals.',
    role: 'estimator',
    notes: ['slate-off-policy-evaluation', 'counterfactual-evaluation-of-recommenders'],
    cite: ['swaminathan2017slate'],
  },
  slate.slatePseudoInverse,
)
fn(
  {
    key: 'slateIps',
    name: 'Slate-level IPS',
    summary: 'IPS on whole slates: unbiased for any reward, but the logged slate rarely matches the target’s.',
    role: 'estimator',
    notes: ['slate-off-policy-evaluation'],
    cite: ['swaminathan2017slate'],
  },
  slate.slateIps,
)
fn(
  {
    key: 'estimatePropensities',
    name: 'Propensity model (multinomial logistic)',
    summary: 'Estimate π₀(a | x) by L2-penalised multinomial logistic regression of the logged action on the context.',
    role: 'fit',
    notes: ['logging-policies-and-propensities', 'propensity-in-recommendation'],
    cite: ['bottou2013'],
  },
  propensity.estimatePropensities,
)
fn(
  {
    key: 'empiricalPropensities',
    name: 'Empirical propensities',
    summary: 'The smoothed share of each action within each discrete context.',
    role: 'estimator',
    notes: ['logging-policies-and-propensities', 'estimating-item-propensities'],
    cite: ['li2011'],
  },
  propensity.empiricalPropensities,
)

type Table = Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

/** The functions of the module, keyed by name. */
export const offPolicyFunctions: Table = entries<FunctionInfo>('function', {
  ...estimators,
  ...slate,
  ...propensity,
}) as Table

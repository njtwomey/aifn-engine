/** The registry of `aifn-methods/learning/survival`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as aft from './aft'
import * as cox from './cox'

const fn = definer<FunctionInfo>('function', 'learning/survival')
const COX = ['cox-proportional-hazards', 'survival-and-hazard-functions']
const AFT = ['parametric-survival-models', 'survival-and-hazard-functions']

fn(
  {
    key: 'coxPh',
    name: 'Cox proportional hazards',
    summary:
      'Log hazard ratios by Newton’s method on the partial likelihood (Breslow or Efron ties), with Breslow’s baseline.',
    role: 'fit',
    notes: COX,
    cite: ['cox1972', 'cox1975', 'breslow1974', 'efron1977'],
  },
  cox.coxPh,
)
fn(
  {
    key: 'coxPartialLikelihood',
    name: 'Cox partial log-likelihood',
    summary: 'ℓ(β) = Σ_events [xᵀβ − log Σ_risk set e^{xᵀβ}], with Efron or Breslow ties.',
    role: 'property',
    notes: COX,
    cite: ['cox1975', 'efron1977', 'breslow1974'],
  },
  cox.coxPartialLikelihood,
)
fn(
  {
    key: 'coxSurvival',
    name: 'Cox survival curve',
    summary: 'S(t | x) = exp(−Ĥ₀(t) e^{xᵀβ}) at the event times.',
    role: 'property',
    notes: COX,
    cite: ['breslow1974'],
  },
  cox.coxSurvival,
)
fn(
  {
    key: 'harrellConcordance',
    name: 'Harrell’s concordance index',
    summary: 'The share of comparable pairs whose earlier failure has the higher risk score.',
    role: 'estimator',
    notes: COX,
  },
  cox.harrellConcordance,
)
fn(
  {
    key: 'aftModel',
    name: 'Accelerated failure time model',
    summary: 'Weibull or log-normal log T = μ + xᵀβ + σε by maximum likelihood with censoring (autodiff, L-BFGS).',
    role: 'fit',
    notes: AFT,
    cite: ['kalbfleisch2002', 'weibull1951'],
  },
  aft.aftModel,
)
fn({ key: 'aftSurvival', name: 'AFT survival curve', role: 'property', notes: AFT }, aft.aftSurvival)
fn({ key: 'timeRatios', name: 'AFT time ratios', role: 'property', notes: AFT }, aft.timeRatios)

/** The functions of the module, keyed by name. */
export const survivalFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', cox, aft) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >

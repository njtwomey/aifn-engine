/**
 * The registry of bijectors (design S §2.5): the fixed monotone maps (exp, log, sigmoid, tanh, softplus, the normal
 * cdf) and the factories that make one from parameters (affine, power, ordered, chains), with their domains and
 * codomains as support names. `Transformed` distributions, constrained parameters and the lab's pickers use this table.
 */

import { definer, entries, type BijectorInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as reparameterise from './reparameterise'
import { real, space } from 'aifn-compute/foundation/space'
import * as maps from './maps'

const bijector = definer<BijectorInfo>('bijector', 'probability/bijectors')
const none = space({})
const notes = ['change-of-variables']
const fixed = { params: none, factory: false, notes } as const

bijector(
  { key: 'expBijector', name: 'exp', tex: 'e^x', domain: 'real', codomain: 'positive', ...fixed },
  maps.expBijector,
)
bijector(
  { key: 'logBijector', name: 'log', tex: '\\log x', domain: 'positive', codomain: 'real', ...fixed },
  maps.logBijector,
)
bijector(
  { key: 'sigmoidBijector', name: 'sigmoid', tex: '\\sigma(x)', domain: 'real', codomain: 'unit-interval', ...fixed },
  maps.sigmoidBijector,
)
bijector(
  { key: 'tanhBijector', name: 'tanh', tex: '\\tanh x', domain: 'real', codomain: 'interval', ...fixed },
  maps.tanhBijector,
)
bijector(
  { key: 'softplusBijector', name: 'softplus', tex: '\\log(1 + e^x)', domain: 'real', codomain: 'positive', ...fixed },
  maps.softplusBijector,
)
bijector(
  {
    key: 'normalCdfBijector',
    name: 'normal cdf',
    tex: '\\Phi(x)',
    domain: 'real',
    codomain: 'unit-interval',
    ...fixed,
  },
  maps.normalCdfBijector,
)
bijector(
  {
    key: 'affineBijector',
    name: 'affine',
    tex: 'a + b x',
    domain: 'real',
    codomain: 'real',
    params: space({
      loc: real(-5, 5, { default: 0, label: 'a', doc: 'shift' }),
      scale: real(-5, 5, { default: 1, label: 'b', doc: 'scale (non-zero; negative reverses the order)' }),
    }),
    factory: true,
    notes,
  },
  maps.affineBijector,
)
bijector(
  {
    key: 'powerBijector',
    name: 'power',
    tex: 'x^p',
    domain: 'positive',
    codomain: 'positive',
    params: space({ p: real(-4, 4, { default: 2, label: 'p', doc: 'exponent (non-zero)' }) }),
    factory: true,
    notes,
  },
  maps.powerBijector,
)
bijector(
  {
    key: 'chainBijectors',
    name: 'chain',
    summary: 'The composition of bijectors, applied left to right.',
    domain: 'varies',
    codomain: 'varies',
    params: none,
    factory: true,
    notes,
  },
  maps.chainBijectors,
)
bijector(
  {
    key: 'orderedBijector',
    name: 'ordered',
    summary: 'Maps ℝᴷ onto increasing vectors: the first entry, then positive gaps (ordinal thresholds).',
    domain: 'real-vector',
    codomain: 'real-vector',
    params: none,
    factory: true,
    notes: ['ordering-constraints-on-ordinal-thresholds'],
  },
  maps.orderedBijector,
)

bijector(
  {
    key: 'affineCouplingBijector',
    name: 'affine coupling',
    summary:
      'Masked coordinates pass through and condition a scale and shift of the rest: a triangular Jacobian (RealNVP, NICE).',
    domain: 'real-vector',
    codomain: 'real-vector',
    params: none,
    factory: true,
    notes: ['normalising-flow', 'change-of-variables'],
    cite: ['dinh2017', 'dinh2015'],
  },
  maps.affineCouplingBijector,
)

/** Every bijector and bijector factory, keyed by export name. */
export const bijectorRegistry: Readonly<Record<string, Entry<object, BijectorInfo>>> = entries<BijectorInfo>(
  'bijector',
  maps,
) as Readonly<Record<string, Entry<object, BijectorInfo>>>

/** The functions of the module, keyed by name. */
export const bijectorFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', {
    transformLogDensity: definer<FunctionInfo>('function', 'probability/bijectors')(
      {
        key: 'transformLogDensity',
        name: 'Transform a log density',
        summary: 'The log density of x = f(z) in z: log p(f(z)) + log |det f′(z)|, so samplers can work unconstrained.',
        role: 'transform',
        notes: ['change-of-variables', 'normalising-flow', 'reparameterisation-trick'],
      },
      reparameterise.transformLogDensity,
    ),
  }) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

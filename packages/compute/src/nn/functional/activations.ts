/**
 * Activation functions, elementwise on numbers, tensors and traced values. `relu` is a primitive defined here; the
 * others are compositions of `aifn-compute/foundation/tensor` and `aifn-compute/numerics/special` primitives (sigmoid, softplus, tanh and Φ come from
 * there), so each exists once and is differentiable wherever its parts are.
 */

import { normalCdf, sigmoid, softplus } from 'aifn-compute/numerics/special'
import {
  add,
  elementwise,
  expm1,
  greater,
  minimum,
  mul,
  pow,
  tanh,
  where,
  type Unary,
  type Value,
} from 'aifn-compute/foundation/tensor'

/**
 * The rectified linear unit max(0, x) (Nair & Hinton, 2010). Its derivative is 1[x > 0], taken as 0 at x = 0 (as in
 * PyTorch); the second derivative is 0 almost everywhere.
 */
export const relu: Unary = elementwise({
  id: 'nn/functional/relu',
  f: (x) => (x > 0 ? x : x === x ? 0 : NaN),
  derivative: [(x) => greater(x, 0)],
  doc: { summary: 'The rectified linear unit max(0, x).', note: 'activation-functions' },
})

/** The leaky ReLU: x for x > 0 and αx otherwise (Maas, Hannun & Ng, 2013). Default α = 0.01. */
export function leakyRelu(x: Value, slope = 0.01): Value {
  return where(greater(x, 0), x, mul(slope, x))
}

/** The exponential linear unit: x for x > 0 and α(eˣ − 1) otherwise (Clevert, Unterthiner & Hochreiter, 2016). */
export function elu(x: Value, alpha = 1): Value {
  // expm1 of min(x, 0): the unused branch must stay finite for large x, or its zero cotangent times e^x is NaN.
  return where(greater(x, 0), x, mul(alpha, expm1(minimum(x, 0))))
}

/**
 * The Gaussian error linear unit x·Φ(x) (Hendrycks & Gimpel, 2016), or with `approximate: 'tanh'` its tanh form
 * ½x(1 + tanh(√(2/π)(x + 0.044715x³))), as in `torch.nn.functional.gelu`.
 */
export function gelu(x: Value, { approximate = 'none' }: { approximate?: 'none' | 'tanh' } = {}): Value {
  if (approximate === 'none') return mul(x, normalCdf(x))
  const inner = mul(Math.sqrt(2 / Math.PI), add(x, mul(0.044715, pow(x, 3))))
  return mul(mul(0.5, x), add(1, tanh(inner)))
}

/** The sigmoid linear unit (SiLU, swish) x·σ(x) (Elfwing, Uchibe & Doya, 2018; Ramachandran et al., 2017). */
export function silu(x: Value): Value {
  return mul(x, sigmoid(x))
}

/** The identity, as an activation (a linear output layer). */
export function identity(x: Value): Value {
  return x
}

/** Every activation by name, for layers configured by name (`mlp(..., { activation: 'gelu' })`). */
export const activations = {
  relu: (x: Value) => relu(x),
  leakyRelu: (x: Value) => leakyRelu(x),
  elu: (x: Value) => elu(x),
  gelu: (x: Value) => gelu(x),
  silu: (x: Value) => silu(x),
  tanh: (x: Value) => tanh(x),
  sigmoid: (x: Value) => sigmoid(x),
  softplus: (x: Value) => softplus(x),
  identity,
} as const satisfies Record<string, (x: Value) => Value>

/** The name of an activation in `activations`. */
export type ActivationName = keyof typeof activations

/** An activation given by name or as a function. */
export type Activation = ActivationName | ((x: Value) => Value)

/** The function of an activation given by name or as a function. */
export function activationFn(a: Activation): (x: Value) => Value {
  return typeof a === 'function' ? a : activations[a]
}

/**
 * Activation functions, elementwise on numbers, tensors and traced values.
 *
 * `relu` is a primitive defined here; the others are compositions of `aifn-compute/foundation/tensor` and
 * `aifn-compute/numerics/special` primitives (`sigmoid`, `softplus`, `tanh` and the standard normal CDF $\Phi$ come
 * from there), so each exists once and is differentiable wherever its parts are. `activations` maps a name to each, for
 * layers configured by name.
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
 * The rectified linear unit $\max(0, x)$ (Nair & Hinton, 2010), a primitive. Its derivative is $\indicator[x > 0]$,
 * taken as 0 at $x = 0$ (as in PyTorch); the second derivative is 0 almost everywhere. NaN stays NaN.
 *
 * @example The negative entries are zeroed, and the gradient is the indicator of $x > 0$
 * const x = tensor([-2, -0.5, 0, 0.5, 2])
 * print('relu(x) =', relu(x))
 * print('d/dx sum(relu(x)) =', grad((v) => sum(relu(v)))(x))
 */
export const relu: Unary = elementwise({
  id: 'nn/functional/relu',
  f: (x) => (x > 0 ? x : x === x ? 0 : NaN),
  derivative: [(x) => greater(x, 0)],
  doc: { summary: 'The rectified linear unit max(0, x).', note: 'activation-functions' },
})

/**
 * The leaky ReLU: $x$ for $x > 0$ and $\alpha x$ otherwise (Maas, Hannun & Ng, 2013). A composition of `where`,
 * so differentiable, with derivative $\alpha$ at $x = 0$.
 *
 * @param x The input, a number or tensor; applied elementwise.
 * @param slope The slope $\alpha$ for $x \le 0$.
 * @returns $x$ where positive and $\alpha x$ elsewhere, with the shape of `x`.
 *
 * @example The default slope, and a steeper one
 * const x = tensor([-2, -1, 0, 1, 2])
 * print('slope 0.01:', leakyRelu(x))
 * print('slope 0.2: ', leakyRelu(x, 0.2))
 */
export function leakyRelu(x: Value, slope = 0.01): Value {
  return where(greater(x, 0), x, mul(slope, x))
}

/**
 * The exponential linear unit: $x$ for $x > 0$ and $\alpha(e^x - 1)$ otherwise (Clevert, Unterthiner & Hochreiter,
 * 2016). Differentiable; the negative branch is evaluated at $\min(x, 0)$, so large $x$ gives no NaN gradient.
 *
 * @param x The input, a number or tensor; applied elementwise.
 * @param alpha The scale $\alpha$ of the negative branch, which tends to $-\alpha$ as $x \to -\infty$.
 * @returns The activation, with the shape of `x`.
 *
 * @example Saturates at $-\alpha$ for large negative inputs
 * const x = tensor([-10, -1, 0, 1])
 * print('alpha 1:', elu(x))
 * print('alpha 2:', elu(x, 2))
 */
export function elu(x: Value, alpha = 1): Value {
  // expm1 of min(x, 0): the unused branch must stay finite for large x, or its zero cotangent times e^x is NaN.
  return where(greater(x, 0), x, mul(alpha, expm1(minimum(x, 0))))
}

/**
 * The Gaussian error linear unit $x\,\Phi(x)$ (Hendrycks & Gimpel, 2016), with $\Phi$ the standard normal CDF, or
 * with `approximate: 'tanh'` its tanh form $\tfrac{1}{2}x\big(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\,x^3))\big)$,
 * as in `torch.nn.functional.gelu`. Differentiable.
 *
 * @param x The input, a number or tensor; applied elementwise.
 * @param options How to evaluate it.
 * @param options.approximate `'none'` for the exact $x\,\Phi(x)$; `'tanh'` for the tanh approximation.
 * @returns The activation, with the shape of `x`.
 *
 * @example The exact form and the tanh approximation agree to about $10^{-3}$
 * const x = tensor([-2, -1, 0, 1, 2])
 * print('exact:', gelu(x))
 * print('tanh: ', gelu(x, { approximate: 'tanh' }))
 */
export function gelu(x: Value, { approximate = 'none' }: { approximate?: 'none' | 'tanh' } = {}): Value {
  if (approximate === 'none') return mul(x, normalCdf(x))
  const inner = mul(Math.sqrt(2 / Math.PI), add(x, mul(0.044715, pow(x, 3))))
  return mul(mul(0.5, x), add(1, tanh(inner)))
}

/**
 * The sigmoid linear unit (SiLU, swish) $x\,\sigma(x)$, with $\sigma$ the logistic sigmoid (Elfwing, Uchibe & Doya,
 * 2018; Ramachandran et al., 2017). Differentiable.
 *
 * @param x The input, a number or tensor; applied elementwise.
 * @returns $x\,\sigma(x)$, with the shape of `x`.
 *
 * @example Close to $x$ for large $x$, and to 0 for large negative $x$
 * print('silu:', silu(tensor([-6, -1, 0, 1, 6])))
 */
export function silu(x: Value): Value {
  return mul(x, sigmoid(x))
}

/**
 * The identity, as an activation (a linear output layer).
 *
 * @param x The input, returned as it is.
 * @returns `x` itself.
 *
 * @example The activation of a linear output layer
 * print('identity:', identity(tensor([-1, 0, 2])))
 */
export function identity(x: Value): Value {
  return x
}

/**
 * Every activation by name, for layers configured by name (`mlp(..., { activation: 'gelu' })`). Each takes the
 * function's defaults (`leakyRelu` with slope 0.01, `elu` with $\alpha = 1$, the exact `gelu`). Exported by the
 * module as `activationFunctions`.
 */
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

/**
 * The function of an activation given by name or as a function.
 *
 * @param a A name in `activations`, or a function of one value, which is returned as it is.
 * @returns The activation as a function of one value.
 *
 * @example By name or as a function
 * const x = tensor([-1, 0, 1])
 * print('by name:', activationFn('relu')(x))
 * print('as a function:', activationFn((v) => mul(2, v))(x))
 */
export function activationFn(a: Activation): (x: Value) => Value {
  return typeof a === 'function' ? a : activations[a]
}

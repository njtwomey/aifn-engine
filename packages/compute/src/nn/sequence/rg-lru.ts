/**
 * The real-gated linear recurrent unit (RG-LRU) of Griffin and Hawk: a diagonal linear recurrence whose decay depends
 * on the input through a recurrence gate, $a_t = a^{c\,r_t}$ with $a = \sigma(\Lambda)$ per channel, and whose input is
 * scaled by $\sqrt{1 - a_t^2}$ so the state keeps the scale of the input whatever the memory length. Neither gate reads
 * the state, so every state is computed at once by `linearRecurrence` (an associative scan).
 *
 * $$
 * \rvec_t = \sigma(\xvec_t \Wmat_a + \bvec_a), \quad \ivec_t = \sigma(\xvec_t \Wmat_x + \bvec_x), \quad
 * \log \avec_t = -c\, \rvec_t \odot \operatorname{softplus}(-\Lambdab),
 * $$
 * $$
 * \hvec_t = \avec_t \odot \hvec_{t-1} + \sqrt{1 - \avec_t^2} \odot (\ivec_t \odot \xvec_t).
 * $$
 *
 * Source: De et al. (2024), "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language
 * Models", §2.4 (arXiv:2402.19427). Differences from the paper: inputs are one sequence, time first ($[T, D]$) with
 * optional leading batch axes handled by broadcasting in the matrix products; the paper's block (convolution, gating
 * branch, projections) is not included. `log a_t` is computed in log space and $\sqrt{1 - a_t^2}$ as
 * $\sqrt{-\mathrm{expm1}(2 \log a_t)}$, as the paper's implementation does, so decays near 1 keep their precision.
 */

import { add, exp, expm1, log, matmul, mul, neg, sqrt, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { logit, sigmoid, softplus } from 'aifn-compute/numerics/special'
import { linearRecurrence } from './ssm'

/** Parameters of `rgLru` for $D$ channels. */
export type RgLruParams = {
  /** Recurrence-gate weights $\Wmat_a$, $D \times D$. */
  Wa: Value
  /** Recurrence-gate bias $\bvec_a$, $D$ values. */
  ba: Value
  /** Input-gate weights $\Wmat_x$, $D \times D$. */
  Wx: Value
  /** Input-gate bias $\bvec_x$, $D$ values. */
  bx: Value
  /** $\Lambda$, $D$ values: the base decay of each channel is $a = \sigma(\Lambda)$ (see `rgLruLambda`). */
  Lambda: Value
}

/** Options of `rgLru`. */
export type RgLruOptions = {
  /** The constant $c$ of $a_t = a^{c r_t}$ (default 8, Griffin's value). */
  c?: number
  /** The state before the first token, $D$ values (default zeros). */
  initial?: Value
}

/** Every intermediate of `rgLru`, time first: each is $[T, D]$. */
export type RgLruResult = {
  /** The states $\hvec_t$, which are the outputs. */
  outputs: Value
  /** The decays $a_t \in (0, 1)$. */
  decay: Value
  /** The recurrence gate $r_t \in (0, 1)$. */
  recurrenceGate: Value
  /** The input gate $i_t \in (0, 1)$. */
  inputGate: Value
  /** The input weight $\sqrt{1 - a_t^2}$. */
  inputScale: Value
}

/**
 * The RG-LRU over inputs $\xvec_t$, $[T, D]$ (time first): the gates from the inputs, then every state at once by
 * `linearRecurrence`. Differentiable in `x` and every parameter.
 *
 * @param x The inputs $\xvec_t$, $T \times D$ (leading batch axes broadcast through the matrix products).
 * @param p The gate weights and biases and $\Lambda$.
 * @param options The constant $c$ and the initial state.
 * @returns The states (the outputs) and every gate, each $T \times D$.
 *
 * @example A closed recurrence gate keeps the memory; an open one lets it decay to the base rate
 * const Lambda = rgLruLambda(tensor([0.25]))
 * const p = { Wa: zeros([1, 1]), ba: tensor([-3]), Wx: zeros([1, 1]), bx: tensor([3]), Lambda }
 * const { outputs, decay } = rgLru(tensor([[2], [0], [0]]), p)
 * print('a_t (gate nearly closed) =', decay)
 * print('h =', outputs)
 * const open = rgLru(tensor([[2], [0], [0]]), { ...p, ba: tensor([3]) })
 * print('a_t (gate nearly open) =', open.decay)
 * print('h =', open.outputs)
 */
export function rgLru(x: Value, p: RgLruParams, options: RgLruOptions = {}): RgLruResult {
  const c = options.c ?? 8
  const r = sigmoid(add(matmul(x, p.Wa), p.ba))
  const i = sigmoid(add(matmul(x, p.Wx), p.bx))
  // log a = log σ(Λ) = −softplus(−Λ); log a_t = c r_t log a.
  const logDecay = mul(mul(-c, r), softplus(neg(p.Lambda)))
  const decay = exp(logDecay)
  const inputScale = sqrt(neg(expm1(mul(2, logDecay))))
  const b = mul(inputScale, mul(i, x))
  const outputs = linearRecurrence(decay, b, options.initial === undefined ? {} : { initial: options.initial })
  return { outputs, decay, recurrenceGate: r, inputGate: i, inputScale }
}

/**
 * The $\Lambda$ that gives each channel the decay `decayAtC` $= a^c$ when the recurrence gate is fully open
 * ($r_t = 1$): $\Lambda = \operatorname{logit}((a^c)^{1/c})$. Griffin draws $a^c$ uniformly in $[0.9, 0.999]$.
 *
 * @param decayAtC The decays $a^c$, one per channel, in $(0, 1)$.
 * @param c The constant $c$ (default 8).
 * @returns $\Lambda$, one value per channel.
 *
 * @example The base decay raised to c returns the decay asked for
 * const Lambda = rgLruLambda(tensor([0.9, 0.999]))
 * print('Lambda =', Lambda)
 * print('sigmoid(Lambda)^8 =', pow(div(1, add(1, exp(neg(Lambda)))), 8))
 */
export function rgLruLambda(decayAtC: Value, c = 8): Tensor {
  return logit(exp(mul(1 / c, log(decayAtC)))) as Tensor
}

/**
 * Pieces shared by the policy-gradient agents: networks as parameter pytrees built from their sizes (the multilayer
 * perceptron of `../dqn`), one Adam step on a loss of the parameters, a categorical policy's log-probabilities, and
 * small array helpers. Everything is plain data, so agent states checkpoint and replay exactly.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { fromData, mul, sum, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { ActivationName } from 'aifn-compute/nn/functional'
import type { Layer } from 'aifn-compute/nn/layers'
import { adamRule, applyUpdates, chainRules, clipByGlobalNorm, type UpdateRule } from 'aifn-compute/optim/first-order'
import { logSoftmax, softmax } from 'aifn-compute/numerics/special'
import { qNetwork } from '../dqn'

/**
 * Networks by their sizes, built once per agent (a network is code, its parameters are state).
 *
 * @param activation The hidden activation of every network the cache builds.
 * @returns A lookup from layer sizes (input, hidden widths, outputs) to the `qNetwork` of those sizes, built on first
 *   use and reused after.
 */
export function networkCache(activation: ActivationName): (sizes: readonly number[]) => Layer<Params[]> {
  const nets = new Map<string, Layer<Params[]>>()
  return (sizes) => {
    const key = sizes.join(',')
    let net = nets.get(key)
    if (!net) nets.set(key, (net = qNetwork(sizes, activation)))
    return net
  }
}

/**
 * Adam with optional global-norm clipping.
 *
 * @param stepSize Adam's step size.
 * @param clipNorm The global gradient norm above which gradients are rescaled to it; `Infinity` for no clipping.
 * @returns The update rule: clipping chained before Adam, or Adam alone.
 */
export function adamWithClip(stepSize: number, clipNorm = Infinity): UpdateRule<unknown> {
  const adam = adamRule({ stepSize })
  return (Number.isFinite(clipNorm) ? chainRules(clipByGlobalNorm(clipNorm), adam) : adam) as UpdateRule<unknown>
}

/**
 * The number in a rank-0 value.
 *
 * @param v A number, a scalar tensor or a traced scalar (unwrapped first).
 * @returns Its value; the first entry for a tensor.
 */
export const scalarOf = (v: Value): number => {
  const u = unwrap(v)
  return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
}

/**
 * One update of `params` on `loss` by `rule`: the new parameters, optimiser state and the loss before the step.
 *
 * @param rule The update rule, such as `adamWithClip` returns.
 * @param params The parameters; not modified.
 * @param optimizer The rule's state for these parameters.
 * @param loss The scalar loss of the parameters, differentiated by `valueAndGrad`.
 * @returns The updated parameters and optimiser state, and the loss at `params`.
 */
export function gradientStep(
  rule: UpdateRule<unknown>,
  params: Params[],
  optimizer: unknown,
  loss: (p: Params[]) => Value,
): { params: Params[]; optimizer: unknown; loss: number } {
  const { value, grad } = valueAndGrad(loss, {})(params)
  const step = rule.update(grad as Params, optimizer as never)
  return { params: applyUpdates(params, step.updates), optimizer: step.state, loss: scalarOf(value) }
}

/**
 * A batch of rows as a tensor $n \times d$.
 *
 * @param data The rows, $nd$ numbers row-major; copied.
 * @param n The number of rows.
 * @param dim The row length $d$.
 * @returns The $n \times d$ tensor.
 */
export const rows = (data: ArrayLike<number>, n: number, dim: number): Tensor =>
  fromData(Float64Array.from(data), [n, dim])

/**
 * $\pi(\cdot \mid o)$ of a categorical policy network at one observation: the softmax of its logits.
 *
 * @param net The policy network, whose outputs are the logits.
 * @param params Its parameters.
 * @param obs The observation, $d$ numbers.
 * @returns One probability per action.
 */
export function actionProbabilities(net: Layer<Params[]>, params: Params[], obs: ArrayLike<number>): Float64Array {
  const logits = net.apply(params, fromData(Float64Array.from(obs), [1, obs.length]))
  return Float64Array.from(toFlat(unwrap(softmax(logits)) as Tensor))
}

/**
 * $\log \pi(a \mid o)$ for a batch, as a traced value of length $n$: the sum over actions of the one-hot mask times
 * $\log \operatorname{softmax}(\text{logits})$.
 *
 * @param logits The policy's logits, $n \times A$ (possibly traced).
 * @param mask The taken actions one-hot, $n \times A$.
 * @returns The log-probability of each row's taken action.
 */
export function logProbabilities(logits: Value, mask: Tensor): Value {
  return sum(mul(logSoftmax(logits), mask), 1)
}

/**
 * The entropy $-\sum \pi \log \pi$ of each row of logits, as a traced value of length $n$.
 *
 * @param logits The policy's logits, $n \times A$ (possibly traced).
 * @returns The entropy of each row's softmax, in nats.
 */
export function entropies(logits: Value): Value {
  const logp = logSoftmax(logits)
  return mul(-1, sum(mul(softmax(logits), logp), 1))
}

/**
 * The index of the largest entry (the first on ties).
 *
 * @param v The values.
 * @returns The index of the largest.
 */
export function argmax(v: ArrayLike<number>): number {
  let best = 0
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i
  return best
}

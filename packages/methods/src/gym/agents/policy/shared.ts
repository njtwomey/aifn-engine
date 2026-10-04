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

/** Networks by their sizes, built once per agent (a network is code, its parameters are state). */
export function networkCache(activation: ActivationName): (sizes: readonly number[]) => Layer<Params[]> {
  const nets = new Map<string, Layer<Params[]>>()
  return (sizes) => {
    const key = sizes.join(',')
    let net = nets.get(key)
    if (!net) nets.set(key, (net = qNetwork(sizes, activation)))
    return net
  }
}

/** Adam with optional global-norm clipping. */
export function adamWithClip(stepSize: number, clipNorm = Infinity): UpdateRule<unknown> {
  const adam = adamRule({ stepSize })
  return (Number.isFinite(clipNorm) ? chainRules(clipByGlobalNorm(clipNorm), adam) : adam) as UpdateRule<unknown>
}

/** The number in a rank-0 value. */
export const scalarOf = (v: Value): number => {
  const u = unwrap(v)
  return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
}

/** One update of `params` on `loss` by `rule`: the new parameters, optimiser state and the loss before the step. */
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

/** A batch of rows as a tensor [n, dim]. */
export const rows = (data: ArrayLike<number>, n: number, dim: number): Tensor =>
  fromData(Float64Array.from(data), [n, dim])

/** π(·|o) of a categorical policy network at one observation. */
export function actionProbabilities(net: Layer<Params[]>, params: Params[], obs: ArrayLike<number>): Float64Array {
  const logits = net.apply(params, fromData(Float64Array.from(obs), [1, obs.length]))
  return Float64Array.from(toFlat(unwrap(softmax(logits)) as Tensor))
}

/** log π(a|o) for a batch, as a traced [n] value: Σ_k onehot · log softmax(logits). */
export function logProbabilities(logits: Value, mask: Tensor): Value {
  return sum(mul(logSoftmax(logits), mask), 1)
}

/** The entropy −Σ π log π of each row of logits, as a traced [n] value. */
export function entropies(logits: Value): Value {
  const logp = logSoftmax(logits)
  return mul(-1, sum(mul(softmax(logits), logp), 1))
}

/** The index of the largest entry (the first on ties). */
export function argmax(v: ArrayLike<number>): number {
  let best = 0
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i
  return best
}

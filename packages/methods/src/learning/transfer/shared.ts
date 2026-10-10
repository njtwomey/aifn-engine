/** Internal helper of the transfer runs: an Adam trainer over any parameter pytree, holding its optimiser state. */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import type { Value } from 'aifn-compute/foundation/tensor'
import { adamRule, applyUpdates, chainRules, clipByGlobalNorm, type UpdateRule } from 'aifn-compute/optim/first-order'

/**
 * Adam with gradients clipped to a global norm, over parameters of any pytree shape. The optimiser state lives in the
 * closure, so one trainer serves one sequence of updates.
 *
 * @param stepSize Adam's step size.
 * @param init Parameters of the shape to be trained, from which the optimiser state is initialised.
 * @param clipNorm The largest global norm of a gradient; larger ones are scaled down to it.
 * @returns `step(params, loss)`, which takes the gradient of `loss` at `params` by reverse mode, applies one update
 *   and returns the new parameters.
 */
export function adamTrainer<P>(
  stepSize: number,
  init: P,
  clipNorm = 10,
): { step: (params: P, loss: (p: P) => Value) => P } {
  const rule = chainRules(clipByGlobalNorm(clipNorm), adamRule({ stepSize })) as UpdateRule<unknown>
  let optimizer = rule.init(init as Params)
  return {
    step(params, loss) {
      const { grad } = valueAndGrad((p: P) => loss(p), {})(params)
      const s = rule.update(grad as Params, optimizer as never)
      optimizer = s.state
      return applyUpdates(params as Params, s.updates) as P
    },
  }
}

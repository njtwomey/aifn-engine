/** Shared by the transfer modules: an Adam trainer over any parameter pytree, holding its optimiser state. */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import type { Value } from 'aifn-compute/foundation/tensor'
import { adamRule, applyUpdates, chainRules, clipByGlobalNorm, type UpdateRule } from 'aifn-compute/optim/first-order'

/** Adam (with gradients clipped to a global norm) over parameters P; `step` applies one update on a loss of P. */
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

/**
 * The auxiliary losses that keep a mixture of experts' router healthy, and plain statistics of a routing.
 *
 * Without them a router trained only on the task loss tends to collapse: the experts it picks early improve, so it
 * picks them more, and the rest receive no tokens and no gradient (Shazeer et al., 2017, §4; Fedus, Zoph and Shazeer,
 * 2022, §2.2). The losses take a `Routing` from `route` and return scalars to add to the task loss, differentiable
 * through the router; the statistics are plain numbers.
 */

import {
  div,
  fromData,
  logsumexp,
  mean,
  mul,
  square,
  sub,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Routing } from './routing'

/**
 * The load-balancing loss of the Switch Transformer (Fedus, Zoph and Shazeer, 2022, eq. 4–6), $N \sum_i f_i P_i$,
 * with $f_i$ the fraction of the gate's assignments that chose expert $i$ (constant; before capacity, summing to 1 for
 * any $k$) and $P_i$ the mean router probability of expert $i$ over the tokens (differentiable). It is 1 when both are
 * uniform and $N$ when one expert takes every token with probability 1. For top-k this is the Switch loss with $f_i$
 * divided by $k$ (Mixtral's implementation leaves the $k$ in, so its value at balance is $k$). Dense softmax gating
 * chooses every expert, so the loss is the constant 1 there: use `importanceLoss`.
 *
 * @param routing The routing, as `route` returns it; its `selected` mask and `probs` are read.
 * @returns The loss, a scalar, differentiable through the router's probabilities.
 *
 * @example Two tokens sent to two experts, then both to one: the loss rises towards N = 2
 * const balanced = route(tensor([[5, 0], [0, 5]]), { gate: 'switch' })
 * const collapsed = route(tensor([[5, 0], [5, 0]]), { gate: 'switch' })
 * print('balanced:', loadBalancingLoss(balanced))
 * print('collapsed:', loadBalancingLoss(collapsed))
 */
export function loadBalancingLoss(routing: Routing): Value {
  const { experts: N } = routing
  const counts = toFlat(sum(routing.selected, 0) as Tensor)
  const total = counts.reduce((a, b) => a + b, 0)
  const f = fromData(
    Float64Array.from(counts, (c) => (total > 0 ? c / total : 0)),
    [N],
  )
  const P = mean(routing.probs, 0)
  return mul(N, sum(mul(f, P)))
}

/**
 * The importance loss of Shazeer et al. (2017, eq. 6): the squared coefficient of variation
 * $\mathrm{CV}^2 = \var / \mathrm{mean}^2$ of the experts' importances $I_i = \sum_t w_{ti}$ (the combine weights;
 * population variance). 0 when every expert carries the same total weight. Differentiable for every gate, dense ones
 * included.
 *
 * @param routing The routing, as `route` returns it; its `combine` weights are read.
 * @returns The loss, a scalar.
 *
 * @example Equal importances give 0; one expert carrying everything of two gives 1
 * const balanced = route(tensor([[5, 0], [0, 5]]), { gate: 'switch' })
 * const collapsed = route(tensor([[5, 0], [5, 0]]), { gate: 'switch' })
 * print('balanced:', importanceLoss(balanced))
 * print('collapsed:', importanceLoss(collapsed))
 */
export function importanceLoss(routing: Routing): Value {
  const importance = sum(routing.combine, 0)
  const m = mean(importance)
  return div(mean(square(sub(importance, m))), square(m))
}

/**
 * The router z-loss of ST-MoE (Zoph et al., 2022, "ST-MoE: designing stable and transferable sparse expert models",
 * eq. 5): the mean over tokens of $(\log \sum_i e^{z_{ti}})^2$, which keeps the router's logits small so its softmax
 * stays in a well-conditioned range.
 *
 * @param routingOrLogits A routing, whose raw `logits` (before temperature and noise) are read, or logits `[T, N]`.
 * @returns The loss, a scalar, differentiable in the logits.
 *
 * @example Zero logits cost (log 2) squared; large ones cost their square
 * print('zero logits:', routerZLoss(tensor([[0, 0], [0, 0]])))
 * print('logits of 10:', routerZLoss(tensor([[10, 0], [0, 10]])))
 */
export function routerZLoss(routingOrLogits: Routing | Value): Value {
  const logits =
    typeof routingOrLogits === 'object' && routingOrLogits !== null && 'combine' in routingOrLogits
      ? routingOrLogits.logits
      : (routingOrLogits as Value)
  return mean(square(logsumexp(logits, -1)))
}

/** Plain numbers describing a routing (primal values; for readouts and training curves). */
export type RoutingStatistics = {
  /** Assignments each expert processed, after capacity. */
  readonly counts: readonly number[]
  /** Each expert's share of the processed assignments (sums to 1; uniform is 1/N). */
  readonly load: readonly number[]
  /** Each expert's mean router probability over the tokens ($P_i$ of the balancing loss). */
  readonly importance: readonly number[]
  /** Mean over tokens of the entropy of the router's probabilities, nats (0: hard; $\log N$: uniform). */
  readonly entropy: number
  /** The entropy of `load`, nats: $\log N$ when balanced, 0 when one expert takes everything. */
  readonly loadEntropy: number
  /** Experts that processed no assignment. */
  readonly idle: number
  /** The share of the gate's assignments dropped for capacity. */
  readonly dropped: number
  /** The share of tokens left with no expert. */
  readonly unrouted: number
}

/**
 * The entropy $-\sum_i p_i \log p_i$ in nats, zero terms skipped.
 *
 * @param p The probabilities.
 * @returns The entropy.
 */
const entropyOf = (p: readonly number[]) => -p.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0)

/**
 * Load per expert, router entropy and dropping of a routing, as plain numbers read from its primal values.
 *
 * @param routing The routing, as `route` returns it.
 * @returns The counts, load, importance, entropies, idle experts, dropped share and unrouted share.
 *
 * @example Four tokens choose expert 0, which has room for two
 * const routing = route(tensor([[3, 0], [2, 0], [1, 0], [4, 0]]), { gate: 'switch', capacityFactor: 1 })
 * print(routingStatistics(routing))
 */
export function routingStatistics(routing: Routing): RoutingStatistics {
  const { tokens: T, experts: N } = routing
  const probs = toFlat(unwrap(routing.probs) as Tensor)
  const sel = toFlat(routing.selected)
  const disp = toFlat(routing.dispatch)
  const counts = new Array<number>(N).fill(0)
  const importance = new Array<number>(N).fill(0)
  let entropy = 0
  let selected = 0
  let processed = 0
  let unrouted = 0
  for (let t = 0; t < T; t++) {
    let any = false
    const row: number[] = []
    for (let i = 0; i < N; i++) {
      const j = t * N + i
      counts[i] += disp[j]
      importance[i] += probs[j] / T
      selected += sel[j]
      processed += disp[j]
      if (disp[j] > 0) any = true
      row.push(probs[j])
    }
    entropy += entropyOf(row) / T
    if (!any) unrouted++
  }
  const load = counts.map((c) => (processed > 0 ? c / processed : 0))
  return {
    counts,
    load,
    importance,
    entropy,
    loadEntropy: entropyOf(load),
    idle: counts.filter((c) => c === 0).length,
    dropped: selected > 0 ? (selected - processed) / selected : 0,
    unrouted: T > 0 ? unrouted / T : 0,
  }
}

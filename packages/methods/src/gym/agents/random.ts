/** The random agent: the baseline every agent beats, on any action domain. */

import type { Agent, AgentInfo, Domain } from 'aifn-compute/foundation/contracts'
import { integers } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { sampleDomain, space } from 'aifn-compute/foundation/space'

/** The random agent's state: the action domain it draws from. */
export interface RandomAgentState {
  /** The environment's action domain, copied at `init`. */
  action: Domain
}

/**
 * An agent acting uniformly at random: over the legal actions when the environment masks some, else over the action
 * domain (an integer of a discrete domain, or a uniform point of a bounded box). It learns nothing. On a discrete
 * domain its decision carries the action probabilities ($1/n$ each, or 1 over the number of legal actions).
 *
 * @returns The agent, named `'random'`; `learn` returns its state unchanged.
 *
 * @example Uniform action counts over three actions
 * const agent = randomAgent()
 * const s = stream(1)
 * const g = agent.init({ action: { kind: 'discrete', n: 3 } }, s)
 * const counts = [0, 0, 0]
 * for (let t = 0; t < 300; t++) counts[agent.act(g, 0, s).action]++
 * print('counts of 300 draws:', counts)
 *
 * @example Only the legal actions are drawn
 * const agent = randomAgent()
 * const s = stream(2)
 * const g = agent.init({ action: { kind: 'discrete', n: 4 } }, s)
 * const d = agent.act(g, 0, s, [1, 3])
 * print('action:', d.action)
 * print('probabilities:', d.probabilities)
 */
export function randomAgent<A = number>(): Agent<RandomAgentState, unknown, A> {
  return {
    name: 'random',
    init: (env) => ({ action: env.action }),
    act: (g, _, stream, legal) => {
      if (g.action.kind !== 'discrete') return { action: sampleDomain(stream, g.action) as A }
      const n = g.action.n
      if (!legal) return { action: sampleDomain(stream, g.action) as A, probabilities: new Float64Array(n).fill(1 / n) }
      const probabilities = new Float64Array(n)
      for (const a of legal as readonly number[]) probabilities[a] = 1 / legal.length
      return { action: legal[integers(stream, legal.length)], probabilities }
    },
    learn: (g) => g,
  }
}

definer<AgentInfo>('agent', 'gym/agents')(
  {
    key: 'randomAgent',
    name: 'Random agent',
    summary: 'Acts uniformly at random over the legal actions or the action domain and learns nothing: the baseline.',
    params: space({}),
    requires: {},
    random: true,
  },
  randomAgent,
)

import { describe, expect, it } from 'vitest'
import { agentRegistry, compatible, environmentRegistry, episodes, rollout, validPairs } from 'aifn-methods/gym'
import type { Agent, Environment } from 'aifn-compute/foundation/contracts'
import { stream } from 'aifn-compute/foundation/random'
import { defaults, domainContains } from 'aifn-compute/foundation/space'
import { run, trace } from 'aifn-compute/foundation/trace'
import { expectInfo } from '../registry'

type AnyEnv = Environment<unknown, unknown, unknown>
type AnyAgent = Agent<unknown, unknown, unknown>

const makeEnv = (key: string) => {
  const e = environmentRegistry[key]
  return (e as unknown as (p: object) => AnyEnv)(defaults(e.info.params))
}
const makeAgent = (key: string) => {
  const a = agentRegistry[key]
  return (a as unknown as (p: object) => AnyAgent)(defaults(a.info.params))
}

describe('gym registries', () => {
  it('have well-formed metadata', () => {
    expectInfo(environmentRegistry, 'environment')
    expectInfo(agentRegistry, 'agent')
  })

  it('every environment builds at its defaults and declares what it is', () => {
    for (const key of Object.keys(environmentRegistry)) {
      const env = makeEnv(key)
      const info = environmentRegistry[key].info
      expect(env.observation.kind, key).toBe(info.observation)
      expect(env.action.kind, key).toBe(info.action)
      for (const c of info.capabilities ?? []) expect(env[c], `${key} ${c}`).toBeDefined()
      const { state, observation } = env.reset(stream(0))
      expect(domainContains(env.observation, observation), key).toBe(true)
      expect(state, key).toBeDefined()
    }
  })

  it('pairs every environment with at least one agent besides the random one', () => {
    const pairs = validPairs()
    for (const key of Object.keys(environmentRegistry))
      expect(
        pairs.some((p) => p.environment === key && p.agent !== 'randomAgent'),
        key,
      ).toBe(true)
    // A tabular learner never meets a continuous observation; a planner only a tabular model.
    for (const p of pairs) {
      const e = environmentRegistry[p.environment].info
      const a = agentRegistry[p.agent].info
      expect(compatible(e, a)).toBe(true)
      if (a.requires.model) expect(e.capabilities).toContain('model')
    }
  })
})

// Generated: every valid environment × agent pair from the registries runs a few episodes at their defaults.
describe.each(validPairs())('$agent in $environment', ({ environment, agent }) => {
  it('runs a few episodes, acting within the action domain', () => {
    const env = makeEnv(environment)
    const ag = makeAgent(agent)
    const horizon = Math.min(env.horizon, 200)
    const capped = { ...env, horizon } as AnyEnv
    const tr = trace(rollout(capped, ag, { episodes: 3 }), undefined, 3 * horizon, { stream: stream(environment) })
    expect(tr.final.episode).toBe(3)
    expect(Number.isFinite(tr.final.totalReward)).toBe(true)
    const last = tr.final.last!
    expect(domainContains(env.action, last.action)).toBe(true)
    const byEpisode = run(episodes(capped, ag), undefined, 2, { stream: stream(agent) })
    expect(byEpisode.t).toBe(2)
  })
})

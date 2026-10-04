import { describe, expect, it } from 'vitest'
import { pendulumLqr, swingUpAgent } from 'aifn-methods/gym/agents/control'
import { pendulumEnvironment, wrapAngle } from 'aifn-methods/gym/environments/control'
import { episodes, rollout } from 'aifn-methods/gym'
import { stream } from 'aifn-compute/foundation/random'
import { toRows } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../../protocol'

const hanging = { theta: Math.PI, thetaDot: 0 }

/** The last `n` states of one episode from `start`. */
function finalStates(env: ReturnType<typeof pendulumEnvironment>, agent: ReturnType<typeof swingUpAgent>) {
  const tr = run(rollout(env, agent), undefined, env.horizon, { stream: stream(1) })
  return tr
}

describe('the LQR controller', () => {
  it('linearises the RK4 step about the upright by autodiff', () => {
    const env = pendulumEnvironment()
    const { A, B, K } = pendulumLqr(env)
    const h = 0.05
    // To first order in h: A ≈ I + h [[0, 1], [15, 0]], B ≈ h [0, 3].
    const a = toRows(A)
    expect(a[0][1]).toBeCloseTo(h, 2)
    expect(a[1][0]).toBeCloseTo(15 * h, 1)
    expect(toRows(B)[1][0]).toBeCloseTo(3 * h, 1)
    expect(K[0]).toBeGreaterThan(0)
    expect(K[1]).toBeGreaterThan(0)
  })

  it('balances from near the top, and fails alone from hanging', () => {
    const near = pendulumEnvironment({ start: { theta: 0.3, thetaDot: 0 } })
    const end = finalStates(near, swingUpAgent({ swingUp: false }))
    expect(Math.abs(wrapAngle(end.envState.theta))).toBeLessThan(1e-3)
    const low = pendulumEnvironment({ start: hanging })
    const stuck = finalStates(low, swingUpAgent({ swingUp: false }))
    expect(Math.abs(wrapAngle(stuck.envState.theta))).toBeGreaterThan(1)
  })
})

describe('the energy swing-up with LQR hand-over', () => {
  it('swings up from hanging and holds upright within the horizon, box or discrete torques', () => {
    for (const torques of [0, 5]) {
      const env = pendulumEnvironment({ start: hanging, torques })
      const s = run(episodes(env, swingUpAgent()), undefined, 1, { stream: stream(2) })
      const obs = s.observations
      const last = obs.slice(-40)
      for (const o of last) expect(Math.abs(Math.atan2(o[1], o[0]))).toBeLessThan(0.05)
      expect(s.episodeReturn).toBeGreaterThan(-1000)
    }
  })

  it('is a pure rollout', () => {
    const env = pendulumEnvironment({ start: hanging })
    expectProtocol(rollout(env, swingUpAgent()), undefined, { n: 30 })
  })
})

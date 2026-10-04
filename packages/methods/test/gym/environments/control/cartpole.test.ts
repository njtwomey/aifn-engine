import { describe, expect, it } from 'vitest'
import { cartPoleEnvironment } from 'aifn-methods/gym/environments/control'
import { rollout } from 'aifn-methods/gym'
import { lqrBangBangAgent, randomAgent } from 'aifn-methods/gym/agents'
import { jacobian } from 'aifn-compute/foundation/autodiff'
import { domainContains } from 'aifn-compute/foundation/space'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../../protocol'

describe('the cart-pole environment', () => {
  it("declares Gymnasium's domains, a dynamics model and a cart-pole render", () => {
    const env = cartPoleEnvironment()
    expect(env.observation).toMatchObject({ kind: 'box', shape: [4] })
    expect(env.action).toEqual({ kind: 'discrete', n: 2, names: ['push left', 'push right'] })
    expect(env.horizon).toBe(500)
    expect(env.model.kind).toBe('dynamics')
    expect(env.render).toMatchObject({ kind: 'cartpole', poleLength: 1, trackLimit: 2.4 })
  })

  it('takes one Euler step of the hand-computed dynamics', () => {
    const env = cartPoleEnvironment()
    const s = { x: 0.1, xDot: -0.2, theta: 0.05, thetaDot: 0.3 }
    // Gymnasium's equations with g = 9.8, m_c = 1, m_p = 0.1, l = 0.5, F = +10, τ = 0.02.
    const [M, pml, l, F, tau] = [1.1, 0.05, 0.5, 10, 0.02]
    const q = (F + pml * s.thetaDot ** 2 * Math.sin(s.theta)) / M
    const thAcc = (9.8 * Math.sin(s.theta) - Math.cos(s.theta) * q) / (l * (4 / 3 - (0.1 * Math.cos(s.theta) ** 2) / M))
    const xAcc = q - (pml * thAcc * Math.cos(s.theta)) / M
    const step = env.step(s, 1, stream(0))
    expect(step.state.x).toBeCloseTo(s.x + tau * s.xDot, 14)
    expect(step.state.xDot).toBeCloseTo(s.xDot + tau * xAcc, 14)
    expect(step.state.theta).toBeCloseTo(s.theta + tau * s.thetaDot, 14)
    expect(step.state.thetaDot).toBeCloseTo(s.thetaDot + tau * thAcc, 14)
    expect(step).toMatchObject({ reward: 1, terminated: false, truncated: false })
    // The model is the same step, differentiable: ∂θ′/∂θ̇ = τ.
    const J = toRows(
      jacobian((v: Value) => env.model.transition(v, fromData(Float64Array.of(10), [1])))(
        env.model.encode(s),
      ) as Tensor,
    )
    expect(J[2][3]).toBeCloseTo(tau, 12)
    expect(
      Array.from((env.model.transition(env.model.encode(s), fromData(Float64Array.of(10), [1])) as Tensor).data),
    ).toEqual([step.state.x, step.state.xDot, step.state.theta, step.state.thetaDot])
  })

  it('terminates past 12° or 2.4 m, and the rollout truncates at 500 steps', () => {
    const env = cartPoleEnvironment()
    expect(env.step({ x: 0, xDot: 0, theta: 0.2, thetaDot: 1 }, 0, stream(0)).terminated).toBe(true)
    expect(env.step({ x: 2.39, xDot: 1, theta: 0, thetaDot: 0 }, 1, stream(0)).terminated).toBe(true)
    expect(env.step({ x: 0, xDot: 0, theta: 0.1, thetaDot: 0 }, 0, stream(0)).terminated).toBe(false)
    // A policy that holds the pole up is cut at the horizon: truncated, not terminated, return 500.
    const s = run(rollout(env, lqrBangBangAgent()), undefined, 500, { stream: stream(1) })
    expect(s.ended && s.last?.truncated && !s.last.terminated).toBe(true)
    expect(s.episodeReturn).toBe(500)
  })

  it('resets within the jitter and runs pure rollouts with the random agent', () => {
    const env = cartPoleEnvironment({ jitter: 0.1 })
    const { observation } = env.reset(stream(4))
    expect(Array.from(observation).every((v) => Math.abs(v) <= 0.1)).toBe(true)
    expect(domainContains(env.observation, observation)).toBe(true)
    expectProtocol(rollout(env, randomAgent()), undefined, { n: 60 })
  })
})

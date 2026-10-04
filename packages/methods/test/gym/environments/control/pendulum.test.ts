import { describe, expect, it } from 'vitest'
import { pendulumEnergy, pendulumEnvironment, wrapAngle } from 'aifn-methods/gym/environments/control'
import { rollout } from 'aifn-methods/gym'
import { randomAgent } from 'aifn-methods/gym/agents'
import { grad } from 'aifn-compute/foundation/autodiff'
import { domainContains } from 'aifn-compute/foundation/space'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, get, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { expectProtocol } from '../../../protocol'

describe('the pendulum environment', () => {
  it("declares Gymnasium's domains, a dynamics model and a pendulum render", () => {
    const env = pendulumEnvironment()
    expect(env.observation).toMatchObject({ kind: 'box', low: [-1, -1, -8], high: [1, 1, 8], shape: [3] })
    expect(env.action).toMatchObject({ kind: 'box', low: [-2], high: [2], shape: [1] })
    expect(env.horizon).toBe(200)
    expect(env.model.kind).toBe('dynamics')
    expect(env.render.kind).toBe('pendulum')
    const discrete = pendulumEnvironment({ torques: 5 })
    expect(discrete.action).toMatchObject({ kind: 'discrete', n: 5 })
    expect(discrete.parameters.levels).toEqual([-2, -1, 0, 1, 2])
  })

  it('takes one RK4 step of the hand-computed dynamics and pays the reward at the start state', () => {
    const env = pendulumEnvironment()
    const [a, b, h] = [15, 3, 0.05]
    const theta = 0.3
    const w = -0.5
    const u = 1.5
    const f = (x: number, v: number) => [v, a * Math.sin(x) + b * u]
    const k1 = f(theta, w)
    const k2 = f(theta + (h / 2) * k1[0], w + (h / 2) * k1[1])
    const k3 = f(theta + (h / 2) * k2[0], w + (h / 2) * k2[1])
    const k4 = f(theta + h * k3[0], w + h * k3[1])
    const expected = [0, 1].map((i) => [theta, w][i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]))
    const step = env.step({ theta, thetaDot: w }, Float64Array.of(u), stream(0))
    expect(step.state.theta).toBeCloseTo(expected[0], 12)
    expect(step.state.thetaDot).toBeCloseTo(expected[1], 12)
    expect(step.reward).toBeCloseTo(-(theta ** 2 + 0.1 * w ** 2 + 0.001 * u ** 2), 12)
    expect(step.observation).toEqual(
      Float64Array.of(Math.cos(step.state.theta), Math.sin(step.state.theta), step.state.thetaDot),
    )
    expect(step.terminated || step.truncated).toBe(false)
  })

  it('clips the torque and the speed, and wraps the angle in the reward', () => {
    const env = pendulumEnvironment()
    const s = { theta: 2 * Math.PI + 0.1, thetaDot: 7.9 }
    const big = env.step(s, Float64Array.of(50), stream(0))
    const capped = env.step(s, Float64Array.of(2), stream(0))
    expect(big.state).toEqual(capped.state)
    expect(big.state.thetaDot).toBe(8)
    expect(big.reward).toBeCloseTo(-(0.1 ** 2 + 0.1 * 7.9 ** 2 + 0.001 * 4), 12)
    expect(wrapAngle(3 * Math.PI)).toBeCloseTo(-Math.PI, 12)
  })

  it('conserves energy without torque or damping, to the accuracy of RK4', () => {
    const env = pendulumEnvironment({ start: { theta: Math.PI - 0.5, thetaDot: 0 } })
    let s = env.reset(stream(0)).state
    const e0 = pendulumEnergy(env.parameters, s)
    let worst = 0
    for (let k = 0; k < 400; k++) {
      s = env.step(s, Float64Array.of(0), stream(k)).state
      worst = Math.max(worst, Math.abs(pendulumEnergy(env.parameters, s) - e0))
    }
    // RK4's local error is O(h⁵): about 1e-4 of e0 over 20 s at h = 0.05; halving h cuts it about 16-fold.
    expect(worst).toBeLessThan(1e-3 * Math.abs(e0))
    const fine = pendulumEnvironment({ dt: 0.025, start: { theta: Math.PI - 0.5, thetaDot: 0 } })
    let t = fine.reset(stream(0)).state
    let worstFine = 0
    for (let k = 0; k < 800; k++) {
      t = fine.step(t, Float64Array.of(0), stream(k)).state
      worstFine = Math.max(worstFine, Math.abs(pendulumEnergy(fine.parameters, t) - e0))
    }
    expect(worstFine).toBeLessThan(worst / 8)
  })

  it('has a differentiable model: the gradient of the reward and of the next state', () => {
    const env = pendulumEnvironment()
    const x = fromData(Float64Array.of(0.3, -0.5), [2])
    const u = fromData(Float64Array.of(0.4), [1])
    const dr = grad((v: Value) => env.model.reward(v, u))(x) as Tensor
    expect(Array.from(dr.data)).toEqual([expect.closeTo(-0.6, 12), expect.closeTo(0.1, 12)])
    const eps = 1e-6
    const next = (du: number) => get(env.model.transition(x, fromData(Float64Array.of(0.4 + du), [1])) as Tensor, 1)
    const fd = ((next(eps) as number) - (next(-eps) as number)) / (2 * eps)
    const ad = grad((v: Value) => get(env.model.transition(x, v), 1))(u) as Tensor
    expect(ad.data[0]).toBeCloseTo(fd, 6)
  })

  it('resets like Gymnasium and runs pure, seekable rollouts with a box-aware random agent', () => {
    const env = pendulumEnvironment()
    const { state, observation } = env.reset(stream(3))
    expect(Math.abs(state.theta)).toBeLessThanOrEqual(Math.PI)
    expect(Math.abs(state.thetaDot)).toBeLessThanOrEqual(1)
    expect(domainContains(env.observation, observation)).toBe(true)
    expectProtocol(rollout(env, randomAgent<Float64Array>()), undefined, { n: 30 })
    expectProtocol(rollout(pendulumEnvironment({ torques: 3 }), randomAgent()), undefined, { n: 30 })
  })
})

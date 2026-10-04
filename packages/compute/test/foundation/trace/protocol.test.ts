/**
 * The runner protocol (`foundation/trace/protocol.ts`): one definition of the stream derivation and the stop rule,
 * shared by the runners and the differentiating drivers, and not part of the package's API.
 */
import { describe, expect, it } from 'vitest'
import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import { child, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import { run } from 'aifn-compute/foundation/trace'
import { initStream, rootKey, stepContext, stopReason } from '../../../src/foundation/trace/protocol'

type S = Status & { x: number; t: number }
const walk: Algorithm<number, S> = {
  name: 'walk',
  init: (x0: number, s: Stream) => ({ x: x0 + (uniform(s) as number), t: 0 }),
  step: (st, ctx) => ({ x: st.x + (uniform(ctx.stream) as number), t: st.t + 1, converged: st.t + 1 >= 4 }),
}

describe('runner protocol', () => {
  it('stepping by hand with the protocol reproduces run', () => {
    const root = stream(7)
    const key = rootKey(root)
    let st = walk.init(1, initStream(key))
    for (let t = 0; !stopReason(walk, st); t++) st = walk.step(st, stepContext(key, t))
    expect(run(walk, 1, 100, { stream: root })).toEqual(st)
  })
  it('derives init from child(root, init) and step t from child(root, step, t), lazily', () => {
    const key = rootKey(stream(3))
    expect(uniform(initStream(key))).toBe(uniform(child(key, 'init')))
    expect(uniform(stepContext(key, 5).stream)).toBe(uniform(child(key, 'step', 5)))
    expect(rootKey(undefined)).toEqual(stream(0).key)
  })
  it('reports divergence before convergence', () => {
    expect(stopReason(walk, { x: 0, t: 0, diverged: true, converged: true })).toBe('diverged')
    expect(stopReason(walk, { x: 0, t: 0, terminated: true })).toBe('done')
    expect(stopReason(walk, { x: 0, t: 0 })).toBeNull()
  })
  it('is internal: the package does not export it', async () => {
    const mod: Record<string, unknown> = await import('aifn-compute/foundation/trace')
    for (const name of ['rootKey', 'initStream', 'stepContext', 'stopReason']) expect(mod[name]).toBeUndefined()
  })
})

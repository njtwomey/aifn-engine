import { describe, expect, it } from 'vitest'
import { fromData, isTensor } from 'aifn-compute/foundation/tensor'
import { call, fromMessage, isTask, toMessage } from './task'

describe('worker tasks', () => {
  it('are plain data naming aifn exports', () => {
    const t = call('foundation/trace/trace', call('data/targets/funnel'), { x0: [0, 1] }, 40)
    expect(isTask(t)).toBe(true)
    expect(structuredClone(t)).toEqual(t)
  })
  it('cross a structured clone with functions dropped and tensors re-branded', () => {
    const t = fromData(Float64Array.from([1, 2, 3, 4]), [2, 2])
    const value = { steps: [{ x: t, f: () => 1 }], extend: () => null, n: 2 }
    const sent = structuredClone(toMessage(value)) as typeof value
    expect('extend' in sent).toBe(false)
    const back = fromMessage<typeof value>(sent)
    expect(isTensor(back.steps[0].x)).toBe(true)
    expect(back.steps[0].x.shape).toEqual([2, 2])
    expect(back.n).toBe(2)
  })
})

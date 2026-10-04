import { describe, expect, it } from 'vitest'
import { huffmanSteps } from 'aifn-methods/information/coding'
import { trace } from 'aifn-compute/foundation/trace'
import { expectProtocol } from '../../protocol'

describe('huffmanSteps', () => {
  it('merges the D least-weight roots (K + d − 1)/(D − 1) times and follows the trace protocol', () => {
    const p = [0.4, 0.2, 0.2, 0.1, 0.1]
    const tr = trace(huffmanSteps(p), undefined, 20)
    expect(tr.meta.steps).toBe(4)
    expect(tr.meta.stopped).toBe('done')
    expect(trace(huffmanSteps(p, { arity: 4 }), undefined, 20).meta.steps).toBe(2)
    expectProtocol(huffmanSteps(p), undefined, { n: 4 })
  })
})

import { describe, expect, it } from 'vitest'
import { isingInference, isingLattice } from 'aifn-methods/inference/lattice-models'
import { expectProtocol } from '../../protocol'

describe('isingInference', () => {
  it('picks the engine by shape, and each engine follows the trace protocol', () => {
    const chain = isingInference(isingLattice(1, 6, 0.4, 0.1))
    expect(chain.shape).toBe('chain')
    expectProtocol(chain.algorithm, undefined, { n: 4 })
    const loopy = isingInference(isingLattice(3, 3, 0.3, 0.05), { damping: 0.2 })
    expect(loopy.shape).not.toBe('chain')
    expectProtocol(loopy.algorithm, undefined, { n: 8 })
  })
})

import { describe, it } from 'vitest'
import { ldaCollapsedGibbs } from 'aifn-methods/inference/topic-models'
import { expectProtocol } from '../../protocol'

describe('ldaCollapsedGibbs', () => {
  it('follows the trace protocol (sweeps draw from each step stream)', () => {
    const documents = [
      [0, 1, 2, 0, 1],
      [3, 4, 5, 4],
      [0, 2, 1, 1],
      [5, 3, 3, 4, 5],
    ]
    expectProtocol(ldaCollapsedGibbs({ documents, topics: 2, vocabulary: 6, alpha: 0.5, beta: 0.1 }), undefined, {
      n: 6,
    })
  })
})

import { describe, it } from 'vitest'
import { trueSkillEp } from 'aifn-methods/inference/rating-models'
import { expectProtocol } from '../../protocol'

describe('trueSkillEp', () => {
  it('follows the trace protocol', () => {
    const players = [0, 1, 2, 3].map(() => ({ mean: 25, sd: 25 / 3 }))
    const matches = [
      { winner: 0, loser: 1 },
      { winner: 1, loser: 2 },
      { winner: 2, loser: 3 },
      { winner: 3, loser: 0 },
      { winner: 0, loser: 2, draw: true },
    ]
    expectProtocol(trueSkillEp({ players, matches }), undefined, { n: 6 })
  })
})

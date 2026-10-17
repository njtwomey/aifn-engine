/**
 * Meta-interpretive learning (aifn-compute/logic/induction `metagol`): ancestor, grandparent and great-grandparent
 * with an invented predicate on the family, each program checked by SLD resolution; predicate invention on the
 * extended family; the event limit.
 */
import { describe, expect, it } from 'vitest'
import { solveQuery } from 'aifn-compute/logic/resolution'
import { metagol, type MilProblem } from 'aifn-compute/logic/induction'

type Pair = readonly [string, string]
const PARENT: Pair[] = [
  ['ann', 'bob'],
  ['ann', 'cid'],
  ['bob', 'dee'],
  ['bob', 'eve'],
  ['cid', 'fay'],
  ['eve', 'gus'],
]
const PARENT_TEXT = PARENT.map(([a, b]) => `parent(${a}, ${b}).`).join(' ')

const TASKS: { target: string; positives: Pair[]; negatives: Pair[]; clauses: number; events: number }[] = [
  {
    target: 'ancestor',
    positives: [
      ['ann', 'bob'],
      ['ann', 'dee'],
      ['ann', 'gus'],
      ['cid', 'fay'],
    ],
    negatives: [
      ['bob', 'ann'],
      ['dee', 'bob'],
      ['cid', 'gus'],
    ],
    clauses: 2,
    events: 69,
  },
  {
    target: 'grandparent',
    positives: [
      ['ann', 'dee'],
      ['bob', 'gus'],
    ],
    negatives: [
      ['ann', 'bob'],
      ['ann', 'gus'],
    ],
    clauses: 1,
    events: 15,
  },
  {
    target: 'greatgrandparent',
    positives: [['ann', 'gus']],
    negatives: [
      ['ann', 'eve'],
      ['bob', 'gus'],
      ['ann', 'dee'],
    ],
    clauses: 2,
    events: 177,
  },
]

describe('metagol', () => {
  it.each(TASKS)('learns $target in $clauses clauses and $events events, sound by SLD', (task) => {
    const r = metagol({ ...task, background: { parent: PARENT }, metarules: ['ident', 'chain'] }, { maxClauses: 3 })
    expect(r.stopped).toBe('found')
    expect(r.program).toHaveLength(task.clauses)
    expect(r.events).toHaveLength(task.events)
    const program = `${PARENT_TEXT} ${r.program!.join(' ')}`
    const proves = ([a, b]: Pair) =>
      solveQuery(program, `${task.target}(${a}, ${b})`, { maxSteps: 5000 }).solutions.length > 0
    expect(task.positives.every(proves)).toBe(true)
    expect(task.negatives.some(proves)).toBe(false)
  })

  it('invents grandparent for great-grandparent after rejecting two programs', () => {
    const t = TASKS[2]
    const r = metagol({ ...t, background: { parent: PARENT }, metarules: ['ident', 'chain'] })
    expect(r.program).toEqual([
      'greatgrandparent(A, B) :- parent(A, C), inv1(C, B).',
      'inv1(A, B) :- parent(A, C), parent(C, B).',
    ])
    expect(r.events.filter((e) => e.kind === 'reject')).toHaveLength(2)
  })

  it('grandparent from mother and father: 4 clauses in 1160 events without invention, 3 in 12 616 with', () => {
    const extended: Pair[] = [...PARENT, ['gus', 'hal'], ['hal', 'ivy']]
    const female = ['ann', 'cid', 'dee', 'eve', 'fay', 'ivy']
    const problem: MilProblem = {
      background: {
        mother: extended.filter(([a]) => female.includes(a)),
        father: extended.filter(([a]) => !female.includes(a)),
      },
      target: 'grandparent',
      positives: [
        ['ann', 'dee'],
        ['ann', 'eve'],
        ['ann', 'fay'],
        ['bob', 'gus'],
        ['eve', 'hal'],
        ['gus', 'ivy'],
      ],
      negatives: [
        ['ann', 'bob'],
        ['ann', 'gus'],
        ['dee', 'ann'],
        ['bob', 'fay'],
        ['eve', 'ivy'],
      ],
      metarules: ['ident', 'chain'],
    }
    const without = metagol(problem, { maxClauses: 4, maxEvents: 30000, maxInvented: 0 })
    expect(without.program).toHaveLength(4)
    expect(without.events).toHaveLength(1160)
    const withInvention = metagol(problem, { maxClauses: 4, maxEvents: 30000, maxInvented: 1 })
    expect(withInvention.program).toEqual([
      'grandparent(A, B) :- inv1(A, C), inv1(C, B).',
      'inv1(A, B) :- mother(A, B).',
      'inv1(A, B) :- father(A, B).',
    ])
    expect(withInvention.events).toHaveLength(12616)
  })

  it('stops at the event limit and says so', () => {
    const t = TASKS[0]
    const r = metagol({ ...t, background: { parent: PARENT }, metarules: ['ident', 'chain'] }, { maxEvents: 20 })
    expect(r.stopped).toBe('limit')
    expect(r.program).toBeNull()
    expect(r.events).toHaveLength(20)
  })
})

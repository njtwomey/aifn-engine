/** aifn-methods/data/synthetic: the toy knowledge graphs and the relation patterns each one shows exactly. */
import { describe, expect, it } from 'vitest'
import { familyKnowledgeGraph, relationPatternGraph, type IndexTriple } from 'aifn-methods/data/synthetic'

const key = ([h, r, t]: IndexTriple) => `${h},${r},${t}`
const has = (triples: readonly IndexTriple[]) => {
  const set = new Set(triples.map(key))
  return (t: IndexTriple) => set.has(key(t))
}
const ofRelation = (triples: readonly IndexTriple[], r: number) => triples.filter((t) => t[1] === r)

describe('familyKnowledgeGraph', () => {
  const kg = familyKnowledgeGraph()
  const holds = has(kg.triples)
  const rel = (name: string) => kg.relations.indexOf(name)

  it('has 36 triples: 10 + 10 + 6 + 4 + 6, all distinct and in range', () => {
    expect(kg.entities.length).toBe(9)
    expect(kg.relations.length).toBe(5)
    expect(kg.triples.length).toBe(36)
    expect(new Set(kg.triples.map(key)).size).toBe(36)
    expect([0, 1, 2, 3, 4].map((r) => ofRelation(kg.triples, r).length)).toEqual([10, 10, 6, 4, 6])
    for (const [h, r, t] of kg.triples) {
      expect(h).toBeLessThan(9)
      expect(t).toBeLessThan(9)
      expect(r).toBeLessThan(5)
      expect(h).not.toBe(t)
    }
  })

  it('spouse_of and sibling_of are symmetric, child_of inverts parent_of, grandparent_of composes parent_of', () => {
    for (const name of ['spouse_of', 'sibling_of'])
      for (const [h, r, t] of ofRelation(kg.triples, rel(name))) expect(holds([t, r, h])).toBe(true)
    const parents = ofRelation(kg.triples, rel('parent_of'))
    const children = ofRelation(kg.triples, rel('child_of'))
    expect(children.length).toBe(parents.length)
    for (const [p, , c] of parents) expect(holds([c, rel('child_of'), p])).toBe(true)
    const composed = new Set<string>()
    for (const [a, , b] of parents) for (const [b2, , c] of parents) if (b === b2) composed.add(`${a},${c}`)
    const grand = ofRelation(kg.triples, rel('grandparent_of')).map(([g, , c]) => `${g},${c}`)
    expect(new Set(grand)).toEqual(composed)
  })

  it('parent_of is one-to-many', () => {
    const counts = new Map<number, number>()
    for (const [p] of ofRelation(kg.triples, rel('parent_of'))) counts.set(p, (counts.get(p) ?? 0) + 1)
    expect(Math.max(...counts.values())).toBeGreaterThan(1)
  })
})

describe('relationPatternGraph', () => {
  it('symmetric: every fact holds reversed (8 triples)', () => {
    const kg = relationPatternGraph('symmetric')
    expect(kg.triples.length).toBe(8)
    const holds = has(kg.triples)
    for (const [h, r, t] of kg.triples) expect(holds([t, r, h])).toBe(true)
  })

  it('inverse: child_of is exactly parent_of reversed (8 triples)', () => {
    const kg = relationPatternGraph('inverse')
    expect(kg.triples.length).toBe(8)
    const reversed = ofRelation(kg.triples, 0).map(([h, , t]) => key([t, 1, h]))
    expect(new Set(ofRelation(kg.triples, 1).map(key))).toEqual(new Set(reversed))
  })

  it('composition: citizen_of is born_in then city_in (9 triples)', () => {
    const kg = relationPatternGraph('composition')
    expect(kg.triples.length).toBe(9)
    const composed = new Set<string>()
    for (const [p, , c] of ofRelation(kg.triples, 0))
      for (const [c2, , k] of ofRelation(kg.triples, 1)) if (c === c2) composed.add(key([p, 2, k]))
    expect(new Set(ofRelation(kg.triples, 2).map(key))).toEqual(composed)
  })

  it('one-to-many: two heads with three tails each, every tail once (6 triples)', () => {
    const kg = relationPatternGraph('one-to-many')
    expect(kg.triples.length).toBe(6)
    expect(kg.triples.filter(([h]) => h === 0).length).toBe(3)
    expect(kg.triples.filter(([h]) => h === 1).length).toBe(3)
    expect(new Set(kg.triples.map(([, , t]) => t)).size).toBe(6)
    expect(() => relationPatternGraph('antisymmetric' as never)).toThrow(/unknown pattern/)
  })
})

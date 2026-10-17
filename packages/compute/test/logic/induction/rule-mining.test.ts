/**
 * Closed rule mining on a knowledge graph (aifn-compute/logic/induction `mineClosedRules`, `predictLinks`): the two
 * worked rules of the note, every rule's counts recomputed by brute force, link prediction, and entity names with
 * spaces.
 */
import { describe, expect, it } from 'vitest'
import { mineClosedRules, predictLinks, type Triple } from 'aifn-compute/logic/induction'

// The people-and-places graph of the knowledge-graph-rule-mining note: eight people, four cities, two countries.
const PEOPLE_AND_PLACES: readonly Triple[] = [
  ['ann', 'marriedTo', 'ben'],
  ['ben', 'marriedTo', 'ann'],
  ['cal', 'marriedTo', 'dia'],
  ['dia', 'marriedTo', 'cal'],
  ['eli', 'marriedTo', 'fay'],
  ['ann', 'hasChild', 'gus'],
  ['ben', 'hasChild', 'gus'],
  ['cal', 'hasChild', 'hal'],
  ['dia', 'hasChild', 'hal'],
  ['ann', 'livesIn', 'paris'],
  ['ben', 'livesIn', 'paris'],
  ['cal', 'livesIn', 'rome'],
  ['dia', 'livesIn', 'rome'],
  ['eli', 'livesIn', 'lyon'],
  ['gus', 'livesIn', 'paris'],
  ['hal', 'livesIn', 'milan'],
  ['paris', 'locatedIn', 'france'],
  ['lyon', 'locatedIn', 'france'],
  ['rome', 'locatedIn', 'italy'],
  ['milan', 'locatedIn', 'italy'],
  ['ann', 'citizenOf', 'france'],
  ['ben', 'citizenOf', 'france'],
  ['cal', 'citizenOf', 'italy'],
  ['dia', 'citizenOf', 'france'],
  ['eli', 'citizenOf', 'france'],
  ['hal', 'citizenOf', 'italy'],
]

describe('mineClosedRules', () => {
  const rules = mineClosedRules(PEOPLE_AND_PLACES, { minSupport: 1, minPcaConfidence: 0 })
  const find = (t: string) => rules.find((r) => r.text === t)!

  it('scores the citizenship rule 5/7 standard and 5/6 PCA', () => {
    const r = find('citizenOf(x, y) ⇐ livesIn(x, z), locatedIn(z, y)')
    expect([r.support, r.bodySize, r.pcaBodySize]).toEqual([5, 7, 6])
    expect(r.confidence).toBeCloseTo(5 / 7, 12)
    expect(r.pcaConfidence).toBeCloseTo(5 / 6, 12)
  })

  it('scores the symmetry rule support 4, body 5, PCA 1', () => {
    const r = find('marriedTo(x, y) ⇐ marriedTo(y, x)')
    expect([r.support, r.bodySize, r.pcaConfidence]).toEqual([4, 5, 1])
  })

  it('agrees with brute-force counts on all 19 rules', () => {
    expect(rules).toHaveLength(19)
    const ents = [...new Set(PEOPLE_AND_PLACES.flatMap(([s, , o]) => [s, o]))]
    const has = (s: string, r: string, o: string) =>
      PEOPLE_AND_PLACES.some(([a, b, c]) => a === s && b === r && c === o)
    const holds = (atom: { relation: string; args: readonly string[] }, env: Record<string, string>) =>
      has(env[atom.args[0]], atom.relation, env[atom.args[1]])
    for (const r of rules) {
      let body = 0
      let supp = 0
      let pca = 0
      for (const x of ents)
        for (const y of ents) {
          if (x === y) continue
          const ok =
            r.body.length === 1
              ? holds(r.body[0], { x, y })
              : ents.some((z) => r.body.every((a) => holds(a, { x, y, z })))
          if (!ok) continue
          body += 1
          if (has(x, r.head, y)) supp += 1
          if (PEOPLE_AND_PLACES.some(([a, b]) => a === x && b === r.head)) pca += 1
        }
      expect([r.bodySize, r.support, r.pcaBodySize]).toEqual([body, supp, pca])
    }
  })

  it('predicts links with the best rule; the best has score 1', () => {
    const p = predictLinks(mineClosedRules(PEOPLE_AND_PLACES))
    expect(p[0].score).toBe(1)
    expect(p.every((x, i) => i === 0 || p[i - 1].score >= x.score)).toBe(true)
  })

  it('keeps entity names with spaces intact', () => {
    const kg: Triple[] = [
      ['ann lee', 'marriedTo', 'ben ng'],
      ['ben ng', 'marriedTo', 'ann lee'],
      ['cal ma', 'marriedTo', 'dia ho'],
    ]
    const sym = mineClosedRules(kg, { minSupport: 1 }).find((r) => r.text === 'marriedTo(x, y) ⇐ marriedTo(y, x)')!
    expect(sym.support).toBe(2)
    expect(sym.predictions).toEqual([['dia ho', 'cal ma']])
  })
})

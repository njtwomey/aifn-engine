import { describe, expect, it } from 'vitest'
import type { Table } from 'aifn-compute/foundation/contracts'
import { stream, units } from 'aifn-compute/foundation/random'
import { fromData } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import {
  binomialQuality,
  bitset,
  bitsetAndCount,
  bitsetCount,
  bitsetIndices,
  bitsetJaccard,
  chiSquareQuality,
  compatibleSelector,
  coverageQuality,
  cutPoints,
  descriptionKey,
  liftQuality,
  meanShiftQuality,
  sdMap,
  selectorKey,
  selectorLanguage,
  standardQuality,
  subgroupDiscovery,
  subgroupDiscoverySteps,
  subgroupOf,
  wraccQuality,
  type Description,
  type QualityMeasure,
  type SelectorLanguage,
} from 'aifn-compute/learning/subgroups'
import { fixture } from '../../fixtures'

// ── A small random table with a planted subgroup ────────────────────────────────────────────────────────────────

function planted(n = 160, seed = 1) {
  const u = units(stream(seed), n * 6)
  const age = Array.from({ length: n }, (_, i) => Math.round(20 + 60 * u[i]))
  const smoker = Array.from({ length: n }, (_, i) => (u[n + i] < 0.35 ? 'yes' : 'no'))
  const region = Array.from({ length: n }, (_, i) => ['north', 'south', 'east'][Math.floor(3 * u[2 * n + i])])
  const income = Array.from({ length: n }, (_, i) => 20 + 80 * u[3 * n + i])
  const inside = age.map((a, i) => a >= 50 && smoker[i] === 'yes')
  const outcome = inside.map((s, i) => (u[4 * n + i] < (s ? 0.85 : 0.15) ? 1 : 0))
  const cost = inside.map((s, i) => (s ? 3 : 1) + u[5 * n + i])
  const table: Table = { age: fromData(Float64Array.from(age)), smoker, region, income, outcome, cost }
  return { table, inside, outcome, cost }
}

/** Every valid description of depth 1..d, by brute force over the language's selectors (fixed discretisation). */
function allDescriptions(lang: SelectorLanguage, d: number): Description[] {
  const out: Description[] = []
  const sels = lang.selectors
  const walk = (cur: Description, from: number) => {
    if (cur.length) out.push(cur)
    if (cur.length >= d) return
    for (let j = from; j < sels.length; j++) {
      const s = sels[j]
      if (!compatibleSelector(cur, s)) continue
      const ge = cur.find((p) => p.attribute === s.attribute && p.op === '≥')
      if (ge && s.op === '≤' && !((s.value as number) > (ge.value as number))) continue
      walk([...cur, s], j + 1)
    }
  }
  walk([], 0)
  return out
}

const sortedDesc = (xs: number[]) => xs.sort((a, b) => b - a)
const round = (xs: number[]) => xs.map((q) => +q.toFixed(10))

describe('bitset covers', () => {
  it('count, intersect, list and compare', () => {
    const a = bitset(70, (i) => i % 3 === 0)
    const b = bitset(70, (i) => i % 5 === 0)
    expect(bitsetCount(a)).toBe(24)
    expect(bitsetAndCount(a, b)).toBe(5)
    expect(Array.from(bitsetIndices(b))).toEqual(Array.from({ length: 14 }, (_, k) => 5 * k))
    expect(bitsetJaccard(a, b)).toBeCloseTo(5 / (24 + 14 - 5), 12)
  })
})

describe('selectorLanguage', () => {
  const { table } = planted()
  const lang = selectorLanguage(table, { exclude: ['outcome', 'cost'], bins: 4, negations: true })

  it('classifies attributes and discretises numeric ones', () => {
    expect(lang.attributes.map((a) => [a.name, a.kind])).toEqual([
      ['age', 'numeric'],
      ['smoker', 'nominal'],
      ['region', 'nominal'],
      ['income', 'numeric'],
    ])
    expect(lang.attributes[0].cuts.length).toBe(3)
    expect(cutPoints([1, 2, 3, 4, 5, 6, 7, 8], 4, 'equal-frequency')).toEqual([3, 5, 7])
    expect(cutPoints([0, 10], 5, 'equal-width')).toEqual([2, 4, 6, 8])
  })

  it('covers equal a direct filter', () => {
    const age = table.age as ReturnType<typeof fromData>
    const d: Description = [
      { attribute: 'age', op: '≥', value: 50 },
      { attribute: 'smoker', op: '=', value: 'yes' },
    ]
    const want = (table.smoker as string[]).map((s, i) => s === 'yes' && (age.data[i] as number) >= 50)
    expect(Array.from(bitsetIndices(lang.cover(d)))).toEqual(want.flatMap((w, i) => (w ? [i] : [])))
  })

  it('canonical refinement reaches every valid conjunction exactly once', () => {
    for (const depth of [1, 2, 3]) {
      const reached: string[] = []
      const walk = (d: Description) => {
        if (d.length) reached.push(descriptionKey(d))
        if (d.length < depth) for (const r of lang.refinements(d)) walk(r)
      }
      walk([])
      expect(new Set(reached).size).toBe(reached.length)
      expect(new Set(reached)).toEqual(
        new Set(allDescriptions(lang, depth).map((d) => descriptionKey(lang.canonical(d)))),
      )
    }
  })

  it('on-the-fly cuts come from the rows being refined', () => {
    const fly = selectorLanguage(table, { exclude: ['outcome', 'cost'], discretisation: 'on-the-fly', bins: 3 })
    const parent: Description = [{ attribute: 'smoker', op: '=', value: 'yes' }]
    const inside = Array.from(bitsetIndices(fly.cover(parent)), (i) => (table.income as number[])[i])
    const cuts = fly
      .extensions(parent)
      .filter((s) => s.attribute === 'income' && s.op === '≥')
      .map((s) => s.value)
    expect(cuts).toEqual(cutPoints(inside, 3, 'equal-frequency'))
  })
})

describe('quality measures', () => {
  const { table, outcome, cost } = planted(120, 3)
  const lang = selectorLanguage(table, { exclude: ['outcome', 'cost'], bins: 3, negations: true })
  const N = outcome.length
  const P = outcome.reduce<number>((a, b) => a + b, 0)
  const p0 = P / N
  const all = allDescriptions(lang, 3)

  it('match their definitions', () => {
    const d: Description = [{ attribute: 'smoker', op: '=', value: 'yes' }]
    const c = lang.cover(d)
    const n = bitsetCount(c)
    const tp = Array.from(bitsetIndices(c)).reduce((a, i) => a + outcome[i], 0)
    const p = tp / n
    expect(wraccQuality(outcome).quality(c)).toBeCloseTo((n / N) * (p - p0), 12)
    expect(standardQuality(outcome, { a: 0.5 }).quality(c)).toBeCloseTo(Math.sqrt(n / N) * (p - p0), 12)
    expect(binomialQuality(outcome).quality(c)).toBeCloseTo((Math.sqrt(n) * (p - p0)) / Math.sqrt(p0 * (1 - p0)), 12)
    expect(liftQuality(outcome).quality(c)).toBeCloseTo(p / p0, 12)
    expect(coverageQuality(outcome).quality(c)).toBeCloseTo(n / N, 12)
    const mean = cost.reduce((a, b) => a + b, 0) / N
    const sd = Math.sqrt(cost.reduce((a, v) => a + (v - mean) ** 2, 0) / N)
    const mu = Array.from(bitsetIndices(c)).reduce((a, i) => a + cost[i], 0) / n
    expect(meanShiftQuality(cost).quality(c)).toBeCloseTo((Math.sqrt(n) * (mu - mean)) / sd, 12)
  })

  const measures: [string, QualityMeasure][] = [
    ['wracc', wraccQuality(outcome)],
    ['wracc lower', wraccQuality(outcome, { direction: 'lower' })],
    ['wracc both', wraccQuality(outcome, { direction: 'both' })],
    ['standard a=0.5', standardQuality(outcome, { a: 0.5 })],
    ['standard a=0', standardQuality(outcome, { a: 0 })],
    ['binomial', binomialQuality(outcome)],
    ['lift (min support 5)', liftQuality(outcome, { minSupport: 5 })],
    ['coverage', coverageQuality(outcome)],
    ['chi-square', chiSquareQuality(outcome)],
    ['chi-square higher', chiSquareQuality(outcome, { direction: 'higher' })],
    ['mean shift', meanShiftQuality(cost)],
    ['mean shift a=1 both', meanShiftQuality(cost, { a: 1, direction: 'both' })],
    ['mean shift lower raw', meanShiftQuality(cost, { direction: 'lower', standardise: false })],
  ]

  const keySets = new Map(all.map((d) => [d, new Set(d.map(selectorKey))]))
  for (const [name, m] of measures)
    it(`${name}: the optimistic estimate bounds every refinement`, () => {
      const minSupport = name.startsWith('lift') ? 5 : 1
      const ok = (d: Description) => bitsetCount(lang.cover(d)) >= minSupport
      for (const d of all.filter((d) => d.length <= 2)) {
        const b = m.bound!(lang.cover(d))
        for (const e of all)
          if (e.length > d.length && ok(e) && d.every((s) => keySets.get(e)!.has(selectorKey(s))))
            expect(m.quality(lang.cover(e))).toBeLessThanOrEqual(b + 1e-9)
      }
    })

  for (const [name, m] of measures)
    for (const strategy of ['depth-first', 'best-first'] as const)
      it(`${name}: ${strategy} branch and bound equals brute force`, () => {
        const minSupport = name.startsWith('lift') ? 5 : 1
        const truth = sortedDesc(
          all.filter((d) => bitsetCount(lang.cover(d)) >= minSupport).map((d) => m.quality(lang.cover(d))),
        ).slice(0, 8)
        const found = subgroupDiscovery(lang, m, { strategy, maxDepth: 3, k: 8, minSupport })
        expect(round(found.map((s) => s.quality))).toEqual(round(truth))
      })
})

describe('subgroup discovery', () => {
  const { table, inside, outcome, cost } = planted(400, 5)
  const lang = selectorLanguage(table, { exclude: ['outcome', 'cost'], bins: 6 })

  it('recovers the planted subgroup with WRAcc and with the mean shift', () => {
    const plantedKey = (s: { description: Description }) => {
      const c = lang.cover(s.description)
      return bitsetJaccard(
        c,
        bitset(inside.length, (i) => inside[i]),
      )
    }
    const top = subgroupDiscovery(lang, wraccQuality(outcome), { strategy: 'beam', beamWidth: 8, maxDepth: 2, k: 5 })
    expect(plantedKey(top[0])).toBeGreaterThan(0.8)
    expect(top[0].description.map((s) => s.attribute).sort()).toEqual(['age', 'smoker'])
    const numeric = subgroupDiscovery(lang, meanShiftQuality(cost), { strategy: 'best-first', maxDepth: 2, k: 3 })
    expect(plantedKey(numeric[0])).toBeGreaterThan(0.8)
  })

  it('SD-Map equals exhaustive search', () => {
    const negated = selectorLanguage(table, { exclude: ['outcome', 'cost'], bins: 4, negations: true })
    for (const m of [wraccQuality(outcome), binomialQuality(outcome), chiSquareQuality(outcome)])
      for (const minSupport of [1, 15]) {
        const opts = { maxDepth: 3, k: 12, minSupport }
        const fp = sdMap(negated, m, opts)
        const dfs = subgroupDiscovery(negated, m, { ...opts, strategy: 'depth-first', prune: false })
        expect(round(fp.map((s) => s.quality))).toEqual(round(dfs.map((s) => s.quality)))
        expect(sdMap(negated, m, { ...opts, prune: false }).map((s) => s.quality)).toEqual(fp.map((s) => s.quality))
      }
  })

  it('filters redundant results by cover and by description', () => {
    const m = wraccQuality(outcome)
    const byCover = subgroupDiscovery(lang, m, {
      strategy: 'depth-first',
      maxDepth: 2,
      k: 6,
      redundancy: { kind: 'cover', threshold: 0.5 },
    })
    for (let i = 0; i < byCover.length; i++)
      for (let j = i + 1; j < byCover.length; j++)
        expect(bitsetJaccard(byCover[i].cover, byCover[j].cover)).toBeLessThan(0.5)
    const byDescription = subgroupDiscovery(lang, m, {
      strategy: 'depth-first',
      maxDepth: 2,
      k: 6,
      redundancy: { kind: 'description' },
    })
    const keys = byDescription.map((s) => new Set(s.description.map((t) => descriptionKey([t]))))
    for (let i = 0; i < keys.length; i++)
      for (let j = 0; j < keys.length; j++)
        if (i !== j) expect([...keys[i]].every((k) => keys[j].has(k)) && keys[i].size <= keys[j].size).toBe(false)
  })

  it('steps a beam search level by level, and its results agree with subgroupOf', () => {
    const m = wraccQuality(outcome)
    const t = trace(subgroupDiscoverySteps(lang, m, { beamWidth: 4, maxDepth: 3 }), undefined, 100)
    expect(t.steps.map((s) => s.level)).toEqual([0, 1, 2, 3])
    for (const s of t.steps.slice(1)) expect(s.frontier.length).toBeLessThanOrEqual(4)
    for (const r of t.final.results) expect(subgroupOf(lang, m, r.node).quality).toBeCloseTo(r.quality, 12)
  })
})

// ── pysubgroup on its Titanic sample ───────────────────────────────────────────────────────────────────────────

type Found = { quality: number; selectors: [string, string][] }
type Fixture = {
  titanic: Record<string, (string | number)[]>
  wracc: Record<string, Found[]>
  standardHalf: Record<string, Found[]>
  addedValue: Found[]
  meanShift: Found[]
  chiSquare: { selectors: [string, string][]; chiSquare: number }[]
}

describe('against pysubgroup (Titanic)', () => {
  const f = fixture<Fixture>('learning/subgroups')
  const { Survived, Fare, ...attributes } = f.titanic
  const lang = selectorLanguage(attributes as Table)
  const y = Survived as number[]
  const fare = Fare as number[]
  const desc = (sels: [string, string][]): Description =>
    lang.canonical(sels.map(([attribute, value]) => ({ attribute, op: '=' as const, value })))
  const check = (m: QualityMeasure, found: Found[], depth: number) => {
    const ours = subgroupDiscovery(lang, m, { strategy: 'best-first', maxDepth: depth, k: found.length })
    expect(round(ours.map((s) => s.quality))).toEqual(round(found.map((s) => s.quality)))
    for (const s of found) expect(m.quality(lang.cover(desc(s.selectors)))).toBeCloseTo(s.quality, 10)
  }

  for (const depth of ['1', '2', '3'])
    it(`WRAcc top 10 at depth ${depth}`, () => check(wraccQuality(y), f.wracc[depth], +depth))
  for (const depth of ['2', '3'])
    it(`Klösgen a = 0.5 top 10 at depth ${depth}`, () =>
      check(standardQuality(y, { a: 0.5 }), f.standardHalf[depth], +depth))
  it('added value (LiftQF) top 10 at depth 2', () => check(standardQuality(y, { a: 0 }), f.addedValue, 2))
  it('numeric mean shift (StandardQFNumeric a = 0.5) top 10 at depth 2', () =>
    check(meanShiftQuality(fare, { standardise: false }), f.meanShift, 2))
  it('χ² equals scipy', () => {
    const m = chiSquareQuality(y)
    for (const c of f.chiSquare) expect(m.quality(lang.cover(desc(c.selectors)))).toBeCloseTo(c.chiSquare, 9)
  })
})

/** Weisfeiler-Leman tests (aifn-compute/graph/isomorphism), against the site's self-check (weisfeiler-leman). */
import { describe, expect, it } from 'vitest'
import { folkloreWeisfeilerLeman2, weisfeilerLeman, weisfeilerLemanSteps } from 'aifn-compute/graph/isomorphism'
import { run } from 'aifn-compute/foundation/trace'
import { bicyclopentyl, decalin, hexagon, path4, rook, shrikhande, star3, twoTriangles } from './pairs'

describe('weisfeilerLeman', () => {
  it('path vs star: distinguished at round 1, path colours [0, 1, 1, 0]', () => {
    const r = weisfeilerLeman([path4, star3])
    expect(r.distinguishedAt).toBe(1)
    expect(Array.from(r.colours[1][0])).toEqual([0, 1, 1, 0])
  })
  it('6-cycle vs two triangles: never distinguished, stable at 1, one class', () => {
    const r = weisfeilerLeman([hexagon, twoTriangles])
    expect([r.distinguishedAt, r.stable, r.classes]).toEqual([-1, 1, [1, 1]])
  })
  it('decalin vs bicyclopentyl tie under 1-WL', () => {
    expect(weisfeilerLeman([decalin, bicyclopentyl]).distinguishedAt).toBe(-1)
  })
  it("rook's graph vs Shrikhande: 48 edges each, tie under 1-WL and 2-FWL", () => {
    expect([rook.edges.length, shrikhande.edges.length]).toEqual([48, 48])
    expect(weisfeilerLeman([rook, shrikhande]).distinguishedAt).toBe(-1)
    expect(folkloreWeisfeilerLeman2([rook, shrikhande]).distinguishedAt).toBe(-1)
  })
  it('the palette does not depend on the order of the graphs', () => {
    const a = weisfeilerLeman([path4, star3])
    const b = weisfeilerLeman([star3, path4])
    expect(Array.from(b.colours[1][1])).toEqual(Array.from(a.colours[1][0]))
  })
  it('initial labels refine: two labelled hexagons separate at round 1', () => {
    const r = weisfeilerLeman([hexagon, hexagon], {
      initial: [
        [0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 1],
      ],
    })
    expect(r.distinguishedAt).toBe(1)
    expect(() => weisfeilerLeman([hexagon], { initial: [[0, 1]] })).toThrow()
  })
  it('the stepped form gives the rounds of the one-call form', () => {
    const all = weisfeilerLeman([decalin, bicyclopentyl])
    for (let t = 0; t <= all.stable; t++) {
      const s = run(weisfeilerLemanSteps([decalin, bicyclopentyl]), undefined, t)
      expect(s.round).toBe(t)
      expect(s.colours.map((c) => Array.from(c))).toEqual(all.colours[t].map((c) => Array.from(c)))
      expect(s.classes).toBe(all.classes[t])
    }
    expect(run(weisfeilerLemanSteps([decalin, bicyclopentyl]), undefined, 100).round).toBe(all.stable)
  })
})

describe('folkloreWeisfeilerLeman2', () => {
  it('separates the 6-cycle from two triangles at round 1, decalin from bicyclopentyl at round 2', () => {
    expect(folkloreWeisfeilerLeman2([hexagon, twoTriangles]).distinguishedAt).toBe(1)
    expect(folkloreWeisfeilerLeman2([decalin, bicyclopentyl]).distinguishedAt).toBe(2)
  })
  it('needs undirected graphs', () => {
    expect(() => folkloreWeisfeilerLeman2([{ kind: 'graph', nodes: 2, edges: [{ from: 0, to: 1 }] }])).toThrow()
  })
})

/**
 * Diffusion maps, self-organising maps and PaCMAP by their laws: the diffusion map's Euclidean distances equal the
 * diffusion distances computed from P^t directly (all coordinates kept), with λ₀ = 1 and π stationary; the batch SOM
 * lowers its quantisation error, orders a 1-D grid along a curve and keeps a low topographic error; PaCMAP's pairs have
 * the documented counts and its embedding keeps blobs apart. The step algorithms pass the trace protocol.
 */
import { describe, expect, it } from 'vitest'
import { normals, stream } from 'aifn-compute/foundation/random'
import { fromRows, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { trace } from 'aifn-compute/foundation/trace'
import { blobs } from 'aifn-methods/data/synthetic'
import {
  diffusionMap,
  selfOrganisingMap,
  selfOrganisingMapSteps,
  somGrid,
} from 'aifn-methods/unsupervised/embedding/manifold'
import { pacmap, pacmapPairs, pacmapWeights } from 'aifn-methods/unsupervised/embedding/neighbour'

describe('diffusionMap', () => {
  const X = normals(stream('dm'), [25, 3])
  for (const time of [1, 3])
    it(`turns diffusion distances into Euclidean ones (t = ${time})`, () => {
      const n = 25
      const m = diffusionMap({ dims: n - 1, time, alpha: 0.5 }).fit(dataset(X))
      expect(toFlat(m.eigenvalues)[0]).toBeCloseTo(1, 10)
      const P = toFlat(m.transition)
      const pi = toFlat(m.stationary)
      // πP = π.
      for (let j = 0; j < n; j++) {
        let s = 0
        for (let i = 0; i < n; i++) s += pi[i] * P[i * n + j]
        expect(s).toBeCloseTo(pi[j], 12)
      }
      let Pt = Float64Array.from(P)
      for (let k = 1; k < time; k++) {
        const next = new Float64Array(n * n)
        for (let i = 0; i < n; i++)
          for (let l = 0; l < n; l++) for (let j = 0; j < n; j++) next[i * n + j] += Pt[i * n + l] * P[l * n + j]
        Pt = next
      }
      const Y = toFlat(m.embedding)
      for (const [a, b] of [
        [0, 1],
        [2, 7],
        [5, 20],
      ]) {
        let direct = 0
        for (let y = 0; y < n; y++) direct += (Pt[a * n + y] - Pt[b * n + y]) ** 2 / pi[y]
        let embedded = 0
        for (let c = 0; c < n - 1; c++) embedded += (Y[a * (n - 1) + c] - Y[b * (n - 1) + c]) ** 2
        expect(embedded).toBeCloseTo(direct, 8)
      }
    })

  it('separates two distant clusters on its first coordinate', () => {
    const data = blobs(stream(3), {
      n: 40,
      centers: [
        [0, 0],
        [8, 0],
      ],
      sd: 0.5,
    })
    const m = diffusionMap({ epsilon: 2 }).fit(dataset(data.x))
    const y = toFlat(data.y!)
    const e = toFlat(m.embedding)
    const side = (i: number) => Math.sign(e[i * 2])
    for (let i = 1; i < 40; i++) expect(side(i) === side(0)).toBe(y[i] === y[0])
  })
})

describe('selfOrganisingMap', () => {
  it('lowers the quantisation error and has small topographic error on a 2-D sheet', () => {
    const X = normals(stream('som'), [200, 2])
    const t = trace(selfOrganisingMapSteps(X, { rows: 6, cols: 6, epochs: 25 }), undefined, 25, { stream: stream(1) })
    expect(t.final.quantisationError).toBeLessThan(t.steps[0].quantisationError)
    expect(t.final.topographicError).toBeLessThan(0.15)
  })

  it('orders a 1-D map along a curve', () => {
    const rows = Array.from({ length: 120 }, (_, i) => {
      const s = (i / 119) * Math.PI
      return [Math.cos(s), Math.sin(s)]
    })
    const m = selfOrganisingMap({ rows: 1, cols: 10, epochs: 40, start: 'random' }).fit(dataset(fromRows(rows)), {
      stream: stream(2),
    })
    const W = toFlat(m.weights)
    const angle = Array.from({ length: 10 }, (_, u) => Math.atan2(W[u * 2 + 1], W[u * 2]))
    const up = angle.every((a, u) => u === 0 || a > angle[u - 1])
    const down = angle.every((a, u) => u === 0 || a < angle[u - 1])
    expect(up || down).toBe(true)
    expect(m.uMatrix.shape).toEqual([1, 10])
    expect(toFlat(m.transform(fromRows([[1, 0]])))[0]).toBe(0)
  })

  it('lists grid positions row by row', () => {
    expect(Array.from(toFlat(somGrid(2, 3)))).toEqual([0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2])
  })
})

describe('pacmap', () => {
  const data = blobs(stream(5), {
    n: 60,
    centers: [
      [0, 0, 0],
      [6, 6, 0],
      [0, 6, 6],
    ],
    sd: 0.6,
  })
  it('builds the documented numbers of pairs, neighbours by scaled distance', () => {
    const p = pacmapPairs(data.x, stream(1), { neighbours: 6, midNearRatio: 0.5, furtherRatio: 2 })
    expect(p.neighbour.length / 2).toBe(60 * 6)
    expect(p.midNear.length / 2).toBe(60 * 3)
    expect(p.further.length / 2).toBe(60 * 12)
    for (let k = 0; k < p.neighbour.length; k += 2) expect(p.neighbour[k]).not.toBe(p.neighbour[k + 1])
  })

  it('follows the three-phase weight schedule', () => {
    expect(pacmapWeights(0)).toEqual({ neighbour: 2, midNear: 1000, further: 1 })
    expect(pacmapWeights(150)).toEqual({ neighbour: 3, midNear: 3, further: 1 })
    expect(pacmapWeights(300)).toEqual({ neighbour: 1, midNear: 0, further: 1 })
  })

  it('keeps blobs apart: each point’s nearest embedded neighbour shares its blob', () => {
    const m = pacmap({ neighbours: 8, iterations: 200 }).fit(dataset(data.x), { stream: stream(7) })
    const Y = toFlat(m.embedding)
    const y = toFlat(data.y!)
    let agree = 0
    for (let i = 0; i < 60; i++) {
      let best = -1
      let bd = Infinity
      for (let j = 0; j < 60; j++) {
        if (j === i) continue
        const dd = (Y[i * 2] - Y[j * 2]) ** 2 + (Y[i * 2 + 1] - Y[j * 2 + 1]) ** 2
        if (dd < bd) {
          bd = dd
          best = j
        }
      }
      if (y[best] === y[i]) agree++
    }
    expect(agree).toBeGreaterThanOrEqual(58)
  })
})

import { describe, expect, it } from 'vitest'
import { layeredLayout } from './layout'
import type { DiagramNode, DiagramSpec } from './types'

/** A deterministic pseudo-random DAG of n nodes with labels of varied length (edges only go from lower to higher ids). */
function randomDag(n: number, seed: number, direction: 'right' | 'down'): DiagramSpec {
  let s = seed
  const next = () => (s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31
  const nodes = Array.from({ length: n }, (_, i) => ({ id: `n${i}`, label: 'x'.repeat(1 + Math.floor(next() * 24)) }))
  const edges: { from: string; to: string }[] = []
  for (let b = 1; b < n; b++)
    for (let a = 0; a < b; a++) if (next() < 2.2 / b) edges.push({ from: `n${a}`, to: `n${b}` })
  return { layout: 'layered', layered: { direction }, nodes, edges }
}

/** A label-fitted size: about a fifth of a grid unit per character, as `Diagram` measures it. */
const fitted = (n: DiagramNode): [number, number] => [Math.max(2.4, 0.6 + 0.2 * (n.label?.length ?? 0)), 0.8]

type Box = { x0: number; y0: number; x1: number; y1: number }

describe('layeredLayout', () => {
  for (const direction of ['right', 'down'] as const)
    for (const seed of [1, 7, 42, 2024, 31337])
      it(`places ${direction} DAG ${seed} without overlaps or edges through nodes`, () => {
        const spec = randomDag(14, seed, direction)
        const out = layeredLayout(spec, fitted)
        const boxes = new Map<string, Box>(
          out.nodes.map((n) => {
            const [w, h] = fitted(n)
            return [n.id, { x0: n.x - w / 2, y0: n.y - h / 2, x1: n.x + w / 2, y1: n.y + h / 2 }]
          }),
        )
        const list = [...boxes.entries()]
        for (let i = 0; i < list.length; i++)
          for (let j = i + 1; j < list.length; j++) {
            const [a, b] = [list[i][1], list[j][1]]
            const overlap = a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1
            expect(overlap, `${list[i][0]} overlaps ${list[j][0]}`).toBe(false)
          }

        // Every edge, from centre to centre through its waypoints, is right-angled and crosses no other node.
        const at = new Map(out.nodes.map((n) => [n.id, n]))
        for (const e of out.edges ?? []) {
          const [a, b] = [at.get(e.from)!, at.get(e.to)!]
          const pts = [[a.x, a.y], ...(e.via ?? []), [b.x, b.y]]
          for (let k = 0; k + 1 < pts.length; k++) {
            const [[x0, y0], [x1, y1]] = [pts[k], pts[k + 1]]
            if (k > 0 && k + 2 < pts.length) expect(x0 === x1 || y0 === y1).toBe(true)
            for (const [id, box] of boxes) {
              if (id === e.from || id === e.to) continue
              const eps = 1e-6
              const hits =
                Math.min(x0, x1) < box.x1 - eps &&
                Math.max(x0, x1) > box.x0 + eps &&
                Math.min(y0, y1) < box.y1 - eps &&
                Math.max(y0, y1) > box.y0 + eps
              expect(hits, `${e.from} → ${e.to} crosses ${id}`).toBe(false)
            }
          }
        }
      })

  it('gives straight edges waypoints at their virtual nodes, as before', () => {
    const spec: DiagramSpec = {
      layout: 'layered',
      nodes: ['a', 'b', 'c'].map((id) => ({ id })),
      edges: [
        { from: 'a', to: 'b' },
        { from: 'b', to: 'c' },
        { from: 'a', to: 'c', route: 'straight' },
      ],
    }
    const out = layeredLayout(spec)
    expect(out.edges![2].via).toHaveLength(1)
    // The straight edge's waypoint is its virtual node, in the middle layer beside b.
    expect(out.edges![2].via![0][0]).toBeCloseTo(out.nodes[1].x)
  })
})

/** Semi-supervised node classification on Zachary's karate club: every layer kind learns the split from four labels. */
import { describe, expect, it } from 'vitest'
import { karateClub } from 'aifn-methods/data/real'
import { nodeClassificationRun, type NodeSnapshot } from 'aifn-methods/neural/graph'

const last = <T>(g: Generator<T>): T => {
  let out: T | undefined
  for (const s of g) out = s
  return out!
}

describe('nodeClassificationRun', () => {
  const data = karateClub()
  it('the karate club has 34 members, 78 friendships and two clubs', () => {
    expect(data.graph.nodes).toBe(34)
    expect(data.graph.edges.length).toBe(78)
    expect(data.x.shape).toEqual([34, 34])
  })
  for (const kind of ['gcn', 'gat', 'sage'] as const) {
    it(`${kind}: the loss falls and most unlabelled members land in their club`, () => {
      const run: NodeSnapshot = last(
        nodeClassificationRun({ data, train: [0, 1, 32, 33], kind, steps: 150, stepSize: 0.03 }),
      )
      const h = run.history
      expect(run.done).toBe(true)
      expect(h.loss.at(-1)!).toBeLessThan(0.5 * h.loss[0])
      expect(h.trainAccuracy.at(-1)).toBe(1)
      expect(h.testAccuracy.at(-1)!).toBeGreaterThan(0.75)
      expect(run.checkpoints[0].step).toBe(0)
      expect(run.checkpoints.at(-1)!.logits.length).toBe(34)
      if (kind === 'gat') {
        expect(run.edges!.source.length).toBe(2 * 78 + 34)
        expect(run.checkpoints[0].attention!.length).toBe(2 * 78 + 34)
      }
    })
  }
})

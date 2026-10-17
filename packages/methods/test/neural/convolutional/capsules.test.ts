/**
 * aifn-methods/neural/convolutional capsules: `squash` lengths and directions, `capsulePredictions`, and the worked
 * example of the capsule-network note traced by `dynamicRoutingTrace` (couplings 0.5, 0.745, 0.916 and lengths 0.733,
 * 0.865, 0.905), priors and equivariance to a common rotation (the site's self-check).
 */
import { describe, expect, it } from 'vitest'
import { norm, tensor } from 'aifn-compute/foundation/tensor'
import { capsulePredictions, dynamicRoutingTrace, squash } from 'aifn-methods/neural/convolutional'

const length = (v: readonly number[]) => norm(tensor([...v])) as number

describe('squash', () => {
  it.each([[[3, 4]], [[0.1, 0]], [[1, 1, 1]]])('keeps the direction of %j and maps |s| to |s|^2 / (1 + |s|^2)', (s) => {
    const n = length(s)
    const v = squash(s)
    expect(length(v)).toBeCloseTo((n * n) / (1 + n * n), 12)
    s.forEach((x, k) => expect(v[k] * n).toBeCloseTo(x * length(v), 12))
  })

  it('maps zero to zero', () => {
    expect(squash([0, 0])).toEqual([0, 0])
  })
})

describe('capsulePredictions', () => {
  it('applies each weight matrix to its lower capsule', () => {
    expect(
      capsulePredictions(
        [
          [
            [
              [0, -1],
              [1, 0],
            ],
          ],
        ],
        [[1, 2]],
      ),
    ).toEqual([[[-2, 1]]])
  })
})

describe('dynamicRoutingTrace', () => {
  const u = [
    [
      [1.5, 0.5],
      [-0.5, 1.5],
    ],
    [
      [1.5, 0.9],
      [1, -1],
    ],
  ]
  const trace = dynamicRoutingTrace(u, 3)

  it('gives the worked example of the note', () => {
    expect(trace.length).toBe(3)
    for (const t of trace) for (const c of t.couplings) expect(c[0] + c[1]).toBeCloseTo(1, 12)
    expect(trace[0].inputs[0][0]).toBeCloseTo(1.5, 12)
    expect(trace[0].inputs[0][1]).toBeCloseTo(0.7, 12)
    expect(length(trace[0].outputs[0])).toBeCloseTo(2.74 / 3.74, 12)
    expect(trace.map((t) => t.couplings[0][0])).toEqual([0.5, expect.closeTo(0.745, 3), expect.closeTo(0.916, 3)])
    expect(trace.map((t) => length(t.outputs[0]))).toEqual([
      expect.closeTo(0.733, 3),
      expect.closeTo(0.865, 3),
      expect.closeTo(0.905, 3),
    ])
  })

  it('is pinned by large priors', () => {
    const pinned = dynamicRoutingTrace(u, 1, [
      [10, 0],
      [0, 10],
    ])
    expect(pinned[0].couplings[0][0]).toBeGreaterThan(0.9999)
    expect(pinned[0].couplings[1][1]).toBeGreaterThan(0.9999)
  })

  it('keeps the couplings and rotates the outputs when every prediction is rotated alike', () => {
    const rot = (v: number[], a: number) => [
      Math.cos(a) * v[0] - Math.sin(a) * v[1],
      Math.sin(a) * v[0] + Math.cos(a) * v[1],
    ]
    const turned = dynamicRoutingTrace(
      u.map((row) => row.map((v) => rot(v, 0.7))),
      3,
    )
    turned.forEach((t, r) =>
      t.couplings.forEach((c, i) => c.forEach((x, j) => expect(x).toBeCloseTo(trace[r].couplings[i][j], 12))),
    )
    turned[2].outputs.forEach((v, j) =>
      rot(trace[2].outputs[j], 0.7).forEach((x, k) => expect(v[k]).toBeCloseTo(x, 12)),
    )
  })
})

/**
 * aifn-methods/neural/operator mesh message passing (the graph-and-mesh note's checks): geometric weights give
 * $u'' = 2$ exactly for $u = x^2$ on the mesh (0, 0.1, 0.35, 0.4, 0.7, 1) where uniform ones give
 * (2.56, -1.88, 7.31, 4.50); on a uniform mesh the two agree; on an irregular one the geometric rollout is closer to
 * the exact solution.
 */
import { describe, expect, it } from 'vitest'
import { meshHeatRollout, messageWeights, type MessageWeights } from 'aifn-methods/neural/operator'

const x = [0, 0.1, 0.35, 0.4, 0.7, 1]
const u = x.map((v) => v * v)
const laplacian = (w: MessageWeights) => {
  const { left, right } = messageWeights(x, w)
  return x.slice(1, -1).map((_, k) => left[k + 1] * (u[k] - u[k + 1]) + right[k + 1] * (u[k + 2] - u[k + 1]))
}
const exact = (v: number, t: number) =>
  Math.exp(-0.1 * Math.PI ** 2 * t) * Math.sin(Math.PI * v) +
  0.5 * Math.exp(-0.1 * 9 * Math.PI ** 2 * t) * Math.sin(3 * Math.PI * v)

describe('mesh message passing', () => {
  it('gives the second difference with geometric weights and not with uniform ones', () => {
    laplacian('geometric').forEach((v) => expect(Math.abs(v - 2)).toBeLessThan(1e-10))
    const want = [2.56, -1.88, 7.31, 4.5]
    laplacian('uniform').forEach((v, i) => expect(Math.abs(v - want[i])).toBeLessThan(0.006))
  })

  it('rolls out the heat equation: equal weights on a uniform mesh, geometric closer on an irregular one', () => {
    const options = { diffusivity: 0.1, time: 0.1, frames: 11 }
    const error = (mesh: number[], weights: MessageWeights) => {
      const run = meshHeatRollout(
        mesh,
        mesh.map((v) => exact(v, 0)),
        { ...options, weights },
      )
      expect(run.times.at(-1)).toBeCloseTo(0.1, 14)
      return Math.max(...mesh.map((v, i) => Math.abs(run.states.at(-1)![i] - exact(v, 0.1))))
    }
    const even = Array.from({ length: 12 }, (_, i) => i / 11)
    expect(Math.abs(error(even, 'geometric') - error(even, 'uniform'))).toBeLessThan(1e-12)
    expect(error(even, 'geometric')).toBeLessThan(0.02)
    const uneven = [0, 0.04, 0.1, 0.15, 0.3, 0.38, 0.5, 0.55, 0.7, 0.85, 0.93, 1]
    expect(error(uneven, 'geometric')).toBeLessThan(error(uneven, 'uniform'))
  })
})

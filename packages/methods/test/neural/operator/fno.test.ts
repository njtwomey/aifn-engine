/**
 * aifn-methods/neural/operator `FourierNeuralOperator1d` (the Fourier neural operator note's check): one network maps
 * $[1, n, 1]$ to $[1, n, 1]$ at any resolution, and gives the same values at 16 and 32 points for a band-limited
 * input only up to its pointwise nonlinearities (so the shapes, not the values, are asserted across grids). Its
 * parameters train by `grad`.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { treeLeaves } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { sum, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { FourierNeuralOperator1d, type FnoParams } from 'aifn-methods/neural/operator'

const at = (n: number) => tensor([Array.from({ length: n }, (_, j) => [Math.cos((2 * Math.PI * j) / n)])])

describe('Fourier neural operator', () => {
  const fno = FourierNeuralOperator1d({ width: 4, modes: 4, layers: 2 })
  const p = fno.init(stream('check/fno'))

  it('maps [1, n, 1] to [1, n, 1] at any resolution', () => {
    expect((fno.apply(p, at(32)) as Tensor).shape).toEqual([1, 32, 1])
    expect((fno.apply(p, at(16)) as Tensor).shape).toEqual([1, 16, 1])
    expect(p.blocks.length).toBe(2)
    expect(fno.label).toBe('FNO1d(width 4, 4 modes, 2 layers)')
  })

  it('has a finite gradient in every parameter', () => {
    // `grad` with the parameter type erased, which keeps the checker from expanding the pytree types.
    const gradOf = grad as unknown as (f: (q: FnoParams) => unknown) => (q: FnoParams) => FnoParams
    const g = gradOf((q) => sum(fno.apply(q, at(8))))(p)
    for (const leaf of treeLeaves(g)) expect(toFlat(leaf.value as Tensor).every(Number.isFinite)).toBe(true)
  })
})

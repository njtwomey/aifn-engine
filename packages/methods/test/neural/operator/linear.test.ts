/**
 * aifn-methods/neural/operator linear fits (the operator-learning notes' checks): a dense map from 32 noise-free heat
 * pairs on 16 points reproduces test outputs (4e-10) while one pair does not generalise (79%); a Fourier multiplier
 * from one pair is $e^{-D(2\pi k)^2 T}$ at $k = 3$ and transfers to a 64-point grid; the residual-only
 * physics-informed multiplier is backward Euler $(1 + D(2\pi k)^2\Delta t)^{-J}$, and data alone recovers the exact
 * one.
 */
import { describe, expect, it } from 'vitest'
import { normals, stream } from 'aifn-compute/foundation/random'
import { fromData, matmul, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { relativeL2Error } from 'aifn-compute/learning/metrics'
import { evaluateSeries, heatSolution, periodicGrid, type FourierSeries1d } from 'aifn-methods/dynamics/pde'
import {
  applyFourierMultiplier,
  applyModeMultiplier,
  fitDenseOperator,
  fitFourierMultiplier,
  fitPhysicsInformedMultiplier,
} from 'aifn-methods/neural/operator'

const D = 0.1
const T = 0.1
const n = 16

/** Random zero-mean series of `modes` modes, decaying as 1/k^2. */
function series(seed: string, count: number, modes: number): FourierSeries1d[] {
  const z = toFlat(normals(stream(seed), [count, 2 * modes]))
  return Array.from({ length: count }, (_, i) => ({
    mean: 0,
    cos: Float64Array.from({ length: modes }, (_, k) => z[i * 2 * modes + k] / (k + 1) ** 2),
    sin: Float64Array.from({ length: modes }, (_, k) => z[i * 2 * modes + modes + k] / (k + 1) ** 2),
  }))
}

const onGrid = (fs: FourierSeries1d[], m = n): Tensor => {
  const x = periodicGrid(m)
  const out = new Float64Array(fs.length * m)
  fs.forEach((f, i) => out.set(evaluateSeries(f, x), i * m))
  return fromData(out, [fs.length, m])
}

const inputs = series('fits', 40, 7)
const outputs = inputs.map((f) => heatSolution(f, D, T))
const A = onGrid(inputs)
const U = onGrid(outputs)
const rows = (t: Tensor, a: number, b: number) => fromData(Float64Array.from(toFlat(t).slice(a * n, b * n)), [b - a, n])

describe('linear operator fits', () => {
  it('fits a dense map from 32 pairs but not from one', () => {
    const G = fitDenseOperator(rows(A, 0, 32), rows(U, 0, 32))
    expect(relativeL2Error(rows(U, 32, 40), matmul(rows(A, 32, 40), G) as Tensor)).toBeLessThan(1e-6)
    const G1 = fitDenseOperator(rows(A, 0, 1), rows(U, 0, 1))
    expect(relativeL2Error(rows(U, 32, 40), matmul(rows(A, 32, 40), G1) as Tensor)).toBeGreaterThan(0.1)
  })

  it('recovers the heat multiplier from one pair, and applies it on a finer grid', () => {
    const m = fitFourierMultiplier(rows(A, 0, 1), rows(U, 0, 1))
    expect(Math.abs(m.re[3] - Math.exp(-D * (2 * Math.PI * 3) ** 2 * T))).toBeLessThan(1e-10)
    expect(Math.abs(m.im[3])).toBeLessThan(1e-10)
    expect(relativeL2Error(rows(U, 32, 40), applyFourierMultiplier(m, rows(A, 32, 40)))).toBeLessThan(1e-10)
    const fine = applyFourierMultiplier(m, onGrid([inputs[35]], 64))
    expect(relativeL2Error(onGrid([outputs[35]], 64), fine)).toBeLessThan(1e-10)
  })

  it('gives backward Euler from the residual alone, and the exact multiplier from data alone', () => {
    const virtual = series('virtual', 10, 5)
    const J = 10
    const fit = fitPhysicsInformedMultiplier({ train: [], virtual, diffusivity: D, time: T, levels: J, weight: 1 })
    expect(Math.abs(fit.final[1] - (1 + D * (2 * Math.PI * 2) ** 2 * (T / J)) ** -J)).toBeLessThan(1e-8)
    expect(fit.trajectory.length).toBe(5 * J)
    const train = inputs.slice(0, 3).map((input, i) => ({
      input: { ...input, cos: input.cos.slice(0, 5), sin: input.sin.slice(0, 5) },
      output: { ...outputs[i], cos: outputs[i].cos.slice(0, 5), sin: outputs[i].sin.slice(0, 5) },
    }))
    const dataOnly = fitPhysicsInformedMultiplier({ train, virtual, diffusivity: D, time: T, levels: J, weight: 0 })
    expect(Math.abs(dataOnly.final[1] - Math.exp(-D * (2 * Math.PI * 2) ** 2 * T))).toBeLessThan(1e-6)
    const out = applyModeMultiplier(dataOnly.final, train[0].input)
    expect(out.cos[1]).toBeCloseTo(train[0].output.cos[1], 6)
  })
})

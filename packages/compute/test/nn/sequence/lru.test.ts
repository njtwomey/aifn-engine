/**
 * aifn-compute/nn/sequence LRU: the ring initialisation (range, uniform by area), the impulse response of one state,
 * the scan against the convolution with complex read-in and read-out, and the unit white-noise gain of a normalised
 * state.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { complex, full, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import {
  lruConvolve,
  lruEigenvalues,
  lruKernel,
  lruLambda,
  lruRingInit,
  lruScan,
  type LruParams,
} from 'aifn-compute/nn/sequence'

const flat = (v: Value) => toFlat(v as Tensor)

const two: LruParams = {
  nuLog: tensor([Math.log(-Math.log(0.9)), Math.log(-Math.log(0.99))]),
  thetaLog: tensor([Math.log(Math.PI / 4), Math.log(Math.PI / 20)]),
  gammaLog: tensor([0.5 * Math.log(1 - 0.81), 0.5 * Math.log(1 - 0.9801)]),
}

describe('LRU', () => {
  it('draws eigenvalues on the ring, uniformly by area (E r^2 = 0.9041 against 0.9040)', () => {
    const p = lruRingInit(4000, { rMin: 0.9, rMax: 0.999, maxPhase: Math.PI / 10 }, stream(0))
    const modulus = flat(lruEigenvalues(p).modulus)
    const phases = flat(p.thetaLog).map(Math.exp)
    expect(modulus.every((r) => r >= 0.9 - 1e-12 && r <= 0.999 + 1e-12)).toBe(true)
    expect(phases.every((t) => t >= 0 && t <= Math.PI / 10 + 1e-12)).toBe(true)
    const meanR2 = modulus.reduce((a, r) => a + r * r, 0) / modulus.length
    expect(Math.abs(meanR2 - (0.81 + 0.998001) / 2)).toBeLessThan(0.003)
  })

  it('gives eigenvalues r e^{i theta} as a complex vector', () => {
    const { re, im } = lruEigenvalues(two)
    expect(flat(re)[0]).toBeCloseTo(0.9 * Math.cos(Math.PI / 4), 14)
    expect(flat(im)[1]).toBeCloseTo(0.99 * Math.sin(Math.PI / 20), 14)
    const lambda = toFlat(lruLambda(two) as Tensor)
    expect(lambda[0]).toBeCloseTo(flat(re)[0], 15)
    expect(lambda[3]).toBeCloseTo(flat(im)[1], 15)
  })

  it('has the impulse response 0.4359 0.9^k cos(k pi/4) through the first state', () => {
    const impulse = tensor([1, 0, 0, 0, 0, 0])
    const { y } = lruScan(impulse, two, { B: tensor([1, 1]), C: tensor([1, 0]) })
    const want = [0, 1, 2, 3, 4, 5].map((k) => Math.sqrt(0.19) * 0.9 ** k * Math.cos((k * Math.PI) / 4))
    flat(y).forEach((v, k) => expect(v).toBeCloseTo(want[k], 12))
  })

  it('scan equals convolution with complex B and C and a skip (16 states, 200 steps)', () => {
    const u = tensor(Array.from({ length: 200 }, (_, k) => Math.sin(0.3 * k) + 0.5 * Math.cos(1.7 * k)))
    const p = lruRingInit(16, { rMin: 0.5, rMax: 0.99, maxPhase: 2 * Math.PI }, stream(3))
    const readout = {
      B: complex(full([16], 1), full([16], 0.5)),
      C: complex(full([16], 0.25), full([16], -0.25)),
      D: 0.3,
    }
    const ys = flat(lruScan(u, p, readout).y)
    const yc = flat(lruConvolve(u, p, readout))
    expect(Math.max(...ys.map((v, k) => Math.abs(v - yc[k])))).toBeLessThan(1e-9)
  })

  it('normalises a state to unit white-noise gain: sum |gamma lambda^k|^2 = 1 at r = 0.99', () => {
    const one: LruParams = {
      nuLog: tensor([Math.log(-Math.log(0.99))]),
      thetaLog: tensor([Math.log(Math.PI / 20)]),
      gammaLog: tensor([0.5 * Math.log(1 - 0.9801)]),
    }
    const K = flat(lruKernel(one, { B: tensor([1]), C: tensor([1]) }, 3000))
    const Kq = flat(lruKernel(one, { B: tensor([1]), C: complex(tensor([0]), tensor([-1])) }, 3000))
    const gain = K.reduce((a, v, k) => a + v * v + Kq[k] * Kq[k], 0)
    expect(Math.abs(gain - 1)).toBeLessThan(1e-6)
  })
})

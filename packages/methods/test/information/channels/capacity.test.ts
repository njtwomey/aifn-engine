/** Channel capacity and rate–distortion by Blahut–Arimoto against known closed forms (BSC 1 − H₂(p), BEC 1 − ε, the Z
 * channel, R(D) = H₂(p) − H₂(D) for a binary source under Hamming distortion). */
import { describe, expect, it } from 'vitest'
import { channelCapacity, rateDistortion } from 'aifn-methods/information/channels'
import { binaryEntropy } from 'aifn-compute/numerics/special'

const close = (a: number, b: number, tol = 1e-12) =>
  expect(Math.abs(a - b) / Math.max(1, Math.abs(b))).toBeLessThan(tol)

describe('info: Blahut–Arimoto', () => {
  it('known capacities', () => {
    close(
      channelCapacity(
        [
          [0.9, 0.1],
          [0.1, 0.9],
        ],
        { base: 2 },
      ).capacity,
      1 - binaryEntropy(0.1, 2),
      1e-9,
    )
    close(
      channelCapacity(
        [
          [0.7, 0.3, 0],
          [0, 0.3, 0.7],
        ],
        { base: 2 },
      ).capacity,
      0.7,
      1e-9,
    )
    const z = channelCapacity(
      [
        [1, 0],
        [0.4, 0.6],
      ],
      { base: 2 },
    )
    expect(z.converged).toBe(true)
    close(z.capacity, Math.log2(1 + 0.6 * 0.4 ** (0.4 / 0.6)), 1e-9)
  })

  it('rate–distortion of a binary source under Hamming distortion', () => {
    for (const beta of [1, 2, 4]) {
      const r = rateDistortion(
        [0.5, 0.5],
        [
          [0, 1],
          [1, 0],
        ],
        beta,
        { base: 2 },
      )
      close(r.rate, 1 - binaryEntropy(r.distortion, 2), 1e-9)
    }
    const p = 0.2
    const r = rateDistortion(
      [1 - p, p],
      [
        [0, 1],
        [1, 0],
      ],
      3,
      { base: 2 },
    )
    close(r.rate, binaryEntropy(p, 2) - binaryEntropy(r.distortion, 2), 1e-8)
  })
})

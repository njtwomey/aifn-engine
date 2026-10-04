import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  boxDomain,
  clipToDomain,
  discreteDomain,
  domainContains,
  domainDimension,
  domainSize,
  sampleDomain,
} from 'aifn-compute/foundation/space'

describe('domains', () => {
  it('a discrete domain holds the integers 0 … n − 1, with optional names', () => {
    const d = discreteDomain(4, ['up', 'right', 'down', 'left'])
    expect(d).toEqual({ kind: 'discrete', n: 4, names: ['up', 'right', 'down', 'left'] })
    expect(domainSize(d)).toBe(4)
    expect([0, 3].map((x) => domainContains(d, x))).toEqual([true, true])
    expect([-1, 4, 1.5, '1', null].map((x) => domainContains(d, x))).toEqual([false, false, false, false, false])
    expect(() => discreteDomain(0)).toThrow()
    expect(() => discreteDomain(2, ['a'])).toThrow()
  })

  it('sampleDomain draws uniformly and reproducibly', () => {
    const d = discreteDomain(3)
    const draws = Array.from({ length: 3000 }, (_, i) => sampleDomain(child(stream(1), i), d))
    expect(draws.every((x) => domainContains(d, x))).toBe(true)
    for (let v = 0; v < 3; v++) expect(Math.abs(draws.filter((x) => x === v).length / 3000 - 1 / 3)).toBeLessThan(0.04)
    expect(sampleDomain(stream(5), d)).toBe(sampleDomain(stream(5), d))
  })

  it('a box holds real arrays inside its bounds and reports itself continuous', () => {
    const b = boxDomain([-1, -1, -8], [1, 1, 8], { names: ['cos', 'sin', 'speed'] })
    expect(b).toEqual({ kind: 'box', low: [-1, -1, -8], high: [1, 1, 8], shape: [3], names: ['cos', 'sin', 'speed'] })
    expect(domainSize(b)).toBe(Infinity)
    expect(domainDimension(b)).toBe(3)
    expect(domainDimension(discreteDomain(5))).toBe(1)
    expect(domainContains(b, Float64Array.of(0.5, -1, 8))).toBe(true)
    expect(domainContains(b, [0, 0, 9])).toBe(false)
    expect(domainContains(b, [0, 0])).toBe(false)
    expect(domainContains(b, 'x')).toBe(false)
    expect(boxDomain(0, 1, { shape: [2, 2] })).toMatchObject({ low: [0, 0, 0, 0], high: [1, 1, 1, 1], shape: [2, 2] })
    expect(() => boxDomain([1], [0])).toThrow()
    expect(() => boxDomain([0, 0], [1], { shape: [2] })).toThrow()
  })

  it('clipToDomain projects onto a box or a discrete range', () => {
    const b = boxDomain([-2], [2])
    expect(Array.from(clipToDomain(b, [5]))).toEqual([2])
    expect(Array.from(clipToDomain(b, [-0.5]))).toEqual([-0.5])
    expect(clipToDomain(discreteDomain(3), 7.2)).toBe(2)
    expect(clipToDomain(discreteDomain(3), -1)).toBe(0)
  })

  it('sampleDomain draws a box uniformly, independently per element, and reproducibly', () => {
    const b = boxDomain([-2, 10], [2, 11])
    const draws = Array.from({ length: 2000 }, (_, i) => sampleDomain(child(stream(2), i), b))
    expect(draws.every((x) => domainContains(b, x))).toBe(true)
    const mean = (k: number) => draws.reduce((s, x) => s + x[k], 0) / draws.length
    expect(mean(0)).toBeCloseTo(0, 1)
    expect(mean(1)).toBeCloseTo(10.5, 1)
    // Independent elements: the two coordinates are uncorrelated.
    const cov = draws.reduce((s, x) => s + (x[0] - mean(0)) * (x[1] - mean(1)), 0) / draws.length
    expect(Math.abs(cov)).toBeLessThan(0.05)
    expect(Array.from(sampleDomain(stream(5), b))).toEqual(Array.from(sampleDomain(stream(5), b)))
    expect(() => sampleDomain(stream(0), boxDomain([-Infinity], [0]))).toThrow()
  })
})

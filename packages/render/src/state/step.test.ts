import { describe, expect, it } from 'vitest'
import { formatField, snapToStep } from './step'

describe('snapToStep', () => {
  it('snaps to multiples of the step, not offsets from min', () => {
    expect(snapToStep(0.4, 0.02, 10, 0.05)).toBe(0.4)
    expect(snapToStep(0.43, 0.02, 10, 0.05)).toBe(0.45)
    expect(snapToStep(12, 0.2, 80, 0.5)).toBe(12)
    expect(snapToStep(12.3, 0.2, 80, 0.5)).toBe(12.5)
  })
  it('keeps off-grid bounds reachable and clamps', () => {
    expect(snapToStep(0.021, 0.02, 10, 0.05)).toBe(0.02)
    expect(snapToStep(-5, 0.02, 10, 0.05)).toBe(0.02)
    expect(snapToStep(9.99, 0.2, 9.9, 0.5)).toBe(9.9)
    expect(snapToStep(9.6, 0.2, 9.9, 0.5)).toBe(9.5)
  })
  it('leaves no floating-point noise', () => {
    expect(snapToStep(0.30000000000000004, 0, 1, 0.1)).toBe(0.3)
    expect(snapToStep(0.7, 0, 1, 0.1)).toBe(0.7)
    expect(snapToStep(1e-4 * 3, 0, 1, 1e-4)).toBe(0.0003)
  })
  it('only clamps without a positive step', () => {
    expect(snapToStep(0.123, 0, 1, 0)).toBe(0.123)
    expect(snapToStep(2, 0, 1, 0)).toBe(1)
  })
})

describe('formatField', () => {
  it('shows whole numbers in full', () => {
    expect(formatField(123456)).toBe('123456')
    expect(formatField(200000)).toBe('200000')
    expect(formatField(-7)).toBe('-7')
  })
  it('keeps large fractions readable and small ones as formatNumber does', () => {
    expect(formatField(123456.75)).toBe('123456.75')
    expect(formatField(0.4)).toBe('0.4')
    expect(formatField(1e-5)).toBe('1e-5')
  })
})

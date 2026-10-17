/**
 * aifn-methods/neural/sequence `convolutionMultiplies`: the long-convolution note's counts at $L = 1024$ (direct
 * 524,800, padded length 2048).
 */
import { describe, expect, it } from 'vitest'
import { convolutionMultiplies } from 'aifn-methods/neural/sequence'

describe('convolution multiplies', () => {
  it('counts L(L + 1)/2 directly and three padded FFTs plus the pointwise products', () => {
    const c = convolutionMultiplies(1024)
    expect(c.direct).toBe(524800)
    expect(c.padded).toBe(2048)
    expect(c.fft).toBe(3 * 1024 * 11 + 2048)
  })
})

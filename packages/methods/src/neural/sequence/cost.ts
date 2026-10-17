/**
 * The multiply counts of one long causal convolution, directly and by FFT, for the long-convolution note's comparison
 * of Hyena's filters with attention. Illustrative, and in its own convention: the direct sum counts its $L(L + 1)/2$
 * real multiplications, and the FFT counts complex multiplications, $\tfrac{M}{2}\log_2 M$ per radix-2 transform of
 * $M$ points (three transforms) plus $M$ complex pointwise products, each counted once. This is not the convention of
 * `convolutionAlgorithmCosts` in `aifn-methods/neural/convolutional`, which counts real multiplications
 * ($P \log_2 P$ per real transform and 4 per complex product over the Hermitian half); the two do not compare
 * number for number.
 */

import type { Size } from 'aifn-compute/foundation/contracts'

/**
 * The multiply counts of one causal convolution of length $L$: directly $L(L + 1)/2$ real multiplications; by FFT
 * with zero padding to the next power of two $M \ge 2L$, three transforms of $\tfrac{M}{2}\log_2 M$ complex
 * multiplications each plus $M$ complex pointwise products (complex multiplications counted as one each, not as the
 * real multiplications of `convolutionAlgorithmCosts`).
 *
 * @param length The sequence (and filter) length $L$.
 * @returns The `direct` and `fft` counts and the `padded` transform length $M$.
 *
 * @example The FFT wins from a few dozen taps
 * for (const L of [16, 64, 1024]) print(L, convolutionMultiplies(L))
 */
export function convolutionMultiplies(length: Size): { direct: number; fft: number; padded: number } {
  let M = 1
  while (M < 2 * length) M *= 2
  return { direct: (length * (length + 1)) / 2, fft: 3 * (M / 2) * Math.log2(M) + M, padded: M }
}

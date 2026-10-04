/** Deterministic test signals: chirps, tones and sample times. */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { readValues } from 'aifn-compute/foundation/fourier'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Sample times t_i = i / fs for i = 0, …, n − 1 (`aifn-compute/signal`'s `sampleTimes` reads them from a `Signal`). */
export function uniformTimes(n: number, fs = 1): Tensor {
  return fromData(Float64Array.from({ length: n }, (_, i) => i / fs))
}

/**
 * A frequency sweep cos(φ(t) + phi), as `scipy.signal.chirp(t, f0, t1, f1, method)` except that `phi` is in radians:
 * the instantaneous frequency goes from f0 at t = 0 to f1 at t = t1, linearly, quadratically (vertex at 0) or
 * logarithmically (geometric; f0, f1 > 0).
 */
export function chirp(
  t: Tensor | ArrayLike<number>,
  f0: number,
  t1: number,
  f1: number,
  { method = 'linear', phi = 0 }: { method?: 'linear' | 'quadratic' | 'logarithmic'; phi?: number } = {},
): Tensor {
  const ts = readValues(t)
  const out = ts.map((ti) => {
    let phase: number
    if (method === 'linear') phase = 2 * Math.PI * (f0 * ti + (0.5 * (f1 - f0) * ti * ti) / t1)
    else if (method === 'quadratic') phase = 2 * Math.PI * (f0 * ti + ((f1 - f0) * ti ** 3) / (3 * t1 * t1))
    else {
      if (!(f0 > 0 && f1 > 0)) throw new DomainError('chirp', 'chirp: a logarithmic sweep needs positive frequencies')
      if (f0 === f1) phase = 2 * Math.PI * f0 * ti
      else {
        const beta = t1 / Math.log(f1 / f0)
        phase = 2 * Math.PI * beta * f0 * ((f1 / f0) ** (ti / t1) - 1)
      }
    }
    return Math.cos(phase + phi)
  })
  return fromData(out)
}

/** A sum of sinusoids Σ a_k sin(2π f_k t + φ_k) at the given times. */
export function tones(
  t: Tensor | ArrayLike<number>,
  components: readonly { frequency: number; amplitude?: number; phase?: number }[],
): Tensor {
  const ts = readValues(t)
  return fromData(
    ts.map((ti) =>
      components.reduce(
        (s, c) => s + (c.amplitude ?? 1) * Math.sin(2 * Math.PI * c.frequency * ti + (c.phase ?? 0)),
        0,
      ),
    ),
  )
}

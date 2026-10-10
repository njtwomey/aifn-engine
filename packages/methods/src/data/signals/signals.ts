/**
 * Deterministic test signals: frequency sweeps (chirps), sums of sinusoids, and the uniform sample times they are
 * evaluated at.
 *
 * Nothing here draws at random: each function maps given times $t_i$ to values, so a signal is reproduced exactly from
 * its arguments. Frequencies are in cycles per unit of time (Hz when $t$ is in seconds) and phases in radians.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { readValues } from 'aifn-compute/foundation/fourier'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Sample times $t_i = i / f_s$ for $i = 0, \dots, n - 1$ (`aifn-compute/signal`'s `sampleTimes` reads them from a
 * `Signal`).
 *
 * @param n The number of samples.
 * @param fs The sample rate $f_s$, in samples per unit of time; the spacing of the times is $1 / f_s$.
 * @returns The $n$ times, a float64 vector starting at 0.
 *
 * @example Eight samples at 4 Hz
 * print('t =', uniformTimes(8, 4))
 */
export function uniformTimes(n: number, fs = 1): Tensor {
  return fromData(Float64Array.from({ length: n }, (_, i) => i / fs))
}

/**
 * A frequency sweep $\cos(\phi(t) + \varphi)$, as `scipy.signal.chirp(t, f0, t1, f1, method)` except that the offset
 * $\varphi$ (`phi`) is in radians, not degrees. The instantaneous frequency $\phi'(t) / 2\pi$ goes from $f_0$ at
 * $t = 0$ to $f_1$ at $t = t_1$ (and carries on past $t_1$ along the same law):
 *
 * - linear: $\phi(t) = 2\pi\left(f_0 t + \tfrac{1}{2}(f_1 - f_0) t^2 / t_1\right)$;
 * - quadratic, with its vertex at $t = 0$: $\phi(t) = 2\pi\left(f_0 t + \tfrac{1}{3}(f_1 - f_0) t^3 / t_1^2\right)$;
 * - logarithmic (geometric): $\phi(t) = 2\pi \beta f_0 \left((f_1 / f_0)^{t / t_1} - 1\right)$ with
 *   $\beta = t_1 / \log(f_1 / f_0)$, and $\phi(t) = 2\pi f_0 t$ when $f_0 = f_1$. Throws `DomainError` unless
 *   $f_0, f_1 > 0$.
 *
 * @param t The times to evaluate at: a tensor (read flat) or an array of numbers.
 * @param f0 The frequency $f_0$ at $t = 0$.
 * @param t1 The time $t_1$ at which the frequency reaches $f_1$.
 * @param f1 The frequency $f_1$ at $t = t_1$.
 * @param options The sweep law and the phase offset.
 * @param options.method How the frequency moves from $f_0$ to $f_1$: `'linear'`, `'quadratic'` or `'logarithmic'`.
 * @param options.phi The phase offset $\varphi$ added inside the cosine, in radians.
 * @returns The sweep at each time, a float64 vector the length of `t`.
 *
 * @example A linear sweep from 0 to 4 Hz over one second
 * const t = uniformTimes(8, 8)
 * print('x =', chirp(t, 0, 1, 4))
 *
 * @example A logarithmic sweep needs positive frequencies
 * try {
 *   chirp([0, 0.5, 1], 0, 1, 10, { method: 'logarithmic' })
 * } catch (e) {
 *   print(e.message)
 * }
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

/**
 * A sum of sinusoids $\sum_k a_k \sin(2\pi f_k t + \varphi_k)$ at the given times.
 *
 * @param t The times to evaluate at: a tensor (read flat) or an array of numbers.
 * @param components One entry per sinusoid: its `frequency` $f_k$, its `amplitude` $a_k$ (default 1) and its `phase`
 *   $\varphi_k$ in radians (default 0). An empty list gives zeros.
 * @returns The sum at each time, a float64 vector the length of `t`.
 *
 * @example A 1 Hz tone plus a weaker 2 Hz tone
 * const t = uniformTimes(4, 4)
 * print('x =', tones(t, [{ frequency: 1 }, { frequency: 2, amplitude: 0.5, phase: Math.PI / 2 }]))
 */
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

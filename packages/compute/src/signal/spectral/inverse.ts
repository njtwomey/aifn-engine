/**
 * The inverse short-time Fourier transform by weighted overlap-add, as `scipy.signal.istft` (Griffin & Lim, 1984,
 * "Signal estimation from modified short-time Fourier transform", IEEE Trans. ASSP 32(2)), and the constant
 * overlap-add (COLA) and nonzero overlap-add (NOLA) tests of a window, as `scipy.signal.check_COLA`/`check_NOLA`.
 */

import { astype, copy, dense, fromData, isTensor, readonlyData, type Tensor } from 'aifn-compute/foundation/tensor'
import { ifft, irfft } from 'aifn-compute/foundation/fourier'
import type { Scalar, Signal, Size, TimeFrequency } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { windowValues, type WindowInput } from 'aifn-compute/signal/windows'
import { signal } from '../signal'

/**
 * Sums over the overlapped positions of one hop: $\sum_k g(w[i + k \cdot \text{step}])$ for $i < \text{step}$, as
 * scipy's `_binsums` (a last partial hop adds to the first positions).
 *
 * @param win The window values; not modified.
 * @param step The hop, in samples.
 * @param g The function applied to each window value before summing (the identity for COLA, the square for NOLA).
 * @returns The `step` sums.
 */
function binSums(win: Float64Array, step: Size, g: (w: number) => number): Float64Array {
  const n = win.length
  const sums = new Float64Array(step)
  for (let k = 0; k < Math.floor(n / step); k++) for (let i = 0; i < step; i++) sums[i] += g(win[k * step + i])
  const rest = n % step
  for (let i = 0; i < rest; i++) sums[i] += g(win[n - rest + i])
  return sums
}

/**
 * The hop of a segmentation, after checking it. Throws `DomainError` unless `nperseg` is a positive integer and
 * `noverlap` an integer in $[0, \text{nperseg})$.
 *
 * @param nperseg The segment length.
 * @param noverlap The samples shared by consecutive segments.
 * @param where The caller's name, for error messages.
 * @returns The hop $\text{nperseg} - \text{noverlap}$.
 */
function checkOverlap(nperseg: Size, noverlap: Size, where: string): Size {
  if (!(Number.isInteger(nperseg) && nperseg >= 1)) throw new DomainError(where, `${where}: nperseg must be ≥ 1`)
  if (!(Number.isInteger(noverlap) && noverlap >= 0 && noverlap < nperseg))
    throw new DomainError(where, `${where}: noverlap must be in [0, nperseg)`)
  return nperseg - noverlap
}

/**
 * The constant overlap-add (COLA) condition: shifted copies of the window, hop $\text{nperseg} - \text{noverlap}$
 * apart, sum to a constant (to within `tolerance` of their median). Under COLA, overlap-adding the unmodified segments
 * of a signal returns the signal times that constant; it is sufficient for an STFT to be invertible but not necessary
 * (see `checkNola`). The window is built periodic, as `stft` builds it. Throws `DomainError` for an invalid
 * segmentation.
 *
 * @param window The window: a spec or explicit values of length `nperseg`.
 * @param nperseg The segment length.
 * @param noverlap The samples shared by consecutive segments, in $[0, \text{nperseg})$.
 * @param options Options.
 * @param options.tolerance The largest deviation of an overlap-added sum from the median allowed (default 1e-10).
 * @returns Whether the window and overlap satisfy COLA.
 *
 * @example Hann at half overlap, Hamming, and too little overlap
 * print('hann, 8, 4:', checkCola('hann', 8, 4))
 * print('hamming, 8, 4:', checkCola('hamming', 8, 4))
 * print('hann, 8, 2:', checkCola('hann', 8, 2))
 */
export function checkCola(window: WindowInput, nperseg: Size, noverlap: Size, { tolerance = 1e-10 } = {}): boolean {
  const step = checkOverlap(nperseg, noverlap, 'checkCola')
  const sums = Array.from(binSums(windowValues(window, nperseg, true), step, (w) => w)).sort((a, b) => a - b)
  const mid = sums.length >> 1
  const median = sums.length % 2 ? sums[mid] : 0.5 * (sums[mid - 1] + sums[mid])
  return sums.every((s) => Math.abs(s - median) < tolerance)
}

/**
 * The nonzero overlap-add (NOLA) condition: the squared window, overlap-added with hop
 * $\text{nperseg} - \text{noverlap}$, is positive everywhere (above `tolerance`). NOLA is necessary and sufficient
 * for `istft` to invert `stft` exactly: the least-squares overlap-add divides by that sum. Throws `DomainError` for an
 * invalid segmentation.
 *
 * @param window The window: a spec (built periodic) or explicit values of length `nperseg`.
 * @param nperseg The segment length.
 * @param noverlap The samples shared by consecutive segments, in $[0, \text{nperseg})$.
 * @param options Options.
 * @param options.tolerance The least overlap-added sum of squares that counts as positive (default 1e-10).
 * @returns Whether the window and overlap satisfy NOLA.
 *
 * @example A window that is zero at its ends needs overlap
 * // The periodic Hann window is 0 at its first sample, so without overlap that sample is lost.
 * print('hann, no overlap:', checkNola('hann', 8, 0))
 * print('hann, half overlap:', checkNola('hann', 8, 4))
 * print('boxcar, no overlap:', checkNola('boxcar', 8, 0))
 */
export function checkNola(window: WindowInput, nperseg: Size, noverlap: Size, { tolerance = 1e-10 } = {}): boolean {
  const step = checkOverlap(nperseg, noverlap, 'checkNola')
  return Math.min(...binSums(windowValues(window, nperseg, true), step, (w) => w * w)) > tolerance
}

/** Options for `istft`. */
export type IstftOptions = {
  /** Sampling frequency. Default: from the input's frequency axis ($\Delta f \cdot n_\text{fft}$), else 1. */
  fs?: Scalar
  /** The window `stft` used. Default `'hann'`. */
  window?: WindowInput
  /**
   * Segment length. Default: the input's `window.length`, else $2(n_\text{freq} - 1)$ for one-sided input
   * ($n_\text{freq}$ two-sided).
   */
  nperseg?: Size
  /** Overlap. Default: from the input's `window.hop`, else $\lfloor \text{nperseg}/2 \rfloor$. */
  noverlap?: Size
  /**
   * FFT length. Default: implied by $\Zmat$, $2(n_\text{freq} - 1)$ one-sided (`nperseg` when that is one less than
   * an odd `nperseg`) or $n_\text{freq}$ two-sided.
   */
  nfft?: Size
  /** Whether the input holds one-sided spectra of a real signal (irfft) or two-sided ones (ifft). Default true. */
  onesided?: boolean
  /** Whether `stft` padded $\lfloor \text{nperseg}/2 \rfloor$ samples at each end (strip them). Default true. */
  boundary?: boolean
}

/**
 * The inverse STFT, as `scipy.signal.istft` (scaling `'spectrum'`): each column of $\Zmat$ is inverse transformed,
 * multiplied by the window's sum (undoing `stft`'s scaling), windowed again and overlap-added; the sum is divided by
 * the overlap-added squared window. That division is the least-squares inverse of Griffin & Lim: for any $\Zmat$ it
 * returns the signal whose STFT is closest to $\Zmat$, and for the STFT of a signal it returns that signal exactly
 * whenever the window and overlap satisfy NOLA (`checkNola`), which `istft` checks and otherwise refuses with a
 * `DomainError`.
 *
 * $\Zmat$ is a `TimeFrequency` from `stft` (complex $[f, t]$; its time axis gives $t_0$) or a complex128 tensor
 * $[f, t]$. Returns a real `Signal` for one-sided input and a complex one for two-sided input; its length is the
 * padded length that `stft` transformed (trim to the original length when `stft` padded the end). Throws `ShapeError`
 * when $\Zmat$ is not two-dimensional and `DomainError` for an invalid segmentation or an `nfft` below `nperseg`.
 *
 * @param Z The STFT: a `TimeFrequency` from `stft`, whose window length, hop and axes supply the defaults, or a bare
 *   complex tensor $[f, t]$.
 * @param options The sample rate, window, segment length, overlap, FFT length, sidedness and boundary; see
 *   `IstftOptions`. The window must be the one `stft` used.
 * @returns The signal, with $f_s$ and $t_0$ from the options or the input's axes.
 *
 * @example Round trip
 * const x = Array.from({ length: 16 }, (_, i) => i)
 * const y = istft(stft(x, { nperseg: 8 }))
 * print('x back =', y.data)
 * print('largest error =', max(abs(sub(y.data, tensor(x)))))
 */
export function istft(Z: TimeFrequency | Tensor, options: IstftOptions = {}): Signal {
  const tf = isTensor(Z) ? undefined : Z
  const values = isTensor(Z) ? Z : Z.values
  if (values.shape.length !== 2) throw new ShapeError('istft', `istft: Z must be [f, t], got [${values.shape}]`)
  const [nfreq, T] = values.shape
  const onesided = options.onesided ?? true
  const nDefault = onesided ? 2 * (nfreq - 1) : nfreq
  const nperseg = options.nperseg ?? tf?.window?.length ?? nDefault
  if (!(Number.isInteger(nperseg) && nperseg >= 1)) throw new DomainError('istft', 'istft: nperseg must be ≥ 1')
  // As scipy: an odd nperseg one more than 2(nfreq − 1) means no FFT padding; otherwise the FFT length is implied by Z.
  const nfft = options.nfft ?? (onesided && nperseg === nDefault + 1 ? nperseg : nDefault)
  if (nfft < nperseg) throw new DomainError('istft', 'istft: nfft must be at least nperseg')
  const noverlap =
    options.noverlap ?? (tf?.window ? tf.window.length - tf.window.hop : undefined) ?? Math.floor(nperseg / 2)
  const step = checkOverlap(nperseg, noverlap, 'istft')
  const windowInput = options.window ?? 'hann'
  if (!checkNola(windowInput, nperseg, noverlap))
    throw new DomainError('istft', 'istft: the window and overlap fail NOLA, so the STFT is not invertible')
  const boundary = options.boundary ?? true
  const f = tf ? dense.data(tf.f) : undefined
  const fs = options.fs ?? (f && f.length > 1 ? (f[1] - f[0]) * nfft : 1)

  // Columns of Z as rows [t, f] (complex), inverse transformed along the frequency axis.
  const z = readonlyData(copy(astype(values, 'complex128')))!
  const rows = new Float64Array(2 * T * nfreq)
  for (let j = 0; j < nfreq; j++)
    for (let t = 0; t < T; t++) {
      rows[2 * (t * nfreq + j)] = z[2 * (j * T + t)]
      rows[2 * (t * nfreq + j) + 1] = z[2 * (j * T + t) + 1]
    }
  const Zt = fromData(rows, [T, nfreq], 'complex128')
  const segs = onesided ? dense.data(irfft(Zt, { n: nfft })) : readonlyData(ifft(Zt, { n: nfft }))!
  const width = onesided ? 1 : 2
  const win = windowValues(windowInput, nperseg, true)
  let winSum = 0
  for (const w of win) winSum += w
  const length = nperseg + (T - 1) * step
  const x = new Float64Array(width * length)
  const norm = new Float64Array(length)
  for (let t = 0; t < T; t++)
    for (let i = 0; i < nperseg; i++) {
      const at = t * step + i
      const w = win[i] * winSum
      for (let c = 0; c < width; c++) x[width * at + c] += segs[width * (t * nfft + i) + c] * w
      norm[at] += win[i] * win[i]
    }
  const cut = boundary ? Math.floor(nperseg / 2) : 0
  const n = Math.max(0, length - 2 * cut)
  const out = new Float64Array(width * n)
  for (let i = 0; i < n; i++) {
    const d = norm[i + cut] > 1e-10 ? norm[i + cut] : 1
    for (let c = 0; c < width; c++) out[width * i + c] = x[width * (i + cut) + c] / d
  }
  const t0Segment = tf ? dense.data(tf.t)[0] : 0
  // stft centres segment 0 on t0 with boundary padding, and on t0 + (nperseg/2)/fs without it.
  const t0 = tf ? (boundary ? t0Segment : t0Segment - nperseg / 2 / fs) : 0
  return signal(fromData(out, [n], onesided ? 'float64' : 'complex128'), { fs, t0 })
}

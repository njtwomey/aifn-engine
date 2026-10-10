/**
 * The shared layer of `aifn-compute/signal`: the constructors and readers of the signal-processing objects defined in
 * `aifn-compute/foundation/contracts` (design S §2.13): `Signal` (samples with a sample rate), `Spectrum` and
 * `TimeFrequency`.
 *
 * Every function of the family takes a `SignalInput` (a `Signal`, or a bare tensor or array, which has a sample rate
 * of 1 unless an `fs` option says otherwise) and returns these objects, so axes and units carry through a chain.
 * Frequencies follow the owner's decision 14: Hz with `fs` for sampled data, always tagged by `axis`. Conventions
 * follow scipy.signal (Virtanen et al., 2020, "SciPy 1.0", Nature Methods 17). The file also registers its functions
 * (`signalFunctions`).
 */

import { abs, angle, complexAbs, dense, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size, Signal, Spectrum, TimeFrequency, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { decibels } from 'aifn-compute/foundation/fourier'
import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'

export type { Signal, Spectrum, TimeFrequency } from 'aifn-compute/foundation/contracts'

/** What a signal-processing function accepts: a `Signal`, or bare samples (a rank-1 tensor or an array). */
export type SignalInput = Signal | VectorLike

/** Options of `signal`. */
export type SignalOptions = {
  /** Samples per second. Default 1 (time in samples). */
  fs?: Scalar
  /** Time of sample 0, in seconds. Default 0. */
  t0?: Scalar
  /** Unit of the values, e.g. `V`, `Pa`, `a.u.`. */
  unit?: string
  /** Channel names, for a [channels, n] signal. */
  channels?: readonly string[]
}

/**
 * True for a `Signal` object: any non-null object with `kind: 'signal'`. Its fields are not checked.
 *
 * @param x Any value.
 * @returns Whether `x` is a `Signal`, narrowing its type.
 *
 * @example A signal, and bare samples
 * print('signal:', isSignal(signal([1, 2, 3])))
 * print('array:', isSignal([1, 2, 3]))
 */
export function isSignal(x: unknown): x is Signal {
  return typeof x === 'object' && x !== null && (x as { kind?: unknown }).kind === 'signal'
}

/**
 * A `Signal` from samples: `data` is $[n]$ or $[\text{channels}, n]$ (a copy is taken of an array; a tensor is kept).
 * A `Signal` passed in keeps its data and has the given options replace its own. Throws `ShapeError` for data of
 * another rank and `DomainError` for a sample rate that is not positive.
 *
 * @param data The samples: an array or a rank-1 tensor of $n$ values, a $[\text{channels}, n]$ tensor, or a `Signal`
 *   to relabel.
 * @param options The sample rate `fs` (default: the signal's, or 1), start time `t0` (default: the signal's, or 0),
 *   `unit` and `channels`; each given one replaces the signal's.
 * @returns The signal.
 *
 * @example One cycle of a 1 Hz sine sampled at 4 Hz
 * const s = signal([0, 1, 0, -1], { fs: 4, unit: 'V' })
 * print('fs =', s.fs, ' unit =', s.unit)
 * print('times =', sampleTimes(s))
 *
 * @example Relabel a signal: the data is kept, the options replace its own
 * const s = signal([0, 1, 0, -1], { fs: 4, unit: 'V' })
 * const faster = signal(s, { fs: 8 })
 * print('fs =', faster.fs, ' unit =', faster.unit)
 */
export function signal(data: SignalInput | Tensor, options: SignalOptions = {}): Signal {
  const base = isSignal(data) ? data : undefined
  const values = base ? base.data : isTensor(data) ? data : dense.vec(dense.toF64(data as VectorLike, 'signal'))
  if (values.shape.length < 1 || values.shape.length > 2)
    throw new ShapeError('signal', `signal: data must be [n] or [channels, n], got [${values.shape.join(', ')}]`)
  const fs = options.fs ?? base?.fs ?? 1
  if (!(fs > 0)) throw new DomainError('signal', 'signal: fs must be positive')
  const unit = options.unit ?? base?.unit
  const channels = options.channels ?? base?.channels
  return {
    kind: 'signal',
    data: values,
    fs,
    t0: options.t0 ?? base?.t0 ?? 0,
    ...(unit !== undefined ? { unit } : {}),
    ...(channels !== undefined ? { channels } : {}),
  }
}

/**
 * The samples of a rank-1 signal input: `values`, a fresh float64 array of them; `fs`, the sample rate in Hz; `t0`,
 * the time of the first sample in seconds; and `unit`, the unit of the values when the input was a `Signal` that has
 * one.
 */
export type Samples = { values: dense.F64; fs: Scalar; t0: Scalar; unit?: string }

/**
 * The samples of a single-channel `SignalInput` as a fresh float64 array (never shared with the input), with the
 * sample rate (the `fs` argument if given, else the signal's, else 1) and start time (the signal's, else 0). Throws
 * `ShapeError` for a multichannel signal.
 *
 * @param x The input: a single-channel `Signal`, or bare samples.
 * @param where The caller's name for error messages.
 * @param fs A sample rate that overrides the input's (an `fs` option of the caller).
 * @returns The samples, sample rate, start time and unit.
 */
export function readSamples(x: SignalInput, where: string, fs?: Scalar): Samples {
  if (isSignal(x)) {
    if (x.data.shape.length !== 1)
      throw new ShapeError(where, `${where}: expected a single-channel signal, got [${x.data.shape.join(', ')}]`)
    return { values: Float64Array.from(dense.toF64(x.data, where)), fs: fs ?? x.fs, t0: x.t0, unit: x.unit }
  }
  return { values: Float64Array.from(dense.toF64(x, where)), fs: fs ?? 1, t0: 0 }
}

/**
 * The sample times $t_0 + k/f_s$, $k = 0, \dots, n - 1$, of a signal, as a rank-1 tensor; $n$ is the length of the
 * last axis, so a multichannel signal gives one time per column.
 *
 * @param s The signal.
 * @returns The $n$ sample times, in seconds (in samples when the sample rate is 1).
 *
 * @example Three samples at 2 Hz, starting at 10 s
 * print('times =', sampleTimes(signal([1, 2, 3], { fs: 2, t0: 10 })))
 */
export function sampleTimes(s: Signal): Tensor {
  const n = s.data.shape[s.data.shape.length - 1]
  return fromData(
    Float64Array.from({ length: n }, (_, k) => s.t0 + k / s.fs),
    [n],
  )
}

/** The fields of a `Spectrum` other than `kind`: what `spectrum` takes. */
export type SpectrumFields = Omit<Spectrum, 'kind'>

/**
 * A `Spectrum` from its fields (the `kind` brand is added). The fields are not checked.
 *
 * @param fields The frequencies `f` and their `axis`, the `values` and what they hold (`quantity`), whether it is
 *   one- or two-sided, and optionally the sample rate and unit.
 * @returns The spectrum.
 *
 * @example A small power spectral density, read in decibels
 * const s = spectrum({ f: tensor([0, 1, 2]), axis: 'hz', values: tensor([4, 1, 0.01]), quantity: 'psd', sided: 'one' })
 * print('kind =', s.kind)
 * print('dB =', spectrumDecibels(s))
 */
export function spectrum(fields: SpectrumFields): Spectrum {
  return { kind: 'spectrum', ...fields }
}

/** The fields of a `TimeFrequency` other than `kind`: what `timeFrequency` takes. */
export type TimeFrequencyFields = Omit<TimeFrequency, 'kind'>

/**
 * A `TimeFrequency` raster from its fields (the `kind` brand is added). The fields are not checked.
 *
 * @param fields The times `t` and frequencies `f`, the $[f, t]$ `values` and what they hold (`quantity`), the
 *   `method` that made them, the `frequencyScale`, and optionally the `window`.
 * @returns The raster.
 *
 * @example A two-by-two raster: one row per frequency, one column per time
 * const tf = timeFrequency({
 *   t: tensor([0, 1]),
 *   f: tensor([0, 0.5]),
 *   values: tensor([[1, 2], [3, 4]]),
 *   quantity: 'power',
 *   method: 'stft',
 *   frequencyScale: 'linear',
 * })
 * print('kind =', tf.kind)
 * print('values, [f, t] =', tf.values)
 */
export function timeFrequency(fields: TimeFrequencyFields): TimeFrequency {
  return { kind: 'time-frequency', ...fields }
}

/**
 * The unit of a density or power spectrum of a signal whose values have unit `u`: `u²/Hz` for a density, `u²` for a
 * power.
 *
 * @param unit The unit of the signal's values; undefined gives undefined.
 * @param density True for a power spectral density, false for a power.
 * @returns The unit string, or undefined.
 */
export function powerUnit(unit: string | undefined, density: boolean): string | undefined {
  if (unit === undefined) return undefined
  return density ? `${unit}²/Hz` : `${unit}²`
}

/**
 * A complex128 tensor of the given shape from its real and imaginary parts. Throws `ShapeError` when the parts differ
 * in length.
 *
 * @param re The real parts, row-major.
 * @param im The imaginary parts, row-major, as many as `re`.
 * @param shape The shape of the result; its size must equal the parts' length.
 * @returns The complex tensor, its values interleaved as real and imaginary pairs.
 */
export function complexValues(re: ArrayLike<number>, im: ArrayLike<number>, shape: readonly Size[]): Tensor {
  if (re.length !== im.length) throw new ShapeError('complexValues', 'complexValues: parts differ in length')
  const d = new Float64Array(2 * re.length)
  for (let k = 0; k < re.length; k++) {
    d[2 * k] = re[k]
    d[2 * k + 1] = im[k]
  }
  return fromData(d, [...shape], 'complex128')
}

// ── Spectrum readers ─────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The values of a spectrum, or a tensor itself.
 *
 * @param s A spectrum or a tensor.
 * @returns `s.values` for a spectrum, `s` for a tensor.
 */
const valuesOf = (s: Spectrum | Tensor): Tensor => (isTensor(s) ? s : s.values)

/**
 * The modulus $\lvert v \rvert$ of a spectrum's values (or of a real or complex128 tensor): the magnitude of a complex
 * response or DFT, or the absolute value of real values.
 *
 * @param s A spectrum, or a tensor of values.
 * @returns A real tensor of the moduli, with the values' shape.
 *
 * @example The moduli of complex values
 * const z = complex(tensor([1, 0, -1, 3]), tensor([1, 2, 0, -4]))
 * print('|z| =', magnitude(z))
 */
export function magnitude(s: Spectrum | Tensor): Tensor {
  const v = valuesOf(s)
  return v.dtype === 'complex128' ? complexAbs(v) : abs(v)
}

/**
 * The phase of a spectrum's values (or of a tensor) in radians: the principal value in $(-\pi, \pi]$, unwrapped along
 * a vector with `unwrap` (as `numpy.unwrap`), in degrees with `degrees`. A real value has phase 0 or $\pi$. Throws
 * `ShapeError` when `unwrap` is asked of values that are not a vector.
 *
 * @param s A spectrum, or a tensor of values.
 * @param options `unwrap`: remove the jumps of $2\pi$ (vectors only); `degrees`: return degrees instead of radians
 *   (applied after unwrapping). Both default to false.
 * @returns A real tensor of phases, with the values' shape.
 *
 * @example Principal values, in radians and in degrees
 * const z = complex(tensor([1, 0, -1, 3]), tensor([1, 2, 0, -4]))
 * print('radians =', phase(z))
 * print('degrees =', phase(z, { degrees: true }))
 *
 * @example A steadily turning phase, wrapped and unwrapped
 * const z = expj(tensor([0, 1, 2, 3, 4, 5]))
 * print('wrapped =', phase(z))
 * print('unwrapped =', phase(z, { unwrap: true }))
 */
export function phase(s: Spectrum | Tensor, options: { unwrap?: boolean; degrees?: boolean } = {}): Tensor {
  let p = angle(valuesOf(s))
  if (options.unwrap) {
    if (p.shape.length !== 1) throw new ShapeError('phase', 'phase: unwrap needs a vector of values')
    p = unwrapPhase(p)
  }
  return options.degrees
    ? fromData(
        Float64Array.from(dense.toF64(p, 'phase'), (v) => (v * 180) / Math.PI),
        p.shape,
      )
    : p
}

/**
 * A spectrum in decibels, by its `quantity`: $10 \log_{10}(\lvert v \rvert / r)$ for powers (`psd`, `power`,
 * `coherence`) and $20 \log_{10}(\lvert v \rvert / r)$ for amplitudes (`amplitude`, `complex`, `response`), so the
 * power rule is never applied to an amplitude. Zero maps to $-\infty$.
 *
 * @param s The spectrum; its `quantity` chooses the rule.
 * @param options The reference.
 * @param options.reference The value $r$ that maps to 0 dB (default 1), in the units of the values.
 * @returns A real tensor of decibels, with the values' shape.
 *
 * @example The same values as powers and as amplitudes
 * const f = tensor([0, 1, 2])
 * const values = tensor([4, 1, 0.01])
 * print('as a PSD =', spectrumDecibels(spectrum({ f, axis: 'hz', values, quantity: 'psd', sided: 'one' })))
 * const amplitude = spectrum({ f, axis: 'hz', values, quantity: 'amplitude', sided: 'one' })
 * print('as amplitudes =', spectrumDecibels(amplitude))
 * print('as amplitudes, re 4 =', spectrumDecibels(amplitude, { reference: 4 }))
 */
export function spectrumDecibels(s: Spectrum, { reference = 1 }: { reference?: Scalar } = {}): Tensor {
  const power = s.quantity === 'psd' || s.quantity === 'power' || s.quantity === 'coherence'
  return decibels(magnitude(s), { power, reference })
}

/**
 * Unwraps a phase sequence, as `numpy.unwrap` (Itoh, 1982, "Analysis of the phase unwrapping algorithm", Applied
 * Optics 21(14)): wherever consecutive values jump by at least `discont`, the rest of the sequence is shifted by the
 * multiple of $2\pi$ that brings the jump into $[-\pi, \pi]$. With the default `discont` of $\pi$, consecutive
 * values then never jump by more than $\pi$.
 *
 * @param phase The phases, in radians.
 * @param options The jump threshold.
 * @param options.discont The smallest jump that is corrected (default $\pi$); a smaller value than $\pi$ acts as
 *   $\pi$, as in numpy.
 * @returns The unwrapped phases, a rank-1 tensor of the same length.
 *
 * @example A jump of nearly a full turn removed
 * print('unwrapped =', unwrapPhase([0, 3, -3, 0]))
 */
export function unwrapPhase(phase: VectorLike, { discont = Math.PI }: { discont?: Scalar } = {}): Tensor {
  const p = dense.toF64(phase, 'unwrapPhase')
  const out = Float64Array.from(p)
  let offset = 0
  for (let i = 1; i < p.length; i++) {
    const d = p[i] - p[i - 1]
    // numpy: map d into [−π, π), keeping +π for positive jumps; correct only when |d| ≥ discont.
    let dm = ((((d + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI
    if (dm === -Math.PI && d > 0) dm = Math.PI
    if (Math.abs(d) >= discont) offset += dm - d
    out[i] = p[i] + offset
  }
  return fromData(out, [out.length])
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const fn = definer<FunctionInfo>('function', 'signal')

fn(
  {
    key: 'signal',
    name: 'Signal',
    summary: 'Samples with a sample rate, start time and unit.',
    role: 'construction',
    returns: 'signal',
    notes: ['sampling-theorem'],
  },
  signal,
)
fn({ key: 'spectrum', name: 'Spectrum', role: 'construction', returns: 'spectrum' }, spectrum)
fn(
  { key: 'timeFrequency', name: 'Time–frequency representation', role: 'construction', returns: 'time-frequency' },
  timeFrequency,
)
fn({ key: 'sampleTimes', name: 'Sample times', role: 'property', notes: ['sampling-theorem'] }, sampleTimes)
fn({ key: 'magnitude', name: 'Magnitude', role: 'transform', notes: ['frequency-response'] }, magnitude)
fn(
  { key: 'phase', name: 'Phase', role: 'transform', notes: ['frequency-response', 'linear-phase-and-group-delay'] },
  phase,
)
fn(
  { key: 'spectrumDecibels', name: 'Spectrum in decibels', role: 'transform', notes: ['frequency-response'] },
  spectrumDecibels,
)
fn(
  {
    key: 'unwrapPhase',
    name: 'Unwrap phase',
    role: 'transform',
    notes: ['linear-phase-and-group-delay', 'instantaneous-frequency'],
  },
  unwrapPhase,
)

/** The functions of the module, keyed by name. */
export const signalFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', {
    signal,
    spectrum,
    timeFrequency,
    sampleTimes,
    magnitude,
    phase,
    spectrumDecibels,
    unwrapPhase,
  }) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

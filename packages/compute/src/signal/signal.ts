/**
 * The shared layer of `aifn-compute/signal`: the constructors and readers of the signal-processing objects defined in
 * `aifn-compute/foundation/contracts` (design S §2.13): `Signal` (samples with a sample rate), `Spectrum` and
 * `TimeFrequency`. Every function of the family takes a `SignalInput` (a `Signal`, or a bare tensor or array, which
 * means fs = 1 unless an `fs` option says otherwise) and returns these objects, so axes and units carry through a
 * chain. Frequencies follow the owner's decision 14: Hz with `fs` for sampled data, always tagged by `axis`.
 * Conventions follow scipy.signal (Virtanen et al., 2020, "SciPy 1.0", Nature Methods 17).
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

/** True for a `Signal` object (`kind: 'signal'`). */
export function isSignal(x: unknown): x is Signal {
  return typeof x === 'object' && x !== null && (x as { kind?: unknown }).kind === 'signal'
}

/**
 * A `Signal` from samples: `data` is [n] or [channels, n] (a copy is taken of an array; a tensor is kept). A `Signal`
 * passed in keeps its data and has the given options replace its own.
 *
 * @example signal([0, 1, 0, -1], { fs: 4 }) // one cycle of a 1 Hz sine sampled at 4 Hz
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

/** The samples of a rank-1 signal input, with its sample rate and start time. */
export type Samples = { values: dense.F64; fs: Scalar; t0: Scalar; unit?: string }

/**
 * The samples of a single-channel `SignalInput` as a fresh float64 array (never shared with the input), with fs (the
 * `fs` option if given, else the signal's, else 1) and t0.
 */
export function readSamples(x: SignalInput, where: string, fs?: Scalar): Samples {
  if (isSignal(x)) {
    if (x.data.shape.length !== 1)
      throw new ShapeError(where, `${where}: expected a single-channel signal, got [${x.data.shape.join(', ')}]`)
    return { values: Float64Array.from(dense.toF64(x.data, where)), fs: fs ?? x.fs, t0: x.t0, unit: x.unit }
  }
  return { values: Float64Array.from(dense.toF64(x, where)), fs: fs ?? 1, t0: 0 }
}

/** The sample times t0 + k/fs of a signal, as a rank-1 tensor. */
export function sampleTimes(s: Signal): Tensor {
  const n = s.data.shape[s.data.shape.length - 1]
  return fromData(
    Float64Array.from({ length: n }, (_, k) => s.t0 + k / s.fs),
    [n],
  )
}

/** The fields of a `Spectrum` other than `kind`. */
export type SpectrumFields = Omit<Spectrum, 'kind'>

/** A `Spectrum` from its fields (the `kind` brand is added). */
export function spectrum(fields: SpectrumFields): Spectrum {
  return { kind: 'spectrum', ...fields }
}

/** The fields of a `TimeFrequency` other than `kind`. */
export type TimeFrequencyFields = Omit<TimeFrequency, 'kind'>

/** A `TimeFrequency` raster from its fields (the `kind` brand is added). */
export function timeFrequency(fields: TimeFrequencyFields): TimeFrequency {
  return { kind: 'time-frequency', ...fields }
}

/** Unit of a density or power spectrum of a signal with value unit `unit`: `u²/Hz` or `u²`. */
export function powerUnit(unit: string | undefined, density: boolean): string | undefined {
  if (unit === undefined) return undefined
  return density ? `${unit}²/Hz` : `${unit}²`
}

/** A complex128 tensor of the given shape from its real and imaginary parts (row-major, equal lengths). */
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

const valuesOf = (s: Spectrum | Tensor): Tensor => (isTensor(s) ? s : s.values)

/** |values| of a spectrum (or of a real or complex128 tensor): the modulus of a complex response or DFT. */
export function magnitude(s: Spectrum | Tensor): Tensor {
  const v = valuesOf(s)
  return v.dtype === 'complex128' ? complexAbs(v) : abs(v)
}

/**
 * The phase of a spectrum's values (or of a tensor) in radians: the principal value in (−π, π], unwrapped along the
 * last axis of a vector with `unwrap` (as `numpy.unwrap`), in degrees with `degrees`. A real value has phase 0 or π.
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
 * A spectrum in decibels, by its `quantity`: 10 log₁₀(v/reference) for powers (`psd`, `power`, `coherence`) and
 * 20 log₁₀(|v|/reference) for amplitudes (`amplitude`, `complex`, `response`), so the power rule is never applied to
 * an amplitude. Zero maps to −∞.
 */
export function spectrumDecibels(s: Spectrum, { reference = 1 }: { reference?: Scalar } = {}): Tensor {
  const power = s.quantity === 'psd' || s.quantity === 'power' || s.quantity === 'coherence'
  return decibels(magnitude(s), { power, reference })
}

/**
 * Unwraps a phase sequence so that consecutive values never jump by more than π, as `numpy.unwrap` (Itoh, 1982,
 * "Analysis of the phase unwrapping algorithm", Applied Optics 21(14)).
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

/**
 * Responses of an `LtiSystem`: the frequency response $H(i\omega)$ or $H(e^{i\omega})$ as a `Spectrum`, Bode data and
 * stability margins, and time responses (simulation as a traceable algorithm; step, impulse and initial-condition
 * responses).
 *
 * Frequencies are in rad/s for a continuous system and rad/sample for a discrete one, and a delay $\tau$ enters as
 * $e^{-i\omega\tau}$. Time responses sample a continuous system every `dt` with the input held over each step, which
 * is exact for such inputs; a discrete system steps by its own `dt`. A MIMO state-space system is evaluated on one
 * channel at a time (`ChannelOptions`) in the frequency domain, and on all its inputs and outputs in time.
 *
 * Sources: Ogata (2010), "Modern Control Engineering", 5th ed., §7-2 (Bode diagrams) and §7-6 (gain and phase
 * margins); Van Loan (1978), "Computing integrals involving the matrix exponential" (the exact zero-order-hold step);
 * Oppenheim & Schafer (2010), "Discrete-Time Signal Processing", §5.1 (frequency response of a difference equation).
 */

import { findRoot } from 'aifn-compute/numerics/roots'
import { polyval } from 'aifn-compute/numerics/polynomial'
import {
  angle,
  complex,
  complexAbs,
  complexItem,
  dense,
  div,
  expj,
  fromData,
  mul,
  slice,
  sub,
  tensor,
  toFlat,
  type Matrix,
  type Tensor,
  type Value,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type {
  ComplexNumber,
  LtiSystem,
  Scalar,
  Size,
  Spectrum,
  Status,
  VectorLike,
} from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { poles, rationalOf, systemZeros, toStateSpace, type ChannelOptions } from './system'
import { discretise } from './transform'

// ── Frequency response ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * $H$ at the points $x$ (complex128, any shape), in the system's own form: products over roots (zpk) or sections
 * (sos), else the SISO rational function $\mathrm{num}(x)/\mathrm{den}(x)$ by Horner's rule (`polyval`).
 * Compositions of primitives, so differentiable in $x$ and in traced coefficients. The delay is not applied.
 *
 * @param sys The system.
 * @param x The points $s$ or $z$ to evaluate at, complex128.
 * @param channel The input and output of a MIMO state-space system.
 * @returns $H(x)$, complex128 of the shape of `x`.
 */
function evaluate(sys: LtiSystem, x: Value, channel: ChannelOptions): Value {
  const r = sys.repr
  if (r.form === 'zpk') {
    let h: Value = mul(r.gain, polyval([1], x))
    const nz = r.zeros.shape[0]
    const np = r.poles.shape[0]
    for (let k = 0; k < nz; k++) h = mul(h, sub(x, slice(r.zeros, k)))
    for (let k = 0; k < np; k++) h = div(h, sub(x, slice(r.poles, k)))
    return h
  }
  if (r.form === 'sos') {
    let h: Value = polyval([1], x)
    for (let j = 0; j < r.sections.shape[0]; j++) {
      const row = slice(r.sections, j)
      h = mul(h, div(polyval(slice(row, [0, 3]), x), polyval(slice(row, [3, 6]), x)))
    }
    return h
  }
  const { num, den } = rationalOf(sys, channel)
  return div(polyval(num, x), polyval(den, x))
}

/**
 * The evaluation points for frequencies $\omega$: $i\omega$ for a continuous system, $e^{i\omega}$ for a discrete one
 * (complex128).
 *
 * @param sys The system, whose domain decides the map.
 * @param w The frequencies $\omega$ (real, any shape).
 * @returns The points, complex128 of the shape of `w`.
 */
function pointsAt(sys: LtiSystem, w: Value): Value {
  return sys.domain === 'discrete' ? expj(w) : complex(mul(0, w), w)
}

/**
 * $H(i\omega)$ or $H(e^{i\omega})$ at the frequencies $\omega$, the delay $\tau$ included as $e^{-i\omega\tau}$.
 *
 * @param sys The system.
 * @param w The frequencies $\omega$ (real, any shape): rad/s, or rad/sample for a discrete system.
 * @param channel The input and output of a MIMO state-space system.
 * @returns The response, complex128 of the shape of `w`.
 */
function responseValues(sys: LtiSystem, w: Value, channel: ChannelOptions = {}): Value {
  const h = evaluate(sys, pointsAt(sys, w), channel)
  return sys.delay ? mul(h, expj(mul(-sys.delay, w))) : h
}

/**
 * The response $H$ at one frequency $\omega$, as a function returning `{ re, im }`: continuous systems at
 * $s = i\omega$ ($\omega$ in rad/s), discrete ones at $z = e^{i\omega}$ ($\omega$ in rad/sample). A delay $\tau$
 * multiplies by $e^{-i\omega\tau}$. For many frequencies use `frequencyResponse`, which evaluates them in one pass.
 *
 * @param sys The system.
 * @param channel The input and output of a MIMO state-space system.
 * @returns A function of $\omega$ giving $H$ there as a complex number.
 *
 * @example The lag $1/(s + 1)$ at $\omega = 1$ is $1/(1 + i)$, and 1 at DC
 * const H = responseAt(transferFunction([1], [1, 1]))
 * print('H(i) =', H(1))
 * print('H(0) =', H(0))
 *
 * @example An integrator with a delay of 0.5 s: $e^{-0.5i}/i$
 * const H = responseAt(transferFunction([1], [1, 0], { delay: 0.5 }))
 * print('H(i) =', H(1))
 */
export function responseAt(sys: LtiSystem, channel: ChannelOptions = {}): (w: Scalar) => ComplexNumber {
  return (w) => complexItem(responseValues(sys, tensor([w]), channel) as Tensor)
}

/**
 * The frequency response at the frequencies `w` as a `Spectrum` (`quantity: 'response'`, complex128 values of shape
 * $[n]$): $H(i\omega)$ with $\omega$ in rad/s for a continuous system, $H(e^{i\omega})$ with $\omega$ in rad/sample
 * for a discrete one (`fs`, $1/dt$, is recorded). `freqz` in `aifn-compute/signal/filters` gives the evenly spaced
 * discrete grid and Hz. Magnitude, phase and dB come from `aifn-compute/signal`'s `magnitude`, `phase` and `decibels`
 * (or `complexAbs`/`angle` directly).
 *
 * @param sys The system.
 * @param w The $n$ frequencies $\omega$, stored as the spectrum's `f`.
 * @param channel The input and output of a MIMO state-space system.
 * @returns The response as a one-sided `Spectrum`.
 *
 * @example The lag $1/(s + 1)$ at DC, its corner and a decade above
 * const r = frequencyResponse(transferFunction([1], [1, 1]), [0, 1, 10])
 * print('axis =', r.axis)
 * print('H =', r.values)
 * print('|H| =', complexAbs(r.values))
 *
 * @example The two-tap moving average passes DC and nulls the Nyquist frequency $\omega = \pi$
 * const r = frequencyResponse(transferFunction([0.5, 0.5], [1], { dt: 0.01 }), [0, Math.PI / 2, Math.PI])
 * print('|H| =', complexAbs(r.values))
 * print('fs =', r.fs)
 */
export function frequencyResponse(sys: LtiSystem, w: VectorLike, channel: ChannelOptions = {}): Spectrum {
  const ws = dense.toF64(w, 'frequencyResponse')
  const f = fromData(ws, [ws.length])
  const values = responseValues(sys, f, channel) as Tensor
  const discrete = sys.domain === 'discrete'
  return {
    kind: 'spectrum',
    f,
    axis: discrete ? 'rad/sample' : 'rad/s',
    values,
    quantity: 'response',
    sided: 'one',
    ...(discrete && sys.dt !== null ? { fs: 1 / sys.dt } : {}),
  }
}

/**
 * A logarithmic frequency grid for a continuous system: two decades either side of the poles and zeros (their
 * nonzero moduli, and $1/\tau$ for a delay), or $10^{-2}$ to $10^2$ rad/s when there are none. For a discrete
 * system: from three decades below $\pi$ up to $\pi$ rad/sample.
 *
 * @param sys The system.
 * @param n The number of points, at least 2.
 * @returns $n$ log-spaced frequencies, ascending.
 *
 * @example Around the lag's pole at 1 rad/s, and up to Nyquist for a discrete system
 * print('lag =', frequencyGrid(transferFunction([1], [1, 1]), 5))
 * print('discrete =', frequencyGrid(transferFunction([1], [1, -0.5], { dt: 1 }), 4))
 */
export function frequencyGrid(sys: LtiSystem, n: Size = 500): Vector {
  if (sys.domain === 'discrete') {
    const lo = Math.log10(Math.PI) - 3
    const hi = Math.log10(Math.PI)
    return fromData(
      Float64Array.from({ length: n }, (_, i) => 10 ** (lo + ((hi - lo) * i) / (n - 1))),
      [n],
    )
  }
  const mods: number[] = []
  for (const rs of [poles(sys), systemZeros(sys)]) for (const m of toFlat(complexAbs(rs))) if (m > 1e-12) mods.push(m)
  if (sys.delay) mods.push(1 / sys.delay)
  const lo = mods.length ? Math.log10(Math.min(...mods)) - 2 : -2
  const hi = mods.length ? Math.log10(Math.max(...mods)) + 2 : 2
  return fromData(
    Float64Array.from({ length: n }, (_, i) => 10 ** (lo + ((hi - lo) * i) / (n - 1))),
    [n],
  )
}

/** Bode data: magnitude and continuous phase over frequency. */
export type Bode = {
  /** Frequencies (rad/s, or rad/sample for a discrete system). */
  w: Vector
  /** $\lvert H \rvert$. */
  magnitude: Vector
  /** $20 \log_{10} \lvert H \rvert$. */
  magnitudeDb: Vector
  /**
   * Phase in degrees, unwrapped to be continuous in $\omega$ (a delay adds $-\omega\tau$ exactly, not modulo
   * $360^\circ$).
   */
  phase: Vector
}

/**
 * Bode data over a frequency grid (default `frequencyGrid(sys)`). The rational part's phase starts at its principal
 * value at the first frequency and is unwrapped between grid points (as numpy's `unwrap`, like scipy's `bode`); the
 * delay's phase $-\omega\tau \cdot 180/\pi$ is added analytically, so a fine grid is needed only for the rational
 * part. SISO (channel 0 to 0 of a MIMO state-space system).
 *
 * @param sys The system.
 * @param w The frequencies (rad/s, or rad/sample when discrete); default `frequencyGrid(sys)`.
 * @returns The frequencies with the magnitude, the magnitude in dB and the phase in degrees at each.
 *
 * @example The lag $1/(s + 1)$: $-3$ dB and $-45^\circ$ at the corner, $-20$ dB a decade above
 * const b = bode(transferFunction([1], [1, 1]), [0.1, 1, 10])
 * print('dB =', b.magnitudeDb)
 * print('phase =', b.phase)
 *
 * @example A delay of 1 s adds $-\omega \cdot 180/\pi$ degrees, unwrapped
 * const b = bode(transferFunction([1], [1, 1], { delay: 1 }), [1, 10])
 * print('phase =', b.phase)
 */
export function bode(sys: LtiSystem, w?: VectorLike): Bode {
  const ws = dense.toF64(w ?? frequencyGrid(sys), 'bode')
  const n = ws.length
  const h = responseValues({ ...sys, delay: 0 }, fromData(ws, [n])) as Tensor
  const mag = Float64Array.from(toFlat(complexAbs(h)))
  const principal = toFlat(angle(h))
  const db = mag.map((m) => 20 * Math.log10(m))
  const ph = new Float64Array(n)
  let prev = 0
  ws.forEach((v, i) => {
    let p = (principal[i] * 180) / Math.PI
    if (i > 0) p += 360 * Math.round((prev - p) / 360)
    prev = p
    ph[i] = p - (v * sys.delay * 180) / Math.PI
  })
  return {
    w: fromData(ws, [n]),
    magnitude: fromData(mag, [n]),
    magnitudeDb: fromData(db, [n]),
    phase: fromData(ph, [n]),
  }
}

/** Stability margins of an open loop L under unity negative feedback. */
export type Margins = {
  /**
   * The gain margin $1/\lvert L(i\omega_{pc}) \rvert$ at the phase crossover with the smallest margin (Infinity if
   * none).
   */
  gainMargin: Scalar
  /** The gain margin in dB, $20 \log_{10}$ of `gainMargin`. */
  gainMarginDb: Scalar
  /** Phase crossover frequency $\omega_{pc}$ (NaN if none). */
  phaseCrossover: Scalar
  /**
   * The phase margin $180^\circ + \angle L(i\omega_{gc})$, in degrees, at the gain crossover with the smallest margin
   * (Infinity if none).
   */
  phaseMargin: Scalar
  /** Gain crossover frequency $\omega_{gc}$ (NaN if none). */
  gainCrossover: Scalar
  /**
   * The delay margin, $\text{phaseMargin} \cdot \pi/180/\omega_{gc}$: the extra delay that destabilises the loop
   * (Infinity with no gain crossover, 0 when the phase margin is not positive).
   */
  delayMargin: Scalar
  /** Every gain crossover found on the grid, refined. */
  gainCrossovers: Vector
  /** Every phase crossover found on the grid, refined. */
  phaseCrossovers: Vector
}

/**
 * Gain and phase margins of an open-loop system $L$ (Ogata, 2010, §7-6): gain crossovers where $\lvert L \rvert = 1$
 * and phase crossovers where $\angle L = -180^\circ$ (mod $360^\circ$). Crossings are bracketed on a fine logarithmic
 * grid (default `frequencyGrid(L, 4000)`) and refined by Brent's method; the reported margins are the smallest over
 * all crossings, as python-control's `stability_margins`. Meaningful when the closed loop's stability is decided by
 * these crossings (e.g. a stable, minimum-phase $L$).
 *
 * @param L The open loop, under unity negative feedback.
 * @param w The grid on which crossings are bracketed (ascending); default 4000 points of `frequencyGrid`. A crossing
 *   outside it is not found.
 * @returns The margins, their crossover frequencies, and every crossover found.
 *
 * @example $L = 1/(s(s + 1)(s + 2))$: gain margin 6 at $\omega = \sqrt 2$, phase margin about $53.4^\circ$
 * const m = margins(transferFunction([1], [1, 3, 2, 0]))
 * print('gain margin =', m.gainMargin, ' at', m.phaseCrossover)
 * print('phase margin =', m.phaseMargin, ' at', m.gainCrossover)
 * print('delay margin =', m.delayMargin)
 */
export function margins(L: LtiSystem, w?: VectorLike): Margins {
  const grid = Array.from(dense.toF64(w ?? frequencyGrid(L, 4000), 'margins'))
  const b = bode(L, grid)
  const mag = toFlat(b.magnitude)
  const phase = toFlat(b.phase)
  const at = responseAt(L)
  const magAt = (v: number) => {
    const h = at(v)
    return Math.hypot(h.re, h.im)
  }
  const phaseNear = (v: number, ref: number) => {
    const h = at(v)
    const p = (Math.atan2(h.im, h.re) * 180) / Math.PI
    return p + 360 * Math.round((ref - p) / 360)
  }
  const gc: number[] = []
  const pc: number[] = []
  for (let i = 0; i + 1 < grid.length; i++) {
    const a = Math.log(mag[i])
    const c = Math.log(mag[i + 1])
    if (a === 0) gc.push(grid[i])
    else if (a * c < 0) gc.push(findRoot((v) => Math.log(magAt(v)), [grid[i], grid[i + 1]], { xtol: 1e-14 }).x)
    const k = Math.round((phase[i] + 180) / 360)
    const target = -180 + 360 * k
    const f0 = phase[i] - target
    const f1 = phase[i + 1] - target
    if (f0 === 0) pc.push(grid[i])
    else if (f0 * f1 < 0) {
      const ref = phase[i]
      pc.push(findRoot((v) => phaseNear(v, ref) - target, [grid[i], grid[i + 1]], { xtol: 1e-14 }).x)
    }
  }
  let gainMargin = Infinity
  let phaseCrossover = NaN
  for (const v of pc) {
    const g = 1 / magAt(v)
    if (Math.abs(Math.log(g)) < Math.abs(Math.log(gainMargin)) || !Number.isFinite(gainMargin)) {
      gainMargin = g
      phaseCrossover = v
    }
  }
  let phaseMargin = Infinity
  let gainCrossover = NaN
  for (const v of gc) {
    let idx = 0
    for (let i = 1; i < grid.length; i++) if (Math.abs(grid[i] - v) < Math.abs(grid[idx] - v)) idx = i
    const p = phaseNear(v, phase[idx])
    const pm = p + 180 - 360 * Math.round((p + 180) / 360)
    if (Math.abs(pm) < Math.abs(phaseMargin)) {
      phaseMargin = pm
      gainCrossover = v
    }
  }
  const delayMargin = Number.isFinite(phaseMargin)
    ? phaseMargin > 0
      ? (phaseMargin * Math.PI) / 180 / gainCrossover
      : 0
    : Infinity
  const v = (xs: number[]) => fromData(Float64Array.from(xs), [xs.length])
  return {
    gainMargin,
    gainMarginDb: 20 * Math.log10(gainMargin),
    phaseCrossover,
    phaseMargin,
    gainCrossover,
    delayMargin,
    gainCrossovers: v(gc),
    phaseCrossovers: v(pc),
  }
}

// ── Time responses ───────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The input to a simulation: a constant (a number drives every input, a vector gives one value per input), or a
 * function $u(\text{time}, \xvec)$ called at the start of each step (which may be state feedback, e.g.
 * $\uvec = -\Kmat\xvec$).
 */
export type Input = Scalar | VectorLike | ((time: Scalar, x: Vector) => Scalar | VectorLike)

/** The state of `simulate`. */
export interface SimulationState extends Status {
  /** Steps taken $k$. */
  t: Size
  /** The time $k \cdot dt$. */
  time: Scalar
  /** The state $\xvec$ at `time`. */
  x: Vector
  /** The input applied from `time` to `time + dt` (held constant over the step). */
  u: Vector
  /** The output $\yvec = \Cmat\xvec + \Dmat\uvec$. */
  y: Vector
  /** Set once `time` reaches `tEnd`. */
  terminated: boolean
  /** True once some component of $\xvec$ is not finite. */
  diverged: boolean
}

/** Options for `simulate`. */
export type SimulationOptions = {
  /** Time step. Required for a continuous system; a discrete system steps by its own `dt`. */
  dt?: Scalar
  /** Stop at this time (inclusive). Omitted, the run lasts as many steps as the runner asks for. */
  tEnd?: Scalar
  /**
   * Apply the input over the first step only (for a continuous system an impulse of area $u$: height $u/dt$).
   */
  impulse?: boolean
}

/**
 * One sampled step: the discretised matrices and the state record. A continuous system is discretised by zero-order
 * hold with the step `options.dt`; a discrete one keeps its own `dt`. Throws `DomainError` for a continuous system
 * without a positive `dt`, an input of the wrong length, or an `x0` of the wrong length.
 *
 * @param sys The system (realised in state space).
 * @param input The input, a constant or a function of time and state.
 * @param options The step, the end time and whether the input is an impulse.
 * @returns `start` (the state at step 0 from an optional `x0`, default zeros), `advance` (one step), the dimensions
 *   `n`, `m`, `p` and the step `dt`.
 */
function stepper(sys: LtiSystem, input: Input, options: SimulationOptions) {
  const ss = toStateSpace(sys)
  const dt = sys.dt ?? options.dt
  if (dt === undefined || !(dt > 0))
    throw new DomainError('simulate', 'simulate: a continuous system needs a positive dt')
  const d = sys.domain === 'continuous' ? discretise(ss, dt, 'zoh') : ss
  const A = dense.data(d.repr.A)
  const B = dense.data(d.repr.B)
  const C = dense.data(d.repr.C)
  const D = dense.data(d.repr.D)
  const n = ss.repr.A.shape[0]
  const m = ss.repr.B.shape[1]
  const p = ss.repr.C.shape[0]
  const tEnd = options.tEnd ?? Infinity
  const vec = (a: dense.F64) => fromData(a, [a.length])
  const inputAt = (k: number, x: Vector): dense.F64 => {
    const time = k * dt
    const v = typeof input === 'function' ? input(time, x) : input
    const raw = typeof v === 'number' ? new Float64Array(m).fill(v) : dense.toF64(v, 'simulate input')
    if (raw.length !== m)
      throw new DomainError('simulate', `simulate: the input has ${raw.length} components, the system ${m}`)
    // An impulse is the input over the first step only; a continuous one has height 1/dt (exact as dt → 0).
    const scale = sys.domain === 'continuous' ? 1 / dt : 1
    return options.impulse ? raw.map((u) => (k === 0 ? u * scale : 0)) : raw
  }
  const make = (k: number, x: dense.F64): SimulationState => {
    const xv = vec(x)
    const u = inputAt(k, xv)
    const y = dense.add(dense.matVec(C, x, p, n), dense.matVec(D, u, p, m))
    const time = k * dt
    return {
      t: k,
      time,
      x: xv,
      u: vec(u),
      y: vec(y),
      terminated: time >= tEnd - 1e-9 * dt,
      diverged: !dense.allFinite(x),
    }
  }
  const advance = (s: SimulationState): SimulationState =>
    make(s.t + 1, dense.add(dense.matVec(A, dense.data(s.x), n, n), dense.matVec(B, dense.data(s.u), n, m)))
  const start = (x0?: VectorLike): SimulationState => {
    const x = x0 === undefined ? new Float64Array(n) : dense.toF64(x0, 'simulate x0')
    if (x.length !== n) throw new DomainError('simulate', `simulate: x0 must have ${n} components`)
    return make(0, x)
  }
  return { start, advance, n, m, p, dt }
}

/**
 * Simulates a system driven by `input` as a traceable algorithm. A continuous system is advanced exactly for an input
 * held constant over each step (zero-order hold, $\xvec_{k+1} = \Amat_d\xvec_k + \Bmat_d\uvec_k$; Van Loan, 1978), so
 * the only approximation is the hold itself; a discrete system steps by its own recursion. `init` takes `{ x0 }`
 * (default zeros). The input is evaluated at the start of each step from $(\text{time}, \xvec)$, so state feedback is
 * closed exactly at the sampling instants. The run stops (`terminated`) at `tEnd`.
 *
 * @param sys The system, in any representation (realised in state space; `x0` is in that realisation's state).
 * @param input The input: a constant, or a function of time and state. Default 0 (the free response).
 * @param options The step (required for a continuous system), the end time and the impulse flag.
 * @returns The algorithm, to run with `run` or step through.
 *
 * @example The lag $1/(s + 1)$ after 10 steps of 0.1 s of a unit step: $1 - e^{-1}$
 * const lag = transferFunction([1], [1, 1])
 * const s = run(simulate(lag, 1, { dt: 0.1 }), {}, 10)
 * print('time =', s.time, ' y =', s.y)
 * print('1 - exp(-1) =', 1 - Math.exp(-1))
 *
 * @example State feedback $u = -2x_1 - 3x_2$ on the double integrator, from $\xvec_0 = (1, 0)$
 * const integrator = stateSpace({ A: [[0, 1], [0, 0]], B: [0, 1], C: [1, 0] })
 * const control = (time, x) => {
 *   const [position, velocity] = toArray(x)
 *   return -2 * position - 3 * velocity
 * }
 * const s = run(simulate(integrator, control, { dt: 0.01, tEnd: 5 }), { x0: [1, 0] }, 1000)
 * print('time =', s.time, ' terminated =', s.terminated)
 * print('x =', s.x)
 * print('exact x =', [2 * Math.exp(-5) - Math.exp(-10), -2 * Math.exp(-5) + 2 * Math.exp(-10)])
 */
export function simulate(
  sys: LtiSystem,
  input: Input = 0,
  options: SimulationOptions = {},
): Algorithm<{ x0?: VectorLike }, SimulationState> {
  const { start, advance } = stepper(sys, input, options)
  return { name: 'lti-simulation', init: ({ x0 } = {}) => start(x0), step: (s) => advance(s) }
}

/** A sampled response: times, outputs, states and inputs stacked over time. */
export type Response = {
  /** Times, length $T$. */
  t: Vector
  /** Outputs, $T \times p$. */
  y: Matrix
  /** States, $T \times n$. */
  x: Matrix
  /** Inputs, $T \times m$. */
  u: Matrix
  /** True when the state stopped being finite; the record ends at that sample. */
  diverged: boolean
}

/**
 * Options for `respond`: `dt`, the step of a continuous system (a discrete one uses its own); `tEnd`, the last time;
 * `x0`, the initial state (default zeros); `impulse`, apply the input over the first step only.
 */
export type RespondOptions = { dt?: Scalar; tEnd: Scalar; x0?: VectorLike; impulse?: boolean }

/**
 * The response to `input` up to `tEnd` (scipy's `lsim` for a held input): `simulate` run and stacked, at the
 * $T = \operatorname{round}(t_\text{end}/dt) + 1$ times $0, dt, \dots$; it stops early if the state diverges.
 *
 * @param sys The system.
 * @param input The input: a constant, or a function of time and state.
 * @param options The horizon, step, initial state and impulse flag.
 * @param options.dt The step of a continuous system (required for one); a discrete system uses its own `dt`.
 * @param options.tEnd The last time.
 * @param options.x0 The initial state, in the state-space realisation of `sys`; default zeros.
 * @param options.impulse Apply the input over the first step only (height $u/dt$ for a continuous system).
 * @returns The times, outputs, states and inputs stacked over time.
 *
 * @example The lag $1/(s + 1)$ driven by $u = 2$: $y = 2(1 - e^{-t})$
 * const r = respond(transferFunction([1], [1, 1]), 2, { dt: 0.5, tEnd: 2 })
 * print('t =', r.t)
 * print('y =', r.y)
 */
export function respond(sys: LtiSystem, input: Input, { dt, tEnd, x0, impulse }: RespondOptions): Response {
  const { start, advance, n, m, p, dt: step } = stepper(sys, input, { dt, tEnd, impulse })
  const T = Math.round(tEnd / step) + 1
  const ts = new Float64Array(T)
  const ys = new Float64Array(T * p)
  const xs = new Float64Array(T * n)
  const us = new Float64Array(T * m)
  let s = start(x0)
  let k = 0
  for (; k < T; k++) {
    ts[k] = s.time
    ys.set(toFlat(s.y), k * p)
    xs.set(toFlat(s.x), k * n)
    us.set(toFlat(s.u), k * m)
    if (s.diverged || k === T - 1) break
    s = advance(s)
  }
  const kept = Math.min(k + 1, T)
  return {
    t: fromData(ts.slice(0, kept), [kept]),
    y: fromData(ys.slice(0, kept * p), [kept, p]),
    x: fromData(xs.slice(0, kept * n), [kept, n]),
    u: fromData(us.slice(0, kept * m), [kept, m]),
    diverged: s.diverged,
  }
}

/**
 * Options for the standard responses: `tEnd`, the horizon; `dt`, the step of a continuous system; `input`, the input
 * channel (default 0).
 */
export type StandardResponseOptions = { tEnd: Scalar; dt?: Scalar; input?: Size }

/**
 * The unit-step response from rest on input `input` (default 0): $u_j(t) = 1$ for $t \ge 0$, the other inputs 0.
 * Exact at the samples, the step being held.
 *
 * @param sys The system.
 * @param options The horizon and step.
 * @param options.tEnd The last time.
 * @param options.dt The step of a continuous system (required for one); a discrete system uses its own `dt`.
 * @param options.input The input $j$ that steps.
 * @returns The response, all outputs.
 *
 * @example The lag $1/(s + 1)$: $1 - e^{-t}$ at $t = 0, 1, 2, 3$
 * const r = stepResponse(transferFunction([1], [1, 1]), { tEnd: 3, dt: 1 })
 * print('t =', r.t)
 * print('y =', r.y)
 *
 * @example The critically damped $1/(s + 1)^2$: $1 - (1 + t)e^{-t}$
 * const r = stepResponse(transferFunction([1], [1, 2, 1]), { tEnd: 2, dt: 1 })
 * print('y =', r.y)
 */
export function stepResponse(sys: LtiSystem, { tEnd, dt, input = 0 }: StandardResponseOptions): Response {
  const m = toStateSpace(sys).repr.B.shape[1]
  const u = new Float64Array(m)
  u[input] = 1
  return respond(sys, u, { dt, tEnd })
}

/**
 * The unit-impulse response from rest on input `input`. Continuous: $\xvec(0^+) = \Bmat\evec_j$ and $\uvec = \zeros$
 * afterwards (exact; the $\Dmat\delta(t)$ term is omitted). Discrete: $\uvec_0 = \evec_j$, then $\zeros$.
 *
 * @param sys The system.
 * @param options The horizon and step.
 * @param options.tEnd The last time.
 * @param options.dt The step of a continuous system (required for one); a discrete system uses its own `dt`.
 * @param options.input The input $j$ that receives the impulse.
 * @returns The response, all outputs.
 *
 * @example The lag $1/(s + 1)$ gives $e^{-t}$; the moving average gives its two taps
 * const r = impulseResponse(transferFunction([1], [1, 1]), { tEnd: 2, dt: 1 })
 * print('y =', r.y)
 * const d = impulseResponse(transferFunction([0.5, 0.5], [1], { dt: 1 }), { tEnd: 3 })
 * print('moving average =', d.y)
 */
export function impulseResponse(sys: LtiSystem, { tEnd, dt, input = 0 }: StandardResponseOptions): Response {
  const ss = toStateSpace(sys).repr
  const n = ss.A.shape[0]
  const m = ss.B.shape[1]
  if (sys.domain === 'discrete') {
    const u = new Float64Array(m)
    u[input] = 1
    return respond(sys, u, { tEnd, impulse: true })
  }
  const B = dense.data(ss.B)
  const x0 = Float64Array.from({ length: n }, (_, i) => B[i * m + input])
  return respond(sys, 0, { dt, tEnd, x0 })
}

/**
 * The free response from `x0` with zero input.
 *
 * @param sys The system.
 * @param x0 The initial state, in the state-space realisation of `sys`.
 * @param options The horizon and step.
 * @param options.tEnd The last time.
 * @param options.dt The step of a continuous system (required for one); a discrete system uses its own `dt`.
 * @returns The response.
 *
 * @example Two modes from $\xvec_0 = (1, 1)$: $y = e^{-t} + e^{-2t}$, exact at the samples
 * const sys = stateSpace({ A: [[-1, 0], [0, -2]], B: [1, 1], C: [1, 1] })
 * const r = initialResponse(sys, [1, 1], { tEnd: 1, dt: 0.5 })
 * print('y =', r.y)
 * print('exact =', [0, 0.5, 1].map((t) => Math.exp(-t) + Math.exp(-2 * t)))
 */
export function initialResponse(sys: LtiSystem, x0: VectorLike, { tEnd, dt }: { tEnd: Scalar; dt?: Scalar }): Response {
  return respond(sys, 0, { dt, tEnd, x0 })
}

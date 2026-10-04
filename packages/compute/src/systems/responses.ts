/**
 * Responses of an `LtiSystem`: the frequency response H(iω) or H(e^{iω}) as a `Spectrum`, Bode data and stability
 * margins, and time responses (simulation as a traceable algorithm; step, impulse and initial-condition responses).
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
 * H at the points x (complex128, any shape), in the system's own form: products over roots (zpk) or sections (sos),
 * else the SISO rational function num(x)/den(x) by Horner's rule (`polyval`). Compositions of primitives, so
 * differentiable in x and in traced coefficients.
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

/** The evaluation points for frequencies w: iω for a continuous system, e^{iω} for a discrete one (complex128). */
function pointsAt(sys: LtiSystem, w: Value): Value {
  return sys.domain === 'discrete' ? expj(w) : complex(mul(0, w), w)
}

/** H(iω) or H(e^{iω}) at the frequencies w, the delay τ included as e^{−iωτ}: complex128 of w's shape. */
function responseValues(sys: LtiSystem, w: Value, channel: ChannelOptions = {}): Value {
  const h = evaluate(sys, pointsAt(sys, w), channel)
  return sys.delay ? mul(h, expj(mul(-sys.delay, w))) : h
}

/**
 * The response H at one frequency ω, as a function returning `{ re, im }`: continuous systems at s = iω (ω in rad/s),
 * discrete ones at z = e^{iω} (ω in rad/sample). A delay τ multiplies by e^{−iωτ}. For many frequencies use
 * `frequencyResponse`, which evaluates them in one pass.
 */
export function responseAt(sys: LtiSystem, channel: ChannelOptions = {}): (w: Scalar) => ComplexNumber {
  return (w) => complexItem(responseValues(sys, tensor([w]), channel) as Tensor)
}

/**
 * The frequency response at the frequencies `w` as a `Spectrum` (`quantity: 'response'`, complex128 values [n]):
 * H(iω) with ω in rad/s for a continuous system, H(e^{iω}) with ω in rad/sample for a discrete one (`fs` = 1/dt is
 * recorded). `freqz` in `aifn-compute/signal/filters` gives the evenly spaced discrete grid and Hz. Magnitude, phase and dB
 * come from `aifn-compute/signal`'s `magnitude`, `phase` and `decibels` (or `complexAbs`/`angle` directly).
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
 * nonzero moduli, and 1/τ for a delay), or 10⁻² … 10² rad/s when there are none, with `n` points (default 500). For a
 * discrete system: three decades below π up to π rad/sample.
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
  /** |H|. */
  magnitude: Vector
  /** 20 log₁₀ |H|. */
  magnitudeDb: Vector
  /** Phase in degrees, unwrapped to be continuous in ω (a delay adds −ωτ exactly, not modulo 360°). */
  phase: Vector
}

/**
 * Bode data over a frequency grid (default `frequencyGrid(sys)`). The rational part's phase starts at its principal
 * value at the first frequency and is unwrapped between grid points (as numpy's `unwrap`, like scipy's `bode`); the
 * delay's phase −ωτ·180/π is added analytically, so a fine grid is needed only for the rational part.
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
  /** The gain margin 1/|L(iω_pc)| at the phase crossover with the smallest margin (Infinity if none). */
  gainMargin: Scalar
  gainMarginDb: Scalar
  /** Phase crossover frequency ω_pc (NaN if none). */
  phaseCrossover: Scalar
  /** The phase margin 180° + ∠L(iω_gc) at the gain crossover with the smallest margin (Infinity if none). */
  phaseMargin: Scalar
  /** Gain crossover frequency ω_gc (NaN if none). */
  gainCrossover: Scalar
  /** The delay margin, phaseMargin·π/180/ω_gc: the extra delay that destabilises the loop (Infinity if none). */
  delayMargin: Scalar
  /** Every crossing found on the grid, refined. */
  gainCrossovers: Vector
  phaseCrossovers: Vector
}

/**
 * Gain and phase margins of an open-loop system L (Ogata, 2010, §7-6): gain crossovers where |L| = 1 and phase
 * crossovers where ∠L = −180° (mod 360°). Crossings are bracketed on a fine logarithmic grid (default
 * `frequencyGrid(L, 4000)`) and refined by Brent's method; the reported margins are the smallest over all crossings,
 * as python-control's `stability_margins`. Meaningful when the closed loop's stability is decided by these crossings
 * (e.g. a stable, minimum-phase L).
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

/** The input to a simulation: a constant, or u(time, x) (which may be state feedback, e.g. u = −Kx). */
export type Input = Scalar | VectorLike | ((time: Scalar, x: Vector) => Scalar | VectorLike)

/** The state of `simulate`. */
export interface SimulationState extends Status {
  /** Steps taken k. */
  t: Size
  /** The time k·dt. */
  time: Scalar
  /** The state x(time). */
  x: Vector
  /** The input applied from time to time + dt (held constant over the step). */
  u: Vector
  /** The output y = Cx + Du. */
  y: Vector
  /** Set once time ≥ `tEnd`. */
  terminated: boolean
  /** True once some component of x is not finite. */
  diverged: boolean
}

/** Options for `simulate`. */
export type SimulationOptions = {
  /** Time step. Required for a continuous system; a discrete system steps by its own `dt`. */
  dt?: Scalar
  /** Stop at this time (inclusive). Omitted, the run lasts as many steps as the runner asks for. */
  tEnd?: Scalar
  /** Apply the input over the first step only (an impulse of area u for a continuous system: height u/dt). */
  impulse?: boolean
}

/** One sampled step: the discretised matrices and the state record. */
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
 * held constant over each step (zero-order hold, x_{k+1} = A_d x_k + B_d u_k; Van Loan, 1978), so the only
 * approximation is the hold itself; a discrete system steps by its own recursion. `init` takes `{ x0 }` (default
 * zeros). The input is evaluated at the start of each step from (time, x), so state feedback is closed exactly at the
 * sampling instants. The run stops (`terminated`) at `tEnd`.
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
  /** Times, length T. */
  t: Vector
  /** Outputs, T×p. */
  y: Matrix
  /** States, T×n. */
  x: Matrix
  /** Inputs, T×m. */
  u: Matrix
  diverged: boolean
}

/** Options for `respond`. */
export type RespondOptions = { dt?: Scalar; tEnd: Scalar; x0?: VectorLike; impulse?: boolean }

/** The response to `input` up to `tEnd` (scipy's `lsim` for a held input): `simulate` run and stacked. */
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

/** Options for the standard responses: the horizon, the step of a continuous system, and the input channel. */
export type StandardResponseOptions = { tEnd: Scalar; dt?: Scalar; input?: Size }

/** The unit-step response from rest on input `input` (default 0): u_j(t) = 1 for t ≥ 0. */
export function stepResponse(sys: LtiSystem, { tEnd, dt, input = 0 }: StandardResponseOptions): Response {
  const m = toStateSpace(sys).repr.B.shape[1]
  const u = new Float64Array(m)
  u[input] = 1
  return respond(sys, u, { dt, tEnd })
}

/**
 * The unit-impulse response from rest on input `input`. Continuous: x(0⁺) = B e_j and u = 0 afterwards (exact; the
 * D δ(t) term is omitted). Discrete: u_0 = e_j, then 0.
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

/** The free response from `x0` with zero input. */
export function initialResponse(sys: LtiSystem, x0: VectorLike, { tEnd, dt }: { tEnd: Scalar; dt?: Scalar }): Response {
  return respond(sys, 0, { dt, tEnd, x0 })
}

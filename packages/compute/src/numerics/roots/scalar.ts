/**
 * Roots of scalar equations f(x) = 0: bracketing methods (bisection, regula falsi with the Illinois modification,
 * Brent's method) and open methods (secant, Newton). Every method is a traceable `Algorithm` whose state records the
 * current estimate, the bracket where there is one, and which kind of step was taken.
 */

import { EPS } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size, Status } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { flagged } from './status'

/** A scalar function of one real variable. */
export type ScalarFunction = (x: Scalar) => Scalar

/** Fields every scalar root-finder state carries: the runner's `Status` (set from `failure`) and the estimate. */
export type RootState = Status & {
  /** Steps taken (0 in the initial state). */
  t: Size
  /** The current estimate of the root and f there. */
  x: number
  fx: number
  /** Calls of f so far. */
  evaluations: number
  /** True once the stopping test passed; the runners stop. */
  converged: boolean
  /**
   * Why the method cannot continue, or null: `'no sign change'` (a bracketing method given a bracket where f does not
   * change sign), `'zero derivative'`, `'flat secant'` or `'not finite'`. `'not finite'` sets `diverged`, the others
   * `terminated`.
   */
  failure: string | null
}

/** Tolerances for the scalar methods: stop when the step (or half-bracket) is at most xtol + rtol·|x|, or f(x) = 0. */
export type RootTolerance = {
  /** Absolute tolerance on x. Default 2e-12 (as scipy's brentq). */
  xtol?: number
  /** Relative tolerance on x. Default 4ε ≈ 8.9e-16. */
  rtol?: number
  /** Also stop when |f(x)| ≤ ftol. Default 0. */
  ftol?: number
}

const tolerances = (o: RootTolerance) => ({ xtol: o.xtol ?? 2e-12, rtol: o.rtol ?? 4 * EPS, ftol: o.ftol ?? 0 })

/** The bracket [a, b] as passed to a bracketing method's `init`. */
export type BracketOptions = { lo: number; hi: number }

/** The state of `bisection` and `regulaFalsi`. */
export type BracketState = RootState & {
  /** The bracket [lo, hi] with f(lo) and f(hi) of opposite signs. */
  lo: number
  hi: number
  flo: number
  fhi: number
  /** hi − lo. */
  width: number
}

function bracketInit(f: ScalarFunction, { lo, hi }: BracketOptions, x: (s: Omit<BracketState, 'x' | 'fx'>) => number) {
  const flo = f(lo)
  const fhi = f(hi)
  let failure: string | null = null
  if (!Number.isFinite(flo) || !Number.isFinite(fhi)) failure = 'not finite'
  else if (flo * fhi > 0) failure = 'no sign change'
  const base = { t: 0, lo, hi, flo, fhi, width: hi - lo, evaluations: 2, converged: false, failure }
  // An endpoint that is a root is returned as is.
  if (flo === 0) return { ...base, x: lo, fx: 0, converged: true, failure: null }
  if (fhi === 0) return { ...base, x: hi, fx: 0, converged: true, failure: null }
  const guess = x(base)
  return { ...base, x: guess, fx: Math.abs(flo) < Math.abs(fhi) ? flo : fhi }
}

/**
 * Bisection: halve the bracket [lo, hi] at its midpoint, keeping the half where f changes sign. The bracket width
 * halves each step, so reaching width w from w₀ takes ⌈log₂(w₀/w)⌉ steps. `init` takes `{ lo, hi }` with f(lo) and
 * f(hi) of opposite signs. `x` is the latest midpoint.
 */
export function bisection(f: ScalarFunction, options: RootTolerance = {}): Algorithm<BracketOptions, BracketState> {
  const { xtol, rtol, ftol } = tolerances(options)
  return flagged<BracketOptions, BracketState>({
    name: 'bisection',
    init: (o) => bracketInit(f, o, (s) => 0.5 * (s.lo + s.hi)),
    step: (s) => {
      const mid = s.lo + 0.5 * (s.hi - s.lo)
      const fm = f(mid)
      const keepLow = Math.sign(fm) === Math.sign(s.flo)
      const lo = keepLow ? mid : s.lo
      const hi = keepLow ? s.hi : mid
      return {
        t: s.t + 1,
        x: mid,
        fx: fm,
        lo,
        hi,
        flo: keepLow ? fm : s.flo,
        fhi: keepLow ? s.fhi : fm,
        width: hi - lo,
        evaluations: s.evaluations + 1,
        converged: fm === 0 || Math.abs(fm) <= ftol || 0.5 * (hi - lo) <= xtol + rtol * Math.abs(mid),
        failure: Number.isFinite(fm) ? null : 'not finite',
      }
    },
  })
}

/** The state of `regulaFalsi`: a bracket and whether the Illinois halving was applied on the last step. */
export type RegulaFalsiState = BracketState & {
  /** Which end the last step replaced, or null at t = 0. */
  replaced: 'lo' | 'hi' | null
  /** True when the retained end's value was halved (Illinois modification). */
  halved: boolean
}

/**
 * Regula falsi (false position): the next point is where the secant through (lo, f(lo)) and (hi, f(hi)) crosses zero,
 * and it replaces the end with the same sign. Plain regula falsi can keep one end fixed and converge slowly; with
 * `illinois` (default true) the retained end's f is halved when the same end is kept twice (Dowell & Jarratt, 1971,
 * "A modified regula falsi method for computing the root of an equation", BIT 11), giving superlinear convergence.
 * Stops when the step moves x by at most xtol + rtol·|x| or f(x) = 0.
 */
export function regulaFalsi(
  f: ScalarFunction,
  options: RootTolerance & { illinois?: boolean } = {},
): Algorithm<BracketOptions, RegulaFalsiState> {
  const { xtol, rtol, ftol } = tolerances(options)
  const illinois = options.illinois ?? true
  const secantZero = (s: { lo: number; hi: number; flo: number; fhi: number }) =>
    s.hi - (s.fhi * (s.hi - s.lo)) / (s.fhi - s.flo)
  return flagged<BracketOptions, RegulaFalsiState>({
    name: illinois ? 'regula-falsi-illinois' : 'regula-falsi',
    init: (o) => ({
      ...bracketInit(f, o, (b) => (Math.abs(b.flo) < Math.abs(b.fhi) ? b.lo : b.hi)),
      replaced: null,
      halved: false,
    }),
    step: (s) => {
      const x = secantZero(s)
      const fx = f(x)
      const replaceLow = Math.sign(fx) === Math.sign(s.flo)
      let { flo, fhi } = s
      let halved = false
      // Illinois: the same end replaced twice in a row means the other end is stuck; halve its value.
      if (illinois && s.replaced === (replaceLow ? 'lo' : 'hi')) {
        halved = true
        if (replaceLow) fhi /= 2
        else flo /= 2
      }
      const lo = replaceLow ? x : s.lo
      const hi = replaceLow ? s.hi : x
      if (replaceLow) flo = fx
      else fhi = fx
      return {
        t: s.t + 1,
        x,
        fx,
        lo,
        hi,
        flo,
        fhi,
        width: hi - lo,
        replaced: replaceLow ? 'lo' : 'hi',
        halved,
        evaluations: s.evaluations + 1,
        converged:
          fx === 0 || Math.abs(fx) <= ftol || Math.abs(x - s.x) <= xtol + rtol * Math.abs(x) || hi - lo <= xtol,
        failure: Number.isFinite(fx) ? null : 'not finite',
      }
    },
  })
}

/** The state of `brent`. */
export type BrentState = RootState & {
  /** The bracket [lo, hi] (the current estimate and the contrapoint, sorted); f changes sign across it. */
  lo: number
  hi: number
  /** The previous estimate and f there. */
  previous: number
  fprevious: number
  /** The contrapoint: f has the opposite sign there to f(x), and |f(x)| ≤ |f(contrapoint)|. */
  contrapoint: number
  fcontrapoint: number
  /** The last two step lengths, used to decide whether interpolation is converging fast enough. */
  stepPrevious: number
  stepCurrent: number
  /** The kind of step taken last. */
  method: 'init' | 'bisection' | 'secant' | 'inverse-quadratic'
}

/**
 * Brent's method (Brent, 1973, "Algorithms for Minimization without Derivatives", ch. 4), as in scipy's `brentq`:
 * inverse quadratic interpolation or the secant step when they are converging, bisection otherwise, so it keeps the
 * bracket's guarantee while usually converging superlinearly. `init` takes `{ lo, hi }` with f(lo), f(hi) of opposite
 * signs.
 */
export function brent(f: ScalarFunction, options: RootTolerance = {}): Algorithm<BracketOptions, BrentState> {
  const { xtol, rtol, ftol } = tolerances(options)

  /** Keep the contrapoint on the other side of the root, and make x the point with the smaller |f|. */
  const normalise = (s: BrentState): BrentState => {
    let { x, fx, previous, fprevious, contrapoint, fcontrapoint, stepPrevious, stepCurrent } = s
    if (fprevious !== 0 && fx !== 0 && Math.sign(fprevious) !== Math.sign(fx)) {
      contrapoint = previous
      fcontrapoint = fprevious
      stepPrevious = stepCurrent = x - previous
    }
    if (Math.abs(fcontrapoint) < Math.abs(fx)) {
      previous = x
      x = contrapoint
      contrapoint = previous
      fprevious = fx
      fx = fcontrapoint
      fcontrapoint = fprevious
    }
    const delta = 0.5 * (xtol + rtol * Math.abs(x))
    const halfBracket = 0.5 * (contrapoint - x)
    return {
      ...s,
      x,
      fx,
      previous,
      fprevious,
      contrapoint,
      fcontrapoint,
      stepPrevious,
      stepCurrent,
      lo: Math.min(x, contrapoint),
      hi: Math.max(x, contrapoint),
      converged: fx === 0 || Math.abs(fx) <= ftol || Math.abs(halfBracket) < delta,
      failure: Number.isFinite(fx) ? s.failure : 'not finite',
    }
  }

  return flagged<BracketOptions, BrentState>({
    name: 'brent',
    init: ({ lo, hi }) => {
      const flo = f(lo)
      const fhi = f(hi)
      let failure: string | null = null
      if (!Number.isFinite(flo) || !Number.isFinite(fhi)) failure = 'not finite'
      else if (flo * fhi > 0) failure = 'no sign change'
      const s: BrentState = {
        t: 0,
        x: hi,
        fx: fhi,
        previous: lo,
        fprevious: flo,
        contrapoint: 0,
        fcontrapoint: 0,
        stepPrevious: 0,
        stepCurrent: 0,
        lo,
        hi,
        method: 'init',
        evaluations: 2,
        converged: false,
        failure,
      }
      if (flo === 0) return { ...s, x: lo, fx: 0, converged: true, failure: null }
      if (fhi === 0) return { ...s, converged: true, failure: null }
      return failure ? s : normalise(s)
    },
    step: (s) => {
      const { x, fx, previous, fprevious, contrapoint, fcontrapoint } = s
      const delta = 0.5 * (xtol + rtol * Math.abs(x))
      const bisect = 0.5 * (contrapoint - x)
      let stepPrevious = bisect
      let stepCurrent = bisect
      let method: BrentState['method'] = 'bisection'
      if (Math.abs(s.stepPrevious) > delta && Math.abs(fx) < Math.abs(fprevious)) {
        let trial: number
        if (previous === contrapoint) {
          trial = (-fx * (x - previous)) / (fx - fprevious)
          method = 'secant'
        } else {
          const dPrevious = (fprevious - fx) / (previous - x)
          const dContra = (fcontrapoint - fx) / (contrapoint - x)
          trial =
            (-fx * (fcontrapoint * dContra - fprevious * dPrevious)) /
            (dContra * dPrevious * (fcontrapoint - fprevious))
          method = 'inverse-quadratic'
        }
        // Accept the interpolation step only if it is short compared with the last steps and the bracket.
        if (2 * Math.abs(trial) < Math.min(Math.abs(s.stepPrevious), 3 * Math.abs(bisect) - delta)) {
          stepPrevious = s.stepCurrent
          stepCurrent = trial
        } else method = 'bisection'
      }
      const next = x + (Math.abs(stepCurrent) > delta ? stepCurrent : bisect > 0 ? delta : -delta)
      const fnext = f(next)
      return normalise({
        ...s,
        t: s.t + 1,
        previous: x,
        fprevious: fx,
        x: next,
        fx: fnext,
        stepPrevious,
        stepCurrent,
        method,
        evaluations: s.evaluations + 1,
      })
    },
  })
}

/** The state of `secant`. */
export type SecantState = RootState & {
  /** The previous estimate and f there. */
  previous: number
  fprevious: number
  /** The last step x − previous. */
  step: number
}

/**
 * The secant method: x_{k+1} = x_k − f(x_k)(x_k − x_{k−1}) / (f(x_k) − f(x_{k−1})). Converges with order
 * (1 + √5)/2 ≈ 1.618 near a simple root, but is not bracketed. `init` takes `{ x0, x1 }`.
 */
export function secant(
  f: ScalarFunction,
  options: RootTolerance = {},
): Algorithm<{ x0: number; x1: number }, SecantState> {
  const { xtol, rtol, ftol } = tolerances(options)
  return flagged<{ x0: number; x1: number }, SecantState>({
    name: 'secant',
    init: ({ x0, x1 }) => {
      const f0 = f(x0)
      const f1 = f(x1)
      return {
        t: 0,
        x: x1,
        fx: f1,
        previous: x0,
        fprevious: f0,
        step: x1 - x0,
        evaluations: 2,
        converged: f1 === 0 || Math.abs(f1) <= ftol,
        failure: Number.isFinite(f0) && Number.isFinite(f1) ? null : 'not finite',
      }
    },
    step: (s) => {
      const denominator = s.fx - s.fprevious
      if (denominator === 0) return { ...s, t: s.t + 1, failure: 'flat secant' }
      const next = s.x - (s.fx * (s.x - s.previous)) / denominator
      const fnext = f(next)
      const step = next - s.x
      return {
        t: s.t + 1,
        x: next,
        fx: fnext,
        previous: s.x,
        fprevious: s.fx,
        step,
        evaluations: s.evaluations + 1,
        converged: fnext === 0 || Math.abs(fnext) <= ftol || Math.abs(step) <= xtol + rtol * Math.abs(next),
        failure: Number.isFinite(fnext) ? null : 'not finite',
      }
    },
  })
}

/** A scalar function with its derivative. */
export type ScalarWithDerivative = (x: number) => { value: number; derivative: number }

/** The state of `newtonRoot`. */
export type NewtonRootState = RootState & {
  /** f′(x). */
  derivative: number
  /** The full Newton step −f(x)/f′(x) computed on the last step (NaN at t = 0). */
  newtonStep: number
  /** The fraction of it taken (1 unless damped). */
  damping: number
}

/**
 * Newton's method for f(x) = 0: x ← x − f(x)/f′(x), converging quadratically near a simple root. With `damped`, the
 * step is halved until |f| decreases (at most 30 halvings), which widens the region of convergence. `init` takes
 * `{ x0 }`; a zero derivative is reported as a failure.
 */
export function newtonRoot(
  f: ScalarWithDerivative,
  options: RootTolerance & { damped?: boolean } = {},
): Algorithm<{ x0: number }, NewtonRootState> {
  const { xtol, rtol, ftol } = tolerances(options)
  return flagged<{ x0: number }, NewtonRootState>({
    name: options.damped ? 'damped-newton-root' : 'newton-root',
    init: ({ x0 }) => {
      const { value, derivative } = f(x0)
      return {
        t: 0,
        x: x0,
        fx: value,
        derivative,
        newtonStep: NaN,
        damping: 1,
        evaluations: 1,
        converged: value === 0 || Math.abs(value) <= ftol,
        failure: Number.isFinite(value) && Number.isFinite(derivative) ? null : 'not finite',
      }
    },
    step: (s) => {
      if (s.derivative === 0) return { ...s, t: s.t + 1, failure: 'zero derivative' }
      const full = -s.fx / s.derivative
      let damping = 1
      let x = s.x + full
      let e = f(x)
      let evaluations = 1
      if (options.damped)
        while (!(Math.abs(e.value) < Math.abs(s.fx)) && evaluations < 31) {
          damping /= 2
          x = s.x + damping * full
          e = f(x)
          evaluations++
        }
      return {
        t: s.t + 1,
        x,
        fx: e.value,
        derivative: e.derivative,
        newtonStep: full,
        damping,
        evaluations: s.evaluations + evaluations,
        converged: e.value === 0 || Math.abs(e.value) <= ftol || Math.abs(damping * full) <= xtol + rtol * Math.abs(x),
        failure: Number.isFinite(e.value) && Number.isFinite(e.derivative) ? null : 'not finite',
      }
    },
  })
}

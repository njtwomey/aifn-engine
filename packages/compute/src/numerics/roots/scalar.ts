/**
 * Roots of scalar equations $f(x) = 0$: bracketing methods (bisection, regula falsi with the Illinois modification,
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
  /** Current root estimate $x$. */
  x: number
  /** Function value evaluated at the current estimate $f(x)$. */
  fx: number
  /** Total number of objective function calls performed so far. */
  evaluations: number
  /** True once the stopping criterion is met. */
  converged: boolean
  /** Failure diagnosis string, or `null` if the method is healthy. */
  failure: string | null
}

/** Tolerances for scalar methods: stop when the step or half-bracket is at most $\text{xtol} + \text{rtol} \cdot |x|$, or $f(x) = 0$. */
export type RootTolerance = {
  /** Absolute tolerance on $x$ (default $2 \cdot 10^{-12}$). */
  xtol?: number
  /** Relative tolerance on $x$ (default $4\varepsilon \approx 8.9 \cdot 10^{-16}$). */
  rtol?: number
  /** Residual tolerance stopping when $|f(x)| \le \text{ftol}$ (default 0). */
  ftol?: number
}

/**
 * Fill omitted root-finding tolerances with default thresholds.
 *
 * @param o User-specified tolerance options.
 * @returns Complete tolerances with `xtol`, `rtol`, and `ftol`.
 */
const tolerances = (o: RootTolerance) => ({ xtol: o.xtol ?? 2e-12, rtol: o.rtol ?? 4 * EPS, ftol: o.ftol ?? 0 })

/** The bracket $[a, b]$ as passed to a bracketing method's `init`. */
export type BracketOptions = {
  /** Lower endpoint of the bracket interval. */
  lo: number
  /** Upper endpoint of the bracket interval. */
  hi: number
}

/** The state of `bisection` and `regulaFalsi`. */
export type BracketState = RootState & {
  /** Current lower bracket endpoint where $f(\text{lo})$ and $f(\text{hi})$ have opposite signs. */
  lo: number
  /** Current upper bracket endpoint where $f(\text{lo})$ and $f(\text{hi})$ have opposite signs. */
  hi: number
  /** Function value $f(\text{lo})$. */
  flo: number
  /** Function value $f(\text{hi})$. */
  fhi: number
  /** Bracket width $\text{hi} - \text{lo}$. */
  width: number
}

/**
 * Initialise bracket state and verify opposite sign condition.
 *
 * @param f Scalar objective function.
 * @param options Bracket interval boundaries.
 * @param options.lo Lower bracket boundary.
 * @param options.hi Upper bracket boundary.
 * @param x Selection function computing initial iterate $x_0$ from bracket endpoints.
 * @returns Initial bracket state with evaluated endpoints.
 */
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
 * Bisection method for scalar root finding: halving the bracket $[a, b]$ at its midpoint.
 *
 * Keeps the half-interval across which $f$ changes sign. The bracket width halves each step,
 * guaranteeing convergence to a root within $\lceil \log_2(w_0 / w) \rceil$ steps.
 * The iterate $x$ is the latest midpoint.
 *
 * @param f Continuous scalar function $f(x)$ whose root is sought.
 * @param options Convergence tolerances on step size and residual.
 * @returns A traceable `Algorithm` stepping through bisection brackets.
 *
 * @example Find root of cubic polynomial
 * const alg = bisection(x => x ** 3 - x - 2)
 * const state = run(alg, { lo: 1, hi: 2 }, 50)
 * print('converged =', state.converged)
 * print('root =', state.x)
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
  /** Which endpoint the previous step replaced (`'lo'`, `'hi'`, or `null` at $t = 0$). */
  replaced: 'lo' | 'hi' | null
  /** Whether Illinois reduction halved the opposite endpoint's value on this step. */
  halved: boolean
}

/**
 * Regula falsi (false position) root finding with optional Illinois modification.
 *
 * Computes the next iterate where the secant line between bracket endpoints crosses zero.
 * Plain regula falsi can retain one endpoint indefinitely; with `illinois` enabled (default true),
 * the retained endpoint's function value is halved whenever the same endpoint is kept twice
 * (Dowell & Jarratt, 1971), restoring superlinear convergence.
 *
 * @param f Continuous scalar function $f(x)$.
 * @param options Convergence tolerances and Illinois adjustment toggle.
 * @returns A traceable `Algorithm` stepping through false position iterates.
 *
 * @example Find root via false position
 * const alg = regulaFalsi(x => x ** 3 - x - 2)
 * const state = run(alg, { lo: 1, hi: 2 }, 50)
 * print('converged =', state.converged)
 * print('root =', state.x)
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
  /** Lower bound of the current bracket. */
  lo: number
  /** Upper bound of the current bracket. */
  hi: number
  /** Previous iterate estimate. */
  previous: number
  /** Function value at the previous iterate. */
  fprevious: number
  /** Contrapoint where $f$ has opposite sign to $f(x)$ with $|f(x)| \le |f(\text{contrapoint})|$. */
  contrapoint: number
  /** Function value at the contrapoint. */
  fcontrapoint: number
  /** Step displacement taken two iterations prior. */
  stepPrevious: number
  /** Step displacement taken in the previous iteration. */
  stepCurrent: number
  /** Interpolation or bisection step strategy applied on the last step. */
  method: 'init' | 'bisection' | 'secant' | 'inverse-quadratic'
}

/**
 * Brent's root-finding method combining bisection, secant, and inverse quadratic interpolation.
 *
 * Combines the robustness of bisection with the superlinear convergence of inverse quadratic
 * interpolation (Brent, 1973), as implemented in SciPy's `scipy.optimize.brentq`.
 * Guaranteed to converge while interpolating rapidly when smooth.
 *
 * @param f Continuous scalar function $f(x)$ with opposite signs on $[lo, hi]$.
 * @param options Convergence tolerances on step length and residual.
 * @returns A traceable `Algorithm` executing Brent root finding.
 *
 * @example Find root with Brent method
 * const alg = brent(x => Math.cos(x) - x)
 * const state = run(alg, { lo: 0, hi: 1 }, 50)
 * print('converged =', state.converged)
 * print('root =', state.x)
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
  /** Previous iterate estimate. */
  previous: number
  /** Function value at the previous iterate. */
  fprevious: number
  /** Last step displacement $x - \text{previous}$. */
  step: number
}

/**
 * Secant method for scalar root finding.
 *
 * Replaces derivatives with finite differences across consecutive iterates:
 * $x_{k+1} = x_k - f(x_k)\frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}$.
 * Converges with order $(1 + \sqrt{5})/2 \approx 1.618$ near a simple root without requiring derivatives.
 *
 * @param f Scalar function $f(x)$.
 * @param options Convergence tolerances on step length and residual.
 * @returns A traceable `Algorithm` executing secant iterations.
 *
 * @example Find root with secant method
 * const alg = secant(x => x ** 2 - 2)
 * const state = run(alg, { x0: 1, x1: 2 }, 50)
 * print('converged =', state.converged)
 * print('root =', state.x)
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
  /** First derivative $f'(x)$ evaluated at current estimate. */
  derivative: number
  /** Full undamped Newton displacement $-f(x) / f'(x)$ on the last step. */
  newtonStep: number
  /** Damping fraction applied to the Newton displacement ($1$ unless damped). */
  damping: number
}

/**
 * Newton-Raphson root finding for $f(x) = 0$ with optional backtracking damping.
 *
 * Updates $x_{k+1} = x_k - f(x_k) / f'(x_k)$, converging quadratically near a simple root.
 * When `damped` is enabled, the step is halved until $|f(x)|$ decreases (up to 30 halvings),
 * significantly widening the basin of attraction.
 *
 * @param f Objective returning function value and first derivative at $x$.
 * @param options Convergence tolerances and damping toggle.
 * @returns A traceable `Algorithm` executing Newton iterations.
 *
 * @example Find root with Newton-Raphson
 * const alg = newtonRoot(x => ({ value: x ** 2 - 2, derivative: 2 * x }))
 * const state = run(alg, { x0: 1.5 }, 20)
 * print('converged =', state.converged)
 * print('root =', state.x)
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

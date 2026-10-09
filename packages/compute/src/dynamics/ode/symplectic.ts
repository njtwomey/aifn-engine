/**
 * Symplectic integrators for separable Hamiltonian systems $H(\qvec, \pvec) = T(\pvec) + V(\qvec)$: symplectic Euler,
 * leapfrog (drift–kick–drift) and velocity Verlet (kick–drift–kick), with the energy tracked at every step. A
 * symplectic method preserves phase-space volume and exactly conserves a nearby "shadow" Hamiltonian, so its energy
 * error stays bounded (oscillates at $O(h^p)$ for a method of order $p$) over exponentially long times instead of
 * drifting (Hairer, Lubich & Wanner, 2006, "Geometric Numerical Integration", 2nd ed., §I.1, §VI.3 and §IX.8).
 */

import { grad } from 'aifn-compute/foundation/autodiff'
import { dense, fromData, toFlat, type Tensor, type Value, type Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm, Scalar, VectorLike } from 'aifn-compute/foundation/contracts'
import { checkDirection, initialState, nextStep, reached } from './explicit'
import type { FixedStepOptions, OdeState, Rhs } from './types'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

const { allFinite, toF64 } = dense
type F64 = dense.F64

/**
 * A separable Hamiltonian $H(\qvec, \pvec) = T(\pvec) + V(\qvec)$ with $\qvec$ and $\pvec$ of length $d$.
 * `potential` $V$ is required; `kinetic` $T$ defaults to $\frac{1}{2} \lVert \pvec \rVert^2$ (unit mass). Their
 * gradients (the force $-\nabla V$ and the velocity $\nabla T$) come from `aifn-compute/foundation/autodiff` unless
 * given, so $V$ and $T$ must then be written with `aifn-compute/foundation/tensor` primitives.
 */
export type SeparableHamiltonian = {
  /** The potential energy $V(\qvec)$: a scalar (a number or a one-element value) for positions of length $d$. */
  potential: (q: Tensor) => Value
  /** The kinetic energy $T(\pvec)$, a scalar. Default $\frac{1}{2} \lVert \pvec \rVert^2$ (gradient $\pvec$). */
  kinetic?: (p: Tensor) => Value
  /** $\nabla V(\qvec)$, if known in closed form. */
  potentialGradient?: (q: Tensor) => VectorLike
  /** $\nabla T(\pvec)$, if known in closed form. */
  kineticGradient?: (p: Tensor) => VectorLike
}

/**
 * The pieces of a separable Hamiltonian as plain functions on working arrays: the energies `V` and `T`, and their
 * gradients `dV` ($\nabla V$) and `dT` ($\nabla T$), each returning a new array.
 */
type Parts = { V: (q: F64) => number; T: (p: F64) => number; dV: (q: F64) => F64; dT: (p: F64) => F64 }

/**
 * A working array as a vector, sharing its data.
 *
 * @param v The array of values; the vector is built on it, not copied.
 * @returns The vector over `v`.
 */
const asTensor = (v: F64): Tensor => fromData(v, [v.length])
/**
 * A scalar value as a number: a number as it is, or the first entry of a tensor.
 *
 * @param v The value of an energy function.
 * @returns The number.
 */
const scalarOf = (v: Value): number => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])

/**
 * The pieces of a separable Hamiltonian as plain functions on working arrays: the energies evaluated through
 * `potential` and `kinetic`, and the gradients from the closed forms when given, else by `grad` (and the identity for
 * the default kinetic energy).
 *
 * @param H The Hamiltonian.
 * @returns The energies `V` and `T` and gradients `dV` and `dT`.
 */
function parts(H: SeparableHamiltonian): Parts {
  const kinetic = H.kinetic ?? ((p: Tensor) => 0.5 * toFlat(p).reduce((s, v) => s + v * v, 0))
  const V = (q: F64) => scalarOf(H.potential(asTensor(q)))
  const T = (p: F64) => scalarOf(kinetic(asTensor(p)))
  const dV = H.potentialGradient
    ? (q: F64) => toF64(H.potentialGradient!(asTensor(q)), 'hamiltonian')
    : (q: F64) =>
        toF64(grad((x: Value) => H.potential(x as Tensor))(asTensor(Float64Array.from(q))) as Tensor, 'hamiltonian')
  const dT = H.kineticGradient
    ? (p: F64) => toF64(H.kineticGradient!(asTensor(p)), 'hamiltonian')
    : H.kinetic
      ? (p: F64) =>
          toF64(grad((x: Value) => H.kinetic!(x as Tensor))(asTensor(Float64Array.from(p))) as Tensor, 'hamiltonian')
      : (p: F64) => Float64Array.from(p)
  return { V, T, dV, dT }
}

/**
 * The first-order system of a separable Hamiltonian, for any ODE solver: the state $\xvec = (\qvec, \pvec)$ of length
 * $2d$ moves by $\qvec' = \nabla T(\pvec)$, $\pvec' = -\nabla V(\qvec)$. Also returns the energy
 * $H(\xvec) = T(\pvec) + V(\qvec)$, for comparing a general solver's energy drift with a symplectic one's. The
 * right-hand side computes on the state's values, not with primitives, so it is not differentiable in $\xvec$.
 *
 * @param H The Hamiltonian, its gradients from `aifn-compute/foundation/autodiff` unless given.
 * @returns `rhs`, the right-hand side $f(t, \xvec)$ (independent of $t$), and `energy`, $H$ at a state
 *   $(\qvec, \pvec)$ of length $2d$.
 *
 * @example The harmonic oscillator as a first-order system
 * // H = q²/2 + p²/2: q′ = p, p′ = −q.
 * const { rhs, energy } = hamiltonianSystem({ potential: (q) => mul(0.5, sum(mul(q, q))) })
 * print('f(0, (1, 2)) =', rhs(0, tensor([1, 2])))
 * print('H(1, 2) =', energy([1, 2]))
 *
 * @example Explicit Euler gains energy where symplectic Euler does not
 * // The same oscillator for 1000 steps of 0.1 from (1, 0), where H = 0.5.
 * const H = { potential: (q) => mul(0.5, sum(mul(q, q))) }
 * const { rhs, energy } = hamiltonianSystem(H)
 * const euler = run(rungeKutta(rhs, 'euler', { stepSize: 0.1 }), { x0: [1, 0] }, 1000)
 * const sympl = run(symplectic(H, 'symplectic-euler', { stepSize: 0.1 }), { q0: [1], p0: [0] }, 1000)
 * print('explicit Euler: H =', energy(euler.x))
 * print('symplectic Euler: H =', sympl.energy)
 */
export function hamiltonianSystem(H: SeparableHamiltonian): { rhs: Rhs; energy: (x: VectorLike) => Scalar } {
  const P = parts(H)
  const split = (x: F64) => {
    const d = x.length / 2
    return { q: x.subarray(0, d) as F64, p: x.subarray(d) as F64 }
  }
  return {
    rhs: (_t, x) => {
      const { q, p } = split(Float64Array.from(toFlat(x)))
      const out = new Float64Array(2 * q.length)
      out.set(P.dT(Float64Array.from(p)), 0)
      out.set(
        P.dV(Float64Array.from(q)).map((v) => -v),
        q.length,
      )
      return fromData(out, [out.length])
    },
    energy: (x) => {
      const { q, p } = split(toF64(x, 'energy'))
      return P.T(Float64Array.from(p)) + P.V(Float64Array.from(q))
    },
  }
}

/**
 * The initial value of a symplectic integrator: positions `q0` ($\qvec_0$) and momenta `p0` ($\pvec_0$), of length $d$
 * each, at time `t0` (default 0).
 */
export type PhaseInitial = { q0: VectorLike; p0: VectorLike; t0?: Scalar }

/** The state of a symplectic integrator. `x` is $(\qvec, \pvec)$ concatenated. */
export interface SymplecticState extends OdeState {
  /** The positions $\qvec$ (length $d$). */
  q: Vector
  /** The momenta $\pvec$ (length $d$). */
  p: Vector
  /** $H(\qvec, \pvec) = T(\pvec) + V(\qvec)$. */
  energy: Scalar
  /** $H(\qvec, \pvec) - H(\qvec_0, \pvec_0)$. */
  energyError: Scalar
  /**
   * $\nabla V(\qvec)$ at the current $\qvec$ (reused by symplectic Euler and velocity Verlet). `evaluations` counts
   * gradient evaluations of $V$ and $T$.
   */
  force: Vector
}

/**
 * One step of a symplectic scheme: from positions `q`, momenta `p` and the force $\nabla V(\qvec)$ at `q`, with step
 * size `h`, to the new positions, momenta and force, and the gradient evaluations it took.
 */
type Scheme = (P: Parts, q: F64, p: F64, force: F64, h: number) => { q: F64; p: F64; force: F64; evaluations: number }

/**
 * $\yvec + a\xvec$ in a new array.
 *
 * @param y The array $\yvec$; not modified.
 * @param a The scalar $a$.
 * @param x The array $\xvec$, as long as $\yvec$.
 * @returns A new array of the values of $\yvec + a\xvec$.
 */
const axpy = (y: F64, a: number, x: F64) => Float64Array.from(y, (v, i) => v + a * x[i])

const SCHEMES: Record<string, Scheme> = {
  // Symplectic Euler (variant "kick then drift"): p ← p − h∇V(q), q ← q + h∇T(p). Order 1.
  'symplectic-euler': (P, q, p, force, h) => {
    const p1 = axpy(p, -h, force)
    const q1 = axpy(q, h, P.dT(p1))
    return { q: q1, p: p1, force: P.dV(q1), evaluations: 2 }
  },
  // Leapfrog as drift–kick–drift (Störmer–Verlet, position form): half drift, full kick, half drift. Order 2.
  leapfrog: (P, q, p, _force, h) => {
    const qh = axpy(q, h / 2, P.dT(p))
    const p1 = axpy(p, -h, P.dV(qh))
    const q1 = axpy(qh, h / 2, P.dT(p1))
    return { q: q1, p: p1, force: P.dV(q1), evaluations: 4 }
  },
  // Velocity Verlet as kick–drift–kick: half kick, full drift, half kick; the force at the end is reused. Order 2.
  'velocity-verlet': (P, q, p, force, h) => {
    const ph = axpy(p, -h / 2, force)
    const q1 = axpy(q, h, P.dT(ph))
    const f1 = P.dV(q1)
    const p1 = axpy(ph, -h / 2, f1)
    return { q: q1, p: p1, force: f1, evaluations: 2 }
  },
}

/** The symplectic methods by name. */
export type SymplecticMethod = 'symplectic-euler' | 'leapfrog' | 'velocity-verlet'

/**
 * A fixed-step symplectic integrator for a separable Hamiltonian system: `'symplectic-euler'` (order 1, kick then
 * drift: $\pvec \leftarrow \pvec - h \nabla V(\qvec)$, $\qvec \leftarrow \qvec + h \nabla T(\pvec)$), `'leapfrog'`
 * (drift–kick–drift, order 2) or `'velocity-verlet'` (kick–drift–kick, order 2; one force evaluation per step, the one
 * at the end of the step being reused). Leapfrog and velocity Verlet are the same Störmer–Verlet map with the roles of
 * $\qvec$ and $\pvec$ exchanged. The state records the energy and its error against the initial energy. `init` takes
 * `{ q0, p0, t0 }`. The `force` field holds $\nabla V(\qvec)$ (the gradient, not its negative). An unknown method or
 * a zero or non-finite step throws `DomainError`; `q0` and `p0` of different lengths throw `ShapeError`.
 *
 * @param H The Hamiltonian, its gradients from `aifn-compute/foundation/autodiff` unless given.
 * @param method The scheme to step with.
 * @param options The step size and the optional end time.
 * @param options.stepSize The step size $h$; negative integrates backwards in time.
 * @param options.tEnd The time to stop at: the last step is shortened to land on it and the run is then `done`. When
 *   left out, the run takes as many steps as the runner asks for.
 * @returns The integrator, an `Algorithm` to run with `run(alg, { q0, p0 }, steps)`.
 *
 * @example A harmonic oscillator over many periods
 * // H = q²/2 + p²/2 from (1, 0), whose exact solution is (cos t, −sin t). Over 16 periods the energy error stays small
 * // and bounded; the phase lags slowly (by about h²t/24), so q and p trail the exact values a little.
 * const H = { potential: (q) => mul(0.5, sum(mul(q, q))) }
 * const s = run(symplectic(H, 'velocity-verlet', { stepSize: 0.1, tEnd: 100 }), { q0: [1], p0: [0] }, 2000)
 * print('time =', s.time)
 * print('q, p =', s.q, s.p)
 * print('cos 100, -sin 100 =', Math.cos(100), -Math.sin(100))
 * print('energy error =', s.energyError)
 *
 * @example A pendulum with each method
 * // V(q) = −cos q, started at rest from 1 rad: the energy error of each scheme after 500 steps of 0.1.
 * const H = { potential: (q) => neg(sum(cos(q))) }
 * for (const m of ['symplectic-euler', 'leapfrog', 'velocity-verlet']) {
 *   const s = run(symplectic(H, m, { stepSize: 0.1 }), { q0: [1], p0: [0] }, 500)
 *   print(m, 'energy error =', s.energyError, 'after', s.evaluations, 'gradient evaluations')
 * }
 */
export function symplectic(
  H: SeparableHamiltonian,
  method: SymplecticMethod,
  { stepSize: h, tEnd }: FixedStepOptions,
): Algorithm<PhaseInitial, SymplecticState> {
  const scheme = SCHEMES[method]
  if (!scheme) throw new DomainError('symplectic', `symplectic: unknown method ${method}`)
  if (!(h !== 0 && Number.isFinite(h)))
    throw new DomainError('symplectic', 'symplectic: the step size must be finite and non-zero')
  const P = parts(H)
  return {
    name: method,
    init: ({ q0, p0, t0 = 0 }) => {
      checkDirection(t0, h, tEnd, method)
      const q = toF64(q0, method)
      const p = toF64(p0, method)
      if (q.length !== p.length) throw new ShapeError('symplectic', 'symplectic: q0 and p0 must have the same length')
      const x = new Float64Array(2 * q.length)
      x.set(q)
      x.set(p, q.length)
      const e = P.T(p) + P.V(q)
      const force = P.dV(q)
      return {
        ...initialState(x, t0),
        q: fromData(q, [q.length]),
        p: fromData(p, [p.length]),
        energy: e,
        energyError: 0,
        force: fromData(force, [force.length]),
        evaluations: 1,
      }
    },
    step: (s) => {
      const hk = nextStep(s.time, h, tEnd)
      const r = scheme(P, dense.data(s.q), dense.data(s.p), dense.data(s.force), hk)
      const x = new Float64Array(2 * r.q.length)
      x.set(r.q)
      x.set(r.p, r.q.length)
      const e = P.T(r.p) + P.V(r.q)
      const finite = allFinite(x) && Number.isFinite(e)
      // The initial energy H(q₀, p₀) is carried as energy − energyError, so the state alone determines the next one.
      const reference = s.energy - s.energyError
      return {
        ...s,
        t: s.t + 1,
        time: s.time + hk,
        x: fromData(x, [x.length]),
        q: fromData(r.q, [r.q.length]),
        p: fromData(r.p, [r.p.length]),
        force: fromData(r.force, [r.force.length]),
        stepSize: hk,
        energy: e,
        energyError: e - reference,
        evaluations: s.evaluations + r.evaluations,
        diverged: !finite,
        failure: finite ? null : 'not finite',
      }
    },
    done: (s) => reached(s.time, h, tEnd),
  }
}

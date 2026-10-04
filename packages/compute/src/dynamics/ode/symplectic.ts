/**
 * Symplectic integrators for separable Hamiltonian systems H(q, p) = T(p) + V(q): symplectic Euler, leapfrog
 * (drift–kick–drift) and velocity Verlet (kick–drift–kick), with the energy tracked at every step. A symplectic method
 * preserves phase-space volume and exactly conserves a nearby "shadow" Hamiltonian, so its energy error stays bounded
 * (oscillates at O(h^order)) over exponentially long times instead of drifting (Hairer, Lubich & Wanner, 2006,
 * "Geometric Numerical Integration", 2nd ed., §I.1, §VI.3 and §IX.8).
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
 * A separable Hamiltonian H(q, p) = T(p) + V(q) with q and p of length d. `potential` V is required; `kinetic` T
 * defaults to ½‖p‖² (unit mass). Their gradients (the force −∇V and the velocity ∇T) come from `aifn-compute/foundation/autodiff`
 * unless given, so V and T must then be written with `aifn-compute/foundation/tensor` primitives.
 */
export type SeparableHamiltonian = {
  potential: (q: Tensor) => Value
  kinetic?: (p: Tensor) => Value
  /** ∇V(q), if known in closed form. */
  potentialGradient?: (q: Tensor) => VectorLike
  /** ∇T(p), if known in closed form. */
  kineticGradient?: (p: Tensor) => VectorLike
}

/** The pieces of a separable Hamiltonian as plain functions on working arrays. */
type Parts = { V: (q: F64) => number; T: (p: F64) => number; dV: (q: F64) => F64; dT: (p: F64) => F64 }

const asTensor = (v: F64): Tensor => fromData(v, [v.length])
const scalarOf = (v: Value): number => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])

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
 * The first-order system of a separable Hamiltonian, for any ODE solver: the state x = (q, p) of length 2d moves by
 * q′ = ∇T(p), p′ = −∇V(q). Also returns the energy H(x) = T(p) + V(q), for comparing a general solver's energy drift
 * with a symplectic one's.
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

/** The initial value of a symplectic integrator: positions q₀ and momenta p₀ (length d each), at time `t0`. */
export type PhaseInitial = { q0: VectorLike; p0: VectorLike; t0?: Scalar }

/** The state of a symplectic integrator. x is (q, p) concatenated. */
export interface SymplecticState extends OdeState {
  q: Vector
  p: Vector
  /** H(q, p) = T(p) + V(q). */
  energy: Scalar
  /** H(q, p) − H(q₀, p₀). */
  energyError: Scalar
  /** ∇V(q) at the current q (reused by velocity Verlet). `evaluations` counts gradient evaluations of V and T. */
  force: Vector
}

type Scheme = (P: Parts, q: F64, p: F64, force: F64, h: number) => { q: F64; p: F64; force: F64; evaluations: number }

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
 * A fixed-step symplectic integrator for a separable Hamiltonian system: `'symplectic-euler'` (order 1),
 * `'leapfrog'` (drift–kick–drift, order 2) or `'velocity-verlet'` (kick–drift–kick, order 2; one force evaluation per
 * step, the one at the end of the step being reused). Leapfrog and velocity Verlet are the same Störmer–Verlet map
 * with the roles of q and p exchanged. The state records the energy and its error against the initial energy.
 * `init` takes `{ q0, p0, t0 }`. The `force` field holds ∇V(q) (the gradient, not its negative).
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

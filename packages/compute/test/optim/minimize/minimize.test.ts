import { describe, expect, it } from 'vitest'
import { minimize, type Method } from 'aifn-compute/optim/minimize'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { bowl, rosenbrock, rosenbrockObjective } from '../problems'

describe('minimize', () => {
  it('runs every method by name', () => {
    const b = bowl({ condition: 4 })
    const methods: Method[] = [
      'gradient-descent',
      'momentum',
      'nesterov',
      'adam',
      'bfgs',
      'lbfgs',
      'conjugate-gradient',
      'nelder-mead',
    ]
    for (const method of methods) {
      const r = minimize(b.objective, b.start, { method, maxSteps: 3000, stepSize: 0.1 } as never)
      expect(r.method).toBe(method)
      expect(r.value, method).toBeLessThan(1e-6)
    }
    expect(minimize(b.objective, b.start, { method: 'newton', hessian: b.hessian }).converged).toBe(true)
    expect(minimize(b.objective, b.start, { method: 'cma-es', stream: stream(1) }).value).toBeLessThan(1e-8)
  })

  it('minimises an Objective written with primitives, differentiating it by autodiff', () => {
    const r = minimize(rosenbrockObjective, [-1.2, 1])
    expect(r.method).toBe('lbfgs')
    expect(r.converged).toBe(true)
    toFlat(r.x).forEach((v) => expect(v).toBeCloseTo(1, 5))
    // Autodiff gradients agree with the analytic ones along the way.
    const byHand = minimize(rosenbrock().objective, [-1.2, 1])
    expect(r.steps).toBe(byHand.steps)
    const nm = minimize(rosenbrockObjective, [-1.2, 1], { method: 'nelder-mead', maxSteps: 2000 })
    expect(nm.value).toBeLessThan(1e-8)
  })
})

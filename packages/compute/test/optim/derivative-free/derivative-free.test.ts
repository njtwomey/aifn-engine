import { describe, expect, it } from 'vitest'
import { cmaEs, nelderMead, simulatedAnnealing } from 'aifn-compute/optim/derivative-free'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, type Vector } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import type { Algorithm } from 'aifn-compute/foundation/contracts'
import type { IterateState, StartOptions } from 'aifn-compute/optim'
import { checkProtocol } from '../../protocol'
import { rosenbrock } from '../problems'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tol)
}
const rosen = rosenbrock()
/** Rastrigin's function 20 + Σ(x_i² − 10 cos 2πx_i), minimum 0 at the origin. */
const rastrigin = (v: Vector) => toFlat(v).reduce((s, x) => s + x * x - 10 * Math.cos(2 * Math.PI * x), 20)

describe('derivative-free methods', () => {
  it('Nelder–Mead finds the Rosenbrock minimum', () => {
    const s = run(nelderMead(rosen.value), { x0: rosen.start }, 2000)
    expect(s.converged).toBe(true)
    close(toFlat(s.x), [1, 1], 1e-4)
  })

  it('CMA-ES finds the Rosenbrock minimum, and simulated annealing lowers Rastrigin', () => {
    const s = run(cmaEs(rosen.value), { x0: rosen.start }, 1000, { stream: stream(1) })
    expect(s.converged).toBe(true)
    close(toFlat(s.x), [1, 1], 1e-4)
    const start = [2.3, -1.7]
    const a = run(simulatedAnnealing(rastrigin, { temperature: 5 }), { x0: start }, 3000, { stream: stream(2) })
    expect(a.bestValue).toBeLessThan(rastrigin(tensor(start)))
  })

  it.each<[string, Algorithm<StartOptions, IterateState>]>([
    ['nelderMead', nelderMead(rosen.value)],
    ['cmaEs', cmaEs(rosen.value)],
    ['simulatedAnnealing', simulatedAnnealing(rosen.value)],
  ])('%s satisfies the Algorithm protocol', (_, alg) => {
    checkProtocol(alg, { x0: rosen.start }, { steps: 12, record: { value: (s: IterateState) => s.value } })
  })

  it('stochastic methods draw from the runner stream: another key, another path', () => {
    const a = run(cmaEs(rosen.value), { x0: rosen.start }, 5, { stream: stream(1) })
    const b = run(cmaEs(rosen.value), { x0: rosen.start }, 5, { stream: stream(2) })
    expect(toFlat(a.x)).not.toEqual(toFlat(b.x))
  })
})

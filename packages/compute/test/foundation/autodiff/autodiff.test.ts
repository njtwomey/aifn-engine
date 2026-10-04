import { describe, expect, it } from 'vitest'
import {
  grad,
  gradCheck,
  hessian,
  hvp,
  jacobian,
  jvp,
  NotDifferentiableError,
  stopGradient,
  traceGraph,
  valueAndGrad,
  vjp,
} from 'aifn-compute/foundation/autodiff'
import { cholesky, choleskyLogDet } from 'aifn-compute/numerics/linalg'
import { digamma, logGamma, normalLogPdf, softplus, sigmoid, trigamma } from 'aifn-compute/numerics/special'
import {
  add,
  cos,
  div,
  dot,
  exp,
  eye,
  get,
  log,
  logsumexp,
  map,
  matmul,
  mul,
  sin,
  square,
  sub,
  sum,
  tanh,
  tensor,
  toArray,
  toFlat,
  transpose,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type Nested = number | Nested[]
type Fx = {
  softplus: { x: number[]; w: number[]; value: number; grad: number[] }
  logsumexp: { x: number[][]; w: number[]; value: number; grad: number[][]; hessian: number[][] }
  matmul: { a: number[][]; b: number[][]; w: number[][]; value: number; gradA: number[][]; gradB: number[][] }
  choleskyLogDet: { c: number[][]; value: number; grad: number[][] }
  normalLogLikelihood: {
    data: number[]
    params: number[]
    value: number
    grad: number[]
    hessian: number[][]
    v: number[]
    hvp: number[]
  }
  logGamma: { x: number[]; d1: number[]; d2: number[]; d3: number[] }
  pytree: {
    W: number[][]
    b: number[]
    u: number[]
    x: number[]
    value: number
    gradW: number[][]
    gradB: number[]
    gradU: number[]
  }
}
const F = fixture<Fx>('foundation/autodiff')

/** Expect two nested arrays (or numbers) to agree elementwise to a relative tolerance. */
function close(actual: unknown, expected: Nested, tol = 1e-12) {
  const a = flatten(actual)
  const e = flatten(expected)
  expect(a.length).toBe(e.length)
  a.forEach((v, k) => {
    expect(Math.abs(v - e[k]), `element ${k}: ${v} vs ${e[k]}`).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(e[k])))
  })
}
function flatten(x: unknown): number[] {
  if (typeof x === 'number') return [x]
  if (Array.isArray(x)) return x.flatMap(flatten)
  return toFlat(unwrap(x as Value) as Tensor)
}

const T = (x: Nested) => tensor(x as number[])

describe('grad against torch', () => {
  it('softplus', () => {
    const w = T(F.softplus.w)
    const f = (x: Value) => sum(mul(w, softplus(x)))
    const { value, grad: g } = valueAndGrad(f)(T(F.softplus.x))
    expect(value).toBeCloseTo(F.softplus.value, 12)
    close(g, F.softplus.grad)
  })
  it('logsumexp along an axis, and its Hessian', () => {
    const w = T(F.logsumexp.w)
    const f = (x: Value) => sum(mul(w, logsumexp(x, 1)))
    close(grad(f)(T(F.logsumexp.x)), F.logsumexp.grad)
    const h = hessian(f)(T(F.logsumexp.x)) as Tensor
    expect(h.shape).toEqual([3, 4, 3, 4])
    close(h, F.logsumexp.hessian, 1e-12)
  })
  it('matmul in both arguments (argnums)', () => {
    const w = T(F.matmul.w)
    const f = (a: Value, b: Value) => sum(mul(w, tanh(matmul(a, b))))
    const [ga, gb] = grad(f, { argnums: [0, 1] })(T(F.matmul.a), T(F.matmul.b))
    close(ga, F.matmul.gradA)
    close(gb, F.matmul.gradB)
    close(grad(f, { argnums: 1 })(T(F.matmul.a), T(F.matmul.b)), F.matmul.gradB)
  })
  it('Cholesky log-determinant', () => {
    const n = F.choleskyLogDet.c.length
    const f = (c: Value) => choleskyLogDet(cholesky(add(matmul(c, transpose(c)), mul(n, eye(n)))).L)
    const { value, grad: g } = valueAndGrad(f)(T(F.choleskyLogDet.c))
    expect(value).toBeCloseTo(F.choleskyLogDet.value, 11)
    close(g, F.choleskyLogDet.grad, 1e-10)
  })
  const { data, params, v } = F.normalLogLikelihood
  const x = T(data)
  const loglik = (p: Value) => {
    const s = exp(get(p, 1))
    return sum(sub(normalLogPdf(div(sub(x, get(p, 0)), s)), log(s)))
  }
  it('normal log-likelihood: value, gradient and Hessian', () => {
    const { value, grad: g } = valueAndGrad(loglik)(T(params))
    expect(value).toBeCloseTo(F.normalLogLikelihood.value, 11)
    close(g, F.normalLogLikelihood.grad)
    close(hessian(loglik)(T(params)), F.normalLogLikelihood.hessian, 1e-11)
  })
  it('hvp equals torch and H·v', () => {
    const hv = hvp(loglik, T(params), T(v))
    close(hv, F.normalLogLikelihood.hvp, 1e-11)
    const H = F.normalLogLikelihood.hessian
    close(hv, [H[0][0] * v[0] + H[0][1] * v[1], H[1][0] * v[0] + H[1][1] * v[1]], 1e-11)
  })
  it('pytree arguments: a gradient with the structure of the argument', () => {
    const P = F.pytree
    const xin = T(P.x)
    const f = (p: { W: Value; b: Value; u: Value }) =>
      add(sum(mul(softplus(add(matmul(p.W, xin), p.b)), p.u)), mul(0.5, sum(square(p.W))))
    const { value, grad: g } = valueAndGrad(f)({ W: T(P.W), b: T(P.b), u: T(P.u) })
    expect(value).toBeCloseTo(P.value, 12)
    close(g.W, P.gradW)
    close(g.b, P.gradB)
    close(g.u, P.gradU)
    // Arrays and static entries: the static label passes through untouched.
    const h = (q: [Value, { k: Value; name: string }]) => mul(q[0], q[1].k)
    const gq = grad(h)([2, { k: 3, name: 'static' }]) as unknown as [number, { k: number; name: string }]
    expect(gq).toEqual([3, { k: 2, name: 'static' }])
  })
})

describe('second and higher derivatives', () => {
  it("logΓ'' = ψ₁ and logΓ''' = ψ₂ through nested grad", () => {
    const d1 = grad((x: Value) => logGamma(x))
    const d2 = grad(d1)
    const d3 = grad(d2)
    F.logGamma.x.forEach((x, k) => {
      expect(d1(x)).toBeCloseTo(F.logGamma.d1[k], 10)
      expect(Math.abs((d2(x) as number) - F.logGamma.d2[k])).toBeLessThan(1e-10 * Math.max(1, F.logGamma.d2[k]))
      expect(Math.abs((d3(x) as number) - F.logGamma.d3[k])).toBeLessThan(
        1e-9 * Math.max(1, Math.abs(F.logGamma.d3[k])),
      )
    })
    // Every special function's derivative is written with primitives, so a fourth derivative exists too: ψ₃ by
    // central differences of ψ₂.
    const h = 1e-4
    const psi3 = ((d3(1.5 + h) as number) - (d3(1.5 - h) as number)) / (2 * h)
    expect(Math.abs((grad(d3)(1.5) as number) - psi3)).toBeLessThan(1e-6 * Math.abs(psi3))
  })
  it('elementary compositions: d²/dx² of x sin x and of sigmoid', () => {
    const f = (x: Value) => mul(x, sin(x))
    const x0 = 0.7
    expect(grad(grad(f))(x0)).toBeCloseTo(2 * Math.cos(x0) - x0 * Math.sin(x0), 14)
    const s = sigmoid(0.3)
    expect(grad(grad((x: Value) => sigmoid(x)))(0.3)).toBeCloseTo(s * (1 - s) * (1 - 2 * s), 14)
  })
  it('closures over an outer variable are not confused with the inner one', () => {
    // d/dx [x · d/dy (x y)] = d/dx x² = 2x.
    const f = (x: Value) => mul(x, grad((y: Value) => mul(x, y))(1))
    expect(grad(f)(3)).toBeCloseTo(6, 14)
    // The inner gradient treats x as a constant: d/dy (x + y) = 1 whatever x is.
    const g = (x: Value) => grad((y: Value) => add(mul(x, x), y))(2)
    expect(grad(g)(5)).toBe(0)
  })
  it('the mixed partial of a function of two arguments', () => {
    const f = (a: Value, b: Value) => mul(square(a), exp(b))
    const dfda = (a: Value, b: Value) => grad(f)(a, b)
    // ∂²f/∂a∂b = 2a eᵇ.
    expect(grad(dfda, { argnums: 1 })(1.5, 0.4)).toBeCloseTo(3 * Math.exp(0.4), 13)
  })
})

describe('forward mode, Jacobians and Hessians', () => {
  const A = tensor([
    [1, 2, 0],
    [0, -1, 3],
  ])
  const f = (x: Value) => tanh(matmul(A, mul(x, x)))
  const x0 = tensor([0.3, -0.5, 0.8])
  it('jacobian has shape [out, in] and matches finite differences', () => {
    const J = jacobian(f)(x0) as Tensor
    expect(J.shape).toEqual([2, 3])
    const h = 1e-6
    for (let j = 0; j < 3; j++) {
      const e = toFlat(x0)
      e[j] += h
      const up = toFlat(f(tensor(e)) as Tensor)
      e[j] -= 2 * h
      const down = toFlat(f(tensor(e)) as Tensor)
      for (let i = 0; i < 2; i++) expect(get(J, i, j)).toBeCloseTo((up[i] - down[i]) / (2 * h), 8)
    }
  })
  it('jvp equals J·v', () => {
    const v = tensor([1, -2, 0.5])
    const { value, tangent } = jvp(f, x0, v)
    close(value, toFlat(f(x0) as Tensor))
    close(tangent, toFlat(matmul(jacobian(f)(x0), v) as Tensor), 1e-14)
    // Numbers in, numbers out.
    const s = jvp((x: Value) => sin(x), 0.4, 2)
    expect(s.tangent).toBeCloseTo(2 * Math.cos(0.4), 15)
  })
  it('vjp equals uᵀJ', () => {
    const u = tensor([0.4, -1.1])
    const { pullback } = vjp(f, x0)
    close(pullback(u), toFlat(matmul(u, jacobian(f)(x0)) as Tensor), 1e-14)
  })
  it('hvp on a quadratic is A·v, and hessian is A', () => {
    const Q = tensor([
      [3, 1, 0],
      [1, 2, -1],
      [0, -1, 4],
    ])
    const q = (x: Value) => mul(0.5, dot(x, matmul(Q, x)))
    const v = tensor([1, 2, -1])
    close(hvp(q, x0, v), toFlat(matmul(Q, v)), 1e-14)
    close(hessian(q)(x0), toArray(Q) as Nested, 1e-14)
    expect(hessian((x: Value) => mul(x, mul(x, x)))(2)).toBeCloseTo(12, 14)
  })
  it('jvp of a pytree argument', () => {
    const g = (p: { a: Value; b: Value }) => mul(p.a, cos(p.b))
    const { tangent } = jvp(g, { a: 2, b: 0.5 }, { a: 1, b: 3 })
    expect(tangent).toBeCloseTo(Math.cos(0.5) - 3 * 2 * Math.sin(0.5), 14)
  })
})

describe('conventions and errors', () => {
  it('a primitive without a derivative on a path to the output is an error', () => {
    expect(() => grad((x: Value) => sum(map(x, Math.floor)))(tensor([1.5, 2]))).toThrow(NotDifferentiableError)
    expect(() => grad((x: Value) => sum(map(x, Math.floor)))(tensor([1.5, 2]))).toThrow(/no derivative: map/)
    // A non-differentiable argument of a binary primitive is an error only when it is differentiated.
    expect(() => grad((k: Value) => trigamma(k))(2)).not.toThrow()
  })
  it('stopGradient makes its argument a constant', () => {
    const f = (x: Value) => mul(x, stopGradient(x))
    expect(grad(f)(3)).toBe(3)
    // A zero-derivative primitive's output is a constant to the reverse interpreter: it is not recorded.
    expect(traceGraph(f, 3).nodes.map((n) => n.op)).toEqual(['input', 'mul'])
  })
  it('an output that does not depend on the argument has a zero gradient of the argument’s shape', () => {
    expect(grad((_x: Value) => 2)(5)).toBe(0)
    expect(toFlat(grad((_x: Value) => sum(tensor([1, 2])))(tensor([1, 2, 3])) as Tensor)).toEqual([0, 0, 0])
  })
  it('grad needs a scalar output', () => {
    expect(() => grad((x: Value) => mul(x, 2))(tensor([1, 2]))).toThrow(/number or a rank-0 tensor/)
  })
  it('top-level results are raw numbers and tensors, not traced values', () => {
    const g = grad((x: Value) => sum(square(x)))(tensor([1, 2]))
    expect(Array.from((g as Tensor).data)).toEqual([2, 4])
    expect(typeof grad((x: Value) => square(x))(3)).toBe('number')
  })
  it('first derivatives of digamma agree with trigamma (scalar rules in a first-order sweep)', () => {
    expect(grad((x: Value) => digamma(x))(2.5)).toBeCloseTo(trigamma(2.5), 13)
  })
})

describe('gradCheck', () => {
  it('passes for a correct gradient and reports every element', () => {
    const f = (p: { w: Value; b: Value }) => sum(softplus(add(mul(p.w, tensor([1, -2, 0.5])), p.b)))
    const report = gradCheck(f, { w: tensor([0.2, -0.4, 1.1]), b: 0.3 })
    expect(report.ok).toBe(true)
    expect(report.entries.map((e) => e.path)).toEqual(['x.w', 'x.w', 'x.w', 'x.b'])
    expect(report.maxRelError).toBeLessThan(1e-7)
  })
  it('fails, and says where, when the gradient is wrong', () => {
    // stopGradient hides the dependence from reverse mode but not from finite differences.
    const bad = gradCheck((x: Value) => add(stopGradient(square(x)), 0), 1.5)
    expect(bad.ok).toBe(false)
    expect(bad.entries[0].analytic).toBe(0)
    expect(bad.entries[0].numeric).toBeCloseTo(3, 6)
  })
})

describe('traceGraph', () => {
  it('records nodes in topological order with values and adjoints', () => {
    // f(a, b) = (a·b + sin a)², at a = 2, b = 3.
    const f = ([a, b]: Value[]) => square(add(mul(a, b), sin(a)))
    const g = traceGraph(f, [2, 3])
    expect(g.nodes.map((n) => n.op)).toEqual(['input', 'input', 'mul', 'sin', 'add', 'square'])
    expect(g.nodes.map((n) => n.label)).toEqual(['x[0]', 'x[1]', undefined, undefined, undefined, undefined])
    expect(g.inputs).toEqual([0, 1])
    expect(g.output).toBe(5)
    for (const node of g.nodes) {
      for (const input of node.inputs) if ('node' in input) expect(input.node).toBeLessThan(node.id)
    }
    const s = 6 + Math.sin(2)
    expect(g.value).toBeCloseTo(s * s, 14)
    expect(g.nodes[4].value).toBeCloseTo(s, 14)
    // Adjoints: ∂f/∂(sum) = 2s, ∂f/∂(ab) = 2s, ∂f/∂(sin a) = 2s, ∂f/∂a = 2s(b + cos a), ∂f/∂b = 2s·a.
    expect(g.nodes[5].adjoint).toBe(1)
    expect(g.nodes[4].adjoint).toBeCloseTo(2 * s, 14)
    expect(g.nodes[0].adjoint).toBeCloseTo(2 * s * (3 + Math.cos(2)), 13)
    expect(g.nodes[1].adjoint).toBeCloseTo(2 * s * 2, 13)
    close(g.grad, [2 * s * (3 + Math.cos(2)), 4 * s], 1e-14)
  })
  it('each edge carries its local partial and its backward message, and messages sum to the adjoints', () => {
    // f(a, b) = (a + b)·(a·b): a and b fan out to two operations each.
    const g = traceGraph(([a, b]: Value[]) => mul(add(a, b), mul(a, b)), [2, 3])
    expect(g.nodes.map((n) => n.op)).toEqual(['input', 'input', 'add', 'mul', 'mul'])
    const edge = (id: number, k: number) => g.nodes[id].inputs[k] as { partial: number; message: number }
    // ∂(ab)/∂a = b, ∂(ab)/∂b = a; the product's partials are the other factor's value.
    expect([edge(3, 0).partial, edge(3, 1).partial]).toEqual([3, 2])
    expect([edge(4, 0).partial, edge(4, 1).partial]).toEqual([6, 5])
    expect([edge(2, 0).partial, edge(2, 1).partial]).toEqual([1, 1])
    // message = adjoint of the node × partial.
    for (const node of g.nodes)
      for (const input of node.inputs)
        if ('node' in input) expect(input.message).toBeCloseTo((node.adjoint as number) * (input.partial as number), 14)
    // An input's adjoint is the sum of the messages on its outgoing edges.
    for (const id of g.inputs) {
      let total = 0
      for (const node of g.nodes)
        for (const input of node.inputs) if ('node' in input && input.node === id) total += input.message as number
      expect(total).toBeCloseTo(g.nodes[id].adjoint as number, 14)
    }
    // f = a²b + ab², so ∇f = (2ab + b², a² + 2ab) = (21, 16).
    expect(g.grad).toEqual([21, 16])
  })
  it('constants are listed as constant inputs, and nodes off the path to the output have no adjoint', () => {
    const g = traceGraph((x: Value) => {
      void exp(x) // computed and recorded, but unused
      return mul(x, 4)
    }, 1)
    expect(g.nodes.map((n) => n.op)).toEqual(['input', 'exp', 'mul'])
    expect(g.nodes[1].adjoint).toBeNull()
    expect(g.nodes[2].inputs).toEqual([{ node: 0, partial: 4, message: 4 }, { constant: 4 }])
    expect(g.nodes[1].inputs).toEqual([{ node: 0, partial: Math.E, message: null }])
    expect(g.grad).toBe(4)
  })
  it('tensor nodes keep tensor values and adjoints', () => {
    const g = traceGraph((x: Value) => sum(exp(x)), tensor([0, 1]))
    const e = g.nodes[1]
    expect(e.op).toBe('exp')
    close(e.value, [1, Math.E], 1e-15)
    close(e.adjoint, [1, 1])
    close(g.grad, [1, Math.E], 1e-15)
  })
  it('marks primitives without derivatives, and throws if the output depends on them', () => {
    expect(() => traceGraph((x: Value) => sum(map(x, Math.round)), tensor([0.2]))).toThrow(NotDifferentiableError)
  })
})

describe('argnums (review 2026-10-01)', () => {
  it('refuses a repeated argnum, which would trace the argument twice and report a zero gradient', () => {
    expect(() => grad((a: Value, b: Value) => mul(a, b), { argnums: [0, 0] })(2, 3)).toThrow(/repeat/)
    expect(grad((a: Value, b: Value) => mul(a, b), { argnums: [1, 0] })(2, 3)).toEqual([2, 3])
  })
})

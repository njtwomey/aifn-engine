/**
 * aifn-compute/nn/sequence against scipy, numpy and torch (`fixtures/nn/sequence.json`): discretisation (zero-order hold,
 * bilinear; a singular A), HiPPO-LegS, the S4 kernel and both modes, Mamba's selective scan with gradients, causal
 * linear attention, Bahdanau and Luong attention, and the hidden-state gradients of backpropagation through time.
 * Laws: the convolutional and recurrent modes agree; linear attention's parallel and recurrent forms agree (with
 * decay too); discretisation differentiates; truncation zeroes gradients beyond the window.
 */
import { describe, expect, it } from 'vitest'
import { grad, gradCheck } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { add, mul, sum, tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { GruCell, type CellParams } from 'aifn-compute/nn/layers'
import {
  BahdanauAttention,
  causalConvolution,
  discretiseDiagonal,
  gradientsThroughTime,
  hippoLegS,
  linearAttention,
  linearAttentionRecurrent,
  linearRecurrence,
  LuongAttention,
  SelectiveSsm,
  selectiveScan,
  ssmKernel,
  ssmRecurrent,
  type BahdanauParams,
  type LuongParams,
  type LuongScore,
} from 'aifn-compute/nn/sequence'
import { discretiseSsm } from 'aifn-compute/systems'
import { fixture } from '../../fixtures'

type Nested = number | Nested[]
type Tree = { [k: string]: Tree | Nested }
const F = fixture<Record<string, Record<string, unknown>>>('nn/sequence')
const T = (x: unknown) => tensor(x as never)
const flat = (v: unknown): number[] => {
  if (Array.isArray(v)) return (v as unknown[]).flatMap((e) => flat(e))
  if (typeof v === 'number') return [v]
  const r = unwrap(v as Value)
  return typeof r === 'number' ? [r] : toFlat(r)
}
function close(actual: unknown, expected: unknown, tol = 1e-10) {
  const a = flat(actual)
  const e = flat(expected)
  expect(a.length).toBe(e.length)
  let worst = 0
  a.forEach((v, i) => (worst = Math.max(worst, Math.abs(v - e[i]) / Math.max(1, Math.abs(e[i])))))
  expect(worst).toBeLessThan(tol)
}
const toTensors = (t: Tree): unknown =>
  Object.fromEntries(Object.entries(t).map(([k, v]) => [k, Array.isArray(v) ? T(v) : toTensors(v as Tree)]))
const weighted = (r: unknown) => (y: Value) => sum(mul(y, T(r)))

describe('state-space discretisation and kernels', () => {
  const D = F.discretise as Record<string, never>

  it('zero-order hold and bilinear match scipy cont2discrete, including a singular A', () => {
    for (const method of ['zoh', 'bilinear'] as const) {
      const r = discretiseSsm(T(D.A), T(D.B), D.dt, method)
      close(r.A, (D[method] as Record<string, unknown>).A, 1e-12)
      close(r.B, (D[method] as Record<string, unknown>).B, 1e-12)
    }
    const s = D.singular as Record<string, never>
    const r = discretiseSsm(T(s.A), T(s.B), s.dt)
    close(r.A, s.Ad, 1e-12)
    close(r.B, s.Bd, 1e-12)
  })

  it('discretisation differentiates in the step', () => {
    const f = (step: Value) => sum(discretiseSsm(T(D.A), T(D.B), step, 'zoh').B)
    const g = flat(grad(f)(0.1) as Value)[0]
    const h = 1e-6
    const fd = (flat(f(0.1 + h))[0] - flat(f(0.1 - h))[0]) / (2 * h)
    expect(Math.abs(g - fd)).toBeLessThan(1e-7)
    for (const method of ['zoh', 'bilinear', 'euler'] as const) {
      const d = discretiseDiagonal(-0.5, 2, 0.1, method)
      const full = discretiseSsm(tensor([[-0.5]]), tensor([[2]]), 0.1, method)
      close(d.A, full.A, 1e-13)
      close(d.B, full.B, 1e-13)
    }
  })

  it('diagonal zero-order hold is accurate for small and zero Δa (review: (e^{Δa} − 1)/a cancelled, and was NaN at a = 0)', () => {
    // B̄ = (e^{Δa} − 1)/a·b → Δb as a → 0.
    expect(flat(discretiseDiagonal(tensor([0]), 1, 0.1, 'zoh').B as Value)[0]).toBe(0.1)
    const tiny = flat(discretiseDiagonal(tensor([-1e-12]), 1, 0.1, 'zoh').B as Value)[0]
    // Exact: 0.1·(1 − Δa/2 + …) with Δa = −1e-13; the naive form was off by 3e-4 relative.
    expect(Math.abs(tiny - 0.1 * (1 - 0.5e-13)) / 0.1).toBeLessThan(1e-15)
    const g = flat(grad((a: Value) => sum(discretiseDiagonal(a, 1, 0.1, 'zoh').B))(tensor([0, -0.5])) as Value)
    expect(g.every(Number.isFinite)).toBe(true)
  })

  it('HiPPO-LegS matches S4’s transition', () => {
    for (const [n, m] of Object.entries(F.hippo as Record<string, { A: Nested; B: Nested }>)) {
      const { A, B } = hippoLegS(Number(n))
      close(A, m.A, 1e-14)
      close(B, m.B, 1e-14)
    }
  })

  it('the S4 kernel, its convolution and the scanned recurrence agree with numpy', () => {
    const s4 = F.s4 as Record<string, Nested>
    const sys = discretiseSsm(T(D.A), T(D.B), D.dt, 'zoh')
    const C = T(flat(D.C))
    const K = ssmKernel(sys.A, sys.B, C, 12)
    close(K, s4.kernel, 1e-12)
    close(causalConvolution(K, T(s4.u)), s4.y, 1e-12)
    const rec = ssmRecurrent(sys, C, T(s4.u))
    close(rec.outputs, s4.y, 1e-12)
    close(rec.states, s4.states, 1e-12)
  })
})

describe('selective scan', () => {
  const S = F.selective as Record<string, Nested>
  const args = () => [T(S.x), T(S.delta), T(S.A), T(S.B), T(S.C)] as const

  it('matches Mamba’s selective_scan_ref with its gradients', () => {
    const opts = { D: T(S.D), z: T(S.z) }
    close(selectiveScan(...args(), opts).outputs, S.y, 1e-11)
    const f = (x: Value, delta: Value, A: Value, B: Value, C: Value) =>
      weighted(S.r)(selectiveScan(x, delta, A, B, C, opts).outputs)
    const expected = [S.gx, S.gdelta, S.gA, S.gB, S.gC]
    for (let i = 0; i < 5; i++) close(grad(f, { argnums: i as 0 })(...args()), expected[i], 1e-10)
  })

  it('a layer trains: correct gradients through Δ, B, C, A and D', () => {
    const layer = SelectiveSsm(4, { state: 3, deltaRank: 2 })
    const p = layer.init(stream('ssm'))
    const x = tensor(Array.from({ length: 6 }, (_, t) => Array.from({ length: 4 }, (_, j) => Math.sin(t + j))))
    expect((unwrap(layer.apply(p, x)) as Tensor).shape).toEqual([6, 4])
    expect(gradCheck((q: typeof p) => sum(layer.apply(q, x)), p, { rtol: 1e-4, atol: 1e-6 }).ok).toBe(true)
  })

  it('linearRecurrence continues from an initial state', () => {
    const a = tensor([0.5, 0.9, 1.1])
    const b = tensor([1, 2, 3])
    close(linearRecurrence(a, b, { initial: 2 }), [0.5 * 2 + 1, 0.9 * 2 + 2, 1.1 * (0.9 * 2 + 2) + 3], 1e-14)
  })
})

describe('linear attention', () => {
  const L = F.linearAttention as Record<string, Nested>

  it('matches the causal cumulative-sum form, with gradients', () => {
    close(linearAttention(T(L.q), T(L.k), T(L.v)), L.y, 1e-12)
    close(linearAttentionRecurrent(T(L.q), T(L.k), T(L.v)).output, L.y, 1e-12)
    const f = (q: Value, k: Value, v: Value) => weighted(L.r)(linearAttentionRecurrent(q, k, v).output)
    const expected = [L.gq, L.gk, L.gv]
    for (let i = 0; i < 3; i++) close(grad(f, { argnums: i as 0 })(T(L.q), T(L.k), T(L.v)), expected[i], 1e-11)
  })

  it('parallel and recurrent forms agree with a decay and without normalisation (retention)', () => {
    const opts = { decay: 0.8, normalise: false }
    close(
      linearAttention(T(L.q), T(L.k), T(L.v), opts),
      flat(linearAttentionRecurrent(T(L.q), T(L.k), T(L.v), opts).output),
      1e-12,
    )
  })
})

describe('sequence-to-sequence attention', () => {
  it('Bahdanau attention matches torch, with a mask and gradients', () => {
    const c = F.bahdanau as Record<string, Nested> & { params: Tree }
    const att = BahdanauAttention(3, 4, 6)
    const p = toTensors(c.params) as BahdanauParams
    const r = att.attend(p, T(c.s), T(c.h), undefined, T(c.mask))
    close(r.context, c.context, 1e-12)
    close(r.weights, c.weights, 1e-12)
    const f = (s: Value, h: Value) => weighted(c.r)(att.attend(p, s, h, undefined, T(c.mask)).context)
    close(grad(f, { argnums: 0 })(T(c.s), T(c.h)), c.gs, 1e-11)
    close(grad(f, { argnums: 1 })(T(c.s), T(c.h)), c.gh, 1e-11)
    expect(flat(att.init(stream('b')).v).length).toBe(6)
  })

  for (const score of ['dot', 'general', 'concat'] as LuongScore[])
    it(`Luong ${score} attention matches torch, with gradients`, () => {
      const c = (F.luong as Record<string, Record<string, Nested> & { params: Tree }>)[score]
      const qDim = score === 'dot' ? 4 : 3
      const att = LuongAttention(qDim, 4, score, 6)
      const p = toTensors(c.params) as LuongParams
      const r = att.attend(p, T(c.s), T(c.h))
      close(r.context, c.context, 1e-12)
      close(r.weights, c.weights, 1e-12)
      const f = (s: Value, h: Value) => weighted(c.r)(att.attend(p, s, h).context)
      close(grad(f, { argnums: 0 })(T(c.s), T(c.h)), c.gs, 1e-11)
      close(grad(f, { argnums: 1 })(T(c.s), T(c.h)), c.gh, 1e-11)
    })
})

describe('backpropagation through time', () => {
  const B = F.bptt as Record<string, Nested>
  const cell = GruCell(2, 3)
  const params: CellParams = {
    inputWeight: T(B.inputWeight),
    hiddenWeight: T(B.hiddenWeight),
    bias: T(B.bias),
    hiddenBias: T(B.hiddenBias),
  }
  // w · h_12, as a weight tensor over every step that is zero except at the last.
  const lastWeights = tensor(Array.from({ length: 12 }, (_, t) => (t === 11 ? flat(B.w) : [0, 0, 0])))
  const lossOf = (hs: Value) => sum(mul(hs, lastWeights))

  it('records ∂L/∂h_t at every step as torch’s retained gradients, and the parameter gradient', () => {
    const loss = (hs: Value) => add(lossOf(hs), mul(0.1, sum(mul(hs, hs))))
    const r = gradientsThroughTime(cell, params, T(B.x), loss)
    expect(r.loss).toBeCloseTo(B.loss as number, 12)
    close(r.hiddenGrads, B.hiddenGrads, 1e-11)
    close(r.paramGrads.inputWeight, B.gInputWeight, 1e-11)
    expect(r.norms).toHaveLength(12)
  })

  it('truncation stops gradients from reaching earlier steps', () => {
    const r = gradientsThroughTime(cell, params, T(B.x), lossOf, { truncate: 4 })
    // The loss reads only h_12; with truncation every 4 steps, h_1 … h_8 get no gradient.
    expect(r.norms.slice(0, 8).every((n) => n === 0)).toBe(true)
    expect(r.norms[11]).toBeGreaterThan(0)
  })
})

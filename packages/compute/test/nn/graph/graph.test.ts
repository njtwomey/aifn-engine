/**
 * Graph neural network layers against plain-torch transcriptions of the papers' formulas (fixed weights, float64):
 * outputs and gradients of sum(r ⊙ y) with respect to the features and a weight, attention weights per edge, and the
 * layers' laws (attention sums to one per node, neighbour samples, layer wrappers).
 */
import { describe, expect, it } from 'vitest'
import { grad, gradCheck } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { mul, sum, tanh, tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { fromEdges } from 'aifn-compute/graph'
import {
  GraphAttention,
  graphAttention,
  GraphConv,
  graphConv,
  MessagePassing,
  messagePassing,
  SageConv,
  sageConv,
  sampleNeighbours,
  type SageParams,
} from 'aifn-compute/nn/graph'
import { Linear, linear } from 'aifn-compute/nn/layers'
import { fixture } from '../../fixtures'

type M = number[][]
type Out = { y: M; r: M; gradH: M; gradW: M }
type Gat = Out & { weight: M; attSource: M; attTarget: M; bias: number[]; attention: M }
type Sage = Out & {
  params: { self: { weight: M; bias: number[] }; neighbour: { weight: M }; pool?: { weight: M; bias: number[] } }
}
const F = fixture('nn/graph') as {
  nodes: number
  edges: [number, number, number][]
  h: M
  gcn: { weight: M; bias: number[] } & Record<'symmetric' | 'random-walk' | 'none', Out>
  gat: Gat
  gatMean: Gat
  gatv2: Gat
  sage: Record<'mean' | 'sum' | 'max' | 'pool' | 'meanNormalised', Sage>
  mpnn: Out & { message: { weight: M; bias: number[] }; update: { weight: M; bias: number[] } }
}

const g = fromEdges(F.nodes, F.edges, { directed: false })
const T = (a: M | number[]) => tensor(a)
const close = (got: Value, want: M, digits = 10) =>
  (toRows(got as Tensor) as M).forEach((r, i) => r.forEach((v, j) => expect(v).toBeCloseTo(want[i][j], digits)))
const weighted = (r: M) => (y: Value) => sum(mul(y, T(r)))

describe('graph convolution against torch', () => {
  for (const normalisation of ['symmetric', 'random-walk', 'none'] as const) {
    it(`${normalisation}: output and gradients`, () => {
      const c = F.gcn[normalisation]
      const f = (h: Value, w: Value) => graphConv(g, h, w, T(F.gcn.bias), { normalisation })
      close(f(T(F.h), T(F.gcn.weight)), c.y)
      close(grad((h: Value) => weighted(c.r)(f(h, T(F.gcn.weight))))(T(F.h)), c.gradH)
      close(grad((w: Value) => weighted(c.r)(f(T(F.h), w)))(T(F.gcn.weight)), c.gradW)
    })
  }
})

describe('graph attention against torch', () => {
  const cases = [
    ['GAT, 3 heads concatenated', F.gat, { heads: 3, concat: true }],
    ['GAT, 2 heads averaged', F.gatMean, { heads: 2, concat: false }],
    ['GATv2, 2 heads', F.gatv2, { heads: 2, concat: true, variant: 'v2' as const }],
  ] as const
  for (const [name, c, options] of cases) {
    it(`${name}: output, gradients and attention weights`, () => {
      const params = (w: Value) =>
        options.heads && 'variant' in options
          ? { weight: w as Tensor, att: T(c.attSource), bias: T(c.bias) }
          : { weight: w as Tensor, attSource: T(c.attSource), attTarget: T(c.attTarget), bias: T(c.bias) }
      const r = graphAttention(g, T(F.h), params(T(c.weight)), options)
      close(r.output, c.y)
      close(r.attention, c.attention, 12)
      close(
        grad((h: Value) => weighted(c.r)(graphAttention(g, h, params(T(c.weight)), options).output))(T(F.h)),
        c.gradH,
      )
      close(
        grad((w: Value) => weighted(c.r)(graphAttention(g, T(F.h), params(w), options).output))(T(c.weight)),
        c.gradW,
      )
      // Each node's incoming attention sums to one, per head.
      const A = toRows(r.attention as Tensor) as M
      const totals = new Map<number, number[]>()
      r.destination.forEach((v, k) =>
        totals.set(
          v,
          (totals.get(v) ?? A[k].map(() => 0)).map((s, h) => s + A[k][h]),
        ),
      )
      for (const t of totals.values()) for (const s of t) expect(s).toBeCloseTo(1, 12)
    })
  }
})

describe('GraphSAGE against torch', () => {
  const kinds = [
    ['mean', { aggregate: 'mean' }],
    ['sum', { aggregate: 'sum' }],
    ['max', { aggregate: 'max' }],
    ['pool', { aggregate: 'pool' }],
    ['meanNormalised', { aggregate: 'mean', normalise: true }],
  ] as const
  for (const [name, options] of kinds) {
    it(`${name}: output and gradients (the isolated node aggregates zeros)`, () => {
      const c = F.sage[name]
      const params = (w: Value): SageParams => ({
        self: { weight: T(c.params.self.weight), bias: T(c.params.self.bias) },
        neighbour: { weight: w as Tensor },
        ...(c.params.pool ? { pool: { weight: T(c.params.pool.weight), bias: T(c.params.pool.bias) } } : {}),
      })
      close(sageConv(g, T(F.h), params(T(c.params.neighbour.weight)), options), c.y)
      close(
        grad((h: Value) => weighted(c.r)(sageConv(g, h, params(T(c.params.neighbour.weight)), options)))(T(F.h)),
        c.gradH,
      )
      close(
        grad((w: Value) => weighted(c.r)(sageConv(g, T(F.h), params(w), options)))(T(c.params.neighbour.weight)),
        c.gradW,
      )
    })
  }
  it('samples at most S incoming neighbours per node, reproducibly', () => {
    const s = sampleNeighbours(g, 1, stream(3))
    const incoming = new Array(F.nodes).fill(0)
    for (const e of s.edges) incoming[e.to]++
    expect(Math.max(...incoming)).toBe(1)
    expect(incoming[7]).toBe(0)
    expect(sampleNeighbours(g, 1, stream(3)).edges).toEqual(s.edges)
    expect(sampleNeighbours(g, 10, stream(3)).edges.length).toBe(2 * F.edges.length)
  })
})

describe('message passing against torch', () => {
  it('a tanh message layer on [h_u, h_v, w] summed, then a linear update', () => {
    const c = F.mpnn
    const f = (h: Value, wm: Value) =>
      messagePassing(
        g,
        h,
        (x) => tanh(linear(x, wm, T(c.message.bias))),
        (x) => linear(x, T(c.update.weight), T(c.update.bias)),
      )
    close(f(T(F.h), T(c.message.weight)), c.y)
    close(grad((h: Value) => weighted(c.r)(f(h, T(c.message.weight))))(T(F.h)), c.gradH)
    close(grad((w: Value) => weighted(c.r)(f(T(F.h), w)))(T(c.message.weight)), c.gradW)
  })
})

describe('layers', () => {
  it('wrap the functional forms, and their gradients pass a finite-difference check', () => {
    const s = stream(1)
    const h = T(F.h)
    for (const layer of [
      GraphConv(g, 3, 4),
      GraphAttention(g, 3, 2, { heads: 2 }),
      GraphAttention(g, 3, 2, { heads: 2, variant: 'v2', concat: false }),
      SageConv(g, 3, 4, { aggregate: 'pool' }),
      MessagePassing(g, Linear(7, 5), Linear(8, 4), { aggregate: 'mean' }),
    ]) {
      const p = layer.init(s)
      const y = layer.apply(p as never, h) as Tensor
      expect(y.shape[0]).toBe(F.nodes)
      expect(toFlat(y).every(Number.isFinite)).toBe(true)
      const report = gradCheck((q: typeof p) => sum(tanh(layer.apply(q as never, h))), p, { rtol: 1e-4, atol: 1e-6 })
      expect(report.ok).toBe(true)
    }
    const gat = GraphAttention(g, 3, 2, { heads: 2 })
    const r = gat.forward(gat.init(s), h)
    expect((r.attention as Tensor).shape).toEqual([r.source.length, 2])
  })
})

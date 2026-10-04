/**
 * Type tests: representative implementations against their contracts in `aifn-compute/foundation/contracts`. The tests run
 * no numerics; `tsc -b` checks them (a contract an implementation stops satisfying fails the type check here).
 */

import { describe, expect, expectTypeOf, it } from 'vitest'
import type * as C from 'aifn-compute/foundation/contracts'
import { transferFunction } from 'aifn-compute/systems'
import { expBijector } from 'aifn-compute/probability/bijectors'
import { Normal } from 'aifn-compute/probability/distributions'
import { welch } from 'aifn-compute/signal/spectral'
import { fromEdges, treeFromParents } from 'aifn-compute/graph'
import { rbf } from 'aifn-compute/learning/kernels'
import { huber } from 'aifn-compute/learning/losses'
import { auroc } from 'aifn-compute/learning/metrics'
import { gradientDescent } from 'aifn-compute/optim/first-order'
import { normal, stream, type Key, type Stream } from 'aifn-compute/foundation/random'
import { softplus } from 'aifn-compute/numerics/special'
import {
  add,
  exp,
  sum,
  tensor,
  type Tensor,
  type Traced,
  type Value,
  type VectorLike,
} from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'

describe('numbers', () => {
  it('aifn-compute/foundation/tensor re-exports the contract types themselves', () => {
    expectTypeOf<Tensor>().toEqualTypeOf<C.Tensor>()
    expectTypeOf<Traced>().toEqualTypeOf<C.Traced>()
    expectTypeOf<Value>().toEqualTypeOf<C.Value>()
    expectTypeOf<VectorLike>().toEqualTypeOf<C.VectorLike>()
  })

  it('scalars and integer metadata are numbers', () => {
    expectTypeOf<C.Scalar>().toEqualTypeOf<number>()
    expectTypeOf<C.Shape>().toEqualTypeOf<readonly number[]>()
  })

  it('an object shaped like a tensor without the brand is not a tensor', () => {
    type Unbranded = Omit<C.Tensor, keyof C.Tensor & symbol>
    expectTypeOf<Unbranded>().not.toExtend<C.Tensor>()
  })

  it('dtypes include bool and complex128', () => {
    expectTypeOf<'bool'>().toExtend<C.DType>()
    expectTypeOf<'complex128'>().toExtend<C.DType>()
  })
})

describe('function families', () => {
  it('elementwise primitives are Unary or Binary', () => {
    expectTypeOf(exp).toExtend<C.Unary>()
    expectTypeOf(softplus).toExtend<C.Unary>()
    expectTypeOf(add).toExtend<C.Binary>()
  })

  it('reductions are Reduction', () => {
    expectTypeOf(sum).toExtend<C.Reduction>()
  })

  it('samplers are Sampler', () => {
    expectTypeOf(normal).toExtend<C.Sampler<[C.Raw, C.Raw]>>()
  })

  it('a kernel evaluates as a KernelFn', () => {
    expectTypeOf(rbf().evaluate).toExtend<C.KernelFn>()
  })

  it('metrics and losses are MetricFn and LossFn', () => {
    expectTypeOf(auroc).toExtend<C.MetricFn>()
    expectTypeOf(huber).toExtend<C.LossFn>()
  })
})

describe('protocols', () => {
  it('distributions carry the kind brand and sample from a plain-data stream', () => {
    expectTypeOf(Normal(0, 1)).toExtend<C.Distribution>()
    expect(Normal(0, 1).kind).toBe('distribution')
    expectTypeOf(expBijector).toExtend<C.Bijector>()
  })

  it('a stream is plain data: a key and a position', () => {
    expectTypeOf<Stream>().toEqualTypeOf<C.Stream>()
    expectTypeOf<Key>().toEqualTypeOf<C.Key>()
    const s = stream(1)
    expect(Object.keys(s).sort()).toEqual(['key', 'position'])
    expect(structuredClone(s)).toEqual(s)
  })

  it('algorithms take a StepContext and their states carry a Status; traces are branded', () => {
    expectTypeOf<Algorithm<number, C.Status>>().toEqualTypeOf<C.Algorithm<number, C.Status>>()
    expectTypeOf(gradientDescent).returns.toExtend<C.Algorithm<never, C.Status>>()
    expectTypeOf<Trace<C.Status>>().toEqualTypeOf<C.Trace<C.Status>>()
    const counter: Algorithm<undefined, { t: number }> = {
      name: 'c',
      init: () => ({ t: 0 }),
      step: (s) => ({ t: s.t + 1 }),
    }
    expect(trace(counter, undefined, 2).kind).toBe('trace')
  })

  it('kernels, graphs and trees carry their kind', () => {
    expectTypeOf(rbf()).toExtend<C.Kernel>()
    expectTypeOf(fromEdges(2, [[0, 1]])).toExtend<C.Graph>()
    expectTypeOf(treeFromParents([-1, 0])).toExtend<C.Tree>()
    expect([rbf().kind, fromEdges(2, [[0, 1]]).kind, treeFromParents([-1, 0]).kind]).toEqual([
      'kernel',
      'graph',
      'tree',
    ])
  })

  it('metric and loss metadata are registry Info', () => {
    expectTypeOf(auroc.info).toExtend<C.MetricInfo>()
    expectTypeOf(huber.info).toExtend<C.LossInfo>()
  })

  it('signal processing returns a Spectrum; systems are LtiSystems', () => {
    expectTypeOf(welch).returns.toExtend<C.Spectrum>()
    expectTypeOf(transferFunction).returns.toExtend<C.LtiSystem>()
    expect(welch(tensor(Array.from({ length: 64 }, (_, k) => Math.sin(k)))).kind).toBe('spectrum')
  })
})

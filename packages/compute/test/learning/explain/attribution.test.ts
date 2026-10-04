/**
 * The attributions added to `aifn-compute/learning/explain` against references (`fixtures/learning/explain.json`, `attribution`):
 * exact tree SHAP interaction values against shap's TreeExplainer (a tree and a forest); DeepLIFT (rescale),
 * DeepSHAP, integrated gradients and occlusion against Captum on exported torch networks. Laws: interaction values
 * against enumeration of the path-dependent value function, rows summing to Shapley values, DeepLIFT's
 * summation-to-delta, occlusion of single features against brute force, expected gradients' completeness in mean,
 * KernelSHAP's spread shrinking with more coalitions.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  deepLift,
  deepShap,
  denseFunction,
  denseOutput,
  ensembleTreeShapInteractions,
  exactShapley,
  expectedGradients,
  integratedGradients,
  interventionalValue,
  kernelShapVariance,
  occlusion,
  pathDependentValue,
  shapleyInteractions,
  treeShap,
  treeShapInteractions,
  type DenseNetwork,
  type ShapTree,
} from 'aifn-compute/learning/explain'
import { fixture } from '../../fixtures'

type Exported = {
  left: number[]
  right: number[]
  feature: number[]
  threshold: number[]
  weight: number[]
  value: number[]
}
type Net = { weights: number[][][]; biases: number[][]; activation: 'tanh' | 'relu' }
type F = {
  trees: {
    queries: number[][]
    tree: Exported & { interactions: number[][][] }
    forest: { trees: Exported[]; interactions: number[][][] }
  }
  deep: {
    x: number[][]
    baseline: number[]
    background: number[][]
    tanh: { net: Net; deepLift: number[][]; deepShap: number[][]; integratedGradients: number[][] }
    relu: { net: Net; deepLift: number[][]; deepShap: number[][]; integratedGradients: number[][] }
  }
  occlusion: {
    net: Net
    image: number[]
    cases: { window: number[]; strides: number[]; baseline: number; values: number[] }[]
  }
}
const F = fixture<{ attribution: F }>('learning/explain').attribution

const asTree = (t: Exported): ShapTree => ({
  root: 0,
  nodes: t.left.map((l, i) => ({
    feature: l === -1 ? -1 : t.feature[i],
    threshold: t.threshold[i],
    weight: t.weight[i],
    value: [t.value[i]],
    children: l === -1 ? [] : [l, t.right[i]],
  })),
})
const close = (got: ArrayLike<number>, want: readonly number[], tol: number) =>
  want.forEach((w, i) => expect(Math.abs(got[i] - w)).toBeLessThanOrEqual(tol * (1 + Math.abs(w))))

describe('tree SHAP interaction values', () => {
  const tree = asTree(F.trees.tree)
  it('match shap on a tree', () => {
    F.trees.queries.forEach((q, k) =>
      close(treeShapInteractions(tree, q).values, F.trees.tree.interactions[k].flat(), 1e-9),
    )
  })
  it('match shap on a forest', () => {
    const trees = F.trees.forest.trees.map(asTree)
    F.trees.queries.forEach((q, k) =>
      close(ensembleTreeShapInteractions(trees, q).values, F.trees.forest.interactions[k].flat(), 1e-9),
    )
  })
  it('equal enumeration of the path-dependent game, and rows sum to TreeSHAP', () => {
    const q = F.trees.queries[1]
    const exact = shapleyInteractions(pathDependentValue(tree, q), 5)
    const fast = treeShapInteractions(tree, q)
    close(fast.values, Array.from(exact.values), 1e-10)
    const shap = treeShap(tree, q).values
    for (let i = 0; i < 5; i++) {
      let row = 0
      for (let j = 0; j < 5; j++) row += fast.values[i * 5 + j]
      expect(row).toBeCloseTo(shap[i], 10)
    }
  })
})

describe('Shapley interaction values of a set function', () => {
  it('give a product its interaction and nothing on the diagonal', () => {
    // v(S) = [0 ∈ S][1 ∈ S]: φ₀ = φ₁ = 1/2, all of it interaction.
    const r = shapleyInteractions((m) => (m[0] && m[1] ? 1 : 0), 3)
    expect(r.values[1]).toBeCloseTo(0.5, 12)
    expect(r.values[0]).toBeCloseTo(0, 12)
    expect(r.values[2 * 3 + 2]).toBeCloseTo(0, 12)
    expect(Array.from(r.values).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })
  it('of an additive game are diagonal', () => {
    const w = [0.3, -1, 2, 0.5]
    const r = shapleyInteractions((m) => m.reduce((a, b, i) => a + (b ? w[i] : 0), 0), 4)
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) expect(r.values[i * 4 + j]).toBeCloseTo(i === j ? w[i] : 0, 12)
  })
})

describe('DeepLIFT and DeepSHAP', () => {
  for (const kind of ['tanh', 'relu'] as const) {
    const ref = F.deep[kind]
    const net = ref.net as DenseNetwork
    it(`match Captum (${kind})`, () => {
      F.deep.x.forEach((x, k) => {
        const r = deepLift(net, x, { baseline: F.deep.baseline, output: 1 })
        close(r.values, ref.deepLift[k], 1e-9)
        expect(Math.abs(r.delta)).toBeLessThan(1e-10)
        close(deepShap(net, x, F.deep.background, { output: 1 }).values, ref.deepShap[k], 1e-9)
      })
    })
    it(`integrated gradients match Captum's midpoint rule (${kind})`, () => {
      const f = denseFunction(net, { output: 1 })
      F.deep.x.forEach((x, k) =>
        close(
          integratedGradients(f, x, { baseline: F.deep.baseline, steps: 20, rule: 'midpoint' }).values,
          ref.integratedGradients[k],
          1e-7,
        ),
      )
    })
  }
  it('equals gradient × input for a linear network', () => {
    const net: DenseNetwork = { weights: [[[2], [-1], [0.5]]], biases: [[0.3]], activation: 'tanh' }
    const r = deepLift(net, [1, 2, 3])
    close(r.values, [2, -2, 1.5], 1e-12)
  })
})

describe('occlusion', () => {
  const net = F.occlusion.net as DenseNetwork
  const model = (X: Tensor) => denseOutput(net, X)
  it('matches Captum with 2-D windows and strides', () => {
    for (const c of F.occlusion.cases)
      close(
        occlusion(model, F.occlusion.image, {
          shape: [5, 6],
          window: c.window,
          strides: c.strides,
          baseline: c.baseline,
        }).values,
        c.values,
        1e-8,
      )
  })
  it('with one-feature windows is the drop from each feature alone', () => {
    const x = F.occlusion.image
    const r = occlusion(model, x)
    const f0 = denseOutput(net, fromData(Float64Array.from(x), [1, 30]))[0]
    for (const i of [0, 7, 29]) {
      const z = Float64Array.from(x)
      z[i] = 0
      expect(r.values[i]).toBeCloseTo(f0 - denseOutput(net, fromData(z, [1, 30]))[0], 12)
    }
  })
})

describe('expected gradients and KernelSHAP spread', () => {
  const net = F.deep.tanh.net as DenseNetwork
  const f = denseFunction(net, { output: 1 })
  it('expected gradients sum to f(x) − mean f(background), in the mean', () => {
    const x = F.deep.x[0]
    const bg = F.deep.background
    const r = expectedGradients(f, x, bg, stream('eg'), { samples: 4000 })
    const fx = denseOutput(net, fromData(Float64Array.from(x), [1, 4]), 1)[0]
    const fb = denseOutput(net, fromData(Float64Array.from(bg.flat()), [5, 4]), 1)
    const target = fx - fb.reduce((a, b) => a + b, 0) / 5
    expect(Math.abs(r.values.reduce((a, b) => a + b, 0) - target)).toBeLessThan(0.05 * (1 + Math.abs(target)))
  })
  it('KernelSHAP spreads less with more coalitions, around the exact values', () => {
    const model = (X: Tensor) => denseOutput(net, X, 1)
    const x = F.deep.x[1]
    const bg = F.deep.background
    const exact = exactShapley(interventionalValue(model, x, bg), 4).values
    const small = kernelShapVariance(model, x, bg, stream('ks'), { samples: 4, repeats: 30 })
    const large = kernelShapVariance(model, x, bg, stream('ks'), { samples: 12, repeats: 30 })
    const total = (v: Float64Array) => v.reduce((a, b) => a + b, 0)
    expect(total(large.sd)).toBeLessThan(total(small.sd))
    expect(toFlat(large.estimates).length).toBe(120)
    for (let i = 0; i < 4; i++) expect(Math.abs(large.mean[i] - exact[i])).toBeLessThan(0.05 + 3 * large.sd[i])
  })
})

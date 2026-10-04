/** The shared scikit-learn fixture of the learning area (`fixtures/learning.json`) and its comparison helpers. */
import { expect } from 'vitest'
import type { Distribution } from 'aifn-compute/foundation/contracts'
import { isTensor, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { classProbabilities } from 'aifn-compute/learning/estimators'
import type { DecisionTree } from 'aifn-methods/learning/trees-and-ensembles'
import { fixture } from '../fixtures'

export type TreeFx = {
  feature: number[]
  threshold: number[]
  left: number[]
  right: number[]
  impurity: number[]
  count: number[]
  importances: number[]
}
export type Fx = {
  x3: number[][]
  y3: number[]
  xq: number[][]
  knn: { uniform: number[][]; distance: number[][] }
  knn_manhattan: number[][]
  gnb: number[][]
  counts: number[][]
  ycounts: number[]
  mnb: number[][]
  bnb: number[][]
  lda: number[][]
  lda_ratio: number[]
  qda: number[][]
  tree: TreeFx
  tree_proba: number[][]
  tree_entropy: TreeFx
  path: { alphas: number[]; impurities: number[] }
  pruned: TreeFx
  yreg: number[]
  rtree: TreeFx
  rtree_predict: number[]
  svc: { decision: number[]; support: number[]; dual: number[] }
  svc_linear: { decision: number[]; coef: number[] }
  lsvc: { coef: number[]; intercept: number }
  cs: { coef: number[][]; intercept: number[]; decision: number[][] }
  perceptron: { coef: number[]; intercept: number }
  ada: { weights: number[]; errors: number[]; predict: number[] }
  gbr: { staged: number[][] }
  gbc: { decision: number[]; proba: number[][] }
  gbm: { proba: number[][] }
  ovr: { decision: number[][]; proba: number[][] }
  ovo: { decision: number[][] }
}
export const fx = fixture<Fx>('learning')
export const X3 = tensor(fx.x3)
export const Y3 = tensor(fx.y3)
export const XQ = tensor(fx.xq)
export const X2 = tensor(fx.x3.slice(0, 40))
export const Y2 = tensor(fx.y3.slice(0, 40))

export function close(a: Tensor | number[] | Distribution, b: number[] | number[][], tol = 1e-8) {
  const got = Array.isArray(a) ? a : isTensor(a) ? toFlat(a) : toFlat(classProbabilities(a))
  const want = (b as number[]).flat() as number[]
  expect(got.length).toBe(want.length)
  got.forEach((v, i) => expect(v).toBeCloseTo(want[i], -Math.log10(tol)))
}

export function sameTree(tree: DecisionTree, want: TreeFx) {
  expect(tree.nodes.length).toBe(want.feature.length)
  tree.nodes.forEach((node, i) => {
    const leaf = want.left[i] < 0
    expect(node.children.length === 0).toBe(leaf)
    if (!leaf) {
      expect(node.feature).toBe(want.feature[i])
      // scikit-learn stores thresholds in float32.
      expect(node.threshold).toBeCloseTo(want.threshold[i], 6)
      expect(node.children).toEqual([want.left[i], want.right[i]])
    }
    expect(node.impurity).toBeCloseTo(want.impurity[i], 10)
    expect(node.count).toBe(want.count[i])
  })
}

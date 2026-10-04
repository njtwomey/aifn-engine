/**
 * Explanations against references (`fixtures/learning/explain.json`): TreeSHAP against shap's TreeExplainer on a
 * scikit-learn tree and forest, and against exact Shapley values of the path-dependent value function by enumeration;
 * KernelSHAP (all coalitions) against shap's KernelExplainer and exact Shapley values, and its sampled estimate
 * converging; permutation importance and partial dependence against scikit-learn. Laws: efficiency (values sum to the
 * output minus the base), linear-model closed forms, integrated-gradient completeness, LIME recovering a linear model.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import {
  fromData,
  fromRows,
  mul,
  sin,
  square,
  sum,
  add,
  get,
  toFlat,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import {
  ensembleTreeShap,
  exactShapley,
  featureGrid,
  inputGradient,
  integratedGradients,
  interventionalValue,
  kernelShap,
  lime,
  partialDependence,
  pathDependentValue,
  permutationImportance,
  shapleyKernelWeight,
  smoothGrad,
  treeEnsembleOutput,
  treeShap,
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
type F = {
  queries: number[][]
  tree: Exported & { shap: number[][]; expected: number; predicted: number[] }
  forest: { trees: Exported[]; shap: number[][]; expected: number }
  kernel: { background: number[][]; point: number[]; shap: number[]; expected: number }
  permutation: {
    x: number[][]
    y: number[]
    permutations: number[][][]
    importances: number[][]
    mean: number[]
    std: number[]
  }
  partialDependence: { grid: number[]; average: number[]; individual: number[][] }
}
const F = fixture<F>('learning/explain')

/** A scikit-learn tree as an aifn `ShapTree` (−1 children at leaves). */
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

/** The fixture's function x₀x₁ + sin x₂ + ½x₃², on rows [m, 4]. */
const fixed = (X: Tensor) => {
  const v = toFlat(X)
  const m = X.shape[0]
  return fromData(
    Float64Array.from(
      { length: m },
      (_, i) => v[i * 4] * v[i * 4 + 1] + Math.sin(v[i * 4 + 2]) + 0.5 * v[i * 4 + 3] ** 2,
    ),
    [m],
  )
}

describe('treeShap', () => {
  const tree = asTree(F.tree)
  F.queries.forEach((q, k) =>
    it(`matches shap's TreeExplainer on query ${k}`, () => {
      const r = treeShap(tree, q)
      close(r.values, F.tree.shap[k], 1e-9)
      expect(r.base).toBeCloseTo(F.tree.expected, 9)
      expect(r.output).toBeCloseTo(F.tree.predicted[k], 9)
      expect(r.base + r.values.reduce((a, b) => a + b, 0)).toBeCloseTo(r.output, 9)
    }),
  )

  it('equals exact Shapley values of the path-dependent value function', () => {
    for (const q of F.queries.slice(0, 3)) {
      const exact = exactShapley(pathDependentValue(tree, q), 5)
      close(treeShap(tree, q).values, Array.from(exact.values), 1e-10)
    }
  })

  it("matches shap's TreeExplainer on a forest (the mean of the trees)", () => {
    const trees = F.forest.trees.map(asTree)
    F.queries.forEach((q, k) => {
      const r = ensembleTreeShap(trees, q)
      close(r.values, F.forest.shap[k], 1e-9)
      expect(r.base).toBeCloseTo(F.forest.expected, 9)
      expect(treeEnsembleOutput(trees, [q])[0]).toBeCloseTo(r.output, 12)
    })
  })
})

describe('Shapley values', () => {
  it('weights the kernel as (d − 1)/(C(d, s) s (d − s))', () => {
    expect(shapleyKernelWeight(4, 1)).toBeCloseTo(3 / (4 * 3), 12)
    expect(shapleyKernelWeight(4, 2)).toBeCloseTo(3 / (6 * 4), 12)
    expect(shapleyKernelWeight(4, 0)).toBe(Infinity)
  })

  it('exact enumeration gives the glove game its textbook values', () => {
    // Players 0 and 1 hold left gloves, 2 a right glove; a pair is worth 1.
    const r = exactShapley((m) => Math.min((m[0] ? 1 : 0) + (m[1] ? 1 : 0), m[2] ? 1 : 0), 3)
    close(r.values, [1 / 6, 1 / 6, 2 / 3], 1e-12)
  })

  it("KernelSHAP over every coalition matches shap's KernelExplainer and exact Shapley values", () => {
    const { background, point } = F.kernel
    const r = kernelShap(fixed, point, background, { samples: 14 })
    expect(r.exact).toBe(true)
    close(r.values, F.kernel.shap, 1e-9)
    expect(r.base).toBeCloseTo(F.kernel.expected, 9)
    const exact = exactShapley(interventionalValue(fixed, point, background), 4)
    close(r.values, Array.from(exact.values), 1e-10)
  })

  it('sampled KernelSHAP converges to the exact values and keeps efficiency', () => {
    const d = 12
    const w = [1, -2, 0.5, 3, 0, 1.5, -1, 2, 0.3, -0.7, 1, 0]
    const model = (X: Tensor) => {
      const v = toFlat(X)
      const m = X.shape[0]
      return fromData(
        Float64Array.from({ length: m }, (_, i) => {
          let s =
            v[i * d] * v[i * d + 1] * v[i * d + 2] + Math.max(v[i * d + 3], v[i * d + 4], v[i * d + 5], v[i * d + 8])
          for (let j = 0; j < d; j++) s += w[j] * v[i * d + j]
          return s
        }),
        [m],
      )
    }
    const x = [1, 2, -1, 0.5, 3, -2, 1, 0, 2, 1, -1, 4]
    const background = [new Array(d).fill(0), new Array(d).fill(1)]
    const exact = exactShapley(interventionalValue(model, x, background), d).values
    const error = (n: number) => {
      const r = kernelShap(model, x, background, { samples: n, stream: stream(n) })
      expect(r.exact).toBe(false)
      expect(r.base + r.values.reduce((a, b) => a + b, 0)).toBeCloseTo(r.output, 9)
      return Math.max(...r.values.map((v, i) => Math.abs(v - exact[i])))
    }
    expect(error(3000)).toBeLessThan(error(60))
    expect(error(3000)).toBeLessThan(0.2)
  })

  it('gives a linear model w_i (x_i − mean background_i)', () => {
    const w = [2, -1, 0.5]
    const model = (X: Tensor) => {
      const v = toFlat(X)
      return fromData(
        Float64Array.from(
          { length: X.shape[0] },
          (_, i) => w[0] * v[i * 3] + w[1] * v[i * 3 + 1] + w[2] * v[i * 3 + 2],
        ),
        [X.shape[0]],
      )
    }
    const bg = [
      [0, 1, 2],
      [2, 3, 0],
    ]
    const r = kernelShap(model, [1, 1, 1], bg)
    close(r.values, [2 * (1 - 1), -1 * (1 - 2), 0.5 * (1 - 1)], 1e-12)
  })
})

describe('permutationImportance', () => {
  it('matches scikit-learn with the same shuffles', () => {
    const P = F.permutation
    const r2 = (y: Float64Array, p: Float64Array) => {
      const mean = y.reduce((a, b) => a + b, 0) / y.length
      let ss = 0
      let st = 0
      for (let i = 0; i < y.length; i++) {
        ss += (y[i] - p[i]) ** 2
        st += (y[i] - mean) ** 2
      }
      return 1 - ss / st
    }
    const r = permutationImportance(fixed, P.x, P.y, r2, { permutations: P.permutations })
    r.importances.forEach((v, j) => close(v, P.importances[j], 1e-10))
    close(r.mean, P.mean, 1e-10)
    close(r.std, P.std, 1e-10)
  })
})

describe('partialDependence', () => {
  it('matches scikit-learn (brute force, custom grid)', () => {
    const P = F.partialDependence
    const r = partialDependence(fixed, F.permutation.x, 2, { grid: P.grid })
    close(r.average, P.average, 1e-10)
    close(toFlat(r.individual), P.individual.flat(), 1e-10)
  })

  it('uses distinct values when there are few, else a percentile grid', () => {
    expect(Array.from(featureGrid([[1], [0], [1], [2]], 0))).toEqual([0, 1, 2])
    const rows = Array.from({ length: 101 }, (_, i) => [i])
    const g = featureGrid(rows, 0, 10)
    expect(g[0]).toBeCloseTo(5, 12)
    expect(g[9]).toBeCloseTo(95, 12)
  })
})

describe('gradient attributions', () => {
  // f(x) = x₀² + x₀x₁ + sin x₂.
  const f = (x: Tensor) => add(add(square(get(x, 0)), mul(get(x, 0), get(x, 1))), sin(get(x, 2)))
  const x = [1, 2, 0.5]

  it('gives the analytic gradient', () => {
    const g = inputGradient(f, x)
    close(g.gradient, [2 * 1 + 2, 1, Math.cos(0.5)], 1e-12)
    close(g.timesInput, [4, 2, 0.5 * Math.cos(0.5)], 1e-12)
  })

  it('integrated gradients are complete and match the closed form from 0', () => {
    const r = integratedGradients(f, x, { steps: 200 })
    expect(Math.abs(r.delta)).toBeLessThan(1e-5)
    // From 0 along αx: ∫ (2αx₀ + αx₁) x₀ dα = x₀² + x₀x₁/2, ∫ αx₀ x₁ dα = x₀x₁/2, ∫ cos(αx₂) x₂ dα = sin x₂.
    close(r.values, [1 + 1, 1, Math.sin(0.5)], 1e-5)
    const mid = integratedGradients(f, x, { steps: 50, rule: 'midpoint' })
    expect(Math.abs(mid.delta)).toBeLessThan(1e-4)
  })

  it('SmoothGrad of a quadratic equals its gradient on average', () => {
    const q = (v: Tensor) => sum(square(v))
    const r = smoothGrad(q, [1, -1], stream(5), { samples: 4000, noise: 0.3 })
    close(r.values, [2, -2], 0.02)
  })
})

describe('lime', () => {
  it('recovers a linear model’s coefficients per standard deviation, with R² = 1', () => {
    const w = [3, -1, 0.5]
    const model = (X: Tensor) => {
      const v = toFlat(X)
      return fromData(
        Float64Array.from(
          { length: X.shape[0] },
          (_, i) => w[0] * v[i * 3] + w[1] * v[i * 3 + 1] + w[2] * v[i * 3 + 2],
        ),
        [X.shape[0]],
      )
    }
    const scale = [2, 1, 0.5]
    const r = lime(model, [1, 1, 1], stream(1), { samples: 2000, scale, ridge: 1e-9 })
    close(r.coefficients, [6, -1, 0.25], 1e-6)
    expect(r.score).toBeCloseTo(1, 9)
    expect(r.intercept).toBeCloseTo(r.output, 6)
    expect(fromRows([[0]]).shape).toEqual([1, 1])
  })
})

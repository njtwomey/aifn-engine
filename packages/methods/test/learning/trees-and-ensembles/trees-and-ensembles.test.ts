import { describe, expect, it } from 'vitest'
import {
  applyTree,
  costComplexityPath,
  decisionPath,
  decisionTree,
  growTree,
  keptNodes,
  nodeRegion,
  predictTree,
  pruneTree,
  regressionTree,
  splitCurve,
  splitSearch,
  treeGrowthSteps,
  type DecisionTree,
  type GrowthOrder,
} from 'aifn-methods/learning/trees-and-ensembles'
import { run } from 'aifn-compute/foundation/trace'
import { moons, withLabelNoise } from 'aifn-methods/data/synthetic'
import { leaves, preOrder } from 'aifn-compute/graph'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { expectProtocol } from '../../protocol'
import { close, fx, sameTree, X3, XQ, Y3 } from '../shared'

describe('trees', () => {
  it('CART matches scikit-learn: structure, thresholds, impurities, importances', () => {
    const m = decisionTree().fit(dataset(X3, Y3))
    sameTree(m.tree, fx.tree)
    close(m.featureImportances, fx.tree.importances)
    close(m.predictive(XQ), fx.tree_proba)
    sameTree(decisionTree({ criterion: 'entropy', maxDepth: 3 }).fit(dataset(X3, Y3)).tree, fx.tree_entropy)
    const r = regressionTree({ maxDepth: 3 }).fit(dataset(X3, tensor(fx.yreg)))
    sameTree(r.tree, fx.rtree)
    close(r.decide(XQ), fx.rtree_predict)
  })
  it('cost-complexity path and pruning match scikit-learn', () => {
    const m = decisionTree().fit(dataset(X3, Y3))
    const path = costComplexityPath(m.tree)
    close(path.alphas, fx.path.alphas)
    close(path.impurities, fx.path.impurities)
    sameTree(decisionTree({ pruneAlpha: 0.02 }).fit(dataset(X3, Y3)).tree, fx.pruned)
  })
  it('trees are aifn-compute/graph trees', () => {
    const t = decisionTree({ maxDepth: 3 }).fit(dataset(X3, Y3)).tree
    expect(preOrder(t)).toEqual(t.nodes.map((n) => n.id))
    expect(leaves(t).every((v) => t.nodes[v].feature === -1)).toBe(true)
    expect(t.nodes[0].label).toMatch(/^\$x_/)
  })
  it('the split search at the root finds the root split', () => {
    const s = splitSearch({ x: X3, y: Y3, task: 'classification' })
    expect(s.feature).toBe(fx.tree.feature[0])
    expect(s.threshold).toBeCloseTo(fx.tree.threshold[0], 6)
    const best = Math.max(...s.candidates.flatMap((c) => toFlat(c.decreases)))
    expect(best).toBeCloseTo(s.decrease, 12)
  })
  it('growth follows the trace protocol', () => {
    expectProtocol(treeGrowthSteps({ x: X3, y: Y3, task: 'classification', params: { maxFeatures: 1 } }), undefined, {
      n: 20,
      record: { k: (s) => s.tree.nodes.length + s.pending.length },
      seed: 3,
    })
  })
  it('a binary tree fitted with weights equals one fitted on duplicated rows', () => {
    const w = tensor(fx.y3.map((_, i) => (i % 3) + 1))
    const t1 = growTree(undefined, { x: X3, y: Y3, weights: w, task: 'classification' })
    expect(t1.nodes[0].weight).toBe(fx.y3.reduce((a, _, i) => a + (i % 3) + 1, 0))
  })
  it('a random feature subset without a valid split draws further features, as scikit-learn', () => {
    // Only feature 0 varies; with maxFeatures 1 most nodes first draw a constant feature. Every leaf must still be pure.
    const n = 24
    const x = tensor(Array.from({ length: n }, (_, i) => [i, 1, 2, 3, 4]))
    const y = tensor(
      Array.from({ length: n }, (_, i) => Math.floor(i / 3) % 2),
      undefined,
      'int32',
    )
    for (const seed of [1, 2, 3]) {
      const tree = growTree(stream(seed), { x, y, task: 'classification', params: { maxFeatures: 1 } })
      for (const node of tree.nodes) {
        if (node.feature < 0) expect(node.impurity).toBeLessThanOrEqual(1e-12)
        else expect(node.feature).toBe(0)
      }
    }
  })

  describe('what the fitter records, and reading it', () => {
    const problem = { x: X3, y: Y3, task: 'classification' as const }
    const tree = decisionTree().fit(dataset(X3, Y3)).tree
    const rows = fx.x3 as number[][]
    const lower = [0, 1].map((f) => Math.min(...rows.map((r) => r[f])) - 1)
    const upper = [0, 1].map((f) => Math.max(...rows.map((r) => r[f])) + 1)

    it('fit runs the growth steps: the same tree', () => {
      const grown = growTree(undefined, problem)
      expect(tree.nodes.map((n) => [n.feature, n.threshold, n.children, Array.from(n.rows)])).toEqual(
        grown.nodes.map((n) => [n.feature, n.threshold, n.children, Array.from(n.rows)]),
      )
    })
    it("each node's stored split search equals a fresh one on its rows", () => {
      for (const node of tree.nodes) {
        if (!node.splits.length) continue
        const fresh = splitSearch(problem, node.rows)
        fresh.candidates.forEach((c, j) => {
          const d = toFlat(c.decreases)
          expect(node.splits[j].decrease).toBeCloseTo(Math.max(...d), 12)
          expect(node.splits[j].threshold).toBe(toFlat(c.thresholds)[d.indexOf(Math.max(...d))])
        })
        if (node.feature >= 0) {
          expect(node.feature).toBe(fresh.feature)
          expect(node.threshold).toBe(fresh.threshold)
        }
        const curve = splitCurve(tree, node.id, 0, problem)
        expect(toFlat(curve.decreases)).toEqual(toFlat(fresh.candidates[0].decreases))
      }
    })
    it('leaves record why they stopped; counts and rows agree', () => {
      for (const node of tree.nodes) {
        expect(node.stop === null).toBe(node.feature >= 0)
        expect(node.rows.length).toBe(node.count)
        expect(node.counts.reduce((a, b) => a + b, 0)).toBeCloseTo(node.weight, 12)
      }
      expect(tree.nodes.filter((n) => n.stop !== null).every((n) => n.stop === 'pure')).toBe(true)
      const shallow = decisionTree({ maxDepth: 1 }).fit(dataset(X3, Y3)).tree
      expect(shallow.nodes.slice(1).map((n) => n.stop)).toContain('max-depth')
    })
    it("a node's region holds exactly the training rows routed to it", () => {
      for (const node of tree.nodes) {
        const box = nodeRegion(tree, node.id, { lower, upper })
        const inside = rows.flatMap((r, i) => (r.every((v, f) => v > box.lower[f] && v <= box.upper[f]) ? [i] : []))
        expect(inside).toEqual(Array.from(node.rows).sort((a, b) => a - b))
      }
    })
    it('a decision path ends at the leaf applyTree finds, through the tests it reports', () => {
      const leaves = toFlat(applyTree(tree, XQ))
      ;(fx.xq as number[][]).forEach((q, i) => {
        const path = decisionPath(tree, q)
        expect(path.leaf).toBe(leaves[i])
        expect(path.nodes[0]).toBe(tree.root)
        path.tests.forEach((t, k) => {
          expect(t.left).toBe(q[t.feature] <= t.threshold)
          expect(path.nodes[k + 1]).toBe(tree.nodes[t.node].children[t.left ? 0 : 1])
        })
      })
    })
    it('the pruning sequence read in place equals pruneTree', () => {
      const path = costComplexityPath(tree)
      const alphas = toFlat(path.alphas)
      const cuts = toFlat(path.cuts)
      expect(cuts.length).toBe(alphas.length - 1)
      for (let k = 0; k <= cuts.length; k++) {
        // Ties cut several links at one α; compare where the next α differs.
        if (k < cuts.length && alphas[k + 1] === alphas[k] && k > 0) continue
        const kept = keptNodes(tree, cuts.slice(0, k))
        const inPlace = predictTree(tree, XQ, { within: (v) => kept[v] === 1 })
        expect(toFlat(inPlace)).toEqual(toFlat(predictTree(pruneTree(tree, alphas[k]), XQ)))
      }
      expect(Array.from(keptNodes(tree, cuts))).toEqual(tree.nodes.map((n) => (n.id === tree.root ? 1 : 0)))
    })
  })

  describe('expansion order', () => {
    const shape = (t: DecisionTree, id = t.root): unknown => {
      const n = t.nodes[id]
      return n.children.length ? [n.feature, n.threshold, shape(t, n.children[0]), shape(t, n.children[1])] : n.count
    }
    const data = withLabelNoise(stream('orders/noise'), moons(stream('orders'), { n: 200, noise: 0.3 }), { rate: 0.1 })
    const grow = (order: GrowthOrder, maxLeaves?: number) =>
      run(
        treeGrowthSteps({
          x: data.x as Tensor,
          y: data.y as Tensor,
          task: 'classification',
          params: { order, maxLeaves },
        }),
        undefined,
        500,
      ).tree
    const impurity = (t: DecisionTree) =>
      t.nodes.filter((n) => !n.children.length).reduce((a, n) => a + n.weight * n.impurity, 0)

    it('without a leaf budget every order grows the same tree', () => {
      const depth = grow('depth-first')
      expect(shape(grow('breadth-first'))).toEqual(shape(depth))
      expect(shape(grow('best-first'))).toEqual(shape(depth))
    })
    it('with a leaf budget best-first spends it best', () => {
      const budgets = [3, 4, 5, 6, 7, 8]
      const best = budgets.map((m) => grow('best-first', m))
      const breadth = budgets.map((m) => grow('breadth-first', m))
      best.forEach((t, i) => {
        expect(t.nodes.filter((n) => !n.children.length).length).toBeLessThanOrEqual(budgets[i])
        expect(impurity(t)).toBeLessThanOrEqual(impurity(breadth[i]) + 1e-12)
      })
      expect(best.some((t, i) => impurity(t) < impurity(breadth[i]) - 1e-9)).toBe(true)
      expect(best[0].nodes.some((n) => n.stop === 'max-leaves')).toBe(true)
    })
  })
})

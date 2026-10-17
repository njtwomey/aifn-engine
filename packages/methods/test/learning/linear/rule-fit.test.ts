/**
 * RuleFit: the 8-row worked example (x = 1..8, y = 1,1,1,1,5,5,5,9, four stump rules), whose path starts at
 * alpha_max = max |z_k| = 1.25 with a support-1/2 rule entering first, against the one-rule lasso by hand
 * (a1 = -(1.25 - 0.7)/0.25 = -2.2 at alpha = 0.7); and rules from a boosted ensemble on Friedman #1: 2(t - 1) rules
 * per tree, no coefficient and the mean as intercept at alpha_max, importances |a| sqrt(s(1 - s)), and predictions
 * equal to the intercept at alpha_max.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { ensembleRules, ruleFeatures, type EnsembleRule } from 'aifn-compute/learning/rules'
import { friedman1 } from 'aifn-methods/data/synthetic'
import { linearFunctions, ruleFitPath, ruleFitPredict, ruleImportances } from 'aifn-methods/learning/linear'
import { gradientBoosting } from 'aifn-methods/learning/trees-and-ensembles'

/** RuleFit's importance of a rule, $|a| \sqrt{s(1 - s)}$. */
const ruleImportance = (a: number, s: number) => Math.abs(a) * Math.sqrt(s * (1 - s))

describe('ruleFitPath on the worked example', () => {
  const rows = [1, 2, 3, 4, 5, 6, 7, 8].map((v) => [v])
  const y = [1, 1, 1, 1, 5, 5, 5, 9]
  const R = (lower: number, upper: number): EnsembleRule => ({
    conditions: [{ feature: 0, lower, upper }],
    tree: 0,
    node: 0,
  })
  const rules = [R(-Infinity, 4.5), R(4.5, Infinity), R(-Infinity, 7.5), R(7.5, Infinity)]
  const fit = ruleFitPath(rows, y, rules, { linear: false, count: 50, span: 3, tolerance: 1e-10, maxSteps: 100000 })

  it('starts at alpha_max = max |z_k| = 1.25 with nothing in the model', () => {
    expect(fit.alphas[0]).toBeCloseTo(1.25, 9)
    expect(fit.nonzero[0]).toBe(0)
    expect(fit.intercept[0]).toBeCloseTo(3.5, 12)
    expect(fit.support).toEqual([0.5, 0.5, 0.875, 0.125])
  })

  it('lets a rule of support 1/2 enter first, not the support-1/8 rule', () => {
    const first = fit.coef.findIndex((c) => c.some((v) => v !== 0))
    const entered = fit.coef[first].map((v, k) => (v !== 0 ? k : -1)).filter((k) => k >= 0)
    expect(entered.length).toBeGreaterThanOrEqual(1)
    expect(entered.every((k) => k <= 1)).toBe(true)
  })

  it('agrees with the one-rule lasso by hand: a1 = -2.2 at alpha = 0.7, and a4 = 0', () => {
    const cols = ruleFeatures(rows, rules)
    const z = cols.columns.map((c, k) => c.reduce((a, r, i) => a + (r - cols.support[k]) * y[i], 0) / 8)
    const one = (k: number, alpha: number) => {
      const s = cols.support[k]
      return (Math.sign(z[k]) * Math.max(Math.abs(z[k]) - alpha, 0)) / (s * (1 - s))
    }
    expect(one(0, 0.7)).toBeCloseTo(-2.2, 12)
    expect(one(3, 0.7)).toBe(0)
    const single = ruleFitPath(rows, y, [rules[0]], { linear: false, count: 50, span: 3, tolerance: 1e-12 })
    single.alphas.forEach((alpha, t) => expect(single.coef[t][0]).toBeCloseTo(one(0, alpha), 8))
  })

  it('predicts the intercept at alpha_max and lists importances |a| sqrt(s(1 - s))', () => {
    expect(ruleFitPredict(fit, 0, rows).every((v) => Math.abs(v - fit.intercept[0]) < 1e-12)).toBe(true)
    const last = fit.alphas.length - 1
    for (const term of ruleImportances(fit, last)) {
      expect(term.kind).toBe('rule')
      if (term.kind === 'rule') expect(term.importance).toBeCloseTo(ruleImportance(term.coefficient, term.support), 12)
    }
  })

  it('is registered', () => {
    expect(linearFunctions.ruleFitPath.info.notes).toEqual(['rule-ensembles-and-rulefit'])
  })
})

describe('ruleFitPath on a boosted ensemble (Friedman #1)', () => {
  const tr = friedman1(stream('rulefit-train-1'), { n: 200, d: 10, noise: 1 })
  const model = gradientBoosting({ stages: 20, learningRate: 0.1, tree: { maxLeaves: 4 }, subsample: 0.5 }).fit(tr, {
    stream: stream('rulefit-boost-1'),
  })
  const trees = model.stages.map((s) => s[0])
  const xtr = toArray(tr.x) as number[][]
  const ytr = Array.from(toFlat(tr.y))

  it('reads 2(t - 1) rules off each tree', () => {
    const leaves = trees.reduce((a, t) => a + t.nodes.filter((n) => n.children.length === 0).length, 0)
    expect(ensembleRules(trees)).toHaveLength(2 * (leaves - trees.length))
  })

  it('starts empty at alpha_max, with the mean as intercept; importances follow the formula', () => {
    const f = ruleFitPath(xtr, ytr, ensembleRules(trees), { count: 12, span: 2.5, tolerance: 1e-5, maxSteps: 500 })
    expect(f.nonzero[0]).toBe(0)
    expect(f.intercept[0]).toBeCloseTo(ytr.reduce((a, b) => a + b) / ytr.length, 9)
    expect(ruleFitPredict(f, 0, xtr).every((v) => Math.abs(v - f.intercept[0]) < 1e-12)).toBe(true)
    const t = f.alphas.length - 1
    expect(f.nonzero[t]).toBeGreaterThan(0)
    const terms = ruleImportances(f, t)
    expect(terms).toHaveLength(f.nonzero[t])
    for (const term of terms) {
      if (term.kind === 'rule') expect(term.importance).toBeCloseTo(ruleImportance(term.coefficient, term.support), 12)
      else {
        const lin = f.linear!
        expect(term.importance).toBeCloseTo(Math.abs(term.coefficient) * lin.sd[term.feature], 12)
      }
    }
    for (let k = 1; k < terms.length; k++) expect(terms[k - 1].importance).toBeGreaterThanOrEqual(terms[k].importance)
  })
})

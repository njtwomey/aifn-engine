import { describe, expect, it } from 'vitest'
import {
  datasetRegistry,
  decodeRecipe,
  describeRecipe,
  encodeRecipe,
  modifierRegistry,
  normaliseRecipe,
  parseRecipe,
  recipe,
  recipeBases,
  recipeOps,
  recipeSpace,
  type Dataset,
  type RecipeInput,
} from 'aifn-methods/data'
import { clamp, defaults } from 'aifn-compute/foundation/space'
import { toFlat } from 'aifn-compute/foundation/tensor'

const counts = (d: Dataset, k = 2) => {
  const c = new Array<number>(k).fill(0)
  toFlat(d.y!).forEach((v) => c[v]++)
  return c
}

const r: RecipeInput = {
  base: 'moons',
  seed: 7,
  knobs: { n: 400, noise: 0.2, prevalence: 0.2 },
  modifiers: [
    { op: 'withOutliers', params: { fraction: 0.02 } },
    { op: 'withNuisanceFeatures', params: { count: 2 } },
    { op: 'withLabelNoise', params: { rate: 0.05 } },
    { op: 'withMissing', params: { rate: 0.1, mechanism: 'mnar' } },
  ],
}

describe('the recipe interpreter', () => {
  it('derives its bases, ops and space from the registries', () => {
    expect(recipeBases).toContain('moons')
    expect(recipeBases).toContain('meanShifts')
    expect(recipeBases).toContain('iris')
    // Generators that do not return a Dataset are not bases.
    expect(recipeBases).not.toContain('casino')
    expect(recipeBases).not.toContain('anscombe')
    for (const k of recipeBases) expect(datasetRegistry[k].info.output).toBe('dataset')
    expect(recipeOps).toEqual(Object.keys(modifierRegistry))
    const base = recipeSpace.dims.base
    expect(base.type).toBe('variants')
    if (base.type === 'variants') {
      expect(Object.keys(base.cases)).toEqual(recipeBases)
      expect(base.cases.moons).toBe(datasetRegistry.moons.info.knobs)
    }
    expect(defaults(recipeSpace).base).toEqual({ case: 'moons', params: defaults(datasetRegistry.moons.info.knobs) })
  })

  it('builds the base and applies the modifiers in order', () => {
    const d = recipe(r)
    expect(d.x.shape).toEqual([400, 4])
    expect(counts(recipe({ ...r, modifiers: [] }))).toEqual([320, 80])
    const made = d.meta.recipe!
    expect(made.base).toBe('moons')
    expect(made.seed).toBe(7)
    expect(made.modifiers.map((m) => m.op)).toEqual([
      'withOutliers',
      'withNuisanceFeatures',
      'withLabelNoise',
      'withMissing',
    ])
    expect(d.meta.ignored).toBeUndefined()
    expect(d.meta.outliers).toBeDefined()
    expect(d.meta.cleanLabels).toBeDefined()
    expect(d.meta.missing).toBeDefined()
  })

  it('records the normalised recipe, which rebuilds the same dataset', () => {
    const d = recipe(r)
    const made = d.meta.recipe!
    expect(made.knobs).toEqual(clamp(datasetRegistry.moons.info.knobs, r.knobs!))
    expect(made.modifiers[0].params).toEqual({ fraction: 0.02, scale: 4 })
    const again = recipe(made)
    expect(toFlat(again.x)).toEqual(toFlat(d.x))
    expect(toFlat(again.y!)).toEqual(toFlat(d.y!))
  })

  it('round-trips through a URL string, leaving default knobs out', () => {
    const text = encodeRecipe(r)
    expect(JSON.parse(decodeURIComponent(text)).knobs).toEqual({ n: 400, noise: 0.2, prevalence: 0.2 })
    const back = decodeRecipe(text)
    expect(back).toEqual(normaliseRecipe(r).recipe)
    expect(toFlat(recipe(back).x)).toEqual(toFlat(recipe(r).x))
  })

  it('is deterministic in its seed, and each step draws from its own substream', () => {
    expect(toFlat(recipe(r).x)).toEqual(toFlat(recipe(r).x))
    expect(toFlat(recipe({ ...r, seed: 8 }).x)).not.toEqual(toFlat(recipe(r).x))
    const base = toFlat(recipe({ base: 'moons', knobs: { n: 100 }, seed: 1 }).x)
    const noisy = recipe({ base: 'moons', knobs: { n: 100 }, seed: 1, modifiers: [{ op: 'withLabelNoise' }] })
    expect(toFlat(noisy.x)).toEqual(base)
  })

  it('reports the knobs and steps it could not use', () => {
    // An unknown knob, a knob inactive under its condition, an unknown modifier parameter.
    const d = recipe({
      base: 'xor',
      knobs: { n: 50, separation: 2, sd: 0.3 },
      modifiers: [{ op: 'withOutliers', params: { fraction: 0.1, typo: 1 } }],
    })
    expect(d.meta.ignored).toEqual(['separation', 'sd', 'withOutliers.typo'])
    // A modifier that needs class labels is skipped on a regression base.
    const reg = recipe({
      base: 'regression1d',
      knobs: { n: 20, prevalence: 0.3 },
      modifiers: [{ op: 'withLabelNoise', params: { rate: 0.1 } }, { op: 'withOutliers' }],
    })
    expect(reg.meta.ignored).toEqual(['prevalence', 'withLabelNoise'])
    expect(reg.meta.outliers).toBeDefined()
  })

  it('describes itself in one line', () => {
    expect(describeRecipe(r)).toBe(
      'Two moons (n 400, noise 0.2, prevalence 0.2), outliers (fraction 0.02, scale 4), nuisance features (count 2, kind gaussian), label noise (rate 0.05), missing values (rate 0.1, mechanism mnar, strength 2), seed 7',
    )
  })

  it('rejects what is not a recipe', () => {
    expect(() => parseRecipe({ base: 'nope' })).toThrow(/unknown base/)
    expect(() => parseRecipe({ base: 'casino' })).toThrow(/unknown base/)
    expect(() => parseRecipe({ base: 'moons', n: 3 })).toThrow(/unknown key/)
    expect(() => parseRecipe({ base: 'moons', modifiers: [{ op: 'withTypo' }] })).toThrow(/unknown modifier/)
    expect(() => recipe({ base: 'moons', modifiers: [{ op: 'withTypo' }] })).toThrow(/unknown modifier/)
    expect(parseRecipe({ base: 'moons', seed: 'a', knobs: {} })).toEqual({ base: 'moons', seed: 'a', knobs: {} })
  })
})

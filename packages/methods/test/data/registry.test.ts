import { describe, expect, it } from 'vitest'
import {
  datasetRegistry,
  generate,
  logDensityRegistry,
  modifierRegistry,
  modify,
  objectiveRegistry,
  type Dataset,
} from 'aifn-methods/data'
import { fontDatasetRegistry } from 'aifn-methods/data/real/fonts'
import { moons, regression1d } from 'aifn-methods/data/synthetic'
import { child, stream } from 'aifn-compute/foundation/random'
import { defaults } from 'aifn-compute/foundation/space'
import { isTensor, toFlat } from 'aifn-compute/foundation/tensor'
import { expectInfo } from '../registry'

/** Generators whose knob defaults differ from the generator's own on purpose: the recipe gives exact class sizes. */
const EXACT_SIZES = new Set(['xor', 'checkerboard'])

describe('dataset registry', () => {
  it('has well-formed metadata', () => {
    expectInfo(datasetRegistry, 'dataset')
    expectInfo(fontDatasetRegistry, 'dataset')
  })

  it('every generator runs at its default knobs and returns what it declares', () => {
    for (const entry of [...Object.values(datasetRegistry), ...Object.values(fontDatasetRegistry)]) {
      const { key, output, task, truth, knobs } = entry.info
      const made = generate(entry, stream(1), defaults(knobs))
      if (output === 'dataset') {
        const d = made as Dataset
        expect(d.kind, key).toBe('dataset')
        expect(isTensor(d.x), key).toBe(true)
        expect(d.meta.task, `${key}: task`).toBe(task)
        expect(d.meta.truth !== undefined, `${key}: truth`).toBe(truth)
      } else if (output === 'datasets') expect(Array.isArray(made), key).toBe(true)
      else if (output === 'image' || output === 'patterns') expect(isTensor(made), key).toBe(true)
      else if (output === 'scene') expect(isTensor((made as { image: unknown }).image), key).toBe(true)
      else expect(typeof made, key).toBe('object')
    }
  })

  it("the knobs' defaults are the generators' own", () => {
    for (const entry of Object.values(datasetRegistry)) {
      const { key, random, output, knobs } = entry.info
      if (!random || output !== 'dataset' || EXACT_SIZES.has(key)) continue
      // A stream is a cursor, so each call gets a fresh one.
      const a = generate(entry, child(stream(2), key), defaults(knobs)) as Dataset
      const b = generate(entry, child(stream(2), key), {}) as Dataset
      expect(toFlat(a.x), key).toEqual(toFlat(b.x))
    }
  })
})

describe('modifier registry', () => {
  it('has well-formed metadata', () => {
    expectInfo(modifierRegistry, 'modifier')
  })

  it('every modifier applies at its default parameters and records itself under its key', () => {
    const labelled = moons(stream(3), { n: 60, noise: 0.2 })
    const real = regression1d(stream(4), { n: 60 })
    for (const entry of Object.values(modifierRegistry)) {
      const { key, needs, params } = entry.info
      for (const d of needs === 'labels' ? [labelled] : [labelled, real]) {
        const out = modify(entry, stream(5), d, defaults(params))
        expect(out.kind, key).toBe('dataset')
        expect(out.meta.recipe!.modifiers.at(-1)!.op, `${key}: recorded op`).toBe(key)
      }
    }
  })
})

describe('objectives and log densities', () => {
  it('have well-formed metadata', () => {
    expectInfo(objectiveRegistry, 'objective')
    expectInfo(logDensityRegistry, 'log-density')
  })

  it('test objectives have the declared dimension and minimisers', () => {
    for (const entry of Object.values(objectiveRegistry)) {
      const { key, dim, truth, params } = entry.info
      const f = (entry as unknown as (p: object) => { dimension: number; minima: unknown[] })(defaults(params))
      if (dim !== null) expect(f.dimension, key).toBe(dim)
      expect(f.minima.length > 0, key).toBe(truth)
    }
  })

  it('log densities without required arguments build at their defaults', () => {
    for (const entry of Object.values(logDensityRegistry)) {
      if (entry.info.key === 'gaussianTarget' || entry.info.key === 'gaussianMixtureTarget') continue
      const target = (entry as unknown as (p: object) => { kind: string })(defaults(entry.info.params))
      expect(target.kind, entry.info.key).toBe('log-density')
    }
  })
})

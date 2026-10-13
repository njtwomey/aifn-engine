import { expect, it } from 'vitest'
import { isEntry } from 'aifn-compute/foundation/registry'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { svmTuningSurface, tuningTables } from 'aifn-methods/data/objectives'

const t = tuningTables()

it('has seven tasks on the 21 by 21 grid of svmTuningSurface, iris first and identical to it', () => {
  expect(t.tasks).toEqual(['iris', 'moons', 'circles', 'xor', 'gaussians', 'spirals', 'diabetes'])
  expect(t.descriptions).toHaveLength(7)
  expect(t.error.shape).toEqual([7, 21, 21])
  expect(t.spread.shape).toEqual([7, 21, 21])
  const f = svmTuningSurface()
  expect(toFlat(t.logC)).toEqual(toFlat(f.logC))
  expect(toFlat(t.logGamma)).toEqual(toFlat(f.logGamma))
  const error = toArray(t.error) as number[][][]
  const spread = toArray(t.spread) as number[][][]
  expect(error[0]).toEqual(toArray(f.error))
  expect(spread[0]).toEqual(toArray(f.spread))
})

it('holds error rates, and its configuration by task table lays out the same values', () => {
  const error = toArray(t.error) as number[][][]
  for (const v of toFlat(t.error)) expect(v >= 0 && v <= 1).toBe(true)
  for (const v of toFlat(t.spread)) expect(v >= 0).toBe(true)
  expect(t.table.shape).toEqual([441, 7])
  expect(t.configurations.shape).toEqual([441, 2])
  const rows = toArray(t.table) as number[][]
  const configurations = toArray(t.configurations) as number[][]
  const [logC, logGamma] = [toFlat(t.logC), toFlat(t.logGamma)]
  for (const [i, j] of [
    [0, 0],
    [3, 17],
    [20, 20],
    [11, 5],
  ]) {
    const k = i * 21 + j
    expect(configurations[k]).toEqual([logC[i], logGamma[j]])
    expect(rows[k]).toEqual(error.map((e) => e[i][j]))
  }
  // The tasks disagree on the best configuration.
  const best = error.map((e) => {
    const f = e.flat()
    return f.indexOf(Math.min(...f))
  })
  expect(new Set(best).size).toBeGreaterThan(3)
})

it('is a registered function', () => {
  expect(isEntry(tuningTables, 'function')).toBe(true)
  expect((tuningTables as unknown as { info: { module: string } }).info.module).toBe('data/objectives')
})

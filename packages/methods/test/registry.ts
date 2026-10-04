/**
 * Registry checks shared by the area tests: every entry's key is unique and names the exported value; and the model
 * protocol (fit each
 * registered estimator on tiny data and compare `capabilities(model)` with the declaration).
 */
import { expect } from 'vitest'
import type { Info } from 'aifn-compute/foundation/contracts'
import { capabilities, type ModelEntry } from 'aifn-compute/learning/estimators'
import { dense, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { MODEL_FIXTURES, fitStream } from './model-fixtures'

/** The metadata of every entry is well formed: kind, key, module and name. */
export function expectInfo(registry: Readonly<Record<string, { info: Info }>>, kind: Info['kind']): void {
  expect(Object.keys(registry).length).toBeGreaterThan(0)
  for (const [key, entry] of Object.entries(registry)) {
    const { info } = entry
    expect(info.key, key).toBe(key)
    expect(info.kind, key).toBe(kind)
    expect(info.module.length, `${key}: module`).toBeGreaterThan(0)
    expect(info.name.length, `${key}: name`).toBeGreaterThan(0)
    if (typeof entry === 'function')
      expect((entry as { name: string }).name, `${key}: the key is the export name`).toBe(key)
  }
}

/** A plain copy of fixture data (tensors as arrays), to check that a fit leaves the shared fixture untouched. */
const snapshot = (d: unknown): unknown =>
  isTensor(d)
    ? Array.from(dense.data(d as Tensor))
    : d !== null && typeof d === 'object'
      ? Object.fromEntries(Object.entries(d).map(([k, v]) => [k, snapshot(v)]))
      : d

/**
 * The model protocol: every registered estimator has a fixture, fits on it without changing the fixture's data (the
 * fixtures share their data, so a fit that wrote into it would make the test depend on the order of the entries), and
 * the fitted model has exactly the declared capabilities (and is transductive exactly when declared).
 */
export function expectModelProtocol(registry: Readonly<Record<string, ModelEntry>>): void {
  expectInfo(registry, 'model')
  for (const [key, entry] of Object.entries(registry)) {
    const fixture = MODEL_FIXTURES[key]
    expect(fixture, `${key}: a fixture in test/model-fixtures.ts`).toBeDefined()
    const before = JSON.stringify(snapshot(fixture.data))
    const model = fixture.make().fit(fixture.data as never, { stream: fitStream() } as never)
    expect(JSON.stringify(snapshot(fixture.data)), `${key}: the fit leaves its data unchanged`).toBe(before)
    expect(capabilities(model), `${key}: capabilities`).toEqual([...entry.info.capabilities])
    expect((model as { transductive?: boolean }).transductive === true, `${key}: transductive`).toBe(
      entry.info.transductive === true,
    )
  }
}

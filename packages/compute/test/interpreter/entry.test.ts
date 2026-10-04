import { describe, expect, test } from 'vitest'
import { entrySignature } from 'aifn-compute/interpreter'
import { defaults } from 'aifn-compute/foundation/space'

const byName = (src: string, entry?: string) =>
  Object.fromEntries(entrySignature(src, entry).params.map((p) => [p.name, p]))

describe('entrySignature', () => {
  test('reads declarations and arrows, with every literal kind', () => {
    const src = `function make(n = 200, noise = 0.3, flip = true, label = 'a', ws = [1, 2], f = (x) => x, rest) { return n }`
    const p = byName(src)
    expect(p.n).toMatchObject({ type: 'int', value: 200, min: 1, max: 1000 })
    expect(p.noise).toMatchObject({ type: 'real', value: 0.3, min: 0, max: 0.6 })
    expect(p.flip).toMatchObject({ type: 'bool', value: true })
    expect(p.label).toMatchObject({ type: 'fixed', value: 'a' })
    expect(p.ws).toMatchObject({ type: 'fixed', value: [1, 2] })
    expect(p.f).toMatchObject({ type: 'fixed', source: '(x) => x' })
    expect(p.rest).toMatchObject({ type: 'fixed' })
    expect(Object.keys(p)).toEqual(['n', 'noise', 'flip', 'label', 'ws', 'f', 'rest'])
    expect(byName('const make = (k = 3, s = -2.5) => k').s).toMatchObject({ type: 'real', min: -5, max: 5 })
    expect(byName('export const make = function (k = 3) { return k }').k).toMatchObject({ type: 'int', value: 3 })
    expect(byName('function build(m = 1.0) {}', 'build').m).toMatchObject({ type: 'real', value: 1 })
  })

  test('nested brackets, strings and comments in defaults do not split parameters', () => {
    const src = `function make(a = f(1, [2, 3], { b: 4 }), s = "x, y)", /* c, d */ n = 5) {}`
    expect(entrySignature(src).params.map((p) => p.name)).toEqual(['a', 's', 'n'])
    expect(byName(src).s).toMatchObject({ value: 'x, y)' })
  })

  test('the JSDoc block refines types, ranges, choices and docs', () => {
    const src = `/**
 * @param {int} n [20, 1000] number of points
 * @param {real:log} rate [0.001, 10] learning rate
 * @param {choice} shape ['sine', 'square'] curve shape
 * @param {bool} flip
 */
function make(n = 200, rate = 0.1, shape = 'square', flip) {}`
    const p = byName(src)
    expect(p.n).toMatchObject({ type: 'int', min: 20, max: 1000, value: 200, doc: 'number of points' })
    expect(p.rate).toMatchObject({ type: 'real', scale: 'log', min: 0.001, max: 10, value: 0.1 })
    expect(p.shape).toMatchObject({ type: 'choice', options: ['sine', 'square'], value: 'square' })
    expect(p.flip).toMatchObject({ type: 'bool', value: false })
  })

  test('missing or malformed JSDoc falls back to inference; defaults are clamped into ranges', () => {
    const src = `/**
 * @param {int} n [oops] broken range
 * @param {real:log} r [-1, 1] log needs min > 0
 * @param {real} z [0, 1]
 * @param nonsense
 */
function make(n = 7, r = 2, z = 5) {}`
    const p = byName(src)
    expect(p.n).toMatchObject({ type: 'int', min: 1, max: 35, value: 7, doc: 'broken range' })
    expect(p.r).toMatchObject({ type: 'real', min: 0, max: 4 })
    expect(p.z).toMatchObject({ type: 'real', min: 0, max: 1, value: 1 })
    // A JSDoc block that is not directly above the function is ignored.
    expect(byName('/** @param {int} n [1, 2] */\nconst other = 1\nfunction make(n = 1.5) {}').n.type).toBe('real')
  })

  test('unreadable source gives no parameters, never an error', () => {
    for (const src of ['', 'function make(n = 1', 'const x = 1', 'function make(', '@@@'])
      expect(entrySignature(src).params).toEqual([])
    expect(entrySignature('function make(n = 1').found).toBe(false)
    expect(entrySignature('function make() {}').found).toBe(true)
  })

  test('the space holds the controllable parameters with their defaults', () => {
    const sig = entrySignature(
      `/** @param {choice} k [1, 2, 3] */\nfunction make(n = 200, noise = 0.3, on = true, k = 2, f = g()) {}`,
    )
    expect(Object.keys(sig.space.dims)).toEqual(['n', 'noise', 'on', 'k'])
    expect(sig.space.dims.n).toMatchObject({ type: 'int', min: 1, max: 1000, default: 200 })
    expect(sig.space.dims.k).toMatchObject({ type: 'choice', options: [1, 2, 3], default: 2 })
    expect(defaults(sig.space)).toEqual({ n: 200, noise: 0.3, on: true, k: 2 })
  })
})

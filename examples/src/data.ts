/** Small data helpers for the recipes: plain maths, no models. */

/** `n` evenly spaced values from `a` to `b`. */
export const grid = (a: number, b: number, n: number) =>
  Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))

/** A seeded generator (mulberry32): `uniform()` in [0, 1) and `normal()` by Box–Muller. Same seed, same numbers. */
export function rng(seed: number) {
  let s = seed >>> 0
  const uniform = () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform())
  return { uniform, normal }
}

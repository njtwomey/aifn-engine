/**
 * Registry conformance (design S §6.3, §7): every compute entry carries the fields `Info` requires, and each kind's
 * declared metadata matches what its values do. Generated from the registries, so a new entry is checked without
 * editing this file (families of distributions and algorithms have their own suites).
 *
 * - windows: the main-lobe width and peak side-lobe level, measured on the periodic window of length 128;
 * - wavelets: filter length, Σh = √2, orthonormality and the declared vanishing moments;
 * - kernels: stationarity as declared, symmetric positive semi-definite Gram matrices, `kernelDiagonal`;
 * - bijectors: domain and codomain as declared, inverse ∘ forward = identity, log|f′| against central differences;
 * - links: inverse ∘ link = identity on the mean space, dμ/dη against central differences;
 * - likelihood families: canonical link and dispersion as declared, links registered;
 * - filter designs: a stable IIR or an FIR (denominator 1) low-pass with unit DC gain and attenuated Nyquist;
 * - KL rules: both families registered, KL(p ‖ p) = 0, and KL(p ‖ q) equal to a Monte Carlo estimate;
 * - functions: a function named by its key, with a known role.
 */
import path from 'node:path'
import fs from 'node:fs'
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { defaults } from 'aifn-compute/foundation/space'
import { tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type {
  Bijector,
  BijectorInfo,
  Distribution,
  DistributionInfo,
  FilterDesignInfo,
  FunctionInfo,
  Interval,
  KernelInfo,
  KlRuleInfo,
  LikelihoodInfo,
  LinkInfo,
  SupportName,
  WaveletInfo,
  WindowInfo,
} from 'aifn-compute/foundation/contracts'
import { eigh } from 'aifn-compute/numerics/linalg'
import { expBijector, sigmoidBijector } from 'aifn-compute/probability/bijectors'
import {
  Bernoulli,
  Categorical,
  Dirichlet,
  distributionRegistry,
  MultivariateNormal,
  type Univariate,
} from 'aifn-compute/probability/distributions'
import { linkRegistry, type Family } from 'aifn-compute/probability/likelihoods'
import { gram, kernelDiagonal, rbf, white, type Kernel } from 'aifn-compute/learning/kernels'
import { stability, toTransferFunction } from 'aifn-compute/systems'
import type { LtiSystem } from 'aifn-compute/foundation/contracts'
import { address, coreEntries, entriesOf } from '../../registries'

const flat = (v: Value): number[] => (typeof v === 'number' ? [v] : Array.from(toFlat(v as Tensor)))
const num = (v: Value): number => flat(v)[0]

const all = await coreEntries()
describe('every compute entry', () => {
  const src = path.resolve(import.meta.dirname, '..', '..', '..', 'src')
  it.each(all.map((e) => [`${e.info.kind} ${address(e)}`, e] as const))('%s has the Info fields', (_, e) => {
    const info = e.info
    for (const f of ['key', 'name', 'module'] as const) expect(typeof info[f] === 'string' && info[f] !== '').toBe(true)
    expect(['stable', 'experimental', 'deprecated']).toContain(info.stability)
    expect(fs.existsSync(path.join(src, info.module, 'index.ts')), info.module).toBe(true)
  })
  it('keys are unique within each kind', () => {
    const seen = new Set<string>()
    for (const e of all) {
      const k = `${e.info.kind} ${address(e)}`
      expect(seen.has(k), k).toBe(false)
      seen.add(k)
    }
  })
})

// ── Windows ──────────────────────────────────────────────────────────────────────────────────────────────────────────

type WindowFn = (n: number, o?: Record<string, number | boolean>) => Tensor
const windows = await entriesOf<WindowInfo>('window')

/** Main-lobe width (bins, first null to first null) and peak side-lobe level (dB) of a window by its DTFT. */
function figures(w: number[]): { width: number; sideLobeDb: number } {
  const n = w.length
  const P = 64
  const mag = (f: number) => {
    let re = 0
    let im = 0
    for (let k = 0; k < n; k++) {
      re += w[k] * Math.cos((2 * Math.PI * f * k) / n)
      im -= w[k] * Math.sin((2 * Math.PI * f * k) / n)
    }
    return Math.hypot(re, im)
  }
  const m = Array.from({ length: (n / 2) * P }, (_, i) => mag(i / P))
  let i = 1
  while (i < m.length - 1 && !(m[i] <= m[i - 1] && m[i] <= m[i + 1])) i++
  let peak = 0
  for (let j = i; j < m.length; j++) peak = Math.max(peak, m[j])
  return { width: (2 * i) / P, sideLobeDb: 20 * Math.log10(peak / m[0]) }
}

describe.each(windows.map((w) => [address(w), w] as const))('window %s', (_, entry) => {
  it('has the declared main-lobe width and side-lobe level', () => {
    const w = (entry as unknown as WindowFn)(128, { periodic: true })
    expect(w.shape).toEqual([128])
    const sym = flat((entry as unknown as WindowFn)(9))
    sym.forEach((v, i) => expect(v).toBeCloseTo(sym[8 - i], 12))
    const f = figures(flat(w))
    if (entry.info.mainLobeWidth !== undefined) expect(Math.abs(f.width - entry.info.mainLobeWidth)).toBeLessThan(0.1)
    if (entry.info.sideLobeDb !== undefined) expect(Math.abs(f.sideLobeDb - entry.info.sideLobeDb)).toBeLessThan(0.5)
  })
})

// ── Wavelets ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const wavelets = await entriesOf<WaveletInfo>('wavelet')
describe.each(wavelets.map((w) => [address(w), w] as const))('wavelet %s', (_, entry) => {
  const info = entry.info
  it.runIf(!info.continuous)('has the declared length, normalisation and vanishing moments', () => {
    const h = flat((entry as unknown as () => { recLo: Tensor }).call(null).recLo)
    expect(h.length).toBe(info.taps)
    expect(h.reduce((a, b) => a + b, 0)).toBeCloseTo(Math.SQRT2, 10)
    for (let k = 0; 2 * k < h.length; k++) {
      let s = 0
      for (let n = 0; n + 2 * k < h.length; n++) s += h[n] * h[n + 2 * k]
      expect(s).toBeCloseTo(k === 0 ? 1 : 0, 10)
    }
    // ψ has p vanishing moments when Σ (−1)ⁿ nᵖ h[n] = 0 for p < N, and not for p = N.
    const moment = (p: number) => h.reduce((a, v, n) => a + (n % 2 ? -1 : 1) * n ** p * v, 0)
    for (let p = 0; p < info.vanishingMoments!; p++) expect(Math.abs(moment(p))).toBeLessThan(1e-6 * 10 ** p)
    expect(Math.abs(moment(info.vanishingMoments!))).toBeGreaterThan(1e-6)
  })
  it.runIf(info.continuous)('evaluates on a grid', () => {
    const t = Array.from({ length: 41 }, (_, i) => -4 + i / 5)
    const omega0 = Number(defaults(info.params).omega0)
    expect((entry as unknown as (t: number[], o: object) => Tensor)(t, { omega0 }).shape[0]).toBe(41)
  })
})

// ── Kernels ──────────────────────────────────────────────────────────────────────────────────────────────────────────

const kernels = await entriesOf<KernelInfo>('kernel')
function buildKernel(key: string, entry: unknown, info: KernelInfo): Kernel {
  const d = defaults(info.hyper) as Record<string, number>
  const f = entry as (...a: unknown[]) => Kernel
  if (key === 'matern') return f(d.nu, { lengthscale: d.lengthscale, variance: d.variance })
  if (key === 'polynomial') return f(d.degree, { variance: d.variance, bias: d.bias })
  if (info.composite) return f(rbf(), white({ variance: 0.1 }))
  return f(d)
}
describe.each(kernels.map((k) => [address(k), k] as const))('kernel %s', (_, entry) => {
  it('is stationary as declared and has a symmetric positive semi-definite Gram matrix', () => {
    const k = buildKernel(entry.info.key, entry, entry.info)
    if (!entry.info.composite) expect(k.stationary).toBe(entry.info.stationary)
    const x = tensor([[0], [0.3], [1.1], [-0.7], [2.0]])
    const K = gram(k, x) as Tensor
    const rows = toRows(K) as number[][]
    rows.forEach((r, i) => r.forEach((v, j) => expect(v).toBeCloseTo(rows[j][i], 12)))
    expect(Math.min(...flat(eigh(K).values as Tensor))).toBeGreaterThan(-1e-9)
    expect(flat(kernelDiagonal(k, x) as Tensor)).toEqual(rows.map((r, i) => r[i]).map((v) => expect.closeTo(v, 12)))
  })
})

// ── Bijectors ────────────────────────────────────────────────────────────────────────────────────────────────────────

const bijectors = await entriesOf<BijectorInfo>('bijector')
/** The interval a support name denotes, or null for one that varies or is not an interval. */
function intervalOf(s: SupportName): [number, number] | 'bounded' | null {
  if (s === 'real') return [-Infinity, Infinity]
  if (s === 'positive' || s === 'non-negative') return [0, Infinity]
  if (s === 'unit-interval') return [0, 1]
  if (s === 'interval') return 'bounded'
  return null
}
const interior = (i: Interval) =>
  Number.isFinite(i.lower) && Number.isFinite(i.upper)
    ? 0.37 * i.lower + 0.63 * i.upper
    : Number.isFinite(i.lower)
      ? i.lower + 0.7
      : Number.isFinite(i.upper)
        ? i.upper - 0.7
        : 0.3
function buildBijector(key: string, entry: unknown, info: BijectorInfo): Bijector | null {
  if (!info.factory) return entry as Bijector
  if (key === 'chainBijectors') return (entry as (...b: Bijector[]) => Bijector)(sigmoidBijector, expBijector)
  // Vector bijectors (eventRank 1) have their own tests.
  if (key === 'orderedBijector' || key === 'affineCouplingBijector') return null
  return (entry as (...a: unknown[]) => Bijector)(...Object.values(defaults(info.params)))
}
describe.each(bijectors.map((b) => [address(b), b] as const))('bijector %s', (_, entry) => {
  const info = entry.info
  const b = buildBijector(info.key, entry, info)
  it.runIf(b !== null)('maps its declared domain onto its codomain, invertibly, with its log-Jacobian', () => {
    for (const [name, i] of [
      [info.domain, b!.domain],
      [info.codomain, b!.codomain],
    ] as const) {
      const want = intervalOf(name)
      if (want === 'bounded') expect(Number.isFinite(i.lower) && Number.isFinite(i.upper)).toBe(true)
      else if (want) expect([i.lower, i.upper]).toEqual(want)
    }
    const x = interior(b!.domain)
    expect(num(b!.inverse(b!.forward(x)))).toBeCloseTo(x, 10)
    const h = 1e-5
    const slope = (num(b!.forward(x + h)) - num(b!.forward(x - h))) / (2 * h)
    expect(num(b!.logAbsDetJacobian(x))).toBeCloseTo(Math.log(Math.abs(slope)), 6)
  })
})

// ── Links and likelihood families ────────────────────────────────────────────────────────────────────────────────────

const links = await entriesOf<LinkInfo>('link')
const meanPoint: Partial<Record<SupportName, number>> = {
  real: 0.3,
  positive: 1.7,
  'non-negative': 1.7,
  'unit-interval': 0.3,
}
describe.each(links.map((l) => [address(l), l] as const))('link %s', (_, entry) => {
  it('inverts on its mean space and has dμ/dη', () => {
    const l = entry as unknown as { link(m: Value): Value; inverse(e: Value): Value; derivative(e: Value): Value }
    const mu = meanPoint[entry.info.meanSpace]!
    const eta = num(l.link(mu))
    expect(num(l.inverse(eta))).toBeCloseTo(mu, 10)
    const h = 1e-6
    expect(num(l.derivative(eta))).toBeCloseTo((num(l.inverse(eta + h)) - num(l.inverse(eta - h))) / (2 * h), 6)
  })
})

const likelihoods = await entriesOf<LikelihoodInfo>('likelihood')
describe.each(likelihoods.map((l) => [address(l), l] as const))('likelihood family %s', (_, entry) => {
  it('has the declared canonical link and dispersion, and registered links', () => {
    const fam = (entry as unknown as (...a: unknown[]) => Family)(...Object.values(defaults(entry.info.params)))
    expect(fam.canonicalLink).toBe(entry.info.canonicalLink)
    expect(fam.dispersion === null).toBe(entry.info.dispersion)
    for (const l of [entry.info.canonicalLink, ...entry.info.links]) expect(l in linkRegistry).toBe(true)
  })
})

// ── Filter designs ───────────────────────────────────────────────────────────────────────────────────────────────────

const designs = await entriesOf<FilterDesignInfo>('filter-design')
const OPTIONS = new Set(['btype', 'window'])
describe.each(designs.map((d) => [address(d), d] as const))('filter design %s', (_, entry) => {
  it('designs a stable low-pass with unit DC gain at its defaults', () => {
    const d = defaults(entry.info.params) as Record<string, string | number>
    const args = Object.entries(d)
      .filter(([k]) => !OPTIONS.has(k))
      .map(([, v]) => v)
    const options = Object.fromEntries(Object.entries(d).filter(([k]) => OPTIONS.has(k)))
    const sys = (entry as unknown as (...a: unknown[]) => LtiSystem)(...args, options)
    const tf = toTransferFunction(sys).repr as { b: Tensor; a: Tensor }
    const [b, a] = [flat(tf.b), flat(tf.a)]
    if (entry.info.family === 'fir') expect(a).toEqual([1])
    else expect(stability(sys).stable).toBe(true)
    const at = (z: number) =>
      Math.abs(b.reduce((s, v, k) => s + v * z ** k, 0) / a.reduce((s, v, k) => s + v * z ** k, 0))
    expect(at(1)).toBeGreaterThan(0.85)
    expect(at(1)).toBeLessThan(1.01)
    expect(at(-1)).toBeLessThan(0.1)
  })
})

// ── KL rules ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const klRules = await entriesOf<KlRuleInfo>('kl-rule')
/** Two instances of a family: its defaults, and its real parameters scaled by 1.3 (vector families by hand). */
function pair(key: string): [Distribution, Distribution] {
  const special: Record<string, () => [Distribution, Distribution]> = {
    Categorical: () => [Categorical(tensor([0.2, 0.5, 0.3])), Categorical(tensor([0.4, 0.4, 0.2]))],
    Bernoulli: () => [Bernoulli(0.3), Bernoulli(0.6)],
    Dirichlet: () => [Dirichlet(tensor([2, 3, 4])), Dirichlet(tensor([1, 1, 2]))],
    MultivariateNormal: () => [
      MultivariateNormal(tensor([1, -1]), {
        covariance: tensor([
          [2, 0.5],
          [0.5, 1],
        ]),
      }),
      MultivariateNormal(tensor([0, 0]), {
        covariance: tensor([
          [1, 0],
          [0, 1],
        ]),
      }),
    ],
  }
  if (special[key]) return special[key]()
  const f = distributionRegistry[key] as unknown as ((...a: unknown[]) => Distribution) & { info: DistributionInfo }
  const d = defaults(f.info.params) as Record<string, number>
  const dims = f.info.params.dims
  const moved = Object.entries(d).map(([k, v]) => (dims[k].type === 'real' ? (v === 0 ? 0.4 : v * 1.3) : v))
  return [f(...Object.values(d)), f(...moved)]
}
describe.each(klRules.map((r) => [address(r), r] as const))('KL rule %s', (_, entry) => {
  it('names registered families, vanishes at p = q and matches a Monte Carlo estimate', () => {
    expect(entry.info.p in distributionRegistry).toBe(true)
    expect(entry.info.q in distributionRegistry).toBe(true)
    const rule = entry as unknown as (p: Distribution, q: Distribution) => Value
    const [p, q] = pair(entry.info.p)
    expect(num(rule(p, p))).toBeCloseTo(0, 10)
    const exact = num(rule(p, q))
    expect(exact).toBeGreaterThan(0)
    const n = 20000
    const x = p.sample(stream(`kl/${entry.info.key}`), { shape: [n] }) as Value
    const lq = flat((q as Univariate).logProb(x) as Value)
    const diff = flat((p as Univariate).logProb(x) as Value).map((v, i) => v - lq[i])
    const m = diff.reduce((a, b) => a + b, 0) / n
    const sd = Math.sqrt(diff.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1))
    expect(Math.abs(m - exact)).toBeLessThan(6 * (sd / Math.sqrt(n)) + 1e-9)
  })
})

// ── Functions ────────────────────────────────────────────────────────────────────────────────────────────────────────

const functions = await entriesOf<FunctionInfo>('function')
const roles = ['transform', 'estimator', 'test', 'construction', 'property', 'fit', 'simulation', 'solver', 'inference']
describe('functions', () => {
  it.each(functions.map((f) => [address(f), f] as const))('%s is a function named by its key, with a role', (_, f) => {
    expect(typeof f).toBe('function')
    expect((f as unknown as { name: string }).name).toBe(f.info.key)
    expect(roles).toContain(f.info.role)
  })
})

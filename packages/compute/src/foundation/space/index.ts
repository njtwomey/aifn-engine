/**
 * `aifn-compute/foundation/space`: parameter spaces as data (design S §2.4). One schema describes hyperparameter
 * searches, recipe knobs, distribution and kernel parameters and figure controls; a `Space` holds no closures, so it
 * serialises, and its conditions are data too.
 *
 * - Dimensions: `real` (linear or log scale, optional step), `int`, `oneOf` (a choice among listed values), `bool`,
 *   `subspace` (a nested space) and `variants` (a family choice with parameters per case); `space(dims)` builds a
 *   space and `when(key, equals)` makes a dimension conditional on an earlier one.
 * - Values: `defaults(space)`, `clamp(space, values)` (fill, clip, round and snap into the space), `clampReport` (the
 *   same, listing the unknown and inactive keys it dropped), `isDimActive`.
 * - Coordinates: `encode(space, values)` maps values into the unit cube $[0, 1]^m$ (log dimensions in log space,
 *   integers and choices as equal bins) and `decode(space, u)` maps any point of the cube back, so `encodedSize`
 *   coordinates drawn uniformly decode to a uniform draw from the space. Inactive dimensions encode their defaults.
 * - Domains of observations and actions (`Domain`): `discreteDomain`, `boxDomain`, `domainContains`, `clipToDomain`,
 *   `domainSize` (`Infinity` for a box), `domainDimension`, `sampleDomain`.
 * - Enumeration and draws: `grid(space, { points })` (every combination, conditions respected) and `sample(s, space)`.
 *
 * The encoding follows the unit-cube convention of Bayesian-optimisation libraries (Snoek, Larochelle and Adams, 2012,
 * "Practical Bayesian optimization of machine learning algorithms", NeurIPS; scikit-optimize's `Space.transform`);
 * grid and random search follow Bergstra and Bengio (2012), "Random search for hyper-parameter optimization", JMLR 13.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Condition, Dim, DimSpec, Size, Space } from 'aifn-compute/foundation/contracts'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type {
  BoxDomain,
  Condition,
  Dim,
  DimSpec,
  DiscreteDomain,
  Domain,
  DomainKind,
  Space,
} from 'aifn-compute/foundation/contracts'
export {
  boxDomain,
  clipToDomain,
  discreteDomain,
  domainContains,
  domainDimension,
  domainSize,
  sampleDomain,
} from './domain'

// ── Values ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The value of a `variants` dimension: the chosen case and that case's parameters. */
export interface VariantValue {
  /** The name of the chosen case, a key of the dimension's `cases`. */
  readonly case: string
  /** A point of the chosen case's space. */
  readonly params: SpaceValues
}

/** The value of one dimension. */
export type SpaceValue = number | string | boolean | SpaceValues | VariantValue

/** A point of a space: one value per active dimension, by key. */
export type SpaceValues = { readonly [key: string]: SpaceValue }

/** The value type of one dimension, read from its literal type. */
export type DimValue<D> = D extends { readonly type: 'real' | 'int' }
  ? number
  : D extends { readonly type: 'choice'; readonly options: readonly (infer O)[] }
    ? O
    : D extends { readonly type: 'bool' }
      ? boolean
      : D extends { readonly type: 'space'; readonly of: infer S extends Space }
        ? ValuesOf<S>
        : D extends { readonly type: 'variants'; readonly cases: infer C }
          ? { [K in keyof C & string]: { case: K; params: C[K] extends Space ? ValuesOf<C[K]> : never } }[keyof C &
              string]
          : never

/** The dimensions of a space, by key, with their literal types. */
type Dims<S extends Space> = S['dims']
/** The keys of the dimensions of a space that carry a `when` condition. */
type Conditional<S extends Space> = {
  [K in keyof Dims<S>]: Dims<S>[K] extends { when: Condition } ? K : never
}[keyof Dims<S>]

/** The values of a space, typed from its literal: conditional dimensions are optional. */
export type ValuesOf<S extends Space> = {
  -readonly [K in Exclude<keyof Dims<S>, Conditional<S>>]: DimValue<Dims<S>[K]>
} & { -readonly [K in Conditional<S>]?: DimValue<Dims<S>[K]> }

// ── Builders ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Presentation-neutral documentation any dimension may carry: a `label` (TeX allowed), a longer `doc`, and a `when`
 * condition under which alone the dimension applies.
 */
export type DimDoc = { readonly label?: string; readonly doc?: string; readonly when?: Condition }

/**
 * `{ when: Condition }` when a builder's options carry a condition, so that the dimension's literal type keeps it and
 * `ValuesOf` makes its value optional.
 */
export type Conditioned<O> = O extends { readonly when: Condition } ? { readonly when: Condition } : unknown

/**
 * A space from its dimensions (keys in the order given, which is the order of `encode` and `grid`). Throws
 * `DomainError` when a condition names a dimension that is not earlier, or a nested space (whose value never equals a
 * scalar).
 *
 * @param dims The dimensions by key, each made by a builder (`real`, `int`, `oneOf`, `bool`, `subspace`,
 *   `variants`). A dimension's `when` may only name a key before it.
 * @returns The space `{ dims }`, keeping the literal types of the dimensions so that `ValuesOf` can type its values.
 *
 * @example A search space with a conditional dimension
 * const s = space({
 *   optimiser: oneOf(['sgd', 'adam']),
 *   lr: real(1e-4, 1, { scale: 'log' }),
 *   momentum: real(0, 0.99, { when: when('optimiser', 'sgd') }),
 * })
 * print('keys:', Object.keys(s.dims))
 * print('defaults:', defaults(s))
 *
 * @example A condition must name an earlier dimension
 * try {
 *   space({ momentum: real(0, 1, { when: when('optimiser', 'sgd') }), optimiser: oneOf(['sgd', 'adam']) })
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function space<const D extends Readonly<Record<string, DimSpec>>>(dims: D): { readonly dims: D } {
  // A condition names an earlier dimension whose value can equal a scalar: a real, int, choice or bool, or a variants
  // dimension (compared by its case name). A nested space's value is an object, which never equals one.
  const seen = new Map<string, DimSpec>()
  for (const [name, dim] of Object.entries(dims)) {
    if (dim.when !== undefined) {
      const ref = seen.get(dim.when.key)
      if (ref === undefined)
        throw new DomainError('space', `space: '${name}' is conditioned on '${dim.when.key}', which is not earlier`)
      if (ref.type === 'space')
        throw new DomainError('space', `space: '${name}' is conditioned on the nested space '${dim.when.key}'`)
    }
    seen.set(name, dim)
  }
  return { dims }
}

/**
 * A real dimension on $[\text{min}, \text{max}]$ with a default (the midpoint of the scale when omitted: the
 * geometric mean on a log scale). `scale: 'log'` needs $\text{min} > 0$ and spaces grids and draws evenly in
 * $\log x$; `step` snaps values to $\text{min} + k \cdot \text{step}$. Throws `DomainError` when
 * $\text{max} < \text{min}$ or a log scale has $\text{min} \le 0$.
 *
 * @param min The lower end of the range.
 * @param max The upper end of the range, at least `min` (equal to it for a fixed value).
 * @param options The `default` value, the `scale` (`'linear'` unless `'log'`), the grid `step` (none when left out),
 *   a display `unit`, and the documentation of `DimDoc` (`label`, `doc`, `when`). The default is kept as given, not
 *   clipped or snapped.
 * @returns The dimension `{ type: 'real', min, max, default, ... }`, carrying the options.
 *
 * @example A log-scale learning rate
 * const lr = real(1e-4, 1, { scale: 'log' })
 * print('dimension:', lr)
 *
 * @example A stepped real snaps to its grid
 * const s = space({ p: real(0, 1, { step: 0.25 }) })
 * print('0.6 ->', clamp(s, { p: 0.6 }).p)
 * print('0.3 ->', clamp(s, { p: 0.3 }).p)
 */
export function real<
  const O extends DimDoc & { default?: number; scale?: 'linear' | 'log'; step?: number; unit?: string } = {},
>(min: number, max: number, options: O = {} as O): DimSpec & { readonly type: 'real' } & Conditioned<O> {
  if (!(max >= min)) throw new DomainError('real', `real: needs min ≤ max, got [${min}, ${max}]`)
  if (options.scale === 'log' && !(min > 0)) throw new DomainError('real', 'real: a log scale needs min > 0')
  const mid = options.scale === 'log' ? Math.sqrt(min * max) : (min + max) / 2
  return { ...options, type: 'real', min, max, default: options.default ?? mid } as DimSpec & {
    readonly type: 'real'
  } & Conditioned<O>
}

/**
 * An integer dimension on $\{\text{min}, \dots, \text{max}\}$ with a default (`min` when omitted). Throws
 * `DomainError` unless `min` and `max` are integers with $\text{min} \le \text{max}$.
 *
 * @param min The smallest value, an integer.
 * @param max The largest value, an integer no smaller than `min`.
 * @param options The `default` value and the documentation of `DimDoc` (`label`, `doc`, `when`).
 * @returns The dimension `{ type: 'int', min, max, default }`, carrying the options.
 *
 * @example Values are rounded and clipped into the range
 * const s = space({ k: int(1, 10) })
 * print('dimension:', s.dims.k)
 * print('3.7 ->', clamp(s, { k: 3.7 }).k)
 * print('99 ->', clamp(s, { k: 99 }).k)
 */
export function int<const O extends DimDoc & { default?: number } = {}>(
  min: number,
  max: number,
  options: O = {} as O,
): DimSpec & { readonly type: 'int' } & Conditioned<O> {
  if (!(Number.isInteger(min) && Number.isInteger(max) && max >= min))
    throw new DomainError('int', `int: needs integers min ≤ max, got [${min}, ${max}]`)
  return { ...options, type: 'int', min, max, default: options.default ?? min } as DimSpec & {
    readonly type: 'int'
  } & Conditioned<O>
}

/**
 * A choice among listed values (strings or numbers), with a default (the first when omitted). Throws `DomainError`
 * when no options are given.
 *
 * @param options The values to choose among, in order (the order of `encode`'s bins and of `grid`).
 * @param doc The `default` value (one of `options`) and the documentation of `DimDoc` (`label`, `doc`, `when`).
 * @returns The dimension `{ type: 'choice', options, default }`, carrying `doc`.
 *
 * @example An activation, and a value that is not an option
 * const s = space({ act: oneOf(['relu', 'tanh', 'gelu'], { default: 'gelu' }) })
 * print('dimension:', s.dims.act)
 * print("'sigmoid' ->", clamp(s, { act: 'sigmoid' }).act)
 */
export function oneOf<
  const O extends readonly (string | number)[],
  const D extends DimDoc & { default?: O[number] } = {},
>(options: O, doc: D = {} as D): DimSpec & { readonly type: 'choice'; readonly options: O } & Conditioned<D> {
  if (options.length === 0) throw new DomainError('oneOf', 'oneOf: needs at least one option')
  const dim = { ...doc, type: 'choice', options, default: doc.default ?? options[0] }
  return dim as DimSpec & { readonly type: 'choice'; readonly options: O } & Conditioned<D>
}

/**
 * A boolean dimension, default false.
 *
 * @param options The `default` value and the documentation of `DimDoc` (`label`, `doc`, `when`).
 * @returns The dimension `{ type: 'bool', default }`, carrying the options.
 *
 * @example Off by default, or on
 * print(bool())
 * print(bool({ default: true, label: 'Shuffle' }))
 */
export function bool<const O extends DimDoc & { default?: boolean } = {}>(
  options: O = {} as O,
): DimSpec & { readonly type: 'bool' } & Conditioned<O> {
  return { ...options, type: 'bool', default: options.default ?? false } as DimSpec & {
    readonly type: 'bool'
  } & Conditioned<O>
}

/**
 * A nested space, e.g. a kernel's sub-kernel parameters. Its value is a point of the nested space, an object. (The
 * result is not intersected with `DimSpec`, whose `of: Space` would add a string index to the literal and widen
 * `ValuesOf`.)
 *
 * @param of The nested space, made by `space`.
 * @param doc The documentation of `DimDoc` (`label`, `doc`, `when`).
 * @returns The dimension `{ type: 'space', of }`, carrying `doc`.
 *
 * @example A kernel's parameters as one dimension
 * const kernel = subspace(space({ lengthscale: real(0.1, 10, { scale: 'log' }), variance: real(0.1, 10) }))
 * print(defaults(space({ kernel })))
 */
export function subspace<const S extends Space, const D extends DimDoc = {}>(
  of: S,
  doc: D = {} as D,
): DimDoc & { readonly type: 'space'; readonly of: S } & Conditioned<D> {
  return { ...doc, type: 'space', of } as DimDoc & { readonly type: 'space'; readonly of: S } & Conditioned<D>
}

/**
 * A family choice whose cases have their own parameters, with a default case (the first when omitted). Its value is a
 * `VariantValue`, `{ case, params }`. Throws `DomainError` when there are no cases.
 *
 * @param cases The cases by name, each with the space of its parameters.
 * @param doc The `default` case's name and the documentation of `DimDoc` (`label`, `doc`, `when`).
 * @returns The dimension `{ type: 'variants', cases, default }`, carrying `doc`.
 *
 * @example A kernel family with parameters per case
 * const s = space({
 *   kernel: variants({
 *     rbf: space({ lengthscale: real(0.1, 10) }),
 *     periodic: space({ lengthscale: real(0.1, 10), period: real(1, 5) }),
 *   }),
 * })
 * print('default:', defaults(s))
 * print('periodic:', clamp(s, { kernel: { case: 'periodic', params: { period: 2 } } }))
 */
export function variants<
  const C extends Readonly<Record<string, Space>>,
  const D extends DimDoc & { default?: keyof C & string } = {},
>(
  cases: C,
  doc: D = {} as D,
): DimDoc & { readonly type: 'variants'; readonly cases: C; readonly default: string } & Conditioned<D> {
  const names = Object.keys(cases)
  if (names.length === 0) throw new DomainError('variants', 'variants: needs at least one case')
  const dim = { ...doc, type: 'variants', cases, default: doc.default ?? names[0] }
  return dim as DimDoc & { readonly type: 'variants'; readonly cases: C; readonly default: string } & Conditioned<D>
}

/**
 * The condition "dimension `key` has the value `equals`", for `DimDoc.when`. `key` must come earlier in the space
 * (`space` checks it); for a variants dimension, `equals` is a case name.
 *
 * @param key The key of the dimension the condition reads.
 * @param equals The value that dimension must have (compared with `===`), or a case name for a variants dimension.
 * @returns The condition `{ key, equals }`, plain data.
 *
 * @example Momentum only for SGD
 * const s = space({
 *   optimiser: oneOf(['sgd', 'adam']),
 *   momentum: real(0, 0.99, { when: when('optimiser', 'sgd') }),
 * })
 * print('condition:', s.dims.momentum.when)
 * print('sgd:', defaults(s))
 * print('adam:', clamp(s, { optimiser: 'adam' }))
 */
export function when(key: string, equals: string | number | boolean): Condition {
  return { key, equals }
}

// ── Values ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * True when a dimension applies under `values`: it has no condition, or its condition holds.
 *
 * @param dim The dimension, with its optional `when`.
 * @param values The values of the space so far; only the key named by the condition is read (for a variants value,
 *   its case name).
 * @returns Whether the dimension is active.
 *
 * @example Active for one optimiser, not the other
 * const s = space({
 *   optimiser: oneOf(['sgd', 'adam']),
 *   lr: real(1e-4, 1, { scale: 'log' }),
 *   momentum: real(0, 0.99, { when: when('optimiser', 'sgd') }),
 * })
 * print('sgd:', isDimActive(s.dims.momentum, { optimiser: 'sgd' }))
 * print('adam:', isDimActive(s.dims.momentum, { optimiser: 'adam' }))
 */
export function isDimActive(dim: DimSpec, values: SpaceValues): boolean {
  if (dim.when === undefined) return true
  const v = values[dim.when.key]
  // A variants dimension's value is { case, params }: its condition compares the case name.
  const scalar = typeof v === 'object' && v !== null && 'case' in v ? (v as VariantValue).case : v
  return scalar === dim.when.equals
}

/**
 * The nearest value of one dimension to `v`: an invalid value takes the default, numbers are clipped (integers
 * rounded, stepped reals snapped to the nearest grid point in range), and nested spaces and variants are clamped
 * recursively.
 *
 * @param dim The dimension.
 * @param v The given value, of any type; `undefined` gives the default.
 * @param dropped Where to append the paths of keys dropped inside a nested space or variant; left out, none are
 *   recorded.
 * @param path The path of this dimension, prefixed to the keys appended to `dropped`.
 * @returns The clamped value.
 */
function clampDim(dim: Dim, v: unknown, dropped?: string[], path = ''): SpaceValue {
  switch (dim.type) {
    case 'real': {
      if (typeof v !== 'number' || Number.isNaN(v)) return dim.default
      let x = Math.min(dim.max, Math.max(dim.min, v))
      if (dim.step !== undefined && dim.step > 0) {
        // The last grid point at or below max, with a little slack so that rounding in (max − min)/step (0.3/0.1 is
        // 2.9999999999999996) does not drop a max that lies on the grid; min + k·step is then capped at max.
        const last = Math.floor((dim.max - dim.min) / dim.step + 1e-9)
        const k = Math.min(last, Math.round((x - dim.min) / dim.step))
        x = Math.min(dim.max, dim.min + k * dim.step)
      }
      return x
    }
    case 'int':
      return typeof v === 'number' && !Number.isNaN(v)
        ? Math.min(dim.max, Math.max(dim.min, Math.round(v)))
        : dim.default
    case 'choice':
      return dim.options.includes(v as string | number) ? (v as string | number) : dim.default
    case 'bool':
      return typeof v === 'boolean' ? v : dim.default
    case 'space':
      return clampInto(dim.of, typeof v === 'object' && v !== null ? (v as SpaceValues) : {}, dropped, `${path}.`)
    case 'variants': {
      const given = typeof v === 'object' && v !== null ? (v as Partial<VariantValue>) : {}
      const name = typeof given.case === 'string' && given.case in dim.cases ? given.case : dim.default
      const params = typeof given.params === 'object' && given.params !== null ? given.params : {}
      return { case: name, params: clampInto(dim.cases[name], params, dropped, `${path}.params.`) }
    }
  }
}

/**
 * Clamp into `space`, appending to `dropped` the path (`prefix` + key) of every unknown or inactive key given.
 *
 * @param space The space to clamp into.
 * @param values The given values by key; not modified.
 * @param dropped Where to append the paths of the keys dropped, or undefined to record none.
 * @param prefix Prepended to each key appended to `dropped` (empty at the top, `outer.` inside a subspace).
 * @returns The clamped point: every active dimension in the space's order.
 */
function clampInto(
  space: Space,
  values: Readonly<Record<string, unknown>>,
  dropped: string[] | undefined,
  prefix: string,
): SpaceValues {
  const out: Record<string, SpaceValue> = {}
  for (const [key, dim] of Object.entries(space.dims))
    if (isDimActive(dim, out)) out[key] = clampDim(dim, values[key], dropped, prefix + key)
  if (dropped) for (const key of Object.keys(values)) if (!(key in out)) dropped.push(prefix + key)
  return out
}

/**
 * The nearest point of the space to `values`: missing or invalid values take their defaults, numbers are clipped to
 * their range (integers rounded, stepped reals snapped), unknown keys and inactive dimensions are dropped.
 * `clampReport` also says which keys were dropped. Conditions are read from the clamped values.
 *
 * @param space The space.
 * @param values Any values by key (a URL's query, a recipe's knobs); not modified.
 * @returns A point of the space, with a value for every active dimension.
 *
 * @example Clip, fill and drop
 * const s = space({
 *   optimiser: oneOf(['sgd', 'adam']),
 *   lr: real(1e-4, 1, { scale: 'log' }),
 *   momentum: real(0, 0.99, { when: when('optimiser', 'sgd') }),
 * })
 * print(clamp(s, { optimiser: 'adam', lr: 5, momentum: 0.9, extra: 1 }))
 */
export function clamp<S extends Space>(space: S, values: Readonly<Record<string, unknown>>): ValuesOf<S>
export function clamp(space: Space, values: Readonly<Record<string, unknown>>): SpaceValues
export function clamp(space: Space, values: Readonly<Record<string, unknown>>): SpaceValues {
  return clampInto(space, values, undefined, '')
}

/** A clamped point and the keys `clamp` dropped. */
export interface ClampReport<V> {
  /** The clamped point, as `clamp` returns it. */
  readonly values: V
  /**
   * The paths of the given keys that are unknown to the space or belong to an inactive dimension, in the order
   * found: `key`, `outer.inner` inside a subspace, `family.params.key` inside a variant's parameters.
   */
  readonly dropped: readonly string[]
}

/**
 * `clamp`, reporting what it dropped: unknown keys and keys of inactive dimensions are listed in `dropped`, so that a
 * caller (a recipe, a URL decoder) can warn rather than ignore them silently.
 *
 * @param space The space.
 * @param values Any values by key; not modified.
 * @returns The clamped `values` and the `dropped` paths.
 *
 * @example An unknown key is reported
 * print(clampReport(space({ n: int(1, 9) }), { n: 3, m: 1 }).dropped)
 *
 * @example An inactive dimension is reported too
 * const s = space({
 *   optimiser: oneOf(['sgd', 'adam']),
 *   lr: real(1e-4, 1, { scale: 'log' }),
 *   momentum: real(0, 0.99, { when: when('optimiser', 'sgd') }),
 * })
 * print(clampReport(s, { optimiser: 'adam', momentum: 0.9 }))
 */
export function clampReport<S extends Space>(
  space: S,
  values: Readonly<Record<string, unknown>>,
): ClampReport<ValuesOf<S>>
export function clampReport(space: Space, values: Readonly<Record<string, unknown>>): ClampReport<SpaceValues>
export function clampReport(space: Space, values: Readonly<Record<string, unknown>>): ClampReport<SpaceValues> {
  const dropped: string[] = []
  return { values: clampInto(space, values, dropped, ''), dropped }
}

/**
 * Every active dimension at its default.
 *
 * @param space The space.
 * @returns The point with every dimension active under the defaults at its default; nested spaces and variants hold
 *   their own defaults.
 *
 * @example The defaults of a search space
 * const s = space({
 *   optimiser: oneOf(['sgd', 'adam']),
 *   lr: real(1e-4, 1, { scale: 'log' }),
 *   momentum: real(0, 0.99, { when: when('optimiser', 'sgd') }),
 * })
 * print(defaults(s))
 */
export function defaults<S extends Space>(space: S): ValuesOf<S>
export function defaults(space: Space): SpaceValues
export function defaults(space: Space): SpaceValues {
  return clamp(space, {})
}

// ── Unit-cube coordinates ────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The number of unit-cube coordinates of one dimension: 1 for a scalar, the nested space's for a subspace, and for
 * variants one for the case plus every case's.
 *
 * @param dim The dimension.
 * @returns Its number of coordinates.
 */
function dimSize(dim: Dim): Size {
  if (dim.type === 'space') return encodedSize(dim.of)
  if (dim.type === 'variants') return 1 + Object.values(dim.cases).reduce((n, c) => n + encodedSize(c), 0)
  return 1
}

/**
 * The number of unit-cube coordinates of a space (a variants dimension holds one for the case plus every case's).
 *
 * @param space The space.
 * @returns The length $m$ of `encode`'s output and of `decode`'s input; inactive dimensions count too.
 *
 * @example Scalars take one coordinate, a variants dimension one more than its cases
 * print('scalars:', encodedSize(space({ a: real(0, 1), b: int(0, 3), c: bool() })))
 * const kernel = variants({
 *   rbf: space({ ell: real(0.1, 10) }),
 *   periodic: space({ ell: real(0.1, 10), p: real(1, 5) }),
 * })
 * print('variants:', encodedSize(space({ kernel })))
 */
export function encodedSize(space: Space): Size {
  return Object.values(space.dims).reduce((n, d) => n + dimSize(d), 0)
}

/**
 * Which of `count` equal bins of $[0, 1]$ holds $u$ ($u = 1$ falls in the last).
 *
 * @param u The coordinate; values outside $[0, 1]$ fall in the first or last bin.
 * @param count The number of bins, at least 1.
 * @returns The bin's index, from 0 to `count - 1`.
 */
const bin = (u: number, count: number) => Math.min(count - 1, Math.max(0, Math.floor(u * count)))

/**
 * Appends the unit-cube coordinates of one dimension's value: a real's position in its range (in $\log x$ on a log
 * scale; 0.5 when the range is a point), the centre of an integer's, choice's or case's bin, 0.25 or 0.75 for a
 * boolean, and recursively for nested spaces and every case of a variants dimension (the cases not chosen at their
 * defaults).
 *
 * @param dim The dimension.
 * @param v Its value, already clamped.
 * @param out The coordinates so far; `dimSize(dim)` are appended.
 */
function encodeDim(dim: Dim, v: SpaceValue, out: number[]): void {
  switch (dim.type) {
    case 'real': {
      const x = v as number
      if (dim.max === dim.min) out.push(0.5)
      else if (dim.scale === 'log') out.push(Math.log(x / dim.min) / Math.log(dim.max / dim.min))
      else out.push((x - dim.min) / (dim.max - dim.min))
      return
    }
    case 'int':
      out.push(((v as number) - dim.min + 0.5) / (dim.max - dim.min + 1))
      return
    case 'choice':
      out.push((dim.options.indexOf(v as string | number) + 0.5) / dim.options.length)
      return
    case 'bool':
      out.push(v ? 0.75 : 0.25)
      return
    case 'space':
      encodeInto(dim.of, v as SpaceValues, out)
      return
    case 'variants': {
      const names = Object.keys(dim.cases)
      const value = v as VariantValue
      out.push((names.indexOf(value.case) + 0.5) / names.length)
      for (const name of names)
        encodeInto(dim.cases[name], name === value.case ? value.params : defaults(dim.cases[name]), out)
    }
  }
}

/**
 * Appends the unit-cube coordinates of a point, clamped first; an inactive dimension encodes its default.
 *
 * @param space The space.
 * @param values The point; not modified.
 * @param out The coordinates so far; `encodedSize(space)` are appended.
 */
function encodeInto(space: Space, values: SpaceValues, out: number[]): void {
  const full = clamp(space, values)
  const fallback = defaults(space)
  for (const [key, dim] of Object.entries(space.dims))
    encodeDim(dim, full[key] ?? fallback[key] ?? clampDim(dim, undefined), out)
}

/**
 * The unit-cube coordinates of a point (clamped first); inactive dimensions encode their defaults. Integers, choices,
 * booleans and cases encode to the centre of their bin, so `decode(space, encode(space, x))` gives back `x` (up to
 * rounding in the reals).
 *
 * @param space The space.
 * @param values The point; any values are clamped into the space first.
 * @returns The `encodedSize(space)` coordinates in $[0, 1]$, in the order of the dimensions.
 *
 * @example A point and its coordinates
 * const s = space({
 *   optimiser: oneOf(['sgd', 'adam']),
 *   lr: real(1e-4, 1, { scale: 'log' }),
 *   momentum: real(0, 0.99, { when: when('optimiser', 'sgd') }),
 * })
 * print(encode(s, { optimiser: 'adam', lr: 0.01 }))
 */
export function encode(space: Space, values: SpaceValues): Float64Array {
  const out: number[] = []
  encodeInto(space, values, out)
  return Float64Array.from(out)
}

/**
 * Reads one dimension's value from unit-cube coordinates, the inverse of `encodeDim`: a real from its position (in
 * $\log x$ on a log scale, then snapped to its step), an integer, choice or case by the bin its coordinate falls in, a
 * boolean by $u \ge 0.5$.
 *
 * @param dim The dimension.
 * @param u All the coordinates of the point.
 * @param at The position of this dimension's first coordinate; advanced past its `dimSize(dim)` coordinates.
 * @returns The dimension's value.
 */
function decodeDim(dim: Dim, u: ArrayLike<number>, at: { i: number }): SpaceValue {
  switch (dim.type) {
    case 'real': {
      const t = Math.min(1, Math.max(0, u[at.i++]))
      const x = dim.scale === 'log' ? dim.min * (dim.max / dim.min) ** t : dim.min + t * (dim.max - dim.min)
      return clampDim(dim, x)
    }
    case 'int':
      return dim.min + bin(u[at.i++], dim.max - dim.min + 1)
    case 'choice':
      return dim.options[bin(u[at.i++], dim.options.length)]
    case 'bool':
      return u[at.i++] >= 0.5
    case 'space':
      return decodeFrom(dim.of, u, at)
    case 'variants': {
      const names = Object.keys(dim.cases)
      const chosen = names[bin(u[at.i++], names.length)]
      let params: SpaceValues = {}
      for (const name of names) {
        const p = decodeFrom(dim.cases[name], u, at)
        if (name === chosen) params = p
      }
      return { case: chosen, params }
    }
  }
}

/**
 * Reads a point of `space` from unit-cube coordinates; every dimension consumes its coordinates, but only the active
 * ones are kept.
 *
 * @param space The space.
 * @param u All the coordinates.
 * @param at The position of the space's first coordinate; advanced past its `encodedSize(space)` coordinates.
 * @returns The point.
 */
function decodeFrom(space: Space, u: ArrayLike<number>, at: { i: number }): SpaceValues {
  const out: Record<string, SpaceValue> = {}
  for (const [key, dim] of Object.entries(space.dims)) {
    const v = decodeDim(dim, u, at)
    if (isDimActive(dim, out)) out[key] = v
  }
  return out
}

/**
 * The point of the space at unit-cube coordinates $\uvec$ (each clipped to $[0, 1]$). Throws `ShapeError` unless
 * there are exactly `encodedSize(space)` coordinates.
 *
 * @param space The space.
 * @param u The coordinates, `encodedSize(space)` numbers in the order of the dimensions.
 * @returns The point; dimensions inactive under it are left out.
 *
 * @example Any point of the cube is a point of the space
 * const s = space({
 *   optimiser: oneOf(['sgd', 'adam']),
 *   lr: real(1e-4, 1, { scale: 'log' }),
 *   momentum: real(0, 0.99, { when: when('optimiser', 'sgd') }),
 * })
 * print(decode(s, [0.1, 0.5, 1]))
 *
 * @example Decoding inverts encoding
 * const s = space({
 *   optimiser: oneOf(['sgd', 'adam']),
 *   lr: real(1e-4, 1, { scale: 'log' }),
 *   momentum: real(0, 0.99, { when: when('optimiser', 'sgd') }),
 * })
 * const x = { optimiser: 'sgd', lr: 0.001, momentum: 0.9 }
 * print(decode(s, encode(s, x)))
 */
export function decode<S extends Space>(space: S, u: ArrayLike<number>): ValuesOf<S>
export function decode(space: Space, u: ArrayLike<number>): SpaceValues
export function decode(space: Space, u: ArrayLike<number>): SpaceValues {
  const m = encodedSize(space)
  if (u.length !== m) throw new ShapeError('decode', `decode: the space has ${m} coordinates, got ${u.length}`)
  return decodeFrom(space, u, { i: 0 })
}

// ── Enumeration and draws ────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The values without repeats (compared by their JSON), keeping the first of each.
 *
 * @param values The values; not modified.
 * @returns A new array of the distinct values, in their first order.
 */
function dedupe(values: SpaceValue[]): SpaceValue[] {
  const seen = new Set<string>()
  return values.filter((v) => {
    const k = JSON.stringify(v)
    if (seen.has(k)) return false
    seen.add(k)
    return true
  })
}

/**
 * The grid values of one dimension: `points` evenly spaced reals or integers (the default alone when `points` is 1;
 * every integer when there are no more than `points`), every option, both booleans, a nested space's grid, and each
 * case's grid for variants.
 *
 * @param dim The dimension.
 * @param points The number of values per real or integer dimension, at least 1.
 * @returns The dimension's grid values, without repeats.
 */
function dimGrid(dim: Dim, points: Size): SpaceValue[] {
  switch (dim.type) {
    case 'real':
    case 'int': {
      if (dim.type === 'int' && dim.max - dim.min + 1 <= points)
        return Array.from({ length: dim.max - dim.min + 1 }, (_, k) => dim.min + k)
      const at = (t: number) => decodeDim(dim, [dim.type === 'int' ? Math.min(t, 1 - 1e-12) : t], { i: 0 })
      return dedupe(points === 1 ? [dim.default] : Array.from({ length: points }, (_, k) => at(k / (points - 1))))
    }
    case 'choice':
      return [...dim.options]
    case 'bool':
      return [false, true]
    case 'space':
      return grid(dim.of, { points })
    case 'variants':
      return Object.entries(dim.cases).flatMap(([name, c]) =>
        grid(c, { points }).map((params): VariantValue => ({ case: name, params })),
      )
  }
}

/**
 * Every combination of the dimensions' grid values, the last dimension varying fastest: `points` evenly spaced values
 * per real (in log space on a log scale; snapped to its step), every integer when there are at most `points` of them
 * (else `points` spread over the range), every option, both booleans, and for `variants` each case's grid in turn. A
 * conditional dimension joins only the combinations where its condition holds. With `points: 1` a real takes only its
 * default.
 *
 * @param space The space.
 * @param options `points`, the number of values per real or integer dimension (default 5): a positive integer, or
 *   `DomainError` is thrown.
 * @returns Every grid point, as a list of points of the space.
 *
 * @example Every combination of an integer and a choice
 * print(grid(space({ k: int(1, 3), act: oneOf(['relu', 'tanh']) })))
 *
 * @example A conditional dimension joins only where it applies
 * const s = space({
 *   optimiser: oneOf(['sgd', 'adam']),
 *   lr: real(1e-4, 1, { scale: 'log' }),
 *   momentum: real(0, 0.99, { when: when('optimiser', 'sgd') }),
 * })
 * const points = grid(s, { points: 3 })
 * print('points:', points.length)
 * print('first:', points[0])
 * print('last:', points[points.length - 1])
 */
export function grid<S extends Space>(space: S, options?: { points?: Size }): ValuesOf<S>[]
export function grid(space: Space, options?: { points?: Size }): SpaceValues[]
export function grid(space: Space, { points = 5 }: { points?: Size } = {}): SpaceValues[] {
  if (!(Number.isInteger(points) && points >= 1)) throw new DomainError('grid', 'grid: points must be an integer ≥ 1')
  let out: Record<string, SpaceValue>[] = [{}]
  for (const [key, dim] of Object.entries(space.dims)) {
    const values = dimGrid(dim, points)
    out = out.flatMap((p) => (isDimActive(dim, p) ? values.map((v) => ({ ...p, [key]: v })) : [p]))
  }
  return out
}

/**
 * A uniform draw from the space: `encodedSize` uniforms from the stream, decoded (so log dimensions are log-uniform,
 * integers and choices uniform over their values).
 *
 * @param s The stream to draw from; the same stream gives the same point.
 * @param space The space.
 * @returns A point of the space.
 *
 * @example Random-search candidates
 * const s = space({
 *   optimiser: oneOf(['sgd', 'adam']),
 *   lr: real(1e-4, 1, { scale: 'log' }),
 *   momentum: real(0, 0.99, { when: when('optimiser', 'sgd') }),
 * })
 * print(sample(stream(0), s))
 * print(sample(stream(1), s))
 */
export function sample<S extends Space>(s: Stream, space: S): ValuesOf<S>
export function sample(s: Stream, space: Space): SpaceValues
export function sample(s: Stream, space: Space): SpaceValues {
  return decode(space, toFlat(uniform(s, 0, 1, { shape: [encodedSize(space)] })))
}

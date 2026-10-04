/**
 * The plain-JSON state behind `useFigureState`, and the pure functions over it: initial values, typed values (with
 * structural sharing, so an unchanged field keeps its identity), setting by path, and the URL form (non-default values
 * only, decoded through aifn's `clampReport`).
 */
import { clampReport, type SpaceValues } from 'aifn-compute/foundation/space'
import {
  coerce,
  initialCase,
  isActive,
  toSpace,
  type LeafDef,
  type ParamDef,
  type ParamDefs,
  type ParamValue,
  type VariantsDef,
} from './schema'
import { parseNumber, validateNumber } from './number'

export type Raw = { readonly [key: string]: RawValue }
/** A variants field: the chosen case, every case's own values (kept when switching away), and the shared values. */
export type RawVariants = {
  readonly case: string
  readonly cases: { readonly [name: string]: Raw }
  readonly shared: Raw
}
export type RawValue = ParamValue | RawVariants | Raw

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null

// ── Initial values and normalisation ─────────────────────────────────────────────────────────────────────────────────

/** Every field at its initial value. */
export function initialRaw(defs: ParamDefs): Raw {
  return normalise(defs, undefined)
}

/** Saved values made valid for `defs`: unknown names dropped, missing or invalid ones reset to their initial value. */
export function normalise(defs: ParamDefs, saved: unknown): Raw {
  const record = isObject(saved) ? saved : {}
  const out: Record<string, RawValue> = {}
  for (const [key, def] of Object.entries(defs)) out[key] = normaliseField(def, record[key])
  return out
}

function normaliseField(def: ParamDef, saved: unknown): RawValue {
  if (def.kind === 'row') return normalise(def.fields, saved)
  if (def.kind === 'variants') {
    const s = isObject(saved) ? saved : {}
    const name = typeof s.case === 'string' && s.case in def.specs ? s.case : initialCase(def)
    const cases = isObject(s.cases) ? s.cases : {}
    return {
      case: name,
      cases: Object.fromEntries(Object.entries(def.specs).map(([k, c]) => [k, normalise(c.params, cases[k])])),
      shared: normalise(def.shared, s.shared),
    }
  }
  return saved === undefined ? coerce(def, def.initial) : coerce(def, saved)
}

// ── Typed values ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Per-field memo of typed values: a field whose raw value and definition signature are unchanged keeps its object. */
export type TypedCache = Map<string, { raw: RawValue; sig: string; value: unknown }>

/** A signature of a definition's data (ranges, options, cases), ignoring labels and closures. */
export function signature(def: ParamDef | ParamDefs): string {
  if ('kind' in def && typeof def.kind === 'string') {
    const d = def as ParamDef
    switch (d.kind) {
      case 'slider':
        return `s${d.min},${d.max},${d.step ?? ''},${d.initial}`
      case 'number':
        return `n${JSON.stringify([d.type, d.gt, d.ge, d.lt, d.le, d.min, d.max, d.step, d.scale, d.initial])}`
      case 'choice':
        return `c${d.options.map((o) => (typeof o === 'object' ? o.value : o)).join('|')}:${d.initial}`
      case 'switch':
        return `b${d.initial}`
      case 'row':
        return `r{${signature(d.fields)}}`
      case 'variants':
        return `v{${Object.entries(d.specs)
          .map(([k, c]) => `${k}:{${signature(c.params)}}`)
          .join(';')}}{${signature(d.shared)}}`
    }
  }
  return Object.entries(def as ParamDefs)
    .map(([k, d]) => `${k}=${signature(d)}`)
    .join(';')
}

/** The typed values of `raw` (see `Values` in schema.ts), reusing cached objects for unchanged fields. */
export function typedValues(defs: ParamDefs, raw: Raw, cache?: TypedCache, prefix = ''): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [key, def] of Object.entries(defs)) {
    const r = raw[key]
    if (def.kind !== 'row' && def.kind !== 'variants') {
      out[key] = coerce(def, r)
      continue
    }
    const sig = cache ? signature(def) : ''
    const hit = cache?.get(prefix + key)
    if (hit && hit.raw === r && hit.sig === sig) {
      out[key] = hit.value
      continue
    }
    const value =
      def.kind === 'row'
        ? typedValues(def.fields, (r as Raw) ?? {}, cache, `${prefix}${key}.`)
        : typedVariant(def, r as RawVariants)
    cache?.set(prefix + key, { raw: r, sig, value })
    out[key] = value
  }
  return out
}

function typedVariant(def: VariantsDef, r: RawVariants | undefined) {
  const v = r ?? (normaliseField(def, undefined) as RawVariants)
  const key = v.case in def.specs ? v.case : initialCase(def)
  const spec = def.specs[key]
  const values = { ...typedValues(spec.params, v.cases[key] ?? {}), ...typedValues(def.shared, v.shared) }
  const run = spec.f as ((x: unknown, p: unknown) => unknown) | undefined
  return { key, label: spec.label, spec, values, params: values, f: run && ((x: unknown) => run(x, values)) }
}

// ── Paths ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A field by path: `x0`, `reveal.jacobian` (in a row), `input.mu` (a parameter of a variants field's chosen case, or a
 * shared one), `input` (the variants field itself, set by case name).
 */
export type Located =
  { kind: 'leaf'; def: LeafDef; raw: RawValue } | { kind: 'variants'; def: VariantsDef; raw: RawVariants } | null

export function locate(defs: ParamDefs, raw: Raw, path: readonly string[]): Located {
  const [head, ...rest] = path
  const def = defs[head]
  if (!def) return null
  const r = raw[head]
  if (def.kind === 'row') return rest.length ? locate(def.fields, (r as Raw) ?? {}, rest) : null
  if (def.kind === 'variants') {
    const v = r as RawVariants
    if (!rest.length) return { kind: 'variants', def, raw: v }
    const own = def.specs[v.case]?.params
    if (own && rest[0] in own) return locate(own, v.cases[v.case] ?? {}, rest)
    return locate(def.shared, v.shared, rest)
  }
  return rest.length ? null : { kind: 'leaf', def, raw: r }
}

/** `raw` with the field at `path` set to `value` (coerced); unchanged (same object) if the path is unknown. */
export function setAt(defs: ParamDefs, raw: Raw, path: readonly string[], value: unknown): Raw {
  const [head, ...rest] = path
  const def = defs[head]
  if (!def) return raw
  const r = raw[head]
  let next: RawValue
  if (def.kind === 'row') {
    if (!rest.length) return raw
    next = setAt(def.fields, (r as Raw) ?? {}, rest, value)
  } else if (def.kind === 'variants') {
    const v = r as RawVariants
    if (!rest.length) {
      if (typeof value !== 'string' || !(value in def.specs) || value === v.case) return raw
      next = { ...v, case: value }
    } else {
      const own = def.specs[v.case]?.params
      if (own && rest[0] in own) {
        const c = setAt(own, v.cases[v.case] ?? {}, rest, value)
        if (c === v.cases[v.case]) return raw
        next = { ...v, cases: { ...v.cases, [v.case]: c } }
      } else {
        const s = setAt(def.shared, v.shared, rest, value)
        if (s === v.shared) return raw
        next = { ...v, shared: s }
      }
    }
  } else {
    // A non-number for a numeric field (an emptied text box) leaves the value as it is.
    if (rest.length || ((def.kind === 'slider' || def.kind === 'number') && !Number.isFinite(value))) return raw
    next = coerce(def, value)
  }
  return Object.is(next, r) ? raw : { ...raw, [head]: next }
}

// ── URL form ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const text = (v: ParamValue) => (typeof v === 'boolean' ? (v ? '1' : '0') : String(v))

/**
 * The non-default values of `raw` as `[path, text]` pairs, e.g. `[['input', 'gamma'], ['input.shape', '2']]`. A
 * variants field writes its case when it is not the initial one, then the chosen case's non-default parameters. Fields
 * whose `when` fails are not written.
 */
export function toEntries(defs: ParamDefs, raw: Raw, prefix = ''): [string, string][] {
  const out: [string, string][] = []
  const flat = flatValues(defs, raw)
  for (const [key, def] of Object.entries(defs)) {
    if (!isActive(def, flat)) continue
    const r = raw[key]
    if (def.kind === 'row') out.push(...toEntries(def.fields, (r as Raw) ?? {}, `${prefix}${key}.`))
    else if (def.kind === 'variants') {
      const v = r as RawVariants
      if (v.case !== initialCase(def)) out.push([prefix + key, v.case])
      out.push(
        ...toEntries(
          { ...def.specs[v.case].params, ...def.shared },
          { ...v.cases[v.case], ...v.shared },
          `${prefix}${key}.`,
        ),
      )
    } else {
      const initial = coerce(def, def.initial)
      if (!Object.is(r, initial)) out.push([prefix + key, text(r as ParamValue)])
    }
  }
  return out
}

/** The leaf values at one level, as a `when` closure or condition reads them. */
export function flatValues(defs: ParamDefs, raw: Raw): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [key, def] of Object.entries(defs)) {
    const r = raw[key]
    out[key] = def.kind === 'variants' ? (r as RawVariants).case : r
  }
  return out
}

/** True when every field is at its initial value. */
export const isInitial = (defs: ParamDefs, raw: Raw) => toEntries(defs, raw).length === 0

/** What `fromEntries` made of a URL: the state, and the keys it could not use. */
export type Decoded = { raw: Raw; dropped: string[] }

/**
 * State from `[path, text]` pairs (the inverse of `toEntries`): texts are parsed by each field's type, then the whole
 * point is clamped into the schema's aifn `Space` by `clampReport`, which lists unknown and inactive keys. Values that
 * do not parse are listed too, as are typed numbers that break their type or a bound (a number field is validated,
 * not clamped).
 */
export function fromEntries(defs: ParamDefs, entries: readonly (readonly [string, string])[]): Decoded {
  const dropped: string[] = []
  const nested = buildNested(defs, entries, dropped)
  const report = clampReport(toSpace(defs), nested)
  dropped.push(...report.dropped)
  return { raw: fromSpaceValues(defs, report.values, initialRaw(defs)), dropped }
}

/** Parsed values in the Space's value shape (variants as `{ case, params }`); unknown keys are kept as text. */
function buildNested(defs: ParamDefs, entries: readonly (readonly [string, string])[], dropped: string[]) {
  const root: Record<string, unknown> = {}
  for (const [path, value] of entries) {
    const parts = path.split('.')
    let level = defs as ParamDefs | undefined
    let node = root
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]
      const def = level?.[part]
      const last = i === parts.length - 1
      if (!def) {
        // Unknown: keep it so that clampReport reports its full path.
        if (last) node[part] = value
        else node = (node[part] ??= {}) as Record<string, unknown>
        level = undefined
        continue
      }
      if (def.kind === 'row') {
        node = (node[part] ??= {}) as Record<string, unknown>
        level = def.fields
        if (last) dropped.push(path)
      } else if (def.kind === 'variants') {
        const v = (node[part] ??= { params: {} }) as { case?: string; params: Record<string, unknown> }
        if (last) v.case = value
        node = v.params
        // Parameters of the case named in the URL (or the initial case), and the shared ones.
        const name = v.case && v.case in def.specs ? v.case : initialCase(def)
        level = { ...def.specs[name].params, ...def.shared }
      } else if (!last) {
        node[part] = value
        level = undefined
        break
      } else {
        const parsed = parseLeaf(def, value)
        if (parsed === undefined) dropped.push(path)
        else node[part] = parsed
      }
    }
  }
  return root
}

function parseLeaf(def: LeafDef, value: string): ParamValue | undefined {
  switch (def.kind) {
    case 'slider': {
      const x = Number(value)
      return value.trim() !== '' && Number.isFinite(x) ? x : undefined
    }
    case 'number': {
      // A typed field is validated, not clamped: a value that breaks its type or a bound is dropped and reported.
      const x = parseNumber(value)
      return x !== null && validateNumber(def, x) === null ? x : undefined
    }
    case 'choice': {
      const v = coerce(def, value)
      return String(v) === value ? v : undefined
    }
    case 'switch':
      return value === '1' || value === 'true' ? true : value === '0' || value === 'false' ? false : undefined
  }
}

/** Space values (as `clampReport` returns them) laid over `base`, so that unchosen cases keep their values. */
function fromSpaceValues(defs: ParamDefs, values: SpaceValues, base: Raw): Raw {
  let out = base
  for (const [key, def] of Object.entries(defs)) {
    const v = values[key]
    if (v === undefined) continue
    if (def.kind === 'row') out = { ...out, [key]: fromSpaceValues(def.fields, v as SpaceValues, out[key] as Raw) }
    else if (def.kind === 'variants') {
      const given = v as { case: string; params: SpaceValues }
      const b = out[key] as RawVariants
      const own = def.specs[given.case].params
      const ownValues = Object.fromEntries(Object.entries(given.params).filter(([k]) => k in own))
      const sharedValues = Object.fromEntries(Object.entries(given.params).filter(([k]) => k in def.shared))
      out = {
        ...out,
        [key]: {
          case: given.case,
          cases: { ...b.cases, [given.case]: fromSpaceValues(own, ownValues, b.cases[given.case]) },
          shared: fromSpaceValues(def.shared, sharedValues, b.shared),
        },
      }
    } else out = { ...out, [key]: coerce(def, v) }
  }
  return out
}

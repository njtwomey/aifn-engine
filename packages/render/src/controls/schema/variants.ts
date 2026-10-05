/**
 * Variants: a figure offers several functions (or models, or methods), each with its own parameters. The reader picks
 * one and the controls change to its parameters. Built on figure state (`aifn-render/state`): a variant set is a `variants`
 * field, and `useVariants` is `useFigureState` with that one field.
 *
 *   const functions = defineVariants({
 *     quadratic: { label: 'Quadratic', params: { a: slider(-2, 2, 1) }, f: (x: Value, p) => mul(p.a, square(x)) },
 *     softplus: { label: 'Softplus', params: { beta: slider(0.1, 5, 1) }, f: (x: Value, p) => ... },
 *   })
 *   const v = useVariants(functions)   // v.key, v.params (narrowed by v.key), v.f (bound to v.params), v.set, v.state
 *   <VariantControls variants={v} />   // the choice of variant, then its controls
 *
 * Each variant keeps its own values, so switching away and back restores them. Parameters in `shared` belong to every
 * variant and keep one value. The whole state is plain JSON (`v.state`, `v.setState`). In a figure, prefer a
 * `variants(...)` field of `useFigureState`, which adds the row, reset and URL state.
 */
import { useMemo, useState } from 'react'
import type { AnyValues, CaseDef, ParamDefs, ParamValue, VariantsDef, VariantValue } from '../../state/schema'
import { useFigureState } from '../../state/useFigureState'
import type { RawVariants } from '../../state/store'

/** One variant: a label, its parameters, and optionally a function of an input and its values. */
export type VariantSpec<
  P extends ParamDefs = ParamDefs,
  S extends ParamDefs = ParamDefs,
  X = never,
  R = unknown,
> = CaseDef<P, S, X, R>

/** Declared variants and shared parameters: a `variants` field of figure state. */
export type Variants<V extends Record<string, ParamDefs>, S extends ParamDefs, X, R> = VariantsDef<V, S, X, R>

/**
 * Declare variants. Every entry's `f` (optional; one signature for the family, e.g. `(x: Value, p) => Value`)
 * receives the entry's own values plus the shared ones, typed. Annotate `x`; put `params` before `f`.
 */
export function defineVariants<
  const V extends Record<string, ParamDefs>,
  X = never,
  R = unknown,
  const S extends ParamDefs = Record<never, never>,
>(specs: { [K in keyof V]: VariantSpec<V[K], S, X, R> }, shared?: S): Variants<V, S, X, R> {
  return { kind: 'variants', specs, shared: (shared ?? {}) as S }
}

/** The plain-JSON state of a variant set: the chosen key, each variant's values, and the shared values. */
export type VariantsJson = { key: string; values: Record<string, Record<string, ParamValue>>; shared: AnyValues }

export type VariantsControl<V extends Record<string, ParamDefs>, S extends ParamDefs, X, R> = VariantValue<
  V,
  S,
  X,
  R
> & {
  variants: Variants<V, S, X, R>
  /** The chosen variant's values and the shared ones, as a plain record (for controls and `when`). */
  values: AnyValues
  setKey: (key: keyof V & string) => void
  /** Set one parameter of the chosen variant (or a shared one), clamped and snapped to its definition. */
  set: (name: string, value: ParamValue) => void
  state: VariantsJson
  /** Restore a saved state; anything invalid falls back to its initial value. */
  setState: (state: unknown) => void
}

const toRaw = (saved: unknown) => {
  const s = (typeof saved === 'object' && saved !== null ? saved : {}) as Partial<VariantsJson>
  return { v: { case: s.key, cases: s.values ?? {}, shared: s.shared ?? {} } }
}

/** The state of a variant set: which is chosen, and every variant's values. */
export function useVariants<V extends Record<string, ParamDefs>, S extends ParamDefs, X, R>(
  variants: Variants<V, S, X, R>,
  initial?: { key?: keyof V & string; state?: unknown },
): VariantsControl<V, S, X, R> {
  // The initial key is read once, as the initial state is.
  const [field] = useState(() => (initial?.key ? { ...variants, initial: initial.key } : variants))
  const fs = useFigureState({ v: field }, { initial: initial?.state === undefined ? undefined : toRaw(initial.state) })
  const chosen = fs.v
  const raw = fs.json.v as RawVariants
  const { set: setPath, setJson } = fs
  return useMemo(() => {
    const state: VariantsJson = {
      key: raw.case,
      values: raw.cases as VariantsJson['values'],
      shared: raw.shared,
    }
    const control = {
      ...(chosen as object),
      variants,
      setKey: (key: string) => setPath('v', key),
      set: (name: string, value: ParamValue) => setPath(`v.${name}`, value),
      state,
      setState: (saved: unknown) => setJson(toRaw(saved)),
    }
    return control as unknown as VariantsControl<V, S, X, R>
  }, [chosen, raw, variants, setPath, setJson])
}

/**
 * A plain conditional control set, without variants: typed values, a clamping setter, and JSON state. Render it with
 * `<ParamControls {...p} />`. In a figure, prefer `useFigureState`, which adds rows, reset and URL state.
 */
export function useParams<P extends ParamDefs>(defs: P, saved?: unknown) {
  const fs = useFigureState(defs, { initial: saved })
  const { values, set, json, setJson } = fs
  return useMemo(
    () => ({
      defs,
      values,
      set: (name: string, value: ParamValue) => set(name, value),
      state: json,
      setState: setJson,
    }),
    [defs, values, set, json, setJson],
  )
}

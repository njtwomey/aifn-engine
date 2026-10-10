/**
 * The prelude: the names a program can call, as data, with the functions that build, extend and look them up.
 *
 * Entries live in namespaces that mirror aifn's module tree (`math.sin`, `random.normal`, `linalg.solve`), so a name
 * says what it is and where it comes from; a few (`seed`, `print`) sit at the top level. Each entry carries its
 * namespace, name, parameters, a one-line doc, its source module and its implementation, so the same data drives
 * evaluation, an editor's completion and hover docs, and a reference list. `withPrelude` extends a prelude
 * (applications add `learn`; a page adds its own helpers). Preludes are frozen: extending one makes a new one.
 */
import type { Stream } from 'aifn-compute/foundation/random'

/**
 * One parameter of an entry: its `name`; when optional, its `default` as source text (`'0'`, `'"full"'`); and `rest`,
 * true for a rest parameter (`...values`).
 */
export type Param = { readonly name: string; readonly default?: string; readonly rest?: boolean }

/** What an entry's implementation can reach while a program runs. */
export type Context = {
  /** A fresh stream for one random call: the run's root stream keyed by the call's position in call order. */
  draw(): Stream
  /** Restart the random draws from seed `s` (the program's `seed(s)`). */
  seed(s: number | string): void
  /** Record a line of output (the program's `print`). */
  log(line: string): void
}

/** A prelude entry. */
export type PreludeEntry = {
  /** Its namespace (`random`), or null for a top-level name. */
  readonly namespace: string | null
  /** Its name within the namespace (`normal`). */
  readonly name: string
  /** Its parameters, in order, for the signature and completion. */
  readonly params: readonly Param[]
  /** One sentence. */
  readonly doc: string
  /** The aifn module it wraps (`aifn-compute/numerics/linalg`), or the interpreter's own for helpers. */
  readonly source: string
  /** What it returns, in words (`number[]`, `number`, `{ weights, intercept }`). */
  readonly returns?: string
  /** The implementation; `ctx` is supplied by the run, the rest are the program's arguments. */
  impl(ctx: Context, ...args: unknown[]): unknown
}

/**
 * A namespace's description: its `name` (`random`), a one-sentence `doc`, and the `source` module (or modules, comma
 * separated) its entries wrap.
 */
export type Namespace = { readonly name: string; readonly doc: string; readonly source: string }

/** A prelude: its `namespaces`, and its `entries` with unique qualified names, in listing order. */
export type Prelude = { readonly namespaces: readonly Namespace[]; readonly entries: readonly PreludeEntry[] }

/**
 * The qualified name of an entry: `random.normal`, or `seed` at the top level.
 *
 * @param e The entry.
 * @returns `namespace.name`, or the bare name when the entry has no namespace.
 *
 * @example Namespaced and top-level names
 * print(qualified(lookup(corePrelude, 'random.normal')), qualified(lookup(corePrelude, 'seed')))
 */
export const qualified = (e: PreludeEntry): string => (e.namespace === null ? e.name : `${e.namespace}.${e.name}`)

/**
 * A prelude from namespaces and entries, frozen. Nothing is checked here: an entry may sit in a namespace declared by
 * the prelude it extends, and `withPrelude` checks the namespaces when they are combined.
 *
 * @param namespaces The namespaces it declares (copied).
 * @param entries Its entries, in listing order (copied).
 * @returns The prelude.
 *
 * @example A one-entry namespace added to the core prelude
 * const double = { namespace: 'my', name: 'double', params: params('x'), doc: 'Twice x.', source: 'page' }
 * const extra = makePrelude(
 *   [{ name: 'my', doc: 'My helpers.', source: 'page' }],
 *   [{ ...double, impl: (_ctx, x) => 2 * x }],
 * )
 * const p = withPrelude(corePrelude, extra)
 * print(signature(lookup(p, 'my.double')), '→', runProgram('return my.double(21)', { prelude: p }).value)
 */
export function makePrelude(namespaces: readonly Namespace[], entries: readonly PreludeEntry[]): Prelude {
  return Object.freeze({ namespaces: Object.freeze([...namespaces]), entries: Object.freeze([...entries]) })
}

/**
 * The namespaces and entries of `base` and then each extra prelude; a later one replaces an earlier by name (a
 * namespace by its name, an entry by its qualified name), keeping the earlier one's place in the listing. Throws when
 * an entry's namespace is declared by none of them.
 *
 * @param base The prelude extended (usually `corePrelude`).
 * @param extra The preludes added, in order; each may replace names of those before it.
 * @returns The combined prelude.
 *
 * @example Replace a core entry
 * const shout = { namespace: null, name: 'print', params: params('...values'), doc: 'Shout.', source: 'page' }
 * const loud = makePrelude([], [{ ...shout, impl: (ctx, ...v) => ctx.log(v.join(' ').toUpperCase()) }])
 * print(runProgram('print("hello")', { prelude: withPrelude(corePrelude, loud) }).output)
 *
 * @example An entry in an undeclared namespace is refused
 * const stray = makePrelude([], [{ namespace: 'nowhere', name: 'f', params: [], doc: '', source: '', impl: () => 0 }])
 * try {
 *   withPrelude(corePrelude, stray)
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function withPrelude(base: Prelude, ...extra: readonly Prelude[]): Prelude {
  const spaces = new Map<string, Namespace>()
  const byName = new Map<string, PreludeEntry>()
  for (const p of [base, ...extra]) {
    for (const n of p.namespaces) spaces.set(n.name, n)
    for (const e of p.entries) byName.set(qualified(e), e)
  }
  for (const e of byName.values())
    if (e.namespace !== null && !spaces.has(e.namespace))
      throw new Error(`withPrelude: ${qualified(e)} is in an undeclared namespace '${e.namespace}'`)
  return makePrelude([...spaces.values()], [...byName.values()])
}

/**
 * The entry with this qualified name (`random.normal`, `seed`), if any.
 *
 * @param p The prelude searched.
 * @param name The qualified name.
 * @returns The entry, or undefined when the prelude has none of that name.
 *
 * @example Look an entry up for its docs
 * const e = lookup(corePrelude, 'linalg.solve')
 * print(signature(e), '-', e.doc)
 * print('missing:', lookup(corePrelude, 'linalg.nothing'))
 */
export function lookup(p: Prelude, name: string): PreludeEntry | undefined {
  return p.entries.find((e) => qualified(e) === name)
}

/**
 * The entries of one namespace, sorted by name.
 *
 * @param p The prelude.
 * @param namespace The namespace's name, or null for the top-level entries.
 * @returns A new array of the entries, in alphabetical order of their names.
 *
 * @example The top level and a namespace
 * print(members(corePrelude, null).map((e) => e.name))
 * print(members(corePrelude, 'stats').map((e) => e.name))
 */
export function members(p: Prelude, namespace: string | null): PreludeEntry[] {
  // Alphabetical, so the reference and the completion list read the same way.
  return p.entries.filter((e) => e.namespace === namespace).sort((a, b) => a.name.localeCompare(b.name))
}

/**
 * The call signature as source text: `random.normal(n, mean = 0, sd = 1)`.
 *
 * @param entry The entry.
 * @returns Its qualified name and parameters, defaults written as `name = default` and a rest parameter as `...name`.
 *
 * @example Signatures for a reference list
 * for (const name of ['random.normal', 'print', 'array.linspace']) print(signature(lookup(corePrelude, name)))
 */
export function signature(entry: PreludeEntry): string {
  const ps = entry.params.map((p) =>
    p.rest ? `...${p.name}` : p.default === undefined ? p.name : `${p.name} = ${p.default}`,
  )
  return `${qualified(entry)}(${ps.join(', ')})`
}

/**
 * Parameters from a compact spec: `'n, mean = 0, sd = 1'` (a leading `...` marks a rest parameter). The spec is
 * split at every comma and `=`, so a default may not itself contain one.
 *
 * @param spec The parameters as source text, comma separated; empty or blank for none.
 * @returns One `Param` per comma-separated part, in order.
 *
 * @example A spec with a default and a rest parameter
 * print(params('x, axis = 0'))
 * print(params('...values'))
 */
export function params(spec: string): Param[] {
  if (spec.trim() === '') return []
  return spec.split(',').map((part) => {
    const [name, value] = part.split('=').map((s) => s.trim())
    if (name.startsWith('...')) return { name: name.slice(3), rest: true }
    return value === undefined ? { name } : { name, default: value }
  })
}

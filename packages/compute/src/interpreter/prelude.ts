/**
 * The prelude: the names a program can call. Entries live in namespaces that mirror aifn's module tree (`math.sin`,
 * `random.normal`, `linalg.solve`), so a name says what it is and where it comes from; a few (`seed`, `print`) sit at
 * the top level. Each entry carries its namespace, name, parameters, a one-line doc, its source module and its
 * implementation, so the same data drives evaluation, an editor's completion and hover docs, and a reference list.
 * `withPrelude` extends a prelude (applications add `learn`; a page adds its own helpers).
 */
import type { Stream } from 'aifn-compute/foundation/random'

/** One parameter of an entry: its name and, when optional, its default as source text. */
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
  readonly name: string
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

/** A namespace's description. */
export type Namespace = { readonly name: string; readonly doc: string; readonly source: string }

/** A prelude: namespaces, and entries with unique qualified names, in listing order. */
export type Prelude = { readonly namespaces: readonly Namespace[]; readonly entries: readonly PreludeEntry[] }

/** The qualified name of an entry: `random.normal`, or `seed` at the top level. */
export const qualified = (e: PreludeEntry): string => (e.namespace === null ? e.name : `${e.namespace}.${e.name}`)

/** A prelude from namespaces and entries. An entry may sit in a namespace declared by the prelude it extends. */
export function makePrelude(namespaces: readonly Namespace[], entries: readonly PreludeEntry[]): Prelude {
  return Object.freeze({ namespaces: Object.freeze([...namespaces]), entries: Object.freeze([...entries]) })
}

/**
 * The namespaces and entries of `base` and then each extra prelude; a later one replaces an earlier by name. Throws
 * when an entry's namespace is declared by none of them.
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

/** The entry with this qualified name (`random.normal`, `seed`), if any. */
export function lookup(p: Prelude, name: string): PreludeEntry | undefined {
  return p.entries.find((e) => qualified(e) === name)
}

/** The entries of one namespace (null: the top level). */
export function members(p: Prelude, namespace: string | null): PreludeEntry[] {
  // Alphabetical, so the reference and the completion list read the same way.
  return p.entries.filter((e) => e.namespace === namespace).sort((a, b) => a.name.localeCompare(b.name))
}

/** The call signature as source text: `random.normal(n, mean = 0, sd = 1)`. */
export function signature(entry: PreludeEntry): string {
  const ps = entry.params.map((p) =>
    p.rest ? `...${p.name}` : p.default === undefined ? p.name : `${p.name} = ${p.default}`,
  )
  return `${qualified(entry)}(${ps.join(', ')})`
}

/** Parameters from a compact spec: `'n, mean = 0, sd = 1'` (a leading `...` marks a rest parameter). */
export function params(spec: string): Param[] {
  if (spec.trim() === '') return []
  return spec.split(',').map((part) => {
    const [name, value] = part.split('=').map((s) => s.trim())
    if (name.startsWith('...')) return { name: name.slice(3), rest: true }
    return value === undefined ? { name } : { name, default: value }
  })
}

/**
 * `aifn-compute/foundation/registry`: one pattern for every named entry of aifn (design S §3.1), metadata attached to
 * the value it describes.
 *
 * - Defining entries: `define(info, value)` attaches a frozen `info` to a value and returns the value; `definer(kind,
 *   module)` makes a `define` for one module's entries of one kind. A value is defined once: a second `define` throws.
 * - Reading them: `isEntry` tests a value for metadata (of a kind), and `entries(kind, ...namespaces)` collects the
 *   entries of one kind from a module's namespaces into a frozen table keyed by `info.key`.
 * - The metadata types (`Info`, `Entry`, and the `Info` of each kind) are those of `aifn-compute/foundation/contracts`,
 *   re-exported.
 *
 * Registries are static and per module: a module's `registry.ts` calls `entries` on its own namespaces.
 */

import { AifnError, DomainError } from 'aifn-compute/foundation/errors'
import type {
  AlgorithmInfo,
  AlgorithmProblem,
  BijectorInfo,
  DistributionInfo,
  Entry,
  EntryKind,
  FilterDesignInfo,
  FunctionInfo,
  TestData,
  TestInfo,
  FunctionRole,
  Info,
  KernelInfo,
  KlRuleInfo,
  LikelihoodInfo,
  LinkInfo,
  PrimitiveInfo,
  Stability,
  StateRoles,
  StatusFlag,
  SupportName,
  WaveletInfo,
  WindowInfo,
} from 'aifn-compute/foundation/contracts'

export type {
  AlgorithmInfo,
  AlgorithmProblem,
  BijectorInfo,
  DistributionInfo,
  Entry,
  EntryKind,
  FilterDesignInfo,
  FunctionInfo,
  TestData,
  TestInfo,
  FunctionRole,
  Info,
  KernelInfo,
  KlRuleInfo,
  LikelihoodInfo,
  LinkInfo,
  PrimitiveInfo,
  Stability,
  StateRoles,
  StatusFlag,
  SupportName,
  WaveletInfo,
  WindowInfo,
}

/**
 * Attach `info` to `value` (the value itself is returned, with a frozen, non-writable `info` added). A value defined
 * twice throws `DomainError`: a second `define` would silently replace the first entry's info (and its key in
 * `entries`).
 *
 * @param info The entry's metadata: `key`, `kind`, `module`, `name`, `stability` and the optional fields of its kind.
 *   A frozen shallow copy is attached, so later changes to the object passed are not seen.
 * @param value The function or object to register; it gains an enumerable `info` property and is otherwise unchanged.
 * @returns `value` itself, typed as an entry of `info`'s kind.
 *
 * @example Register a function
 * const square = define(
 *   { key: 'square', kind: 'function', module: 'demo', name: 'Square', stability: 'stable' },
 *   (x) => x * x,
 * )
 * print('square(3) =', square(3))
 * print('info =', square.info)
 * print('info frozen:', Object.isFrozen(square.info))
 *
 * @example A value is defined once
 * const f = define({ key: 'f', kind: 'function', module: 'demo', name: 'F', stability: 'stable' }, () => 1)
 * try {
 *   define({ key: 'g', kind: 'function', module: 'demo', name: 'G', stability: 'stable' }, f)
 * } catch (e) {
 *   print(e.name, ':', e.message)
 * }
 */
export function define<T extends object, I extends Info>(info: I, value: T): Entry<T, I> {
  if (Object.prototype.hasOwnProperty.call(value, 'info')) {
    const prior = (value as { info?: { kind?: unknown; key?: unknown } }).info
    throw new DomainError(
      'define',
      `define: '${String(info.key)}' (${info.kind}) is already defined as '${String(prior?.key)}' (${String(prior?.kind)})`,
    )
  }
  Object.defineProperty(value, 'info', { value: Object.freeze({ ...info }), enumerable: true, writable: false })
  return value as Entry<T, I>
}

/** What a definition states: its info without `kind` and `module` (fixed by the definer), `stability` optional. */
export type Spec<I extends Info> = Omit<I, 'kind' | 'module' | 'stability'> & { readonly stability?: Stability }

/**
 * A `define` for one kind and module, so a module's registry states only what differs per entry: `stability` defaults
 * to `experimental`. In TypeScript the kind's `Info` is the type argument, e.g. `definer<WindowInfo>('window',
 * 'signal/windows')`, so each spec is checked against it.
 *
 * @param kind The kind given to every entry it defines.
 * @param module The module given to every entry it defines, e.g. `'signal/windows'`.
 * @returns A function of a spec (the info without `kind` and `module`, `stability` optional) and a value, which calls
 *   `define` with the kind and module filled in.
 *
 * @example One definer for a module's entries
 * const fn = definer('function', 'demo')
 * const double = fn({ key: 'double', name: 'Double' }, (x) => 2 * x)
 * const half = fn({ key: 'half', name: 'Half', stability: 'stable' }, (x) => x / 2)
 * print('double(5) =', double(5))
 * print('double:', double.info.module, double.info.stability)
 * print('half:', half.info.module, half.info.stability)
 */
export function definer<I extends Info>(
  kind: I['kind'],
  module: string,
): <T extends object>(spec: Spec<I>, value: T) => Entry<T, I> {
  return (spec, value) => define({ stability: 'experimental', ...spec, kind, module } as unknown as I, value)
}

/**
 * True when `x` carries registry metadata of `kind` (any kind when omitted): a function or object with an `info`
 * object whose `kind` is a string.
 *
 * @param x The value to test; anything.
 * @param kind The kind the entry must have; left out, an entry of any kind passes.
 * @returns Whether `x` is an entry (of `kind`), narrowing its type.
 *
 * @example Entries and plain values
 * const double = definer('function', 'demo')({ key: 'double', name: 'Double' }, (x) => 2 * x)
 * print('an entry:', isEntry(double))
 * print('a function entry:', isEntry(double, 'function'))
 * print('a kernel entry:', isEntry(double, 'kernel'))
 * print('a plain function:', isEntry((x) => 2 * x))
 */
export function isEntry<I extends Info = Info>(x: unknown, kind?: I['kind']): x is Entry<unknown, I> {
  if ((typeof x !== 'function' && typeof x !== 'object') || x === null || !('info' in x)) return false
  const info = (x as { info?: unknown }).info
  if (typeof info !== 'object' || info === null) return false
  const k = (info as { kind?: unknown }).kind
  return typeof k === 'string' && (kind === undefined || k === kind)
}

/**
 * The entries of `kind` among the values of `namespaces` (module namespaces or plain objects), keyed by `info.key` in
 * the order the namespaces list their values (`Object.values`, namespace by namespace). A key defined by two different
 * values throws `AifnError`, so each entry is defined once; the same value reached through two namespaces is kept once.
 *
 * @param kind The kind to collect; values that are not entries, or are entries of another kind, are skipped.
 * @param namespaces The objects whose own enumerable values are scanned, in order (typically `import * as` namespaces).
 * @returns A frozen table from each entry's `info.key` to the entry.
 *
 * @example Collect a module's entries
 * const fn = definer('function', 'demo')
 * const ns = {
 *   double: fn({ key: 'double', name: 'Double' }, (x) => 2 * x),
 *   half: fn({ key: 'half', name: 'Half' }, (x) => x / 2),
 *   helper: (x) => x + 1,
 * }
 * const table = entries('function', ns)
 * print('keys =', Object.keys(table))
 * print('table.half(8) =', table.half(8))
 * print('kernels =', Object.keys(entries('kernel', ns)))
 */
export function entries<I extends Info>(
  kind: I['kind'],
  ...namespaces: readonly object[]
): Readonly<Record<string, Entry<unknown, I>>> {
  const out: Record<string, Entry<unknown, I>> = {}
  for (const ns of namespaces)
    for (const value of Object.values(ns)) {
      if (!isEntry<I>(value, kind)) continue
      const known = out[value.info.key]
      if (known !== undefined && known !== value) {
        throw new AifnError(
          'registry',
          `registry: two ${kind} entries are keyed '${value.info.key}' (each entry is defined once)`,
        )
      }
      out[value.info.key] = value
    }
  return Object.freeze(out)
}

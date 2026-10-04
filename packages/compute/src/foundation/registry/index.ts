/**
 * `aifn-compute/foundation/registry`: one pattern for every named entry of aifn (design S §3.1): `define(info, value)`
 * attaches metadata to a value and returns it; `entries(kind, ...namespaces)` collects the entries of one kind from a
 * module's namespaces into a frozen table keyed by `info.key`; `isEntry`. Registries are static and per module.
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
 * twice throws: a second `define` would silently replace the first entry's info (and its key in `entries`).
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
 * to `experimental`.
 *
 * ```ts
 * const window = definer<WindowInfo>('window', 'signal/windows')
 * export const hann = window({ key: 'hann', name: 'Hann', … }, (n: Size) => getWindow('hann', n))
 * ```
 */
export function definer<I extends Info>(
  kind: I['kind'],
  module: string,
): <T extends object>(spec: Spec<I>, value: T) => Entry<T, I> {
  return (spec, value) => define({ stability: 'experimental', ...spec, kind, module } as unknown as I, value)
}

/** True when `x` carries registry metadata of `kind` (any kind when omitted). */
export function isEntry<I extends Info = Info>(x: unknown, kind?: I['kind']): x is Entry<unknown, I> {
  if ((typeof x !== 'function' && typeof x !== 'object') || x === null || !('info' in x)) return false
  const info = (x as { info?: unknown }).info
  if (typeof info !== 'object' || info === null) return false
  const k = (info as { kind?: unknown }).kind
  return typeof k === 'string' && (kind === undefined || k === kind)
}

/**
 * The entries of `kind` among the values of `namespaces` (module namespaces or plain objects), keyed by `info.key` in
 * definition order. A key defined by two different values throws, so each entry is defined once.
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

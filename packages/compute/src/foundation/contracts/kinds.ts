/**
 * The `kind` brand: every displayable protocol object carries a string literal naming what it is, so a view registry
 * (design S §4.2) can pick a view by `kind` without guessing from field names.
 */

/** The kinds of displayable objects, one per protocol that has a view. */
export type ObjectKind =
  | 'distribution'
  | 'log-density'
  | 'model'
  | 'dataset'
  | 'trace'
  | 'graph'
  | 'tree'
  | 'signal'
  | 'spectrum'
  | 'time-frequency'
  | 'lti'
  | 'decomposition'
  | 'curve'
  | 'objective'
  | 'kernel'

/** An object branded with its kind. */
export interface Kinded<K extends ObjectKind = ObjectKind> {
  /** What the object is, which picks its view. */
  readonly kind: K
}

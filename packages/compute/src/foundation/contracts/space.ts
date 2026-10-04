/**
 * Parameter spaces (design S §2.4): one serialisable schema for hyperparameter search, recipe knobs, distribution
 * family parameters, kernel and estimator hyperparameters, figure controls and URL state. Pure data, no closures, so a
 * space can be sent to a worker, written to the catalog and read by a check.
 */

/** One dimension of a space. */
export type Dim =
  | {
      readonly type: 'real'
      readonly min: number
      readonly max: number
      readonly default: number
      readonly scale?: 'linear' | 'log'
      readonly step?: number
      readonly unit?: string
    }
  | { readonly type: 'int'; readonly min: number; readonly max: number; readonly default: number }
  | { readonly type: 'choice'; readonly options: readonly (string | number)[]; readonly default: string | number }
  | { readonly type: 'bool'; readonly default: boolean }
  /** Nested, e.g. a kernel's sub-kernels. */
  | { readonly type: 'space'; readonly of: Space }
  /** A family choice with its own parameters per case. */
  | { readonly type: 'variants'; readonly cases: Readonly<Record<string, Space>>; readonly default: string }

/** A dimension applies only when another dimension has a given value. Data, not a closure, so it serialises. */
export type Condition = { readonly key: string; readonly equals: string | number | boolean }

/** A dimension with its presentation-neutral documentation. */
export type DimSpec = Dim & {
  /** A label; TeX allowed. */
  readonly label?: string
  readonly doc?: string
  readonly when?: Condition
}

/** A parameter space: named dimensions. */
export interface Space {
  readonly dims: Readonly<Record<string, DimSpec>>
}

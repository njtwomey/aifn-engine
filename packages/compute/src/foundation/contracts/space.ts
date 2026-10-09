/**
 * Parameter spaces (design S §2.4): one serialisable schema for hyperparameter search, recipe knobs, distribution
 * family parameters, kernel and estimator hyperparameters, figure controls and URL state. Pure data, no closures, so a
 * space can be sent to a worker, written to the catalog and read by a check.
 */

/** One dimension of a space, told apart by `type`. */
export type Dim =
  /** A real number in a range. */
  | {
      /** The brand of a real dimension. */
      readonly type: 'real'
      /** The lower end of the range. */
      readonly min: number
      /** The upper end of the range, at least `min`. */
      readonly max: number
      /** The value when none is given. */
      readonly default: number
      /**
       * How grids and draws are spaced: evenly in $x$ (`linear`, the default) or in $\log x$ (`log`, which needs
       * $\text{min} > 0$).
       */
      readonly scale?: 'linear' | 'log'
      /** When set, values snap to $\text{min} + k \cdot \text{step}$. */
      readonly step?: number
      /** The unit of the value, for display. */
      readonly unit?: string
    }
  /** An integer from `min` to `max` inclusive, with a `default`. */
  | { readonly type: 'int'; readonly min: number; readonly max: number; readonly default: number }
  /** One of the listed `options` (strings or numbers), with a `default` among them. */
  | { readonly type: 'choice'; readonly options: readonly (string | number)[]; readonly default: string | number }
  /** True or false, with a `default`. */
  | { readonly type: 'bool'; readonly default: boolean }
  /** Nested, e.g. a kernel's sub-kernels: the value is a point of the space `of`. */
  | { readonly type: 'space'; readonly of: Space }
  /**
   * A family choice with its own parameters per case: the value is a case name of `cases` with a point of that case's
   * space, and `default` names the case taken when none is given.
   */
  | { readonly type: 'variants'; readonly cases: Readonly<Record<string, Space>>; readonly default: string }

/**
 * A dimension applies only when another dimension has a given value: the dimension named `key` must equal `equals`
 * (a case name, for a variants dimension). Data, not a closure, so it serialises.
 */
export type Condition = { readonly key: string; readonly equals: string | number | boolean }

/** A dimension with its presentation-neutral documentation. */
export type DimSpec = Dim & {
  /** A label; TeX allowed. */
  readonly label?: string
  /** A longer description, for a tooltip or the catalog. */
  readonly doc?: string
  /** The condition under which alone the dimension applies; it always applies when absent. */
  readonly when?: Condition
}

/** A parameter space: named dimensions. */
export interface Space {
  /** The dimensions by key, in order (the order of encoding and of grids). */
  readonly dims: Readonly<Record<string, DimSpec>>
}

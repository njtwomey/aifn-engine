/**
 * Draggable handles: bind a parameter to its place on a chart.
 *
 * A handle has a position in data coordinates and an `onDrag` setter. A `Plot` takes them as `Handle` layers; `EChart` accepts `handles`.
 * Pointer events only: charts never emit when their props change, so a handle and a slider bound to the same parameter
 * cannot trigger each other. Bind handles to things that have a natural place on the chart (a centroid, a threshold, a
 * start point), not to distribution parameters.
 */
import { HandleLayer } from './plot/layers/marks'

export type Vec2 = [number, number]

type Placement =
  /** A point to drag in two dimensions, e.g. a start position or a pair of parameters. */
  | {
      kind: 'point'
      at: Vec2
      onDrag: (p: Vec2) => void
      /** The ink marker's shape (an ECharts symbol such as a `MARKER_SHAPES` entry; default `circle`), to tell kinds of
       * point apart (poles and zeros). */
      symbol?: string
    }
  /** A vertical guide line; dragging sets its x value, e.g. a rank, a threshold or an iteration. */
  | { kind: 'x'; at: number; onDrag: (x: number) => void }
  /** A horizontal guide line; dragging sets its y value, e.g. a probability on a cdf. */
  | { kind: 'y'; at: number; onDrag: (y: number) => void }

export type Handle = Placement & {
  label?: string
  /** The marker's or guide line's colour (a resolved CSS colour); ink by default. A `Handle` layer sets it from `slot`. */
  color?: string
  /** Called once when the pointer lets go, e.g. to start work too slow to redo on every move. */
  onRelease?: () => void
}

/** How near, in pixels, the pointer must be to grab a handle when a chart has several. */
export const GRAB_RADIUS = 28

/**
 * The `Handle` layer of a `Plot` (same props as the `Handle` type): `<Handle kind="x" at={x0} onDrag={setX0} />`.
 * Declared here so the layer and the handle type share one name.
 */
export const Handle = HandleLayer

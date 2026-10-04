/**
 * A diagram is specified, not charted: nodes are placed by hand on a grid (or by `layout: 'layered'` for graphs
 * generated from data), groups are drawn around nodes or at given rectangles, and edges are routed through ports and
 * optional waypoints. Coordinates are grid units (one unit is `unit` pixels, 40 by default); y grows downwards. Node
 * positions are centres.
 */

/** A palette slot (0–7, the categorical data colours), or a neutral tone. */
export type Tone = number | 'neutral' | 'ink'

export type Side = 'n' | 's' | 'e' | 'w'

/**
 * Where an element is in a process the diagram steps through: `idle` (not reached: dimmed), `active` (the current
 * step: drawn in its tone with a heavier outline and fill) or `done` (reached). Unset means no process: drawn plainly.
 */
export type ElementState = 'idle' | 'active' | 'done'

export type Shape =
  | 'box' // rounded rectangle, tinted with its tone
  | 'pill' // fully rounded rectangle
  | 'circle' // a variable or state
  | 'latent' // a latent variable: circle with a double border
  | 'noise' // a random input: dashed circle
  | 'op' // a small operator circle holding a symbol (⊕, ⊗, σ, tanh)
  | 'factor' // a small filled square: a factor in a factor graph
  | 'encoder' // trapezoid narrowing along `dir`: a projection into a smaller space
  | 'decoder' // trapezoid widening along `dir`: a projection back out
  | 'stack' // a box with layered copies behind it: a block repeated N times
  | 'dot' // a junction point where lines meet or split
  | 'text' // a label with no outline

export type Direction = 'right' | 'left' | 'up' | 'down'

export type DiagramNode = {
  id: string
  /** Centre in grid units. Required unless the spec uses `layout: 'layered'`, which assigns it. */
  x?: number
  y?: number
  shape?: Shape
  /** Width and height in grid units; each shape has a default. */
  w?: number
  h?: number
  /** Text with `$…$` maths, rendered with KaTeX. `\n` breaks lines. */
  label?: string
  tone?: Tone
  /** Flow direction for encoder and decoder trapezoids. */
  dir?: Direction
  dashed?: boolean
  /** Shade the node: an observed variable in a graphical model. */
  filled?: boolean
  /** Shade the node in proportion to a value in [0, 1], e.g. a probability; overrides `filled`. */
  shade?: number
  /** Draw in the accent colour with a heavier outline, e.g. the variables a sentence or a control is about. */
  highlight?: boolean
  /** Put the label outside the shape instead of inside (defaults to `n` for factors and dots). */
  labelSide?: Side
  /** Smaller label text, e.g. for annotations. */
  small?: boolean
  /**
   * Annotations outside the shape, one per side: e.g. a value above (`n`) and an adjoint below (`s`). Text with `$…$`
   * maths; they follow the node's state (dimmed when idle).
   */
  notes?: Partial<Record<Side, string>>
  /** Step state; see `ElementState`. */
  state?: ElementState
  /** For `layout: 'layered'`: pin the node to this column (layer) instead of its topological depth. */
  layer?: number
  /**
   * A stacked bar of shares by palette slot (share k in slot k, e.g. class proportions), drawn along the bottom of a
   * box or pill (`barHeight` pixels, default 5) under its label, or filling a node that has no label.
   */
  bar?: readonly number[]
  barHeight?: number
  /** The node the reader has chosen: an ink ring around it. */
  selected?: boolean
  /** What a screen reader announces for a clickable node (default its label). */
  ariaLabel?: string
}

export type DiagramGroup = {
  id: string
  label?: string
  tone?: Tone
  /** Draw around these nodes (with `pad` grid units of margin), or at an explicit rectangle. */
  around?: string[]
  pad?: number
  rect?: { x: number; y: number; w: number; h: number }
  /** Where the label sits. */
  labelAt?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'
  dashed?: boolean
}

export type DiagramEdge = {
  /**
   * A node id, optionally with a side: `mha:s`. Without a side the side facing the other end is used. An edge from a
   * node to itself is a loop drawn outside the side given on `from` (default `n`), e.g. a self-transition.
   */
  from: string
  to: string
  /** Waypoints in grid units; the edge passes through them in order. */
  via?: [number, number][]
  /**
   * `ortho` (default) routes with right angles: through the waypoints if given, else with one or two elbows chosen
   * from the port sides. `straight` draws a direct line (the usual choice in graphical models). `curve` bows the line
   * sideways by `bend` grid units, e.g. for two edges between the same pair of nodes.
   */
  route?: 'ortho' | 'straight' | 'curve'
  bend?: number
  label?: string
  /** Where the label sits along the edge, as a fraction of its length (default: middle of the longest straight run). */
  labelPos?: number
  /** Which side of the edge the label sits on, relative to the direction of travel (default left: above a rightward edge). */
  labelSide?: 'left' | 'right'
  /** Gap between the edge and its label, in grid units. */
  labelOffset?: number
  /** Turn the label to run along the edge, kept upright (default true). */
  labelRotate?: boolean
  /**
   * An annotation riding on the edge itself (a chip over the line): a message, value or gradient travelling along it.
   * `at` is the fraction of the way from `from` to `to` (default 0.5).
   */
  note?: { text: string; at?: number; tone?: Tone }
  dashed?: boolean
  /**
   * Arrowheads: at the `end` (at `to`), the `start`, `both`, `none`, or `mid` (one head halfway, pointing to `to`; the
   * usual choice on a factor graph, whose ends are small).
   */
  arrow?: 'end' | 'start' | 'both' | 'none' | 'mid'
  /** Flip the direction the arrows point without rerouting the edge, e.g. for a backward pass over forward edges. */
  reverse?: boolean
  tone?: Tone
  highlight?: boolean
  /** Step state; see `ElementState`. */
  state?: ElementState
}

/** Options of the automatic layered layout. */
export type LayeredOptions = {
  /** Flow direction: layers are columns left to right (default) or rows top to bottom. */
  direction?: 'right' | 'down'
  /** Distance between layers, in grid units (default 2). */
  layerGap?: number
  /** Distance between nodes within a layer, in grid units (default 1.6). */
  nodeGap?: number
  /** Barycentre sweeps used to order each layer (default 8). */
  sweeps?: number
  /**
   * Move each source (a node without incoming edges) to the layer just before its first consumer, so an input used
   * late does not stretch across the diagram (default true).
   */
  compactSources?: boolean
}

export type DiagramSpec = {
  nodes: DiagramNode[]
  edges?: DiagramEdge[]
  groups?: DiagramGroup[]
  /**
   * `manual` (default): every node has `x` and `y`. `layered`: nodes are placed in layers by topological depth along
   * the edges, ordered within each layer to reduce crossings; edges spanning several layers get waypoints.
   */
  layout?: 'manual' | 'layered'
  layered?: LayeredOptions
  /** Pixels per grid unit at natural size. */
  unit?: number
  /** How far the diagram may scale up beyond its natural size to fill its container. */
  maxScale?: number
  /**
   * A box (grid units) the view always includes, so the drawing keeps its scale while elements come and go (e.g. a
   * tree drawn node by node).
   */
  frame?: { x0: number; y0: number; x1: number; y1: number }
  /**
   * The colour of active and highlighted elements: the accent data colour (default), or `ink`, for diagrams whose data
   * colours carry meaning (e.g. class colours in a tree's nodes).
   */
  accent?: 'data' | 'ink'
  /**
   * Multiplies every position (nodes, waypoints, group rectangles) to space the layout out without resizing the shapes;
   * a pair scales x and y separately.
   */
  spread?: number | [number, number]
  /** Grow nodes whose label does not fit (default true). The label is measured after the first render. */
  fitLabels?: boolean
}

/** A node with its position resolved (after layout). */
export type PlacedNode = DiagramNode & { x: number; y: number }

/**
 * The lab's diagrams: hand-specified SVG diagrams (architecture diagrams, flow charts, graphical models, factor graphs)
 * with grid placement, ports, orthogonal, straight and curved edges, KaTeX labels, notes, edge chips and step states,
 * plus an automatic layered layout for graphs generated from data and a tidy layout for trees (`treeLayout`).
 * Diagrams are not charts: use `@render/viz` for data. Import from '@render/diagram'.
 */
export { Diagram, type DiagramProps } from './Diagram'
export { layeredLayout } from './layout'
export { treeLayout, type TreeLayoutOptions } from './tree'
export { TreeView, type NodeSummary, type PerItem, type TreeViewProps } from './TreeView'
export { circleLayout, forceLayout, type ForceOptions, type Point } from './force'
export { MathText } from './MathText'
export { factor, gate, link, merge, op, projector, reparam, variable } from './components'
export type {
  DiagramEdge,
  DiagramGroup,
  DiagramNode,
  DiagramSpec,
  Direction,
  ElementState,
  LayeredOptions,
  PlacedNode,
  Shape,
  Side,
  Tone,
} from './types'
export * from './specs'
export * as diagramSpecs from './specs'

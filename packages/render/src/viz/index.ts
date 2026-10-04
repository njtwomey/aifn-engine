/** The lab's visual layer. Everything outside src/viz imports from '@render/viz' only. */
export { EChart, type EChartProps, type EChartClick, type PlotPointer } from './EChart'
export { Readout, ReadoutGroup, Readouts } from './Readout'
export { zoomRange, panRange, type Range } from './viewport'
export {
  FrameContext,
  DEFAULT_HEIGHT,
  useChartHeight,
  useElementSize,
  type FrameContextValue,
  type HoverInfo,
  type HoverRow,
} from './frame'
export { Handle, type Vec2 } from './handles'
export type { Vector } from './vectors'
export { useScaleColor } from './useScaleColor'
export { formatNumber, formatPower, niceStep, stepDecimals } from './format'
export * from './plot'
export { ScaleBar } from './ScaleBar'
export { argmaxMargins } from './contours'

/** Reusable figure layouts and frames. */
export { Figure, type FigureProps } from './Figure'
export { FigurePage } from './FigurePage'
export { Columns } from './Columns'
export { ControlGroup, ControlRow, Controls } from './Controls'
export { Dashboard, DashboardCell, DashboardRow } from './Dashboard'
export { Equation, EquationSteps, type EquationStep } from './Equation'
export { Tex } from './Tex'
export { live, tex, toTex, type EquationTemplate, type LiveValue } from './equation-tex'
export { createFigureIds, FigureIdsContext, useFigureId } from './figure-ids'
export { FigureScope, FIGURE_SIZES, type FigureSize } from './figure-size'
export { PanelSlot } from './slots'
export { FrameSlotsContext, type SlotName } from './slots-context'
export { Providers } from './Providers'
export { slugify } from './slugify'
export {
  defaultMathMacros,
  registerMathMacros,
  getGlobalMathMacros,
  RenderMathContext,
  RenderMathProvider,
  useRenderMathMacros,
  renderKatex,
} from './math-macros'

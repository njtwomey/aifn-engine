/**
 * `aifn-render/gym`: views of sequential decisions on the compute gym contract (`aifn-compute/foundation/contracts`: environments,
 * agents, their render specs, trajectories and training runs). `GridView` draws any grid (a maze, a gridworld, the
 * cliff, a search: colour only, or with policy arrows and a path); `CartPoleView` and `PendulumView` classic control;
 * `GymTrainer` trains in the compute worker and plays any episode, drawn by the renderer for the environment's
 * `render.kind` (`GYM_RENDERERS`), with `StepSeries` beneath. Environments and agents arrive as values (a compute
 * `GymSetup`), so this module never imports `aifn-methods`.
 */
export { GymTrainer, type GymTrainerProps } from './GymTrainer'
export { trainingRun, trainingRunOf, type TrainingRunDefaults, type TrainingRunValues } from './trainingRun'
export { GYM_RENDERERS, renderKind, WIDE_KINDS } from './registry'
export { GridView, type GridTones, type GridValueField, type GridViewProps } from './GridView'
export {
  cellXY,
  GRID_KINDS,
  kindRows,
  LANE_OFFSET,
  pathMoves,
  policyArrows,
  toneRows,
  valueRows,
  type PathMovesOptions,
} from './grid'
export {
  BanditRenderer,
  CartPoleRenderer,
  GridRenderer,
  PendulumRenderer,
  type GridOverlay,
  type GridRendererOptions,
  type GymRenderer,
  type GymRenderProps,
} from './renderers'
export { actionSeries, hasStepSeries, stateSeries, type ActionSeries } from './series'
export { StepSeries, type StepSeriesProps } from './StepSeries'
export { CartPoleView, type CartPoleViewProps } from './CartPoleView'
export { PendulumView, type PendulumViewProps } from './PendulumView'

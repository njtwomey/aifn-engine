/**
 * Figure state (DESIGN.md §4, §6, §8a): `useFigureState` and its field builders, probes and the `useComputed`
 * scheduler. Import from '@render/state'.
 */
export {
  choice,
  float,
  fromSpace,
  int,
  number,
  row,
  setting,
  slider,
  toggle,
  toSpace,
  variants,
  when,
  type AnyValues,
  type CaseDef,
  type ChoiceDef,
  type ChosenCase,
  type NumberDef,
  type ParamDef,
  type ParamDefs,
  type ParamValue,
  type RowDef,
  type SliderDef,
  type SwitchDef,
  type ValueOf,
  type Values,
  type VariantsDef,
  type VariantValue,
} from './schema'
export { useFigureState, type FigureState, type FigureStateApi, type HandleOptions, type Param } from './useFigureState'
export { fromEntries, toEntries, type Raw } from './store'
export { useStreamed, type Streamed } from './useStreamed'
export {
  useComputed,
  type Computed,
  type ComputeMode,
  type ComputeOptions,
  type WorkerComputeOptions,
} from './useComputed'
export { call, type Task } from './task'
export { formatField, snapToStep } from './step'
export {
  checkNumber,
  clampNumber,
  formatNumberValue,
  numberBounds,
  parseNumber,
  stepNumber,
  validateNumber,
  type NumberOptions,
  type NumberScale,
  type NumberType,
} from './number'
export { isPointerHeld, onceReleased } from './pointer'
export { useProbe, type ProbeModel } from './probe'
export { namedPinField, packPair, pinField, unpackPair, usePinned, usePinnedName, type Pinned } from './usePinned'
export { ProbeReadout } from './ProbeReadout'

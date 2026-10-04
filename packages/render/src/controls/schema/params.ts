/**
 * Declarative parameters: the field builders and types of figure state (`@render/state`), re-exported where controls
 * have always imported them. `useParams` and `defineVariants` (variants.ts) and `useFigureState` share them, so a
 * parameter is described one way everywhere.
 */
export {
  choice,
  coerce,
  float,
  int,
  isActive,
  number,
  row,
  setting,
  slider,
  toggle,
  variants,
  when,
  type AnyValues,
  type ChoiceDef,
  type NumberDef,
  type ParamDef,
  type ParamDefs,
  type ParamValue,
  type SliderDef,
  type SwitchDef,
  type ValueOf,
  type Values,
} from '@render/state/schema'

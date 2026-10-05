/**
 * Structured interactive controls for aifn figures and specimens:
 * - visual: SwatchPicker, RampPicker, ThemeToggle, RevealToggle
 * - numeric: Slider, NumberField, useNumberDraft
 * - selection: Select, Segmented, Combobox, MultiCombobox, Choice, options
 * - playback: Player, StepControls, usePlayhead
 * - schema: params, ParamControls, variants, VariantControls, useParam
 * - code: CodeEditor, CodeBlock (read-only, highlighted, copyable), prologLanguage; programs: useProgram,
 *   useEntryArgs, EntryControls, ProgramStatus
 * - base: ControlLabel, StatusText, Switch
 */
export * from './visual'
export * from './numeric'
export * from './selection'
export * from './playback'
export * from './schema'
export * from './code'
export * from './base'

// Buttons re-exported from UI primitives so figures have a single controls import point
export { Button } from '../ui/button'
export { ButtonGroup, ButtonGroupSeparator, ButtonGroupText } from '../ui/button-group'

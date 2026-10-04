import type { ReactNode } from 'react'
import type { ParamDefs } from '@render/state/schema'
import { Choice } from '../selection/Choice'
import { ParamControls } from './ParamControls'
import type { VariantsControl } from './variants'

/**
 * The choice of variant, then the chosen variant's controls and the shared ones (the same controls a `variants` field
 * of figure state draws). Renders a fragment for the surrounding `Controls` grid or a Figure's controls slot.
 */
export function VariantControls<V extends Record<string, ParamDefs>, S extends ParamDefs, X, R>({
  variants: v,
  label = 'function',
}: {
  variants: VariantsControl<V, S, X, R>
  label?: ReactNode
}) {
  const specs = v.variants.specs
  const options = Object.keys(specs).map((key) => ({ value: key, label: specs[key].label }))
  return (
    <>
      <Choice label={label} value={v.key as string} onChange={v.setKey} options={options} />
      <ParamControls defs={{ ...specs[v.key].params, ...v.variants.shared }} values={v.values} set={v.set} />
    </>
  )
}

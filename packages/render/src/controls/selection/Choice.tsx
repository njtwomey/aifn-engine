import { Combobox, type ComboboxProps } from './Combobox'
import { Select } from './Select'

/** Lists longer than this get a searchable combobox. */
export const SEARCHABLE_FROM = 12

export type ChoiceProps<T extends string> = ComboboxProps<T> & {
  /** Force a searchable combobox (true) or a plain dropdown (false); by default decided by the list's length. */
  searchable?: boolean
}

/** A categorical choice: a `Select` for short lists, a searchable `Combobox` for long ones. */
export function Choice<T extends string>({ searchable, placeholder, ...props }: ChoiceProps<T>) {
  return (searchable ?? props.options.length > SEARCHABLE_FROM) ? (
    <Combobox {...props} placeholder={placeholder} />
  ) : (
    <Select {...props} />
  )
}

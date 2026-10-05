import { useId, useMemo, type ReactNode } from 'react'
import {
  Combobox as ComboboxRoot,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from '../../ui/combobox'
import { cn } from '../../lib/utils'
import { ControlLabel } from '../base/ControlLabel'
import { normalise, type NormalOption, type Options } from './options'

export type ComboboxProps<T extends string> = {
  label: ReactNode
  value: T
  onChange: (value: T) => void
  options: Options<T>
  placeholder?: string
  disabled?: boolean
  className?: string
}

/** A searchable dropdown for a long list of categorical choices: type to filter by label or keywords. */
export function Combobox<T extends string>({
  label,
  value,
  onChange,
  options,
  placeholder = 'Search…',
  disabled,
  className,
}: ComboboxProps<T>) {
  const id = useId()
  const items = useMemo(() => normalise(options), [options])
  const selected = items.find((o) => o.value === value) ?? null
  return (
    <div className={cn('flex w-full max-w-xs min-w-40 flex-col gap-1.5', className)}>
      <ControlLabel htmlFor={id}>{label}</ControlLabel>
      <ComboboxRoot<NormalOption<T>>
        items={items}
        value={selected}
        onValueChange={(o) => o && onChange(o.value)}
        itemToStringLabel={(o) => (typeof o.label === 'string' ? o.label : o.value)}
        isItemEqualToValue={(a, b) => a.value === b.value}
        filter={(o, query) => o.text.toLowerCase().includes(query.trim().toLowerCase())}
        disabled={disabled}
      >
        <ComboboxInput id={id} placeholder={placeholder} className="h-8 w-full text-xs" disabled={disabled} />
        <ComboboxContent>
          <ComboboxEmpty>No matches</ComboboxEmpty>
          <ComboboxList>
            {(o: NormalOption<T>) => (
              <ComboboxItem key={o.value} value={o} disabled={o.disabled} className="text-xs">
                {o.label}
              </ComboboxItem>
            )}
          </ComboboxList>
        </ComboboxContent>
      </ComboboxRoot>
    </div>
  )
}

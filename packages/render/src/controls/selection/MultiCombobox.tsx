import { Fragment, useId, useMemo, type ReactNode } from 'react'
import {
  Combobox as ComboboxRoot,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from '@render/ui/combobox'
import { cn } from '@render/lib/utils'
import { ControlLabel } from '../base/ControlLabel'
import { normalise, type NormalOption, type Options } from './options'

export type MultiComboboxProps<T extends string> = {
  label: ReactNode
  value: readonly T[]
  onChange: (value: T[]) => void
  options: Options<T>
  /** At most this many may be chosen; further options are disabled until one is removed. */
  max?: number
  /** At least this many stay chosen: their chips lose the remove button. Default 1. */
  min?: number
  placeholder?: string
  className?: string
}

/**
 * A searchable multi-select: the chosen options as removable chips, the rest in a filtered dropdown. The chips keep the
 * order of `options`, not the order of choosing, so a figure drawn from them is stable.
 */
export function MultiCombobox<T extends string>({
  label,
  value,
  onChange,
  options,
  max = Infinity,
  min = 1,
  placeholder = 'Add…',
  className,
}: MultiComboboxProps<T>) {
  const id = useId()
  const anchor = useComboboxAnchor()
  const items = useMemo(() => normalise(options), [options])
  const chosen = useMemo(() => items.filter((o) => value.includes(o.value)), [items, value])
  const full = chosen.length >= max
  return (
    <div className={cn('col-span-full flex w-full flex-col gap-1.5', className)}>
      <ControlLabel htmlFor={id}>{label}</ControlLabel>
      <ComboboxRoot<NormalOption<T>, true>
        multiple
        items={items}
        value={chosen}
        onValueChange={(next) => {
          const set = new Set(next.map((o) => o.value))
          if (set.size < min || set.size > max) return
          onChange(items.filter((o) => set.has(o.value)).map((o) => o.value))
        }}
        itemToStringLabel={(o) => o.text}
        isItemEqualToValue={(a, b) => a.value === b.value}
        filter={(o, query) => o.text.toLowerCase().includes(query.trim().toLowerCase())}
      >
        <ComboboxChips ref={anchor} className="text-xs">
          <ComboboxValue>
            {(values: NormalOption<T>[]) => (
              <Fragment>
                {values.map((o) => (
                  <ComboboxChip key={o.value} showRemove={values.length > min}>
                    {o.label}
                  </ComboboxChip>
                ))}
                <ComboboxChipsInput id={id} placeholder={full ? `${max} at most` : placeholder} disabled={full} />
              </Fragment>
            )}
          </ComboboxValue>
        </ComboboxChips>
        <ComboboxContent anchor={anchor}>
          <ComboboxEmpty>No matches</ComboboxEmpty>
          <ComboboxList>
            {(o: NormalOption<T>) => (
              <ComboboxItem
                key={o.value}
                value={o}
                disabled={o.disabled || (full && !value.includes(o.value))}
                className="text-xs"
              >
                {o.label}
              </ComboboxItem>
            )}
          </ComboboxList>
        </ComboboxContent>
      </ComboboxRoot>
    </div>
  )
}

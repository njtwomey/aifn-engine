import { useId, type ReactNode } from 'react'
import { Select as SelectRoot, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@render/ui/select'
import { cn } from '@render/lib/utils'
import { ControlLabel } from '../base/ControlLabel'
import { normalise, type Options } from './options'

export type SelectProps<T extends string> = {
  label: ReactNode
  value: T
  onChange: (value: T) => void
  options: Options<T>
  disabled?: boolean
  className?: string
}

/** A dropdown for a short list of categorical choices (up to about a dozen). For longer lists use `Combobox`. */
export function Select<T extends string>({ label, value, onChange, options, disabled, className }: SelectProps<T>) {
  const id = useId()
  const items = normalise(options)
  return (
    <div className={cn('flex w-full max-w-xs min-w-0 flex-col gap-1.5', className)}>
      <ControlLabel id={id}>{label}</ControlLabel>
      <SelectRoot
        value={value}
        onValueChange={(v) => v !== null && onChange(v as T)}
        items={items.map((o) => ({ value: o.value, label: o.label }))}
        disabled={disabled}
      >
        <SelectTrigger className="w-full min-w-0 text-xs" aria-labelledby={id}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {items.map((o) => (
            <SelectItem key={o.value} value={o.value} disabled={o.disabled} className="text-xs">
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </SelectRoot>
    </div>
  )
}

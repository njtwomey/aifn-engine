import { useId, type ReactNode } from 'react'
import { ToggleGroup, ToggleGroupItem } from '../../ui/toggle-group'
import { cn } from '../../lib/utils'
import { ControlLabel } from '../base/ControlLabel'
import { normalise, type Options } from './options'

export type SegmentedProps<T extends string> = {
  label: ReactNode
  value: T
  onChange: (value: T) => void
  options: Options<T>
  disabled?: boolean
  className?: string
}

/**
 * A choice among two to five short options shown side by side as one segmented button row, every option visible
 * (a `Select` hides them behind a menu). Exactly one is pressed: pressing the pressed one keeps it. Arrow keys move
 * between options.
 */
export function Segmented<T extends string>({
  label,
  value,
  onChange,
  options,
  disabled,
  className,
}: SegmentedProps<T>) {
  const id = useId()
  const items = normalise(options)
  return (
    <div className={cn('flex min-w-0 flex-col gap-1.5', className)}>
      <ControlLabel id={id}>{label}</ControlLabel>
      <ToggleGroup
        aria-labelledby={id}
        value={[value]}
        onValueChange={(v) => v.length > 0 && onChange(v[0] as T)}
        variant="outline"
        size="sm"
        spacing={0}
        disabled={disabled}
      >
        {items.map((o) => (
          <ToggleGroupItem key={o.value} value={o.value} disabled={o.disabled} className="text-xs">
            {o.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  )
}

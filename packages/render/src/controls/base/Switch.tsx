import { useId, type ReactNode } from 'react'
import { Switch as SwitchPrimitive } from '@render/ui/switch'
import { cn } from '@render/lib/utils'
import { ControlLabel } from './ControlLabel'

/** An on/off setting, label to the right. */
export function Switch({
  label,
  checked,
  onChange,
  disabled,
  className,
}: {
  label: ReactNode
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
  className?: string
}) {
  const id = useId()
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <SwitchPrimitive id={id} checked={checked} onCheckedChange={onChange} disabled={disabled} size="sm" />
      <ControlLabel htmlFor={id}>{label}</ControlLabel>
    </div>
  )
}

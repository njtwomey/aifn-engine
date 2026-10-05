import { Eye, EyeOff } from 'lucide-react'
import type { ReactNode } from 'react'
import { Toggle } from '../../ui/toggle'

export type RevealToggleProps = {
  label: ReactNode
  pressed: boolean
  onChange: (pressed: boolean) => void
  className?: string
}

/**
 * A revealing toggle button: an eye/eye-off toggle that turns on or off
 * visual layers or ingredients being inspected (e.g. Jacobian, Contours, Vector fields).
 */
export function RevealToggle({ label, pressed, onChange, className }: RevealToggleProps) {
  return (
    <div className="flex items-end self-end">
      <Toggle
        variant="outline"
        size="sm"
        aria-label={typeof label === 'string' ? label : undefined}
        pressed={pressed}
        onPressedChange={onChange}
        className={
          className ?? 'data-pressed:border-primary/60 data-pressed:bg-primary/10 data-pressed:text-foreground'
        }
      >
        {pressed ? <Eye className="size-3.5" /> : <EyeOff className="size-3.5" />}
        {label}
      </Toggle>
    </div>
  )
}

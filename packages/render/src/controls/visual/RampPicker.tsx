import { type ReactNode } from 'react'
import { diverging, sequential } from '../../design/palette'
import { useTheme } from '../../design/theme'
import { cn } from '../../lib/utils'
import { ControlLabel } from '../ControlLabel'

export type RampType = 'sequential' | 'diverging'

export type RampPickerProps = {
  label?: ReactNode
  value?: RampType
  onChange?: (mode: RampType) => void
  className?: string
}

/**
 * Visual display and picker for continuous data color ramps: sequential (0 → 1 magnitude)
 * or diverging (−1 … 0 … +1 signed deviation).
 */
export function RampPicker({ label = 'Colour ramp', value = 'sequential', onChange, className }: RampPickerProps) {
  const { resolved } = useTheme()
  const colours = value === 'sequential' ? sequential(resolved) : diverging(resolved)

  return (
    <div className={cn('col-span-full flex flex-col gap-1.5', className)}>
      <div className="flex items-center justify-between">
        <ControlLabel>{label}</ControlLabel>
        {onChange && (
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => onChange('sequential')}
              className={cn(
                'rounded px-2 py-0.5 text-[11px] font-medium transition-colors',
                value === 'sequential'
                  ? 'bg-primary/10 font-semibold text-primary'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              Sequential
            </button>
            <button
              type="button"
              onClick={() => onChange('diverging')}
              className={cn(
                'rounded px-2 py-0.5 text-[11px] font-medium transition-colors',
                value === 'diverging'
                  ? 'bg-primary/10 font-semibold text-primary'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              Diverging
            </button>
          </div>
        )}
      </div>
      <div
        className="h-6 w-full rounded-md ring-1 ring-foreground/10"
        style={{ background: `linear-gradient(to right, ${colours.join(', ')})` }}
      />
    </div>
  )
}

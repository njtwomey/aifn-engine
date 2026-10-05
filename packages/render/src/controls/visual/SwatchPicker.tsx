import { type ReactNode } from 'react'
import { palette } from '../../design/palette'
import { useTheme } from '../../design/theme'
import { cn } from '../../lib/utils'
import { ControlLabel } from '../ControlLabel'

export type SwatchPickerProps = {
  label?: ReactNode
  value: number
  onChange: (slot: number) => void
  /** Defaults to the active theme's categorical palette slots (0–9). */
  colours?: readonly string[]
  className?: string
}

/**
 * A categorical color swatch picker: displays the 10 data palette slots for the current theme,
 * highlighting the active slot with a checkmark and showing its hex value.
 */
export function SwatchPicker({ label = 'Colour slot', value, onChange, colours, className }: SwatchPickerProps) {
  const { resolved } = useTheme()
  const swatches = colours ?? palette.categorical[resolved]
  const activeColor = swatches[value % swatches.length]

  return (
    <div className={cn('col-span-full flex flex-col gap-1.5', className)}>
      {label && <ControlLabel>{label}</ControlLabel>}
      <div className="flex flex-wrap items-center gap-1.5">
        {swatches.map((hex, i) => {
          const selected = value === i
          return (
            <button
              key={i}
              type="button"
              onClick={() => onChange(i)}
              title={`Slot ${i}: ${hex}`}
              aria-label={`Slot ${i}: ${hex}`}
              className={cn(
                'relative size-7 rounded-md ring-1 ring-foreground/10 transition-all hover:scale-110 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-hidden',
                selected && 'scale-105 shadow-xs ring-2 ring-foreground',
              )}
              style={{ backgroundColor: hex }}
            >
              {selected && (
                <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]">
                  ✓
                </span>
              )}
            </button>
          )
        })}
        <span className="ml-1.5 font-mono text-[11px] text-muted-foreground">{activeColor}</span>
      </div>
    </div>
  )
}

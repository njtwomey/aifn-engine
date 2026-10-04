import { categorical, scaleStops, seriesColor, useTheme } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Theme colours',
  question: 'How do I use the palette outside a chart, in both themes?',
  explain:
    "`useTheme().resolved` is `light` or `dark`; `seriesColor(mode, slot)`, `categorical(mode)` and `scaleStops(scale, mode)` give that theme's data colours, so custom marks follow the theme toggle.",
}

export default function ThemeColours() {
  const { resolved: mode } = useTheme()
  return (
    <div className="space-y-3 rounded-lg border p-4 text-sm">
      <div className="flex flex-wrap gap-2">
        {categorical(mode).map((_, slot) => (
          <span
            key={slot}
            className="rounded px-2 py-1 font-mono text-xs text-white"
            style={{ background: seriesColor(mode, slot) }}
          >
            slot {slot}
          </span>
        ))}
      </div>
      {(['sequential', 'diverging'] as const).map((scale) => (
        <div key={scale} className="flex items-center gap-3">
          <span className="w-20 text-muted-foreground">{scale}</span>
          <span
            className="h-4 flex-1 rounded"
            style={{ background: `linear-gradient(to right, ${scaleStops(scale, mode).join(', ')})` }}
          />
        </div>
      ))}
    </div>
  )
}

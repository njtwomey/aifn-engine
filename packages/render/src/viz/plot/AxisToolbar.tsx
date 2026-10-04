import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Maximize,
  Minus,
  MoveHorizontal,
  MoveVertical,
  Plus,
} from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { Button } from '@render/ui/button'
import { ButtonGroup } from '@render/ui/button-group'
import { Popover, PopoverContent, PopoverTrigger } from '@render/ui/popover'
import { cn } from '@render/lib/utils'
import { formatNumber } from '../format'
import { validRange, type Range } from '../viewport'
import type { AxisModel } from './axis'

/** Zoom factor of one press; pan moves a quarter of the span. */
const STEP = 2
const PAN = 0.25

export type ToolbarAxis = { axis: AxisModel; direction: 'x' | 'y' }

/**
 * The compact axis toolbar: one small button per axis (↔ or ↕, with the axis's name when there are several) that opens
 * its full controls in a popover (typed range, pan, zoom), then one auto-scale button that refits every axis listed.
 * A Plot on its own draws one for its two axes; a `Plots` grid draws one for all its axes, each once.
 */
export function AxisToolbar({ axes, className }: { axes: readonly ToolbarAxis[]; className?: string }) {
  const zoomable = axes.filter((a) => a.axis.zoomable)
  // Names when a direction repeats (two y axes in a grid), so each button says which axis it is.
  const named =
    zoomable.filter((a) => a.direction === 'x').length > 1 || zoomable.filter((a) => a.direction === 'y').length > 1
  if (!axes.length) return null
  return (
    <div className={cn('flex h-6 items-center justify-end gap-2', className)}>
      <ButtonGroup aria-label="Axes">
        {zoomable.map(({ axis, direction }) => (
          <AxisButton key={axis.id} axis={axis} direction={direction} named={named} />
        ))}
        <Button
          variant="outline"
          size="icon-xs"
          aria-label="Auto-scale: fit every axis to the data"
          title="Auto-scale: fit every axis to the data"
          onClick={() => axes.forEach(({ axis }) => axis.reset())}
        >
          <Maximize />
        </Button>
      </ButtonGroup>
    </div>
  )
}

function IconButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <Button variant="outline" size="icon-xs" aria-label={label} title={label} onClick={onClick}>
      {children}
    </Button>
  )
}

function AxisButton({ axis, direction, named }: { axis: AxisModel; direction: 'x' | 'y'; named: boolean }) {
  const Icon = direction === 'x' ? MoveHorizontal : MoveVertical
  const name = axis.options.label ?? direction
  const label = `${name} axis: range, pan and zoom`
  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button
            variant="outline"
            size={named ? 'xs' : 'icon-xs'}
            aria-label={label}
            title={label}
            className={named ? 'max-w-32 gap-1 px-1.5 text-[11px] font-normal' : undefined}
          />
        }
      >
        <Icon />
        {named && <span className="truncate">{name}</span>}
      </PopoverTrigger>
      <PopoverContent align="end" className="w-auto p-2">
        <AxisControls axis={axis} direction={direction} name={name} />
      </PopoverContent>
    </Popover>
  )
}

function AxisControls({ axis, direction, name }: { axis: AxisModel; direction: 'x' | 'y'; name: string }) {
  const [Back, Forward] = direction === 'x' ? [ArrowLeft, ArrowRight] : [ArrowDown, ArrowUp]
  const [back, forward] = direction === 'x' ? ['left', 'right'] : ['down', 'up']
  return (
    <span className="flex items-center gap-1.5">
      <RangeFields name={name} range={axis.resolve()} log={axis.log} onChange={(r) => axis.setView(r)} />
      <ButtonGroup aria-label={`${name} axis`}>
        <IconButton label={`Pan ${back}`} onClick={() => axis.pan(-PAN)}>
          <Back />
        </IconButton>
        <IconButton label={`Zoom ${name} out`} onClick={() => axis.zoom(STEP)}>
          <Minus />
        </IconButton>
        <IconButton label={`Zoom ${name} in`} onClick={() => axis.zoom(1 / STEP)}>
          <Plus />
        </IconButton>
        <IconButton label={`Pan ${forward}`} onClick={() => axis.pan(PAN)}>
          <Forward />
        </IconButton>
      </ButtonGroup>
    </span>
  )
}

/** Two typed fields for an axis range. Enter or blur commits both if they form a valid range; Escape reverts. */
function RangeFields({
  name,
  range,
  log,
  onChange,
}: {
  name: string
  range: Range | undefined
  log: boolean
  onChange: (r: Range) => void
}) {
  const [draft, setDraft] = useState<[string, string] | null>(null)
  const shown: [string, string] = draft ?? (range ? [formatNumber(range[0]), formatNumber(range[1])] : ['', ''])
  const commit = () => {
    if (!draft) return
    const next: Range = [Number(draft[0].replace('−', '-')), Number(draft[1].replace('−', '-'))]
    setDraft(null)
    if (validRange(next, log)) onChange(next)
  }
  const field = (i: 0 | 1) => (
    <input
      aria-label={`${name} ${i === 0 ? 'minimum' : 'maximum'}`}
      className="h-6 w-16 rounded-md border border-input bg-transparent px-1.5 text-right font-mono text-[11px] tabular-nums outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-50 dark:bg-input/30"
      value={shown[i]}
      disabled={!range}
      onFocus={(e) => {
        setDraft(shown)
        e.currentTarget.select()
      }}
      onChange={(e) => setDraft(i === 0 ? [e.target.value, shown[1]] : [shown[0], e.target.value])}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') e.currentTarget.blur()
        else if (e.key === 'Escape') {
          setDraft(null)
          requestAnimationFrame(() => (e.target as HTMLInputElement).blur())
        }
      }}
    />
  )
  return (
    <span className="flex items-center gap-1 font-mono text-[11px] text-muted-foreground">
      <span className="max-w-24 truncate">{name}</span>
      {field(0)}
      <span aria-hidden>–</span>
      {field(1)}
    </span>
  )
}

import { Slider as Rail } from '@base-ui/react/slider'
import { ChevronDown, ChevronLeft, ChevronRight } from 'lucide-react'
import { useId, type ReactNode } from 'react'
import { Button } from '../../ui/button'
import { ButtonGroup } from '../../ui/button-group'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '../../ui/dropdown-menu'
import { cn } from '../../lib/utils'
import {
  checkNumber,
  defaultSuggestions,
  formatHeaderValue,
  formatNumberValue,
  railFraction,
  railRange,
  railValue,
  stepNumber,
  type NumberOptions,
} from '../../state/number'
import { ControlLabel } from '../base/ControlLabel'
import type { Param } from '../schema/param'
import { useNumberDraft } from './useNumberDraft'

type Common = NumberOptions & {
  label: ReactNode
  /** How the value shows when not being edited (default: by type and scale, e.g. 1e-3 on a log10 scale). */
  format?: (v: number) => string
  disabled?: boolean
  className?: string
  /** Text shown as an uncommitted draft at first, validated like typing (for demonstrations). */
  initialDraft?: string
}

export type NumberFieldProps = Common & {
  param?: Param
  value?: number
  onChange?: (value: number) => void
}

/**
 * A typed number in one tight row, `‹ [ value ⌄ ] ›`: step buttons either side of the field, a menu of `suggestions`
 * inside it at the right, and a rail along its bottom edge to drag. For a seed, an episode count, a learning rate.
 * One field for every typed number: `type` float or int, bounds `gt`/`ge`/`lt`/`le` (`min`/`max` alias `ge`/`le`),
 * `scale` linear or log10, `spacing` ('lin' | 'log').
 *
 * The rail appears when the number has a range to drag across: both bounds, or `suggestions` standing in for a
 * missing one (a rate over decades with `gt: 0` runs from its smallest suggestion). It follows the field's scale, so
 * a log10 field drags evenly through the decades, and lands on readable values. A number with no range has no rail.
 *
 * Typing is validated, not clamped: a draft that does not parse or breaks the type or a bound turns the field red with
 * a short message ("must be > 0"), Enter is refused, and Escape or blur reverts to the last valid value. The buttons
 * and ↑/↓ step (linear ± step or points, × 10 with Shift; log10 mantissas [1, 3, 10, 30] with points_per_decade=2, a decade with Shift).
 */
export function NumberField(props: NumberFieldProps) {
  const { label, disabled, className } = props
  const value = props.param ? props.param.value : (props.value ?? 0)
  const onChange = props.param ? props.param.set : (props.onChange ?? (() => {}))

  const options: NumberOptions = {
    type: props.type,
    gt: props.gt,
    ge: props.ge,
    lt: props.lt,
    le: props.le,
    min: props.min ?? finite(props.param?.min),
    max: props.max ?? finite(props.param?.max),
    scale: props.scale,
    spacing: props.spacing,
    increment: props.increment,
    points_per_decade: props.points_per_decade,
    points: props.points,
    step: props.step ?? props.param?.step,
    suggestions: props.suggestions,
    logTransform: props.logTransform,
    headerValue: props.headerValue,
  }

  const format = props.format ?? ((v: number) => formatNumberValue(options, v))
  const headerText = formatHeaderValue(options, value)
  const suggestions = props.suggestions !== undefined ? props.suggestions : defaultSuggestions(options)

  const id = useId()
  const errorId = useId()
  const commit = (v: number) => {
    if (v !== value) onChange(v)
  }
  const step = (dir: 1 | -1, big = false) => stepNumber(options, value, dir, big)
  const { props: field, error } = useNumberDraft({
    value,
    onCommit: commit,
    step: options.step ?? 1,
    format,
    check: (text) => checkNumber(options, text),
    stepBy: (base, dir, big) => stepNumber(options, base, dir, big),
    initialDraft: props.initialDraft,
  })
  const down = step(-1)
  const up = step(1)
  const rail = railRange(options)

  return (
    <div className={cn('flex w-full max-w-xs min-w-32 flex-col gap-1.5', className)}>
      <div className="flex items-center justify-between gap-2">
        <ControlLabel
          htmlFor={id}
          className="min-w-0 flex-1 truncate"
          title={typeof label === 'string' ? label : undefined}
        >
          {label}
        </ControlLabel>
        {headerText ? (
          <span className="shrink-0 font-mono text-xs text-muted-foreground tabular-nums">{headerText}</span>
        ) : null}
      </div>
      <ButtonGroup className="w-full">
        <Button
          variant="outline"
          size="icon"
          aria-label="Decrease"
          disabled={disabled || down === null}
          onClick={(e) => {
            const next = step(-1, e.shiftKey)
            if (next !== null) commit(next)
          }}
        >
          <ChevronLeft />
        </Button>
        {/* The field: the typed value, the suggestions menu at its right, and the rail on its bottom edge. */}
        <div
          data-slot="input"
          className={cn(
            'relative flex h-8 min-w-0 flex-1 items-center border border-input bg-transparent focus-within:z-10 focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/50 dark:bg-input/30',
            disabled && 'opacity-50',
            error && 'z-10 border-destructive focus-within:border-destructive focus-within:ring-destructive/30',
          )}
        >
          <input
            id={id}
            disabled={disabled}
            aria-describedby={error ? errorId : undefined}
            className={cn(
              'h-full w-0 min-w-0 flex-1 bg-transparent pl-2 text-right font-mono text-xs tabular-nums outline-none',
              suggestions && suggestions.length > 0 ? 'pr-0.5' : 'pr-2',
            )}
            {...field}
          />
          {suggestions && suggestions.length > 0 ? (
            <DropdownMenu>
              <DropdownMenuTrigger
                disabled={disabled}
                aria-label="Suggested values"
                title="Suggested values"
                className="flex h-full shrink-0 items-center pr-1.5 pl-0.5 text-muted-foreground outline-none hover:text-foreground focus-visible:text-foreground data-popup-open:text-foreground"
              >
                <ChevronDown className="size-3.5" />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="max-h-60 w-auto min-w-24 overflow-y-auto">
                {suggestions.map((s) => (
                  <DropdownMenuItem
                    key={s}
                    onClick={() => commit(s)}
                    className={cn(
                      'justify-end font-mono text-xs tabular-nums',
                      s === value && 'bg-accent font-semibold',
                    )}
                  >
                    {format(s)}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}
          {rail ? (
            // The track is centred on the field's bottom border; a press anywhere along it jumps there, a drag follows.
            <Rail.Root
              className="absolute inset-x-0 -bottom-[5px] z-10"
              value={[railFraction(options, value)]}
              min={0}
              max={1}
              step={0.001}
              disabled={disabled}
              thumbAlignment="edge"
              onValueChange={(v) => {
                const next = railValue(options, Array.isArray(v) ? v[0] : (v as number))
                if (Number.isFinite(next)) commit(next)
              }}
            >
              <Rail.Control className="relative flex h-[10px] w-full cursor-pointer touch-none items-center select-none data-disabled:cursor-default">
                <Rail.Track className="relative h-0.5 w-full select-none">
                  <Rail.Indicator className="h-full bg-primary select-none" />
                </Rail.Track>
                <Rail.Thumb
                  aria-label={typeof label === 'string' ? `${label} rail` : 'rail'}
                  className="relative block size-2.5 shrink-0 rounded-full border border-ring bg-white ring-ring/50 transition-[color,box-shadow] select-none after:absolute after:-inset-1.5 hover:ring-3 focus-visible:ring-3 focus-visible:outline-hidden active:ring-3"
                />
              </Rail.Control>
            </Rail.Root>
          ) : null}
        </div>
        <Button
          variant="outline"
          size="icon"
          aria-label="Increase"
          disabled={disabled || up === null}
          onClick={(e) => {
            const next = step(1, e.shiftKey)
            if (next !== null) commit(next)
          }}
        >
          <ChevronRight />
        </Button>
      </ButtonGroup>
      {error ? (
        <p id={errorId} role="alert" className="text-xs leading-none text-destructive">
          {error}
        </p>
      ) : null}
    </div>
  )
}

/** Aliases for NumberField across different conventions */
export const NumberSelector = NumberField
export const NumericSelector = NumberField
export const NumericControl = NumberField
export const ParamNumberField = NumberField

const finite = (x: number | undefined) => (x !== undefined && Number.isFinite(x) ? x : undefined)

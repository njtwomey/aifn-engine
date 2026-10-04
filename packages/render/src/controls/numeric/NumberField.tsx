import { ChevronDown, Minus, Plus } from 'lucide-react'
import { useId, type ReactNode } from 'react'
import { Button } from '@render/ui/button'
import { ButtonGroup } from '@render/ui/button-group'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@render/ui/dropdown-menu'
import { cn } from '@render/lib/utils'
import {
  checkNumber,
  defaultSuggestions,
  formatHeaderValue,
  formatNumberValue,
  stepNumber,
  type NumberOptions,
} from '@render/state/number'
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
 * A typed number with − and + buttons, for values with no natural track: a seed, an episode count, a learning rate.
 * One field for every typed number: `type` float or int, bounds `gt`/`ge`/`lt`/`le` (`min`/`max` alias `ge`/`le`),
 * `scale` linear or log10, `spacing` ('lin' | 'log'), and `suggestions` (a menu of common values beside the field).
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

  return (
    <div className={cn('flex w-full max-w-xs min-w-40 flex-col gap-1.5', className)}>
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
          <Minus />
        </Button>
        <input
          id={id}
          disabled={disabled}
          data-slot="input"
          aria-describedby={error ? errorId : undefined}
          className={cn(
            'h-8 min-w-0 flex-1 border border-input bg-transparent px-2 text-right font-mono text-xs tabular-nums outline-none focus-visible:z-10 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-50 dark:bg-input/30',
            error && 'z-10 border-destructive focus-visible:border-destructive focus-visible:ring-destructive/30',
          )}
          {...field}
        />
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
          <Plus />
        </Button>
        {suggestions && suggestions.length > 0 ? (
          <DropdownMenu>
            <DropdownMenuTrigger
              disabled={disabled}
              render={<Button variant="outline" size="icon" aria-label="Suggested values" title="Suggested values" />}
            >
              <ChevronDown className="size-3.5" />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="max-h-60 w-auto min-w-24 overflow-y-auto">
              {suggestions.map((s) => (
                <DropdownMenuItem
                  key={s}
                  onClick={() => commit(s)}
                  className={cn('justify-end font-mono text-xs tabular-nums', s === value && 'bg-accent font-semibold')}
                >
                  {format(s)}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        ) : null}
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

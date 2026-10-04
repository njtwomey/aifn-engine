/**
 * Equations (DESIGN.md §7): the figure's equation band, set large in KaTeX with the live values substituted and
 * highlighted, and `EquationSteps` for walkthroughs (backprop's forward assignments, a chain of identities) driven by
 * a `Player`.
 *
 *   <Figure equation={<Equation>{tex`p_Y(${y}) = p_X(${x}) \cdot |1/g'(${x})| = ${pY}`}</Equation>} …>
 *
 * A slot is a number (formatted with the lab's number format), a string of TeX, or `live(value, { digits, strong })`.
 */
import katex from 'katex'
import { useMemo, type ReactNode } from 'react'
import { Player } from '@render/controls/Player'
import { cn } from '@render/lib/utils'
import { toTex, type EquationTemplate } from './equation-tex'
import { useRenderMathMacros } from './math-macros'

function render(t: EquationTemplate | string, display: boolean, macros?: Record<string, string>) {
  return katex.renderToString(toTex(t), {
    displayMode: display,
    throwOnError: false,
    output: 'html',
    // \htmlClass marks the live values; nothing else in a template is trusted.
    trust: (ctx) => ctx.command === '\\htmlClass',
    strict: false,
    macros: { ...(macros ?? {}) },
  })
}

/** One large equation with live values (about 1.6 × the body size). Children: a `tex` template or a TeX string. */
export function Equation({
  children,
  size = 'lg',
  className,
  macros: propMacros,
}: {
  children: EquationTemplate | string
  /** `lg` (default) for the band, `md` for an equation among other text. */
  size?: 'lg' | 'md'
  className?: string
  macros?: Record<string, string>
}) {
  const contextMacros = useRenderMathMacros()
  const effectiveMacros = useMemo(
    () => (propMacros ? { ...contextMacros, ...propMacros } : contextMacros),
    [contextMacros, propMacros],
  )
  const html = useMemo(() => render(children, true, effectiveMacros), [children, effectiveMacros])
  return (
    <div
      className={cn(
        'eq-left font-prose [&_.katex-display]:my-0 [&_.katex-display]:text-left',
        size === 'lg' ? 'text-[1.35rem]' : 'text-base',
        className,
      )}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

export type EquationStep = { tex: EquationTemplate | string; note?: ReactNode }

/**
 * A sequence of equations for a walkthrough: the steps up to `step` are shown, the current one at full strength and
 * the earlier ones faded, each with an optional note. Given `onStep`, a `Player` above it moves through the steps;
 * otherwise the figure's own player drives `step`.
 */
export function EquationSteps({
  steps,
  step,
  onStep,
  label = 'step',
  className,
  macros: propMacros,
}: {
  steps: readonly EquationStep[]
  step: number
  onStep?: (step: number) => void
  label?: ReactNode
  className?: string
  macros?: Record<string, string>
}) {
  const current = Math.max(0, Math.min(steps.length - 1, Math.round(step)))
  const contextMacros = useRenderMathMacros()
  const effectiveMacros = useMemo(
    () => (propMacros ? { ...contextMacros, ...propMacros } : contextMacros),
    [contextMacros, propMacros],
  )
  const html = useMemo(() => steps.map((s) => render(s.tex, true, effectiveMacros)), [steps, effectiveMacros])
  return (
    <div className={cn('flex flex-col gap-3', className)}>
      {onStep && (
        <Player
          label={label}
          value={current}
          onChange={onStep}
          count={steps.length}
          format={(p) => `${Math.round(p) + 1} / ${steps.length}`}
        />
      )}
      <ol className="flex flex-col gap-1.5">
        {steps.slice(0, current + 1).map((s, i) => (
          <li
            key={i}
            aria-current={i === current ? 'step' : undefined}
            className={cn(
              'grid grid-cols-[2rem_minmax(0,1fr)] items-baseline gap-2 transition-opacity',
              i === current ? 'opacity-100' : 'opacity-45',
            )}
          >
            <span className="text-right font-mono text-xs text-muted-foreground tabular-nums">{i + 1}</span>
            <div className="min-w-0">
              <div
                className={cn(
                  'eq-left font-prose [&_.katex-display]:my-0 [&_.katex-display]:text-left',
                  i === current ? 'text-[1.25rem]' : 'text-base',
                )}
                dangerouslySetInnerHTML={{ __html: html[i] }}
              />
              {s.note && i === current && <p className="mt-0.5 text-xs text-muted-foreground">{s.note}</p>}
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}

import { useMemo } from 'react'
import { cn } from '../lib/utils'
import { renderKatex, useRenderMathMacros } from './math-macros'

/**
 * Maths set by KaTeX, for labels, readouts and descriptions: `<Tex>{String.raw`\sigma^2`}</Tex>`. Invalid input renders
 * in red rather than throwing. `display` sets it as a centred block.
 */
export function Tex({
  children,
  display = false,
  className,
  macros: propMacros,
}: {
  children: string
  display?: boolean
  className?: string
  macros?: Record<string, string>
}) {
  const contextMacros = useRenderMathMacros()
  const effectiveMacros = useMemo(
    () => (propMacros ? { ...contextMacros, ...propMacros } : contextMacros),
    [contextMacros, propMacros],
  )
  const html = useMemo(
    () => renderKatex(children, { displayMode: display, throwOnError: false, output: 'html', macros: effectiveMacros }),
    [children, display, effectiveMacros],
  )
  const Tag = display ? 'div' : 'span'
  return <Tag className={cn('font-prose', className)} dangerouslySetInnerHTML={{ __html: html }} />
}

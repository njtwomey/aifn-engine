import { Tex } from '../layout'

/** Text with inline `$…$` maths: the maths is set by KaTeX, the rest as plain text. */
export function MathText({
  text,
  className,
  macros,
}: {
  text: string
  className?: string
  macros?: Record<string, string>
}) {
  const parts = text.split(/(\$[^$]+\$)/g)
  return (
    <>
      {parts.map((part, i) =>
        part.length > 2 && part.startsWith('$') && part.endsWith('$') ? (
          <Tex key={i} className={className} macros={macros}>
            {part.slice(1, -1)}
          </Tex>
        ) : (
          part
        ),
      )}
    </>
  )
}

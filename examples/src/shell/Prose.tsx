import { Fragment } from 'react'

/** A recipe's sentences, with `code` in backticks set as inline code. */
export function Prose({ text, className }: { text: string; className?: string }) {
  return (
    <p className={className}>
      {text.split(/(`[^`]+`)/).map((part, i) =>
        part.startsWith('`') ? (
          <code key={i} className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em] text-foreground">
            {part.slice(1, -1)}
          </code>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </p>
  )
}

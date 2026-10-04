import type { ReactNode } from 'react'
import { cn } from '@render/lib/utils'

/** The small muted label every control sits under. */
export function ControlLabel({
  children,
  htmlFor,
  id,
  title,
  className,
}: {
  children: ReactNode
  htmlFor?: string
  id?: string
  title?: string
  className?: string
}) {
  return (
    <label
      id={id}
      htmlFor={htmlFor}
      title={title}
      className={cn('text-xs leading-none text-muted-foreground select-none', className)}
    >
      {children}
    </label>
  )
}

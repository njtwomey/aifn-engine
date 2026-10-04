import { CircleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '@render/lib/utils'

/**
 * A one-line status beside a control (a training run's progress, a computation's state). `tone="error"` shows it in the
 * destructive colour with an alert icon, so a failure never reads like ordinary progress; `tone="attention"` is for a
 * state that asks the reader to act (not trained yet, settings changed); `muted` is ordinary progress.
 */
export function StatusText({
  tone = 'muted',
  children,
  className,
}: {
  tone?: 'muted' | 'attention' | 'error'
  children: ReactNode
  className?: string
}) {
  return (
    <span
      role={tone === 'error' ? 'alert' : 'status'}
      className={cn(
        'inline-flex items-center gap-1 text-xs tabular-nums',
        tone === 'error'
          ? 'font-medium text-destructive'
          : tone === 'attention'
            ? 'font-medium text-foreground'
            : 'text-muted-foreground',
        className,
      )}
    >
      {tone === 'error' && <CircleAlert className="size-3.5 shrink-0" aria-hidden />}
      <span className="break-words">{children}</span>
    </span>
  )
}

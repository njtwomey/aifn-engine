import type { ReactNode } from 'react'
import { cn } from '../lib/utils'

/** A labelled value under a figure, e.g. the current loss. Numbers are set in tabular mono so they do not jitter. */
export function Readout({ label, value, color }: { label: ReactNode; value: ReactNode; color?: string }) {
  return (
    <span className="inline-flex items-baseline gap-1.5">
      {color && <span aria-hidden className="size-2 shrink-0 self-center rounded-full" style={{ background: color }} />}
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono text-foreground tabular-nums">{value}</span>
    </span>
  )
}

/** A wrapping row of readouts. */
export function Readouts({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('flex flex-wrap gap-x-4 gap-y-1 text-[11px]', className)}>{children}</div>
}

/** A labelled group of readouts (a Figure's `readouts` given as a record draws one per entry). */
export function ReadoutGroup({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <div className="text-[11px] font-medium text-muted-foreground">{label}</div>
      <Readouts>{children}</Readouts>
    </div>
  )
}

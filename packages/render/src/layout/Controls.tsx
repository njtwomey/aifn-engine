import { useState, type ComponentType, type ReactNode } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '@render/lib/utils'

/**
 * The layout for a set of controls: a responsive multi-column grid whose items align neatly,
 * keeping controls space-efficient and avoiding vertical stretching.
 */
export function Controls({
  children,
  className,
  columns = 'default',
}: {
  children: ReactNode
  className?: string
  columns?: 'default' | 'compact' | 'wide' | 'single'
}) {
  const gridCols =
    columns === 'single'
      ? 'grid-cols-1'
      : columns === 'compact'
        ? 'grid-cols-[repeat(auto-fill,minmax(11rem,1fr))]'
        : columns === 'wide'
          ? 'grid-cols-[repeat(auto-fill,minmax(15rem,1fr))]'
          : 'grid-cols-[repeat(auto-fill,minmax(13rem,1fr))]'

  return (
    <div className={cn('grid w-full items-end gap-x-4 gap-y-2.5 *:w-full *:max-w-none', gridCols, className)}>
      {children}
    </div>
  )
}

/**
 * One full-width row of related controls inside a `Controls` grid.
 */
export function ControlRow({
  label,
  children,
  className,
}: {
  label?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <div className={cn('col-span-full flex w-full flex-col gap-1.5', className)}>
      {label && <div className="text-xs font-medium text-muted-foreground">{label}</div>}
      <Controls>{children}</Controls>
    </div>
  )
}

export type ControlGroupProps = {
  title?: ReactNode
  description?: ReactNode
  badge?: ReactNode
  icon?: ComponentType<{ className?: string }>
  collapsible?: boolean
  defaultCollapsed?: boolean
  collapsed?: boolean
  onToggleCollapsed?: (collapsed: boolean) => void
  children: ReactNode
  className?: string
  contentClassName?: string
}

/**
 * A structured group of related controls (e.g. Data, Model, Playback, Configuration),
 * with a small group title, subtle border, compact multi-column layout, and optional collapsing.
 */
export function ControlGroup({
  title,
  description,
  badge,
  icon: Icon,
  collapsible = true,
  defaultCollapsed = false,
  collapsed: controlledCollapsed,
  onToggleCollapsed,
  children,
  className,
  contentClassName,
}: ControlGroupProps) {
  const [internalCollapsed, setInternalCollapsed] = useState(defaultCollapsed)
  const isControlled = controlledCollapsed !== undefined
  const isCollapsed = isControlled ? controlledCollapsed : internalCollapsed

  const toggle = () => {
    if (!collapsible) return
    const next = !isCollapsed
    if (!isControlled) setInternalCollapsed(next)
    onToggleCollapsed?.(next)
  }

  const hasHeader = Boolean(title || description || badge || Icon || collapsible)

  return (
    <div
      className={cn(
        'col-span-full flex flex-col rounded-lg border border-border/60 bg-muted/20 text-card-foreground shadow-2xs transition-colors',
        isCollapsed ? 'px-3 py-1.5' : 'px-3 pt-1.5 pb-2.5',
        className,
      )}
    >
      {hasHeader && (
        <div
          role={collapsible ? 'button' : undefined}
          tabIndex={collapsible ? 0 : undefined}
          onClick={collapsible ? toggle : undefined}
          onKeyDown={
            collapsible
              ? (e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    toggle()
                  }
                }
              : undefined
          }
          className={cn(
            'flex items-center justify-between gap-2 select-none',
            collapsible && 'group cursor-pointer hover:text-foreground',
            !isCollapsed && 'mb-2 border-b border-border/30 pb-1',
          )}
        >
          <div className="flex min-w-0 items-center gap-1.5">
            {collapsible && (
              <span className="text-muted-foreground transition-transform group-hover:text-foreground">
                {isCollapsed ? <ChevronRight className="size-3" /> : <ChevronDown className="size-3" />}
              </span>
            )}
            {Icon && <Icon className="size-3 shrink-0 text-muted-foreground" />}
            <div className="flex min-w-0 items-baseline gap-2">
              {title && (
                <span className="font-sans text-[11px] font-semibold tracking-wider text-muted-foreground/90 uppercase group-hover:text-foreground">
                  {title}
                </span>
              )}
              {description && <p className="truncate font-sans text-[11px] text-muted-foreground/75">{description}</p>}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {badge && (
              <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] font-medium text-muted-foreground">
                {badge}
              </span>
            )}
            {collapsible && isCollapsed && (
              <span className="font-sans text-[10px] text-muted-foreground/70 italic">collapsed</span>
            )}
          </div>
        </div>
      )}
      {!isCollapsed && (
        <div className={cn('w-full', contentClassName)}>
          <Controls>{children}</Controls>
        </div>
      )}
    </div>
  )
}

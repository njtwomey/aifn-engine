import { ChevronRight, Search, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Button, Input, ThemeToggle, cn } from 'aifn-render'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from 'aifn-render/ui/collapsible'
import { SECTIONS } from '@examples/recipe'
import { ENTRIES } from './registry'
import { onLink } from './paths'

/**
 * The index: the gallery link, then each section's recipes as an accordion (one section open at a time; the current
 * page's section opens on navigation; its header closes it). A search opens every section with a match.
 */
export function Sidebar({ current, onClose }: { current: string; onClose: () => void }) {
  const [query, setQuery] = useState('')
  const currentSection = ENTRIES.find((e) => e.path === current)?.section ?? null
  const [open, setOpen] = useState<string | null>(currentSection)
  useEffect(() => {
    if (currentSection) setOpen(currentSection)
  }, [currentSection])
  const q = query.trim().toLowerCase()
  const shown = useMemo(
    () => ENTRIES.filter((e) => !q || `${e.title} ${e.question} ${e.section}`.toLowerCase().includes(q)),
    [q],
  )
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b px-4 py-3">
        <a href="/" onClick={onLink('')} className="flex-1 truncate text-sm font-semibold tracking-tight">
          aifn-render examples
        </a>
        <ThemeToggle />
        <Button variant="ghost" size="icon-sm" className="md:hidden" aria-label="Close index" onClick={onClose}>
          <X />
        </Button>
      </div>
      <div className="relative px-3 pt-3">
        <Search className="pointer-events-none absolute top-1/2 left-5.5 mt-1.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search recipes"
          aria-label="Search recipes"
          className="h-8 pl-7 text-sm"
        />
      </div>
      <nav aria-label="Recipes" className="min-h-0 flex-1 overflow-y-auto px-2 py-3 text-sm">
        <a
          href="/"
          onClick={onLink('')}
          aria-current={current === '' ? 'page' : undefined}
          className={cn(
            'mb-2 block rounded-md px-2 py-1.5 font-medium hover:bg-sidebar-accent',
            current === '' && 'bg-sidebar-accent',
          )}
        >
          Gallery
        </a>
        {SECTIONS.map((s) => {
          const items = shown.filter((e) => e.section === s.id)
          if (!items.length) return null
          return (
            <Collapsible
              key={s.id}
              open={q ? true : open === s.id}
              onOpenChange={(o) => setOpen(o ? s.id : null)}
              className="mb-1"
            >
              <CollapsibleTrigger className="group flex w-full items-center gap-1 rounded-md px-2 py-1.5 text-left text-xs font-medium tracking-wide text-muted-foreground uppercase outline-none hover:bg-sidebar-accent hover:text-sidebar-foreground focus-visible:ring-2 focus-visible:ring-ring/50">
                <ChevronRight className="size-3.5 shrink-0 transition-transform group-data-[panel-open]:rotate-90" />
                <span className="flex-1 truncate">{s.title}</span>
                <span className="font-normal normal-case tabular-nums">{items.length}</span>
              </CollapsibleTrigger>
              {/* Kept mounted (hidden when closed) so every recipe link stays in the DOM for the screenshot tool. */}
              <CollapsibleContent keepMounted className="pb-2 pl-3 data-[closed]:hidden">
                {items.map((e) => (
                  <a
                    key={e.path}
                    href={`/${e.path}`}
                    onClick={onLink(e.path)}
                    aria-current={e.path === current ? 'page' : undefined}
                    className={cn(
                      'block truncate rounded-md px-2 py-1 text-sidebar-foreground/85 hover:bg-sidebar-accent',
                      e.path === current && 'bg-sidebar-accent font-medium text-sidebar-foreground',
                    )}
                  >
                    {e.title}
                  </a>
                ))}
              </CollapsibleContent>
            </Collapsible>
          )
        })}
        {!shown.length && <p className="px-2 text-muted-foreground">No recipe matches.</p>}
      </nav>
    </div>
  )
}

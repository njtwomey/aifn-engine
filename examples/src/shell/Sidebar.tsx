import { ChevronRight, Search, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Button, Input, cn } from 'aifn-render'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from 'aifn-render/ui/collapsible'
import { SECTIONS } from '@examples/recipe'
import { filePath, nodePath, TREE, type DocPackage, type DocTreeNode } from '../docs/data'
import { ENTRIES } from './registry'
import { hrefOf, onLink } from './paths'
import type { Area } from './Shell'

const linkClass = (active: boolean, strong = false) =>
  cn(
    'block truncate rounded-md px-2 py-1 hover:bg-sidebar-accent',
    strong ? 'mb-2 py-1.5 font-medium' : 'text-sidebar-foreground/85',
    active && 'bg-sidebar-accent font-medium text-sidebar-foreground',
  )

const triggerClass =
  'group flex w-full items-center gap-1 rounded-md px-2 py-1.5 text-left text-xs font-medium tracking-wide text-muted-foreground uppercase outline-none hover:bg-sidebar-accent hover:text-sidebar-foreground focus-visible:ring-2 focus-visible:ring-ring/50'

/** The index of the package the page belongs to (the top bar links the packages), with a close button on narrow screens. */
export function Sidebar({ area, current, onClose }: { area: Area; current: string; onClose: () => void }) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex justify-end border-b px-2 py-2 md:hidden">
        <Button variant="ghost" size="icon-sm" aria-label="Close index" onClick={onClose}>
          <X />
        </Button>
      </div>
      {area === 'render' && <RecipeNav current={current} />}
      {(area === 'compute' || area === 'methods') && <DocsNav key={area} pkg={area} current={current} />}
    </div>
  )
}

/**
 * The gallery link, then each section's recipes as an accordion (one section open at a time; the current page's
 * section opens on navigation; its header closes it). A search opens every section with a match.
 */
function RecipeNav({ current }: { current: string }) {
  const [query, setQuery] = useState('')
  const currentSection = ENTRIES.find((e) => e.path === current)?.section ?? null
  const [open, setOpen] = useState<string | null>(currentSection)
  // Navigation opens the page's section: state adjusted during render when the section changes, not in an effect.
  const [seen, setSeen] = useState(currentSection)
  if (seen !== currentSection) {
    setSeen(currentSection)
    if (currentSection) setOpen(currentSection)
  }
  const q = query.trim().toLowerCase()
  const shown = useMemo(
    () => ENTRIES.filter((e) => !q || `${e.title} ${e.question} ${e.section}`.toLowerCase().includes(q)),
    [q],
  )
  return (
    <>
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
          href={hrefOf('render')}
          onClick={onLink('render')}
          aria-current={current === 'render' ? 'page' : undefined}
          className={linkClass(current === 'render', true)}
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
              <CollapsibleTrigger className={triggerClass}>
                <ChevronRight className="size-3.5 shrink-0 transition-transform group-data-[panel-open]:rotate-90" />
                <span className="flex-1 truncate">{s.title}</span>
                <span className="font-normal normal-case tabular-nums">{items.length}</span>
              </CollapsibleTrigger>
              {/* Kept mounted (hidden when closed) so every recipe link stays in the DOM for the screenshot tool. */}
              <CollapsibleContent keepMounted className="pb-2 pl-3 data-[closed]:hidden">
                {items.map((e) => (
                  <a
                    key={e.path}
                    href={hrefOf(e.path)}
                    onClick={onLink(e.path)}
                    aria-current={e.path === current ? 'page' : undefined}
                    className={linkClass(e.path === current)}
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
    </>
  )
}

/** A package's tree: each top-level node an accordion of its descendants; the current page's opens on navigation. */
function DocsNav({ pkg, current }: { pkg: DocPackage; current: string }) {
  const top = current.split('/')[1] ?? null
  const [open, setOpen] = useState<string | null>(top)
  const [seen, setSeen] = useState(top)
  if (seen !== top) {
    setSeen(top)
    if (top) setOpen(top)
  }
  const indent = (depth: number) => ({ paddingLeft: `${0.5 + depth * 0.75}rem` })
  const link = (n: DocTreeNode, depth: number) => {
    const path = nodePath(n)
    // A module's source files open beneath it while it (or one of them) is the page, and the page's file lists its functions.
    const inside = current === path || current.startsWith(`${path}/`)
    return (
      <div key={n.path}>
        <a
          href={hrefOf(path)}
          onClick={onLink(path)}
          aria-current={path === current ? 'page' : undefined}
          className={linkClass(path === current)}
          style={indent(depth)}
        >
          {n.name}
          {n.examples > 0 && <span className="ml-1.5 text-[10px] text-muted-foreground">● examples</span>}
        </a>
        {n.children.map((c) => link(c, depth + 1))}
        {inside &&
          n.files.map((f) => {
            const at = filePath(n, f)
            return (
              <div key={f.name}>
                <a
                  href={hrefOf(at)}
                  onClick={onLink(at)}
                  aria-current={at === current ? 'page' : undefined}
                  className={cn(linkClass(at === current), 'font-mono text-xs')}
                  style={indent(depth + 1)}
                >
                  {f.name}.ts
                  {f.examples > 0 && <span className="ml-1.5 text-[10px] text-muted-foreground">●</span>}
                </a>
                {at === current &&
                  [
                    ...f.key.map((name) => ({ name, key: true })),
                    ...f.supporting.map((name) => ({ name, key: false })),
                  ].map(({ name, key }) => (
                    <a
                      key={name}
                      href={`${hrefOf(at)}#${name}`}
                      onClick={onLink(at, name)}
                      className={cn(
                        'block truncate rounded-md py-0.5 font-mono text-xs hover:bg-sidebar-accent hover:text-sidebar-foreground',
                        key ? 'text-sidebar-foreground/80' : 'text-sidebar-foreground/45',
                      )}
                      style={indent(depth + 2)}
                    >
                      {name}
                    </a>
                  ))}
              </div>
            )
          })}
      </div>
    )
  }
  return (
    <nav aria-label="Modules" className="min-h-0 flex-1 overflow-y-auto px-2 py-3 text-sm">
      <a
        href={hrefOf(pkg)}
        onClick={onLink(pkg)}
        aria-current={current === pkg ? 'page' : undefined}
        className={linkClass(current === pkg, true)}
      >
        Overview
      </a>
      {TREE[pkg].map((n) => (
        <Collapsible
          key={n.path}
          open={open === n.name}
          onOpenChange={(o) => setOpen(o ? n.name : null)}
          className="mb-1"
        >
          <CollapsibleTrigger className={triggerClass}>
            <ChevronRight className="size-3.5 shrink-0 transition-transform group-data-[panel-open]:rotate-90" />
            <span className="flex-1 truncate">{n.name}</span>
            <span className="font-normal normal-case tabular-nums">{n.children.length || ''}</span>
          </CollapsibleTrigger>
          <CollapsibleContent className="pb-2 pl-3">
            {n.children.length === 0 && (
              <a
                href={hrefOf(nodePath(n))}
                onClick={onLink(nodePath(n))}
                aria-current={nodePath(n) === current ? 'page' : undefined}
                className={linkClass(nodePath(n) === current)}
              >
                {n.name}
              </a>
            )}
            {n.children.map((c) => link(c, 0))}
          </CollapsibleContent>
        </Collapsible>
      ))}
    </nav>
  )
}

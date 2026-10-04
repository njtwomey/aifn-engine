import { Menu } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Button, cn } from 'aifn-render'
import { Gallery } from './Gallery'
import { currentPath } from './paths'
import { RecipePage } from './RecipePage'
import { ENTRIES } from './registry'
import { Sidebar } from './Sidebar'

/** The app: the sidebar index beside the gallery or one recipe, chosen by the URL path. */
export function Shell() {
  const [path, setPath] = useState(() => (typeof location === 'undefined' ? '' : currentPath()))
  const [open, setOpen] = useState(false)
  const main = useRef<HTMLElement>(null)
  useEffect(() => {
    const onPop = () => {
      setPath(currentPath())
      setOpen(false)
      if (!location.hash) main.current?.scrollTo({ top: 0 })
    }
    addEventListener('popstate', onPop)
    return () => removeEventListener('popstate', onPop)
  }, [])
  // A gallery link to a section (`/#lines`) or a figure anchor (`#<figure-id>`): scroll it into view once drawn.
  useEffect(() => {
    if (!location.hash) return
    const frame = requestAnimationFrame(() =>
      document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView({ block: 'start' }),
    )
    return () => cancelAnimationFrame(frame)
  }, [path])
  const entry = ENTRIES.find((e) => e.path === path)
  return (
    <div className="flex h-svh overflow-hidden bg-background text-foreground">
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 w-72 shrink-0 border-r bg-sidebar text-sidebar-foreground transition-transform md:static md:translate-x-0',
          open ? 'translate-x-0 shadow-xl' : '-translate-x-full',
        )}
      >
        <Sidebar current={entry ? entry.path : ''} onClose={() => setOpen(false)} />
      </aside>
      {open && <div className="fixed inset-0 z-30 bg-black/30 md:hidden" onClick={() => setOpen(false)} />}
      <main ref={main} className="min-w-0 flex-1 overflow-y-auto overscroll-contain">
        <div className="sticky top-0 z-20 flex items-center gap-2 border-b bg-background/90 px-4 py-2 backdrop-blur md:hidden">
          <Button variant="ghost" size="icon-sm" aria-label="Open index" onClick={() => setOpen(true)}>
            <Menu />
          </Button>
          <span className="truncate text-sm font-medium">{entry?.title ?? 'Gallery'}</span>
        </div>
        <div className="mx-auto flex w-full max-w-[1100px] flex-col gap-6 px-4 py-6 md:px-8">
          {entry ? <RecipePage entry={entry} /> : <Gallery />}
        </div>
      </main>
    </div>
  )
}

import { Menu } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Button, cn } from 'aifn-render'
import { fileAt, nodeAt, PACKAGES, type DocPackage } from '../docs/data'
import { FilePage, NodePage, PackagePage } from '../docs/DocsPages'
import { Gallery } from './Gallery'
import { Home } from './Home'
import { currentPath } from './paths'
import { RecipePage } from './RecipePage'
import { ENTRIES } from './registry'
import { Sidebar } from './Sidebar'

export type Area = 'home' | DocPackage | 'render'

/** The page a URL path names: its area (which the sidebar follows), a title and the page. */
function route(path: string): { area: Area; title: string; page: ReactNode } {
  const head = path.split('/')[0]
  const entry = ENTRIES.find((e) => e.path === path)
  if (entry) return { area: 'render', title: entry.title, page: <RecipePage entry={entry} /> }
  if (head === 'render') return { area: 'render', title: 'Render Gallery', page: <Gallery /> }
  if (head === 'compute' || head === 'methods') {
    const node = nodeAt(path)
    const at = node ? undefined : fileAt(path)
    if (at) return { area: head, title: `${at.file.name}.ts`, page: <FilePage node={at.node} file={at.file} /> }
    return node
      ? { area: head, title: node.name, page: <NodePage node={node} /> }
      : { area: head, title: PACKAGES[head].title, page: <PackagePage pkg={head} /> }
  }
  return { area: 'home', title: 'AIFN Engine', page: <Home /> }
}

/** The app: the sidebar beside the page the URL path names. */
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
  // A link to a gallery section (`/render#lines`), an export or a figure anchor: scroll it into view once drawn.
  useEffect(() => {
    if (!location.hash) return
    const frame = requestAnimationFrame(() =>
      document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView({ block: 'start' }),
    )
    return () => cancelAnimationFrame(frame)
  }, [path])
  const { area, title, page } = route(path)
  return (
    <div className="flex h-svh overflow-hidden bg-background text-foreground">
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 w-72 shrink-0 border-r bg-sidebar text-sidebar-foreground transition-transform md:static md:translate-x-0',
          open ? 'translate-x-0 shadow-xl' : '-translate-x-full',
        )}
      >
        <Sidebar area={area} current={path} onClose={() => setOpen(false)} />
      </aside>
      {open && <div className="fixed inset-0 z-30 bg-black/30 md:hidden" onClick={() => setOpen(false)} />}
      <main ref={main} className="min-w-0 flex-1 overflow-y-auto overscroll-contain">
        <div className="sticky top-0 z-20 flex items-center gap-2 border-b bg-background/90 px-4 py-2 backdrop-blur md:hidden">
          <Button variant="ghost" size="icon-sm" aria-label="Open index" onClick={() => setOpen(true)}>
            <Menu />
          </Button>
          <span className="truncate text-sm font-medium">{title}</span>
        </div>
        <div className="mx-auto flex w-full max-w-[1100px] flex-col gap-6 px-4 py-6 md:px-8">{page}</div>
      </main>
    </div>
  )
}

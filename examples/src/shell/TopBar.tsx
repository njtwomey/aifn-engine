import { Code, Menu } from 'lucide-react'
import { Button, ThemeToggle, cn } from 'aifn-render'
import { hrefOf, onLink } from './paths'
import type { Area } from './Shell'

const REPOSITORY = 'https://github.com/njtwomey/aifn-engine'

/** The packages, in the order the bar lists them; `short` is the label on narrow screens. */
const NAV = [
  { id: 'compute', label: 'Compute', short: 'Compute' },
  { id: 'methods', label: 'Methods', short: 'Methods' },
  { id: 'render', label: 'Render Gallery', short: 'Render' },
] as const

/**
 * The site's top bar on every page: the name (home), the three packages with the current one marked, the theme toggle
 * and the source. `onMenu`, on pages with a sidebar, adds the button that opens it on narrow screens; `contained`
 * keeps the bar's content to the landing page's width.
 */
export function TopBar({ area, onMenu, contained }: { area: Area; onMenu?: () => void; contained?: boolean }) {
  return (
    <header className="sticky top-0 z-20 shrink-0 border-b bg-background/80 backdrop-blur">
      <div
        className={cn(
          'flex h-14 w-full items-center gap-1 px-3 md:px-6',
          contained && 'mx-auto max-w-[1200px] md:px-8',
        )}
      >
        {onMenu && (
          <Button variant="ghost" size="icon-sm" className="md:hidden" aria-label="Open index" onClick={onMenu}>
            <Menu />
          </Button>
        )}
        <a
          href={hrefOf('')}
          onClick={onLink('')}
          className="mr-auto flex items-center gap-2 px-1 text-sm font-semibold tracking-tight"
        >
          <span className="size-3 rounded-full bg-gradient-to-br from-[#2a78d6] via-[#1baf7a] to-[#eb6834]" />
          <span className="hidden sm:inline">AIFN Engine</span>
        </a>
        <nav aria-label="Packages" className="flex items-center gap-0.5 text-sm">
          {NAV.map((n) => (
            <a
              key={n.id}
              href={hrefOf(n.id)}
              onClick={onLink(n.id)}
              aria-current={area === n.id ? 'page' : undefined}
              className={cn(
                'rounded-md px-2 py-1.5 hover:bg-accent hover:text-foreground sm:px-3',
                area === n.id ? 'bg-accent font-medium text-foreground' : 'text-muted-foreground',
              )}
            >
              <span className="sm:hidden">{n.short}</span>
              <span className="hidden sm:inline">{n.label}</span>
            </a>
          ))}
        </nav>
        <ThemeToggle />
        <Button
          variant="ghost"
          size="icon-sm"
          className="hidden sm:inline-flex"
          aria-label="Source on GitHub"
          nativeButton={false}
          render={<a href={REPOSITORY} />}
        >
          <Code />
        </Button>
      </div>
    </header>
  )
}

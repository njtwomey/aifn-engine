/**
 * URLs: `/` is the front page; `/render` the gallery and `/render/<section>/<slug>` a recipe (`#<figure-id>` a figure
 * on it); `/compute/<family>/<module>` and `/methods/<area>/…` the generated pages of those packages, all under Vite's `base` (`/`
 * in development; the repository path when the gallery is published to GitHub Pages).
 */

const BASE = import.meta.env.BASE_URL

/** The URL of an in-app path (`''` is the front page). */
export const hrefOf = (path: string) => `${BASE}${path}`

/** The current path without slashes at either end. */
export const currentPath = () => {
  const { pathname } = location
  const local = pathname.startsWith(BASE) ? pathname.slice(BASE.length) : pathname
  return decodeURIComponent(local.replace(/^\/+|\/+$/g, ''))
}

/** Opens a path in place (the shell listens for `popstate`), at the element with id `hash` when one is given. */
export function go(path: string, hash = '') {
  const url = hash ? `${hrefOf(path)}#${hash}` : hrefOf(path)
  if (url !== `${location.pathname}${location.hash}`) history.pushState(null, '', url)
  dispatchEvent(new PopStateEvent('popstate'))
  // The shell scrolls to a hash when the path changes; on the same page, scroll here.
  if (hash) requestAnimationFrame(() => document.getElementById(hash)?.scrollIntoView({ block: 'start' }))
}

/** Follow a plain click on an in-app link without reloading. */
export function onLink(path: string, hash = '') {
  return (e: { metaKey: boolean; ctrlKey: boolean; shiftKey: boolean; button: number; preventDefault: () => void }) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
    e.preventDefault()
    go(path, hash)
  }
}

/** The theme-specific thumbnail of a recipe (`public/thumbs`). */
export const thumbnailOf = (entry: { section: string; slug: string }, theme: 'light' | 'dark') =>
  `${BASE}thumbs/${entry.section}/${entry.slug}-${theme}.png`

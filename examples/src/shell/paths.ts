/**
 * URLs: `/` is the gallery, `/<section>/<slug>` a recipe, `#<figure-id>` a figure on it, all under Vite's `base` (`/`
 * in development; the repository path when the gallery is published to GitHub Pages).
 */

const BASE = import.meta.env.BASE_URL

/** The URL of an in-app path (`''` is the gallery). */
export const hrefOf = (path: string) => `${BASE}${path}`

/** The current path without slashes at either end. */
export const currentPath = () => {
  const { pathname } = location
  const local = pathname.startsWith(BASE) ? pathname.slice(BASE.length) : pathname
  return decodeURIComponent(local.replace(/^\/+|\/+$/g, ''))
}

/** Opens a path in place (the shell listens for `popstate`). */
export function go(path: string) {
  if (hrefOf(path) !== location.pathname || location.hash) history.pushState(null, '', hrefOf(path))
  dispatchEvent(new PopStateEvent('popstate'))
}

/** Follow a plain click on an in-app link without reloading. */
export function onLink(path: string) {
  return (e: { metaKey: boolean; ctrlKey: boolean; shiftKey: boolean; button: number; preventDefault: () => void }) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
    e.preventDefault()
    go(path)
  }
}

/** The theme-specific thumbnail of a recipe (`public/thumbs`). */
export const thumbnailOf = (path: string, theme: 'light' | 'dark') => `${BASE}thumbs/${path}-${theme}.png`

/** URLs: `/` is the gallery, `/<section>/<slug>` a recipe, `#<figure-id>` a figure on it. */

/** The current path without slashes at either end. */
export const currentPath = () => decodeURIComponent(location.pathname.replace(/^\/+|\/+$/g, ''))

/** Opens a path in place (the shell listens for `popstate`). */
export function go(path: string) {
  if (`/${path}` !== location.pathname || location.hash) history.pushState(null, '', `/${path}`)
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
export const thumbnailOf = (path: string, theme: 'light' | 'dark') => `/thumbs/${path}-${theme}.png`

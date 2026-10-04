/**
 * Figure state in the URL (DESIGN.md §4, decision 1): each figure's non-default values are query parameters named by
 * the figure's anchor, `?change-of-variables.input=gamma&change-of-variables.input.shape=2#change-of-variables`, so a
 * link reproduces the view and several figures on one page keep their own values. Navigating to another page drops
 * them (the shell pushes a bare path).
 */

const hasLocation = () => typeof location !== 'undefined' && typeof history !== 'undefined'

/** The `[path, text]` pairs the URL holds for figure `id`. */
export function readFigureEntries(id: string): [string, string][] {
  if (!hasLocation()) return []
  const prefix = `${id}.`
  return [...new URLSearchParams(location.search)]
    .filter(([k]) => k.startsWith(prefix))
    .map(([k, v]): [string, string] => [k.slice(prefix.length), v])
}

const pending = new Map<string, [string, string][]>()
let pendingPath = ''
let timer: ReturnType<typeof setTimeout> | undefined

/**
 * Replace figure `id`'s parameters in the URL with `entries` (an empty list removes them). Writes are gathered and
 * applied once the values have been still for a moment, so a drag does not flood the history API; `replaceState`
 * keeps the back button for pages, not slider positions.
 */
export function writeFigureEntries(id: string, entries: [string, string][]): void {
  if (!hasLocation()) return
  if (pendingPath !== location.pathname) pending.clear()
  pendingPath = location.pathname
  pending.set(id, entries)
  clearTimeout(timer)
  timer = setTimeout(flush, 150)
}

function flush() {
  // The reader left the page before the write: its figures' values do not belong to the new page.
  if (pendingPath !== location.pathname) return pending.clear()
  const params = new URLSearchParams(location.search)
  for (const [id, entries] of pending) {
    const prefix = `${id}.`
    for (const k of [...params.keys()]) if (k.startsWith(prefix)) params.delete(k)
    for (const [k, v] of entries) params.append(prefix + k, v)
  }
  pending.clear()
  const query = params.toString()
  const url = `${location.pathname}${query ? `?${query}` : ''}${location.hash}`
  if (url !== `${location.pathname}${location.search}${location.hash}`) history.replaceState(history.state, '', url)
}

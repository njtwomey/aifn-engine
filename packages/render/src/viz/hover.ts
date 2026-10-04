/**
 * Linked hover. Charts given the same `hoverGroup` share the hovered x value: hovering one moves the axis pointer (and
 * tooltip) of the others to the same x, e.g. a trace's loss, weight and timing panels. A plain module-level bus; it
 * carries data values, not pixels, so linked charts may differ in size and range.
 */
type Listener = (x: number | null, source: symbol) => void

const groups = new Map<string, Set<Listener>>()

export function subscribeHover(group: string, listener: Listener): () => void {
  let set = groups.get(group)
  if (!set) groups.set(group, (set = new Set()))
  set.add(listener)
  return () => {
    set.delete(listener)
    if (!set.size) groups.delete(group)
  }
}

export function publishHover(group: string, x: number | null, source: symbol): void {
  for (const listener of groups.get(group) ?? []) listener(x, source)
}

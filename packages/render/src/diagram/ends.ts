import type { Side } from './types'

const SIDES = new Set<string>(['n', 's', 'e', 'w'])

/**
 * An edge end `id` or `id:side`. Node ids may themselves contain `:` (compute's structured-model diagrams use variable
 * names as ids), so only a final `:n|s|e|w` is read as a port; anything else is part of the id.
 */
export function parseEnd(ref: string): { id: string; side?: Side } {
  const at = ref.lastIndexOf(':')
  if (at >= 0 && SIDES.has(ref.slice(at + 1))) return { id: ref.slice(0, at), side: ref.slice(at + 1) as Side }
  return { id: ref }
}

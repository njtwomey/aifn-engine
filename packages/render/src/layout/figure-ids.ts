import { createContext, useContext, useId } from 'react'
import { slugify } from './slugify'

/** Hands out figure ids within one page: a title's slug, then `-2`, `-3`, … for repeats. */
export type FigureIds = { claim: (key: string, base: string) => string }

export function createFigureIds(): FigureIds {
  const byKey = new Map<string, string>()
  const taken = new Set<string>()
  return {
    claim(key, base) {
      const known = byKey.get(key)
      if (known) return known
      let id = base
      for (let n = 2; taken.has(id); n++) id = `${base}-${n}`
      taken.add(id)
      byKey.set(key, id)
      return id
    },
  }
}

export const FigureIdsContext = createContext<FigureIds | null>(null)

/** A figure's anchor id: `id`, or the slug of `title`, made unique within its page. */
export function useFigureId(title: string, id?: string): string {
  const ids = useContext(FigureIdsContext)
  const key = useId()
  const base = id ?? slugify(title)
  return ids ? ids.claim(key, base) : base
}

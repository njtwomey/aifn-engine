/**
 * Every recipe, found by globbing `recipes/<section>/<NN-slug>.tsx`: the module (metadata and live example) and its own
 * source as text. Adding a recipe is adding a file; nothing here changes.
 */
import type { ComponentType } from 'react'
import { SECTIONS, type Recipe, type SectionId } from '@examples/recipe'
import { snippetOf } from './snippet'

type RecipeModule = { recipe: Recipe; default: ComponentType }

const modules = import.meta.glob<RecipeModule>('../recipes/*/*.tsx', { eager: true })
const sources = import.meta.glob<string>('../recipes/*/*.tsx', { eager: true, query: '?raw', import: 'default' })

export type Entry = Recipe & {
  section: SectionId
  /** The file name without its order prefix, e.g. `line-chart`. */
  slug: string
  /** The URL path without the leading slash: `<section>/<slug>`. */
  path: string
  /** The file, relative to the app, e.g. `src/recipes/lines/01-line-chart.tsx`. */
  file: string
  Example: ComponentType
  snippet: string
}

const sectionIds = new Set<string>(SECTIONS.map((s) => s.id))

export const ENTRIES: readonly Entry[] = Object.entries(modules)
  .map(([key, mod]): Entry => {
    const [, section, name] = /\/recipes\/([^/]+)\/([^/]+)\.tsx$/.exec(key)!
    if (!sectionIds.has(section)) throw new Error(`${key}: no section '${section}' in SECTIONS (src/recipe.ts)`)
    if (!mod.recipe) throw new Error(`${key}: exports no \`recipe\``)
    const slug = name.replace(/^\d+-/, '')
    return {
      ...mod.recipe,
      section: section as SectionId,
      slug,
      path: `${section}/${slug}`,
      file: `src/recipes/${section}/${name}.tsx`,
      Example: mod.default,
      snippet: snippetOf(sources[key] ?? ''),
    }
  })
  .sort(
    (a, b) =>
      SECTIONS.findIndex((s) => s.id === a.section) - SECTIONS.findIndex((s) => s.id === b.section) ||
      a.file.localeCompare(b.file),
  )

export const entriesOf = (section: SectionId) => ENTRIES.filter((e) => e.section === section)

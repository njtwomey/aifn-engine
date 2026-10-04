/**
 * The name-collision lint (design S §8, consolidation N5): no two aifn modules (compute or applications) export different
 * values under one name, so an import by name is never ambiguous across modules. Re-exports of the same value are
 * fine. Today's known collisions are allowlisted below; the list may only shrink. Run by `make names`.
 */
import fs from 'node:fs'
import path from 'node:path'
import { expect, it } from 'vitest'

type AppNode = { module: string; status?: string } | { group: string; children: AppNode[] }
type Tree = {
  compute: { families: { family: string; modules: { module: string; status?: string }[] }[] }
  applications: { areas: { area: string; children: AppNode[] }[] }
  aliases: { path: string }[]
}
const spec = JSON.parse(fs.readFileSync(path.join(import.meta.dirname, '../../../modules.json'), 'utf8')) as Tree

/** The public paths of an application node's descendants (groups and modules). */
const leaves = (prefix: string, list: readonly AppNode[]): string[] =>
  list.flatMap((n) =>
    'module' in n
      ? n.status === 'gap'
        ? []
        : [`${prefix}/${n.module}`]
      : [`${prefix}/${n.group}`, ...leaves(`${prefix}/${n.group}`, n.children)],
  )

// Names allowed to collide.
const ALLOWED: readonly string[] = ['bernoulliKl']

// Importing every module of both packages takes several seconds when it runs alongside the rest of the suite.
it('no two aifn modules export different values under one name', async () => {
  const modules = [
    // Every node: family and group indexes hold shared layers (and re-export their children's values unchanged).
    ...spec.compute.families.flatMap((f) => [
      `aifn-compute/${f.family}`,
      ...f.modules.filter((m) => m.status !== 'gap').map((m) => `aifn-compute/${f.family}/${m.module}`),
    ]),
    ...spec.applications.areas.flatMap((a) => [
      `aifn-methods/${a.area}`,
      ...leaves(`aifn-methods/${a.area}`, a.children),
    ]),
    ...spec.aliases.map((a) => a.path),
  ]
  const owners = new Map<string, Map<unknown, string[]>>()
  for (const id of modules) {
    const ns = (await import(id)) as Record<string, unknown>
    for (const [name, value] of Object.entries(ns)) {
      if (value === undefined) continue
      const byValue = owners.get(name) ?? new Map<unknown, string[]>()
      byValue.set(value, [...(byValue.get(value) ?? []), id])
      owners.set(name, byValue)
    }
  }
  const collisions = [...owners]
    .filter(([, byValue]) => byValue.size > 1)
    .map(([name, byValue]) => `${name}: ${[...byValue.values()].map((ids) => ids.join(' = ')).join(' ≠ ')}`)
    .sort()
  const names = collisions.map((c) => c.slice(0, c.indexOf(':')))
  expect(collisions.filter((_, k) => !ALLOWED.includes(names[k]))).toEqual([])
  expect(
    ALLOWED.filter((n) => !names.includes(n)),
    'resolved collisions to remove from ALLOWED',
  ).toEqual([])
}, 60_000)

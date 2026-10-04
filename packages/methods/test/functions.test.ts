/**
 * Every `function` entry registered by the applications (found as the catalog finds entries: exported values with an
 * `info`, or the values of an exported table of entries) is a function named by its key, with a known role.
 * Compute's functions are checked by `compute/test/foundation/registry/conformance.test.ts`.
 */
import fs from 'node:fs'
import path from 'node:path'
import { expect, it } from 'vitest'
import { isEntry, type FunctionInfo } from 'aifn-compute/foundation/registry'

const src = path.join(import.meta.dirname, '../src')
const modules: string[] = []
const walk = (dir: string) => {
  for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!d.isDirectory() || d.name.startsWith('_')) continue
    const full = path.join(dir, d.name)
    if (fs.existsSync(path.join(full, 'index.ts'))) modules.push(path.relative(src, full).split(path.sep).join('/'))
    walk(full)
  }
}
walk(src)

const ROLES = ['transform', 'estimator', 'test', 'construction', 'property', 'fit', 'simulation', 'solver', 'inference']

it('every application function is a function named by its key, with a role', async () => {
  const found = new Map<unknown, string>()
  for (const mod of modules.sort()) {
    const ns = (await import(`aifn-methods/${mod}`)) as Record<string, unknown>
    for (const v of Object.values(ns)) {
      const candidates =
        v !== null && typeof v === 'object' && !isEntry(v) && Object.getPrototypeOf(v) === Object.prototype
          ? Object.values(v)
          : [v]
      for (const c of candidates) if (isEntry<FunctionInfo>(c, 'function') && !found.has(c)) found.set(c, mod)
    }
  }
  expect(found.size).toBeGreaterThan(0)
  const problems: string[] = []
  for (const [value, mod] of found) {
    const { info } = value as { info: FunctionInfo }
    const where = `${mod}: ${info.module}/${info.key}`
    if (typeof value !== 'function') problems.push(`${where} is not a function`)
    else if ((value as { name: string }).name !== info.key)
      problems.push(`${where} is named ${(value as { name: string }).name}`)
    if (!ROLES.includes(info.role)) problems.push(`${where} has role ${info.role}`)
  }
  expect(problems).toEqual([])
}, 60_000)

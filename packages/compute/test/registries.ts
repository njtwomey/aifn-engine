/**
 * The registry entries of the compute, loaded as a catalog would load them: every module with an
 * `index.ts` is imported, and an entry is an exported value carrying `info` with a `kind`, or a value of an exported
 * table of entries. The generated tests (`foundation/trace/algorithms.test.ts`, `probability/distributions/families.test.ts`, `foundation/registry/conformance.test.ts`) iterate
 * these, so a newly registered entry is tested without editing a list.
 */
import fs from 'node:fs'
import path from 'node:path'
import { isEntry, type Entry, type Info } from 'aifn-compute/foundation/registry'

const src = path.resolve(import.meta.dirname, '..', 'src')

/** Module paths of the compute: directories under src holding an index.ts. */
function modules(): string[] {
  const found: string[] = []
  const walk = (dir: string) => {
    for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
      if (!d.isDirectory() || d.name.startsWith('_')) continue
      const full = path.join(dir, d.name)
      if (fs.existsSync(path.join(full, 'index.ts'))) found.push(path.relative(src, full).split(path.sep).join('/'))
      walk(full)
    }
  }
  walk(src)
  return found.sort()
}

let cache: Entry<unknown, Info>[] | null = null

/** Every registry entry of the compute, each once. */
export async function coreEntries(): Promise<Entry<unknown, Info>[]> {
  if (cache) return cache
  const seen = new Set<Entry<unknown, Info>>()
  for (const mod of modules()) {
    const ns = (await import(/* @vite-ignore */ `aifn-compute/${mod}`)) as Record<string, unknown>
    for (const v of Object.values(ns)) {
      if (isEntry(v)) seen.add(v)
      else if (v !== null && typeof v === 'object' && Object.getPrototypeOf(v) === Object.prototype) {
        const values = Object.values(v)
        if (values.length > 0 && values.every((x) => isEntry(x))) values.forEach((x) => seen.add(x))
      }
    }
  }
  cache = [...seen]
  return cache
}

/** The entries of one kind, with their info typed. */
export async function entriesOf<I extends Info>(kind: I['kind']): Promise<Entry<unknown, I>[]> {
  return (await coreEntries()).filter((e) => e.info.kind === kind) as Entry<unknown, I>[]
}

/** `module/key`, the address of an entry within its kind. */
export const address = (e: { info: Info }): string => `${e.info.module}/${e.info.key}`

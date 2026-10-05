/**
 * How the compute worker finds an engine module by its path (`numerics/linalg`, or `applied/learning` for one of
 * `aifn-methods`): a loader that imports it on first use.
 *
 * In this repository the modules are found by globbing the sibling packages' source, so a new module is reachable
 * without listing it. In the packaged `aifn-render` this file is replaced by one generated at build time, with one
 * `import('aifn-compute/…')` or `import('aifn-methods/…')` per module (`scripts/package.ts`), because a package cannot
 * reach into another's source.
 */
type Loader = () => Promise<Record<string, unknown>>

// Every engine module (a directory with an index.ts; `_` folders are private), imported lazily by path.
const MODULES = {
  ...import.meta.glob(['../../../compute/src/**/index.ts', '!**/_*/**']),
  ...import.meta.glob(['../../../methods/src/**/index.ts', '!**/_*/**']),
} as Record<string, Loader>

const COMPUTE = '../../../compute/src/'
const METHODS = '../../../methods/src/'

/** The loader of the module at `path`, from compute first, else from methods (with or without `applied/`). */
export function loaderOf(path: string): Loader | undefined {
  return MODULES[`${COMPUTE}${path}/index.ts`] ?? MODULES[`${METHODS}${path.replace(/^applied\//, '')}/index.ts`]
}

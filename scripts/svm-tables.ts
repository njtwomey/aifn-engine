/**
 * `node scripts/svm-tables.ts [task ...]`: tabulate the cross-validated error of an RBF support vector machine over
 * the grid of `scripts/svm-surface.ts` on several small binary classification tasks, once, for `tuningTables`
 * (`aifn-methods/data/objectives`): the configuration-by-task tables of warm-starting and portfolio construction.
 *
 * - Tasks: the first is iris (versicolor against virginica), whose table is `svm-table.ts` itself and is not
 *   recomputed here. The others are listed in `TASKS` below, each with its data, 100 to 150 rows, the labels 0 and
 *   1, and each feature standardised by its mean and population standard deviation over the task's rows.
 * - Model and validation as `scripts/svm-surface.ts`: `supportVectorMachine` with $C$ and the RBF kernel
 *   $\exp(-\gamma \lVert \xvec - \xvec' \rVert^2)$, scored by `errorRate` over ten repeats of shuffled stratified
 *   5-fold cross-validation, the same splits for every cell of a task (one stream per task, `svm-tables/<task>`).
 * - Grid: $\log_{10} C$ from $-2$ to 3 and $\log_{10}\gamma$ from $-3$ to 2, in steps of $\tfrac14$ (21 by 21).
 *
 * With task names as arguments, only those tasks are computed and the others are kept from the existing file, so the
 * tables can be built one task at a time; with none, every task is computed. It writes
 * `packages/methods/src/data/objectives/svm-tables.ts`, formatted by Prettier; each task takes a few minutes. The
 * source is loaded through Vite's SSR loader, as in `scripts/svm-surface.ts`.
 */
import fs from 'node:fs'
import path from 'node:path'
import { format, resolveConfig } from 'prettier'
import { createServer } from 'vite'

const root = path.join(import.meta.dirname, '..')
const out = path.join(root, 'packages', 'methods', 'src', 'data', 'objectives', 'svm-tables.ts')

const LOG_C = Array.from({ length: 21 }, (_, i) => -2 + i / 4)
const LOG_GAMMA = Array.from({ length: 21 }, (_, i) => -3 + i / 4)
const FOLDS = 5
const REPEATS = 10

type Load = (id: string) => Promise<Record<string, any>>
type Task = { name: string; description: string; data: (load: Load) => Promise<{ rows: number[][]; labels: number[] }> }

/** Rows and labels of a generated `Dataset`. */
const rowsOf = async (load: Load, d: { x: unknown; y: unknown }) => {
  const { toArray } = await load('aifn-compute/foundation/tensor')
  return { rows: toArray(d.x) as number[][], labels: (toArray(d.y) as number[]).map(Number) }
}

/** The tasks after iris, in the order of the table. Each draws from its own stream, so each is reproducible alone. */
const TASKS: Task[] = [
  {
    name: 'moons',
    description: 'two moons, 100 points with noise sd 0.2 (`moons`)',
    data: async (load) => {
      const { moons } = await load('aifn-methods/data/synthetic')
      const { stream } = await load('aifn-compute/foundation/random')
      return rowsOf(load, moons(stream('svm-tables/moons/data'), { n: 100, noise: 0.2 }))
    },
  },
  {
    name: 'circles',
    description: 'two concentric circles, 100 points, inner radius 0.5, noise sd 0.15 (`circles`)',
    data: async (load) => {
      const { circles } = await load('aifn-methods/data/synthetic')
      const { stream } = await load('aifn-compute/foundation/random')
      return rowsOf(load, circles(stream('svm-tables/circles/data'), { n: 100, factor: 0.5, noise: 0.15 }))
    },
  },
  {
    name: 'xor',
    description: 'Gaussian XOR, 50 points per class, blob sd 0.5 (`xor`)',
    data: async (load) => {
      const { xor } = await load('aifn-methods/data/synthetic')
      const { stream } = await load('aifn-compute/foundation/random')
      return rowsOf(load, xor(stream('svm-tables/xor/data'), { n: [50, 50], kind: 'gaussian', sd: 0.5 }))
    },
  },
  {
    name: 'gaussians',
    description: 'two isotropic Gaussian classes, 100 points, separation 2 (Bayes error 0.159; `gaussians`)',
    data: async (load) => {
      const { gaussians } = await load('aifn-methods/data/synthetic')
      const { stream } = await load('aifn-compute/foundation/random')
      return rowsOf(load, gaussians(stream('svm-tables/gaussians/data'), { n: 100, separation: 2 }))
    },
  },
  {
    name: 'spirals',
    description: 'two interleaved spirals of one turn, 100 points, noise sd 0.05 (`spirals`)',
    data: async (load) => {
      const { spirals } = await load('aifn-methods/data/synthetic')
      const { stream } = await load('aifn-compute/foundation/random')
      return rowsOf(load, spirals(stream('svm-tables/spirals/data'), { n: 100, turns: 1, noise: 0.05 }))
    },
  },
  {
    name: 'diabetes',
    description: 'the first 150 patients of `diabetes`, labelled 1 when the progression is above their median',
    data: async (load) => {
      const { diabetes } = await load('aifn-methods/data/real')
      const { rows, labels } = await rowsOf(load, diabetes())
      const y = labels.slice(0, 150)
      const sorted = [...y].sort((a, b) => a - b)
      const median = (sorted[74] + sorted[75]) / 2
      return { rows: rows.slice(0, 150), labels: y.map((v) => (v > median ? 1 : 0)) }
    },
  },
]

type Table = { name: string; description: string; error: number[][]; sd: number[][] }

const asked = process.argv.slice(2)
for (const name of asked)
  if (!TASKS.some((t) => t.name === name))
    throw new Error(`svm-tables: unknown task '${name}'; tasks: ${TASKS.map((t) => t.name).join(', ')}`)

const server = await createServer({
  root,
  configFile: false,
  logLevel: 'error',
  server: { middlewareMode: true, hmr: false, watch: null, ws: false },
  appType: 'custom',
})
try {
  const load: Load = (id) => server.ssrLoadModule(id) as Promise<Record<string, any>>
  const { supportVectorMachine } = await load('aifn-methods/learning/kernel-methods')
  const { crossValidate, repeated, stratifiedKFold } = await load('aifn-compute/learning/validate')
  const { errorRate } = await load('aifn-compute/learning/metrics')
  const { stream } = await load('aifn-compute/foundation/random')
  const { rbf } = await load('aifn-compute/learning/kernels')
  const { tensor } = await load('aifn-compute/foundation/tensor')

  // The tables already written, kept for the tasks not asked for.
  const kept = new Map<string, Table>()
  if (asked.length > 0 && fs.existsSync(out)) {
    const { SVM_TABLES } = await load(out)
    for (const t of SVM_TABLES.tasks as Table[]) kept.set(t.name, t)
  }

  const tables: Table[] = []
  for (const task of TASKS) {
    if (asked.length > 0 && !asked.includes(task.name)) {
      const old = kept.get(task.name)
      if (old) tables.push({ ...old, description: task.description })
      continue
    }
    const { rows, labels } = await task.data(load)
    const d = rows[0].length
    const centre = Array.from({ length: d }, (_, j) => rows.reduce((s, r) => s + r[j], 0) / rows.length)
    const spread = Array.from({ length: d }, (_, j) =>
      Math.sqrt(rows.reduce((s, r) => s + (r[j] - centre[j]) ** 2, 0) / rows.length),
    )
    const x = tensor(rows.map((r) => r.map((v, j) => (v - centre[j]) / spread[j])))
    const y = tensor(labels)
    const counts = [0, 1].map((c) => labels.filter((v) => v === c).length)
    console.log(`${task.name}: ${rows.length} rows, ${d} features, classes ${counts.join(' / ')}`)

    const error: number[][] = []
    const sd: number[][] = []
    const start = performance.now()
    for (const logC of LOG_C) {
      const errorRow: number[] = []
      const sdRow: number[] = []
      for (const logGamma of LOG_GAMMA) {
        const estimator = supportVectorMachine({
          C: 10 ** logC,
          kernel: rbf({ lengthscale: 1 / Math.sqrt(2 * 10 ** logGamma) }),
        })
        const cv = crossValidate(
          estimator,
          { x, y },
          repeated(stratifiedKFold({ k: FOLDS, shuffle: true }), REPEATS),
          [errorRate],
          { stream: stream(`svm-tables/${task.name}`) },
        )
        errorRow.push(Number(cv.mean.errorRate.toFixed(10)))
        sdRow.push(Number(cv.std.errorRate.toFixed(6)))
      }
      error.push(errorRow)
      sd.push(sdRow)
      console.log(`  log10 C = ${logC.toFixed(2)}: ${errorRow.map((e) => e.toFixed(3)).join(' ')}`)
    }
    console.log(`  ${((performance.now() - start) / 1000).toFixed(0)} s`)
    tables.push({ name: task.name, description: task.description, error, sd })
  }

  const list = (v: number[]) => `[${v.join(', ')}]`
  const table = (m: number[][]) => `[\n${m.map((r) => `      ${list(r)},`).join('\n')}\n    ]`
  const entry = (t: Table) => `  {
    name: ${JSON.stringify(t.name)},
    description: ${JSON.stringify(t.description)},
    error: ${table(t.error)},
    sd: ${table(t.sd)},
  },`
  const text = `/**
 * Generated by \`node scripts/svm-tables.ts\`; do not edit. The validation error of an RBF support vector machine on
 * several small binary classification tasks (standardised), by ${REPEATS} repeats of stratified ${FOLDS}-fold
 * cross-validation, on the grid of \`svm-table.ts\`: the data of \`tuningTables\` after its first task (iris, which is
 * \`svm-table.ts\`). The script documents the settings and the tasks.
 */

/**
 * The tabulated tasks: in each table, row $i$ is the $i$-th value of $\\log_{10} C$ and column $j$ the $j$-th of
 * $\\log_{10}\\gamma$.
 */
export const SVM_TABLES = {
  /** The values of $\\log_{10} C$, increasing and equally spaced. */
  logC: ${list(LOG_C)},
  /** The values of $\\log_{10}\\gamma$, increasing and equally spaced. */
  logGamma: ${list(LOG_GAMMA)},
  /** One table per task: its name, a description, the mean error over the ${FOLDS * REPEATS} folds and their sd. */
  tasks: [
${tables.map(entry).join('\n')}
  ],
} as const
`
  fs.writeFileSync(out, await format(text, { ...(await resolveConfig(out)), filepath: out }))
  console.log(`wrote ${path.relative(root, out)} (${tables.map((t) => t.name).join(', ')})`)
} finally {
  await server.close()
}

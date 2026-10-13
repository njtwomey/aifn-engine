/**
 * Tuning tables on several tasks: the cross-validated error of an RBF support vector machine on the 21 by 21 grid of
 * `svmTuningSurface` ($\log_{10} C \in [-2, 3]$, $\log_{10}\gamma \in [-3, 2]$, steps of $\tfrac14$), tabulated on
 * seven small binary classification tasks, the data of warm-starting, portfolio construction and hyperparameter
 * importance across data sets (Feurer, Springenberg and Hutter, 2015; van Rijn and Hutter, 2018).
 *
 * The tasks, each standardised feature by feature and validated by ten repeats of stratified 5-fold cross-validation
 * with the same splits for every cell:
 *
 * - `iris`: versicolor against virginica, 100 flowers and four measurements: the table of `svmTuningSurface` itself.
 * - `moons`: two moons, 100 points, noise sd 0.2.
 * - `circles`: two concentric circles, 100 points, inner radius 0.5, noise sd 0.15.
 * - `xor`: Gaussian XOR, 50 points per class, blob sd 0.5.
 * - `gaussians`: two isotropic Gaussian classes, 100 points, separation 2 (a linear task with Bayes error 0.159).
 * - `spirals`: two interleaved spirals of one turn, 100 points, noise sd 0.05.
 * - `diabetes`: the first 150 patients of `diabetes`, ten features, labelled 1 when the progression is above their
 *   median.
 *
 * The best settings differ by task (large $C$ with small $\gamma$ for the linear tasks, $\gamma$ near 1 for the
 * spirals), which is what makes a configuration that is good on average different from the best on any one task.
 * The tables are written by `scripts/svm-tables.ts` (`svm-tables.ts`) and `scripts/svm-surface.ts` (`svm-table.ts`).
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { dense, fromData, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { SVM_TABLE } from './svm-table'
import { SVM_TABLES } from './svm-tables'

const { mat, vec } = dense

/** The description of the first task, whose table is `svm-table.ts`. */
const SVM_DESCRIPTION = 'iris, versicolor against virginica, 100 flowers and four measurements (`iris`)'

/** The tables `tuningTables` returns, on a shared grid of $n_C = 21$ values of $C$ by $n_\gamma = 21$ of $\gamma$. */
export interface TuningTables {
  /** The task names, in the order of the tables' first axis. */
  tasks: string[]
  /** One line on each task's data. */
  descriptions: string[]
  /** The grid's values of $\log_{10} C$ (the second axis of `error`). */
  logC: Vector
  /** The grid's values of $\log_{10}\gamma$ (the third axis of `error`). */
  logGamma: Vector
  /** The mean validation error over the 50 folds, of shape $T \times n_C \times n_\gamma$ for $T$ tasks. */
  error: Tensor
  /** The standard deviation of the validation error over the folds, laid out as `error`. */
  spread: Tensor
  /**
   * The configurations as rows, $n_C n_\gamma \times 2$: row $k = i n_\gamma + j$ is
   * $(\log_{10} C_i, \log_{10}\gamma_j)$.
   */
  configurations: Matrix
  /**
   * The error as a configuration by task table, $n_C n_\gamma \times T$, rows ordered as `configurations`: the layout
   * `greedyPortfolio` (`aifn-methods/optimisation/portfolio`) and `fanova` per task read.
   */
  table: Matrix
}

/**
 * The SVM tuning tables of seven tasks (see the file comment): the mean cross-validated error and its spread over the
 * folds on the $21 \times 21$ grid of $(\log_{10} C, \log_{10}\gamma)$, by task, and the same errors as a
 * configuration by task table. The first task, `iris`, is the table of `svmTuningSurface`. Nothing is random and
 * nothing is computed: the values are tabulated.
 *
 * @returns The tables, as `TuningTables`.
 *
 * @example The best setting differs from task to task
 * const t = tuningTables()
 * const nG = t.logGamma.shape[0]
 * const err = toArray(t.error)
 * const [C, G] = [toArray(t.logC), toArray(t.logGamma)]
 * t.tasks.forEach((task, k) => {
 *   const flat = err[k].flat()
 *   const b = flat.indexOf(Math.min(...flat))
 *   print(task, 'lowest error', flat[b], 'at log10 C, log10 gamma =', [C[Math.floor(b / nG)], G[b % nG]])
 * })
 *
 * @example The configuration by task table: the best configuration on average
 * const t = tuningTables()
 * const rows = toArray(t.table)
 * const mean = rows.map((r) => r.reduce((a, b) => a + b) / r.length)
 * const k = mean.indexOf(Math.min(...mean))
 * print('table:', t.table.shape, ' best on average:', toArray(t.configurations)[k])
 * print('its error per task:', rows[k])
 */
export function tuningTables(): TuningTables {
  const sources = [
    { name: 'iris', description: SVM_DESCRIPTION, error: SVM_TABLE.error, sd: SVM_TABLE.sd },
    ...SVM_TABLES.tasks,
  ]
  const { logC, logGamma } = SVM_TABLES
  const nC = logC.length
  const nG = logGamma.length
  const T = sources.length
  const cells = nC * nG
  const error = new Float64Array(T * cells)
  const spread = new Float64Array(T * cells)
  const table = new Float64Array(cells * T)
  sources.forEach((s, t) => {
    for (let i = 0; i < nC; i++)
      for (let j = 0; j < nG; j++) {
        const k = i * nG + j
        error[t * cells + k] = s.error[i][j]
        spread[t * cells + k] = s.sd[i][j]
        table[k * T + t] = s.error[i][j]
      }
  })
  const configurations = new Float64Array(cells * 2)
  for (let i = 0; i < nC; i++)
    for (let j = 0; j < nG; j++) {
      configurations[2 * (i * nG + j)] = logC[i]
      configurations[2 * (i * nG + j) + 1] = logGamma[j]
    }
  return {
    tasks: sources.map((s) => s.name),
    descriptions: sources.map((s) => s.description),
    logC: vec(Float64Array.from(logC)),
    logGamma: vec(Float64Array.from(logGamma)),
    error: fromData(error, [T, nC, nG]),
    spread: fromData(spread, [T, nC, nG]),
    configurations: mat(configurations, cells, 2),
    table: mat(table, cells, T),
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

definer<FunctionInfo>('function', 'data/objectives')(
  {
    key: 'tuningTables',
    name: 'SVM tuning tables on several tasks',
    summary:
      'The cross-validated error of an RBF SVM over log C and log γ on seven small classification tasks, tabulated.',
    role: 'construction',
    cite: ['feurer2015warmstart', 'vanrijn2018importance'],
  },
  tuningTables,
)

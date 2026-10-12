/**
 * The diabetes data of Efron, Hastie, Johnstone and Tibshirani (2004): ten baseline measurements of 442 patients and
 * their disease progression one year later, the example of "Least angle regression" and of the regularised paths that
 * followed it.
 *
 * The values are the raw table of the paper, vendored verbatim by `scripts/diabetes.py` into `diabetes-source.ts`
 * (pinned by its sha256) and parsed here; nothing is scaled. scikit-learn's default copy is the same table with each
 * feature centred and scaled to unit norm; the field notes standardise it with `standardScaler` instead.
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { space } from 'aifn-compute/foundation/space'
import { DIABETES_LINES, DIABETES_SHA256, DIABETES_URL } from './diabetes-source'
import { matrix, vector, type Dataset, type DatasetMeta } from '../types'

/** The feature names, in column order: scikit-learn's names for the ten baseline measurements. */
const FEATURES = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'] as const

/**
 * The diabetes data of Efron, Hastie, Johnstone and Tibshirani (2004), "Least angle regression", Annals of Statistics
 * 32(2): 442 patients with ten baseline measurements and a quantitative measure of disease progression one year later.
 * The columns are age (years), sex (1 or 2), body mass index, mean blood pressure, and six blood serum measurements:
 * `s1` total cholesterol, `s2` low-density lipoproteins, `s3` high-density lipoproteins, `s4` the ratio of total
 * cholesterol to HDL, `s5` the log of the serum triglycerides level and `s6` blood sugar. The values are raw, as
 * published; nothing is random. `s1`, `s2`, `s4` and `s5` are strongly correlated, which is what makes the data a
 * classic for the lasso, the elastic net and least-angle regression.
 *
 * @returns The dataset: `x` ($442 \times 10$, raw units), `y` the disease progression (442 values, 25 to 346), and
 *   `meta` with the feature names, the source, its URL and `sha256`, the hash of the vendored source file.
 *
 * @example Shape, columns and the first patient
 * const d = diabetes()
 * print('x:', d.x.shape, ' features:', d.meta.featureNames)
 * print('first patient:', toArray(d.x)[0], ' progression:', toArray(d.y)[0])
 *
 * @example Standardise the features and centre the target, as the field notes do
 * import { standardScaler } from 'aifn-methods/learning/preprocessing'
 * const d = diabetes()
 * const x = standardScaler().fit({ x: d.x }).transform(d.x)
 * const y = toArray(d.y)
 * const mean = y.reduce((a, b) => a + b) / y.length
 * print('target mean:', mean, ' first centred targets:', y.slice(0, 3).map((v) => v - mean))
 * print('first standardised row:', toArray(x)[0])
 */
export function diabetes(): Dataset & { readonly meta: DatasetMeta & { readonly sha256: string } } {
  const n = DIABETES_LINES.length - 1
  const d = FEATURES.length
  const x = new Float64Array(n * d)
  const y = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const cells = DIABETES_LINES[i + 1].split('\t').map(Number)
    for (let j = 0; j < d; j++) x[i * d + j] = cells[j]
    y[i] = cells[d]
  }
  return {
    kind: 'dataset',
    x: matrix(x, n, d),
    y: vector(y),
    meta: {
      name: 'Diabetes',
      description:
        'Age, sex, body mass index, blood pressure and six blood serum measurements of 442 diabetes patients, with a measure of disease progression one year later.',
      task: 'regression',
      featureNames: [...FEATURES],
      targetName: 'disease progression',
      source:
        'Efron, Hastie, Johnstone and Tibshirani (2004), "Least angle regression", Annals of Statistics 32(2); the raw data of the paper',
      url: DIABETES_URL,
      sha256: DIABETES_SHA256,
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

definer<DatasetInfo>('dataset', 'data/real')(
  {
    key: 'diabetes',
    name: 'Diabetes',
    summary: 'Ten baseline measurements of 442 diabetes patients and their disease progression a year later.',
    task: 'regression',
    output: 'dataset',
    knobs: space({}),
    truth: false,
    random: false,
    cite: ['efron2004'],
  },
  diabetes,
)

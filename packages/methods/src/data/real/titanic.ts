/**
 * The Titanic passengers and crew by class, sex, age and survival: R's `datasets::Titanic`, a
 * $4 \times 2 \times 2 \times 2$ table of counts for the 2201 people aboard (Dawson, 1995), expanded to one row per
 * person. All attributes are nominal; the target is `survived` (1 yes, 0 no).
 */

import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { space } from 'aifn-compute/foundation/space'
import type { TableData } from '../types'

/** The values of the `class` column, in the order of the last index of `COUNTS`. */
const CLASSES = ['1st', '2nd', '3rd', 'crew'] as const

/** Counts by [survived][age][sex][class]: survived no/yes, age child/adult, sex male/female, class 1st/2nd/3rd/crew. */
const COUNTS = [
  [
    [
      [0, 0, 35, 0],
      [0, 0, 17, 0],
    ],
    [
      [118, 154, 387, 670],
      [4, 13, 89, 3],
    ],
  ],
  [
    [
      [5, 11, 13, 0],
      [1, 13, 14, 0],
    ],
    [
      [57, 14, 75, 192],
      [140, 80, 76, 20],
    ],
  ],
]

/**
 * The 2201 people aboard the Titanic: class, sex, age (child or adult) and whether they survived, one row per person,
 * expanded from the table of counts in R's `datasets::Titanic` (Dawson, 1995, "The 'unusual episode' data revisited",
 * Journal of Statistics Education 3(3)). Rows come in blocks of identical people, ordered by survival, then age, sex
 * and class. Nothing is random.
 *
 * @returns The table: string columns `class` (`1st`, `2nd`, `3rd`, `crew`), `sex` (`male`, `female`) and `age`
 *   (`child`, `adult`), and the target `survived` (1 yes, 0 no), each of length 2201; no planted patterns.
 *
 * @example Columns, size and the first rows
 * const d = titanic()
 * print('columns:', Object.keys(d.table), ' rows:', d.table.class.length, ' target:', d.targets)
 * for (let i = 0; i < 3; i++) print(d.table.class[i], d.table.sex[i], d.table.age[i], d.table.survived[i])
 *
 * @example Survival by class
 * const { table } = titanic()
 * print('survived in all:', table.survived.filter((s) => s === 1).length)
 * for (const c of ['1st', '2nd', '3rd', 'crew']) {
 *   const rows = table.survived.filter((_, i) => table.class[i] === c)
 *   print(c, rows.filter((s) => s === 1).length, 'of', rows.length)
 * }
 */
export function titanic(): TableData {
  const cls: string[] = []
  const sex: string[] = []
  const age: string[] = []
  const survived: number[] = []
  COUNTS.forEach((byAge, s) =>
    byAge.forEach((bySex, a) =>
      bySex.forEach((byClass, x) =>
        byClass.forEach((count, c) => {
          for (let k = 0; k < count; k++) {
            cls.push(CLASSES[c])
            sex.push(x === 0 ? 'male' : 'female')
            age.push(a === 0 ? 'child' : 'adult')
            survived.push(s)
          }
        }),
      ),
    ),
  )
  return {
    kind: 'table',
    table: { class: cls, sex, age, survived },
    targets: ['survived'],
    planted: [],
    meta: {
      name: 'Titanic',
      description:
        'The 2201 passengers and crew of the Titanic by class, sex and age (child or adult), with whether each survived (711 did).',
      source:
        'Dawson (1995), "The ‘unusual episode’ data revisited", Journal of Statistics Education 3(3); R datasets::Titanic',
      url: 'https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/Titanic.html',
    },
  }
}

definer<DatasetInfo>('dataset', 'data/real')(
  {
    key: 'titanic',
    name: 'Titanic',
    summary: 'Class, sex, age and survival of the 2201 people aboard the Titanic (R’s Titanic table, one row each).',
    task: 'classification',
    output: 'table',
    knobs: space({}),
    truth: false,
    notes: ['simpsons-paradox'],
  },
  titanic,
)

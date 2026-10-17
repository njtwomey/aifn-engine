/**
 * The Titanic table for the rule-learning tests, inlined so that the compute tests need not import
 * `aifn-methods/data/real`: R's `datasets::Titanic` counts (Dawson, 1995), expanded in the same order as `titanic()`
 * (survived, then age, sex and class), coded as the site's figures code it: class 0 to 3 (1st, 2nd, 3rd, crew), sex
 * (0 male, 1 female), age (0 child, 1 adult), and the label survived (1 yes).
 */

/** Counts by [survived][age][sex][class]. */
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

/** The 2201 rows `[class, sex, age]`, their labels, and the names and level names `ruleText` prints them with. */
export function titanicRows(): {
  rows: number[][]
  y: number[]
  names: string[]
  levels: Record<number, string[]>
} {
  const rows: number[][] = []
  const y: number[] = []
  COUNTS.forEach((byAge, s) =>
    byAge.forEach((bySex, a) =>
      bySex.forEach((byClass, x) =>
        byClass.forEach((count, c) => {
          for (let k = 0; k < count; k++) {
            rows.push([c, x, a])
            y.push(s)
          }
        }),
      ),
    ),
  )
  return {
    rows,
    y,
    names: ['class', 'sex', 'age'],
    levels: { 0: ['1st', '2nd', '3rd', 'crew'], 1: ['male', 'female'], 2: ['child', 'adult'] },
  }
}

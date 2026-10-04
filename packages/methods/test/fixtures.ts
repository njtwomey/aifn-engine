import fs from 'node:fs'
import path from 'node:path'

/**
 * Load a Python-generated fixture by its node path (`fixture('numerics/linalg')` reads `fixtures/numerics/linalg.json`,
 * written by `fixtures/gen/numerics/linalg.py`); "inf", "-inf" and "nan" strings become numbers.
 */
export function fixture<T = unknown>(name: string): T {
  const file = path.join(import.meta.dirname, 'fixtures', `${name}.json`)
  return JSON.parse(fs.readFileSync(file, 'utf8'), (_k, v) =>
    v === 'inf' ? Infinity : v === '-inf' ? -Infinity : v === 'nan' ? NaN : v,
  ) as T
}

/**
 * Propositionalisation: one feature vector per individual of a one-to-many relational database, so that an
 * attribute-value learner can be trained on it.
 *
 * Kramer, Lavrač and Flach (2001), "Propositionalization approaches to relational data mining", in Relational Data
 * Mining, §§11.3 to 11.5. An individual has any number of rows in child tables; three constructions summarise them.
 * `existentialFeatures` builds first-order features in the style of RSD (Železný and Lavrač, 2006, §3): a conjunction
 * of attribute tests on one row of a child table, true of an individual when some row of it satisfies all of them,
 * $f(I) \leftarrow \mathrm{order}(I, O) \land \mathrm{late}(O) \land \mathrm{high}(O)$. Features below a minimum
 * support, true of every individual, or with the same coverage as a shorter feature are dropped (RSD's irrelevance
 * filter, simplified to equal coverage). `aggregateFeatures` builds aggregates in the style of RELAGGS (Krogel and
 * Wrobel, 2001): per child table, the number of rows and the number of rows with each attribute value. `wordify` is
 * Wordification (Perovšek et al., 2015, §3): each individual becomes a document whose words are
 * `table_attribute_value` for every attribute value of every row, and conjunctions of up to `maxWords` such words
 * from one row, weighted by term frequency, TF-IDF or presence.
 *
 * Differences from the sources: one level of child tables (no chains of foreign keys), equality tests on categorical
 * attributes only, and RSD's filters reduced to support and equal coverage. Feature values are in the database's order
 * of individuals; `featureTable` lays them out as rows for a learner such as `ruleList`.
 */

/** An individual: its key, its own attributes and an optional label. */
export interface RelationalIndividual {
  /** The key child rows point to. */
  readonly id: string
  /** Its own categorical attributes. */
  readonly attributes: Readonly<Record<string, string>>
  /** Its class, if known. */
  readonly label?: number
}

/** A child table: rows that point to an individual by `parent`, each with categorical attributes. */
export interface ChildTable {
  /** The table's name. */
  readonly name: string
  /** The rows: an id, the parent individual's id and the attributes. */
  readonly rows: readonly {
    readonly id: string
    readonly parent: string
    readonly attributes: Readonly<Record<string, string>>
  }[]
}

/** A one-to-many database: the individuals and their child tables. */
export interface RelationalDatabase {
  /** The individuals, in the order features list their values. */
  readonly individuals: readonly RelationalIndividual[]
  /** The child tables. */
  readonly children: readonly ChildTable[]
}

/** An attribute test, attribute $=$ value. */
export interface AttributeTest {
  /** The attribute. */
  readonly attribute: string
  /** The value it must have. */
  readonly value: string
}

/** A propositional feature: its name, its column of values (one per individual) and, when relational, its witnesses. */
export interface PropositionalFeature {
  /** The feature's name, such as `has order: delivery = late and value = high`. */
  readonly name: string
  /** The kind of construction that made it. */
  readonly kind: 'attribute' | 'existential' | 'aggregate' | 'word'
  /** One value per individual, in the database's order (0 or 1 for Boolean features). */
  readonly values: readonly number[]
  /** The child table it reads, if any. */
  readonly table?: string
  /** The tests on one row (existential features, counts by value and conjunctions of words). */
  readonly tests?: readonly AttributeTest[]
  /** For each individual, the ids of the rows that make it true or that it counts. */
  readonly witnesses?: readonly (readonly string[])[]
}

/** Options of `existentialFeatures`. */
export interface ExistentialOptions {
  /** Tests per feature at most (default 2). */
  maxTests?: number
  /** Individuals a feature must be true of (default 1). */
  minSupport?: number
  /** Also the feature "has a row of the table" (default true). */
  includeExists?: boolean
}

/** Options of `wordify`. */
export interface WordifyOptions {
  /** Words per conjunction at most (default 2). */
  maxWords?: number
  /**
   * `tf` (counts, default), `tfidf` (count $\times \log(N / \mathrm{df})$, $N$ individuals and $\mathrm{df}$ those
   * whose document has the word) or `binary` (presence).
   */
  weighting?: 'tf' | 'tfidf' | 'binary'
}

/**
 * The rows of a child table that belong to each individual.
 *
 * @param db The database.
 * @param table The child table.
 * @returns For each individual, in order, its rows.
 */
const rowsOf = (db: RelationalDatabase, table: ChildTable) =>
  db.individuals.map((ind) => table.rows.filter((r) => r.parent === ind.id))

/**
 * Whether a row passes every test.
 *
 * @param row The row.
 * @param tests The tests.
 * @returns True when each tested attribute has the tested value.
 */
const matches = (row: { attributes: Readonly<Record<string, string>> }, tests: readonly AttributeTest[]) =>
  tests.every((t) => row.attributes[t.attribute] === t.value)

/**
 * The individuals' own attributes as one-hot features (`plan = basic`), attribute by attribute, values sorted.
 *
 * @param db The database.
 * @returns One 0/1 feature per attribute value.
 *
 * @example One-hot attributes
 * const db = {
 *   individuals: [{ id: 'amy', attributes: { plan: 'basic' } }, { id: 'dan', attributes: { plan: 'premium' } }],
 *   children: [],
 * }
 * for (const f of attributeFeatures(db)) print(f.name, f.values)
 */
export function attributeFeatures(db: RelationalDatabase): PropositionalFeature[] {
  const names = [...new Set(db.individuals.flatMap((i) => Object.keys(i.attributes)))]
  const out: PropositionalFeature[] = []
  for (const a of names) {
    const values = [...new Set(db.individuals.map((i) => i.attributes[a]).filter((v) => v !== undefined))].sort()
    for (const v of values)
      out.push({
        name: `${a} = ${v}`,
        kind: 'attribute',
        values: db.individuals.map((i) => (i.attributes[a] === v ? 1 : 0)),
        tests: [{ attribute: a, value: v }],
      })
  }
  return out
}

/**
 * Every set of tests on distinct attributes, of 1 to $k$ tests, from the attribute values a table uses; shortest
 * first, attributes and values in sorted order.
 *
 * @param table The child table.
 * @param k Tests per set at most.
 * @returns The sets of tests.
 */
function testSets(table: ChildTable, k: number): AttributeTest[][] {
  const byAttr = new Map<string, Set<string>>()
  for (const r of table.rows)
    for (const [a, v] of Object.entries(r.attributes)) {
      if (!byAttr.has(a)) byAttr.set(a, new Set())
      byAttr.get(a)!.add(v)
    }
  const attrs = [...byAttr.keys()].sort()
  const out: AttributeTest[][] = []
  const grow = (start: number, acc: AttributeTest[]) => {
    if (acc.length) out.push(acc)
    if (acc.length === k) return
    for (let i = start; i < attrs.length; i++)
      for (const v of [...byAttr.get(attrs[i])!].sort()) grow(i + 1, [...acc, { attribute: attrs[i], value: v }])
  }
  grow(0, [])
  return out.sort((x, y) => x.length - y.length)
}

/**
 * First-order existential features (RSD style): "has a row of table $T$ with $a_1 = v_1$ and … and $a_k = v_k$",
 * table by table, shortest first, after the filters of the file comment.
 *
 * @param db The database.
 * @param options The number of tests, the support and whether to include "has a row" (see `ExistentialOptions`).
 * @returns The features that survive the filters, with the witnessing rows of each individual.
 *
 * @example Features of two customers' orders
 * const db = {
 *   individuals: [{ id: 'amy', attributes: {} }, { id: 'ben', attributes: {} }],
 *   children: [{ name: 'order', rows: [
 *     { id: 'o1', parent: 'amy', attributes: { delivery: 'late', value: 'high' } },
 *     { id: 'o2', parent: 'ben', attributes: { delivery: 'late', value: 'low' } },
 *   ] }],
 * }
 * for (const f of existentialFeatures(db)) print(f.name, f.values)
 */
export function existentialFeatures(db: RelationalDatabase, options: ExistentialOptions = {}): PropositionalFeature[] {
  const k = options.maxTests ?? 2
  const minSupport = options.minSupport ?? 1
  const n = db.individuals.length
  const out: PropositionalFeature[] = []
  const coverage = new Set<string>()
  for (const table of db.children) {
    const rows = rowsOf(db, table)
    const sets: AttributeTest[][] = [...(options.includeExists === false ? [] : [[]]), ...testSets(table, k)]
    for (const tests of sets) {
      const witnesses = rows.map((rs) => rs.filter((r) => matches(r, tests)).map((r) => r.id))
      const values: number[] = witnesses.map((w) => (w.length ? 1 : 0))
      const support = values.reduce((s, x) => s + x, 0)
      if (support < minSupport || support === n) continue
      const key = values.join('')
      if (coverage.has(key)) continue
      coverage.add(key)
      const name = tests.length
        ? `has ${table.name}: ${tests.map((t) => `${t.attribute} = ${t.value}`).join(' and ')}`
        : `has ${table.name}`
      out.push({ name, kind: 'existential', values, table: table.name, tests, witnesses })
    }
  }
  return out
}

/**
 * Aggregate features (RELAGGS style): per child table, the number of rows and the number of rows with each attribute
 * value.
 *
 * @param db The database.
 * @returns The count features, with the rows each one counts.
 *
 * @example Counts of orders
 * const db = {
 *   individuals: [{ id: 'amy', attributes: {} }, { id: 'ben', attributes: {} }],
 *   children: [{ name: 'order', rows: [
 *     { id: 'o1', parent: 'amy', attributes: { delivery: 'late' } },
 *     { id: 'o2', parent: 'amy', attributes: { delivery: 'on-time' } },
 *     { id: 'o3', parent: 'ben', attributes: { delivery: 'late' } },
 *   ] }],
 * }
 * for (const f of aggregateFeatures(db)) print(f.name, f.values)
 */
export function aggregateFeatures(db: RelationalDatabase): PropositionalFeature[] {
  const out: PropositionalFeature[] = []
  for (const table of db.children) {
    const rows = rowsOf(db, table)
    out.push({
      name: `count ${table.name}`,
      kind: 'aggregate',
      values: rows.map((rs) => rs.length),
      table: table.name,
      tests: [],
      witnesses: rows.map((rs) => rs.map((r) => r.id)),
    })
    for (const tests of testSets(table, 1)) {
      const witnesses = rows.map((rs) => rs.filter((r) => matches(r, tests)).map((r) => r.id))
      out.push({
        name: `count ${table.name}: ${tests[0].attribute} = ${tests[0].value}`,
        kind: 'aggregate',
        values: witnesses.map((w) => w.length),
        table: table.name,
        tests,
        witnesses,
      })
    }
  }
  return out
}

/**
 * Wordification: each individual is a document of words `table_attribute_value` from its rows, plus conjunctions of
 * up to `maxWords` words of one row, joined by `__`.
 *
 * @param db The database.
 * @param options The conjunction length and the weighting (see `WordifyOptions`).
 * @returns One feature per word of the vocabulary, sorted by word.
 *
 * @example The words of two orders
 * const db = {
 *   individuals: [{ id: 'amy', attributes: {} }, { id: 'ben', attributes: {} }],
 *   children: [{ name: 'order', rows: [
 *     { id: 'o1', parent: 'amy', attributes: { delivery: 'late', value: 'high' } },
 *     { id: 'o2', parent: 'ben', attributes: { delivery: 'late', value: 'low' } },
 *   ] }],
 * }
 * for (const f of wordify(db, { weighting: 'tfidf' })) print(f.name, f.values)
 */
export function wordify(db: RelationalDatabase, options: WordifyOptions = {}): PropositionalFeature[] {
  const k = options.maxWords ?? 2
  const weighting = options.weighting ?? 'tf'
  const n = db.individuals.length
  const counts = new Map<string, { tests: AttributeTest[]; table: string; rows: string[][] }>()
  for (const table of db.children) {
    const rows = rowsOf(db, table)
    rows.forEach((rs, i) => {
      for (const r of rs) {
        const words = Object.entries(r.attributes)
          .map(([attribute, value]) => ({ attribute, value }))
          .sort((x, y) => (x.attribute < y.attribute ? -1 : 1))
        const subsets: AttributeTest[][] = []
        const grow = (start: number, acc: AttributeTest[]) => {
          if (acc.length) subsets.push(acc)
          if (acc.length === k) return
          for (let j = start; j < words.length; j++) grow(j + 1, [...acc, words[j]])
        }
        grow(0, [])
        for (const s of subsets) {
          const name = s.map((t) => `${table.name}_${t.attribute}_${t.value}`).join('__')
          if (!counts.has(name)) counts.set(name, { tests: s, table: table.name, rows: db.individuals.map(() => []) })
          counts.get(name)!.rows[i].push(r.id)
        }
      }
    })
  }
  return [...counts.keys()].sort().map((name) => {
    const c = counts.get(name)!
    const df = c.rows.filter((r) => r.length > 0).length
    const idf = Math.log(n / df)
    const values = c.rows.map((r) =>
      weighting === 'binary' ? (r.length ? 1 : 0) : weighting === 'tfidf' ? r.length * idf : r.length,
    )
    return { name, kind: 'word' as const, values, table: c.table, tests: c.tests, witnesses: c.rows }
  })
}

/**
 * The flat table: one row per individual, one column per feature, for an attribute-value learner.
 *
 * @param db The database (for the number of individuals).
 * @param features The features, as the constructions above return them.
 * @returns The rows, `rows[i][j]` the value of feature $j$ for individual $i$.
 *
 * @example A two-column table
 * const db = {
 *   individuals: [{ id: 'amy', attributes: {} }, { id: 'ben', attributes: {} }],
 *   children: [{ name: 'order', rows: [{ id: 'o1', parent: 'amy', attributes: { delivery: 'late' } }] }],
 * }
 * print(featureTable(db, aggregateFeatures(db)))
 */
export function featureTable(db: RelationalDatabase, features: readonly PropositionalFeature[]): number[][] {
  return db.individuals.map((_, i) => features.map((f) => f.values[i]))
}

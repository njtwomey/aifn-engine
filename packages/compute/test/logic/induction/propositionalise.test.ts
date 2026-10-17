/**
 * Propositionalisation (aifn-compute/logic/induction `propositionalise.ts`) on the customers database of the
 * propositionalisation note: eight customers and fifteen orders; a customer churned exactly when one of their orders
 * was both late and of high value.
 */
import { describe, expect, it } from 'vitest'
import {
  aggregateFeatures,
  attributeFeatures,
  existentialFeatures,
  featureTable,
  wordify,
  type RelationalDatabase,
} from 'aifn-compute/logic/induction'

const ORDERS: readonly (readonly [string, string, string, string, string])[] = [
  ['o1', 'amy', 'late', 'high', 'web'],
  ['o2', 'amy', 'on-time', 'low', 'store'],
  ['o3', 'ben', 'late', 'low', 'store'],
  ['o4', 'ben', 'on-time', 'high', 'web'],
  ['o5', 'cat', 'on-time', 'high', 'store'],
  ['o6', 'dan', 'late', 'high', 'store'],
  ['o7', 'eli', 'late', 'low', 'store'],
  ['o8', 'eli', 'on-time', 'low', 'web'],
  ['o9', 'fin', 'on-time', 'high', 'web'],
  ['o10', 'fin', 'late', 'high', 'web'],
  ['o11', 'fin', 'on-time', 'low', 'store'],
  ['o12', 'gia', 'late', 'low', 'web'],
  ['o13', 'gia', 'late', 'low', 'store'],
  ['o14', 'gia', 'on-time', 'high', 'store'],
  ['o15', 'hal', 'on-time', 'low', 'web'],
]
const PEOPLE: readonly (readonly [string, string, number])[] = [
  ['amy', 'basic', 1],
  ['ben', 'basic', 0],
  ['cat', 'basic', 0],
  ['dan', 'premium', 1],
  ['eli', 'basic', 0],
  ['fin', 'basic', 1],
  ['gia', 'premium', 0],
  ['hal', 'basic', 0],
]
const CUSTOMERS: RelationalDatabase = {
  individuals: PEOPLE.map(([id, plan, label]) => ({ id, attributes: { plan }, label })),
  children: [
    {
      name: 'order',
      rows: ORDERS.map(([id, parent, delivery, value, channel]) => ({
        id,
        parent,
        attributes: { delivery, value, channel },
      })),
    },
  ],
}

describe('propositionalisation', () => {
  it('builds 17 existential features with up to 2 tests and 5 with 1', () => {
    const ex = existentialFeatures(CUSTOMERS, { maxTests: 2 })
    expect(ex).toHaveLength(17)
    expect(existentialFeatures(CUSTOMERS, { maxTests: 1 })).toHaveLength(5)
    const target = ex.find((f) => f.name === 'has order: delivery = late and value = high')!
    expect(target.values).toEqual(CUSTOMERS.individuals.map((i) => i.label))
    expect(target.witnesses![0]).toEqual(['o1'])
  })

  it('gives Amy and Ben the same aggregates', () => {
    const agg = aggregateFeatures(CUSTOMERS)
    const row = (name: string) => agg.map((f) => f.values[CUSTOMERS.individuals.findIndex((i) => i.id === name)])
    expect(row('amy')).toEqual([2, 1, 1, 1, 1, 1, 1])
    expect(row('ben')).toEqual(row('amy'))
  })

  it('has 18 Wordification words, weighted three ways', () => {
    const words = wordify(CUSTOMERS, { maxWords: 2 })
    expect(words).toHaveLength(18)
    const lateHigh = wordify(CUSTOMERS, { weighting: 'tfidf' }).find(
      (w) => w.name === 'order_delivery_late__order_value_high',
    )!
    expect(lateHigh.values[0]).toBeCloseTo(Math.log(8 / 3), 12)
    expect(wordify(CUSTOMERS, { weighting: 'binary' }).every((w) => w.values.every((v) => v === 0 || v === 1))).toBe(
      true,
    )
  })

  it('one-hot attributes and the flat table', () => {
    const att = attributeFeatures(CUSTOMERS)
    expect(att.map((f) => f.name)).toEqual(['plan = basic', 'plan = premium'])
    const table = featureTable(CUSTOMERS, att)
    expect(table).toHaveLength(8)
    expect(table[3]).toEqual([0, 1])
  })
})

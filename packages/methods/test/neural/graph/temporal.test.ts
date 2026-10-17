/** Temporal memory (aifn-methods/neural/graph), against the site's self-check (temporal-memory). */
import { describe, expect, it } from 'vitest'
import {
  memoryTrace,
  staticReach,
  timeEncoding,
  timeProjection,
  timeRespectingReach,
  type TimedEvent,
} from 'aifn-methods/neural/graph'
import { GruCell } from 'aifn-compute/nn/layers'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

// The site's temporal-graph-networks/events.ts: five users, three items, ten rated interactions.
const USERS = 5
const NODES = USERS + 3
const user = (u: number) => u - 1
const item = (i: number) => USERS + i - 1
const RAW: [number, number, number, number][] = [
  [1, 1, 1, 4],
  [2, 2, 2, 2],
  [2, 1, 3, 0],
  [3, 2, 5, 1],
  [4, 3, 6, 3],
  [1, 3, 8, 5],
  [5, 1, 9, 2],
  [4, 2, 11, 4],
  [3, 3, 12, 1],
  [5, 2, 14, 3],
]
const EVENTS: TimedEvent[] = RAW.map(([u, i, time, rating]) => ({
  source: user(u),
  destination: item(i),
  time,
  feature: [rating],
}))
const average = (self: readonly number[], other: readonly number[], feature: readonly number[]) => [
  0.5 * (self[0] + other[0] + feature[0]),
]

describe('memoryTrace', () => {
  const trace = memoryTrace(NODES, 1, EVENTS, average)
  it('the worked example: memories after events 1 to 4', () => {
    expect([trace.memory[1][user(1)][0], trace.memory[1][item(1)][0]]).toEqual([2, 2])
    expect([trace.memory[2][user(2)][0], trace.memory[2][item(2)][0]]).toEqual([1, 1])
    expect([trace.memory[3][user(2)][0], trace.memory[3][item(1)][0]]).toEqual([1.5, 1.5])
    expect([trace.memory[4][user(3)][0], trace.memory[4][item(2)][0]]).toEqual([1, 1])
  })
  it('gaps of event 3 are [1, 2]; u1 was last updated at time 1 after five events', () => {
    expect(trace.gaps[2]).toEqual([1, 2])
    expect(trace.lastUpdate[5][user(1)]).toBe(1)
  })
  it('a GRU cell as the updater keeps the memories in (-1, 1)', () => {
    const cell = GruCell(3, 1)
    const params = cell.init(stream(3))
    const gru = memoryTrace(NODES, 1, EVENTS, (self, other, x, dt) => {
      const input = tensor([other[0], x[0], Math.cos(0.3 * dt)])
      return Array.from(toFlat(cell.step(params, input, { h: tensor(self) }).h as Tensor))
    })
    expect(gru.memory[EVENTS.length].every((m) => Math.abs(m[0]) < 1)).toBe(true)
  })
  it('rejects an event from a node to itself', () => {
    expect(() => memoryTrace(2, 1, [{ source: 0, destination: 0, time: 1, feature: [1] }], average)).toThrow()
  })
})

describe('reach', () => {
  it("u1's rating reaches u2 through i1, but not u3, whose path to u1 runs back in time", () => {
    expect(timeRespectingReach(NODES, EVENTS, user(1))[4].map(Number)).toEqual([1, 1, 0, 0, 0, 1, 0, 0])
    expect(staticReach(NODES, EVENTS, user(1))[4].map(Number)).toEqual([1, 1, 1, 0, 0, 1, 1, 0])
  })
})

describe('time encodings', () => {
  it('cos encoding (1, 1) at 0 and (cos 10, cos 1) at 10', () => {
    expect(timeEncoding(0, [1, 0.1])).toEqual([1, 1])
    const e = timeEncoding(10, [1, 0.1])
    expect(e[0]).toBeCloseTo(Math.cos(10), 12)
    expect(e[1]).toBeCloseTo(Math.cos(1), 12)
  })
  it('projection [2, -4] to [1, -2] at dt = 50 with w = -0.01', () => {
    expect(timeProjection([2, -4], 50, [-0.01, -0.01])).toEqual([1, -2])
  })
})

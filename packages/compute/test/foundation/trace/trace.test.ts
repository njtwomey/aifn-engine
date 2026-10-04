import { describe, expect, it } from 'vitest'
import {
  decimate,
  extend,
  live,
  profile,
  run,
  seek,
  seriesComponents,
  timeSliced,
  trace,
  type Algorithm,
  type Trace,
} from 'aifn-compute/foundation/trace'
import { stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import { add, isContiguous, tensor, toFlat, transpose } from 'aifn-compute/foundation/tensor'
import { checkProtocol } from '../../protocol'

// ---------------------------------------------------------------------------------------------------------------------
// Toy algorithms.

/** Counts up by one; done at `stop` when given. */
const counter = (stop?: number): Algorithm<{ start: number }, { n: number; t: number }> => ({
  name: 'counter',
  init: ({ start }) => ({ n: start, t: 0 }),
  step: (s) => ({ n: s.n + 1, t: s.t + 1 }),
  done: stop === undefined ? undefined : (s) => s.n >= stop,
})

type GdState = { x: number[]; loss: number; t: number; diverged: boolean }

/** Gradient descent on f(x) = ½ Σ a_i x_i², flagging divergence once |x| exceeds 1e8. */
function gradientDescent(
  options: { flag: boolean } = { flag: true },
): Algorithm<{ a: number[]; x0: number[]; lr: number }, GdState & { a: number[]; lr: number }> {
  const loss = (a: number[], x: number[]) => 0.5 * x.reduce((s, xi, i) => s + a[i] * xi * xi, 0)
  return {
    name: 'gradient-descent',
    init: ({ a, x0, lr }) => ({ a, lr, x: x0, loss: loss(a, x0), t: 0, diverged: false }),
    step: (s) => {
      const x = s.x.map((xi, i) => profile('gradient', () => xi - s.lr * s.a[i] * xi))
      const l = profile('loss', () => loss(s.a, x))
      return { ...s, x, loss: l, t: s.t + 1, diverged: options.flag && x.some((v) => Math.abs(v) > 1e8) }
    },
    done: (s) => s.loss < 1e-12,
  }
}

/** A keyed stream from `aifn-compute/random`: values depend only on the key and the draw count. */
const toyStream = (key: string): Stream => stream(key)

type WalkState = { x: number; t: number }
/** A ±1 random walk; step t draws from the stream the runner derives for it, so steps are pure and states plain data. */
const randomWalk: Algorithm<{ x0: number }, WalkState> = {
  name: 'random-walk',
  init: ({ x0 }) => ({ x: x0, t: 0 }),
  step: (s, ctx) => ({ x: s.x + (uniform(ctx.stream) < 0.5 ? -1 : 1), t: s.t + 1 }),
}

const plain = <S>(t: Trace<S>) => ({
  steps: t.steps,
  index: Array.from(t.index),
  series: Object.fromEntries(
    Object.entries(t.series).map(([k, v]) => [k, { shape: v.shape, data: Array.from(v.data) }]),
  ),
  stopped: t.meta.stopped,
  steps_: t.meta.steps,
  checkpoints: t.checkpoints.index,
})

// ---------------------------------------------------------------------------------------------------------------------

describe('run', () => {
  it('returns the state after n steps', () => {
    expect(run(counter(), { start: 3 }, 10)).toEqual({ n: 13, t: 10 })
    expect(run(counter(), { start: 3 }, 0)).toEqual({ n: 3, t: 0 })
  })
  it('stops early on done', () => {
    expect(run(counter(7), { start: 0 }, 100)).toEqual({ n: 7, t: 7 })
  })
  it('stops on a diverged flag', () => {
    const s = run(gradientDescent(), { a: [1], x0: [1], lr: 5 }, 1000)
    expect(s.diverged).toBe(true)
    expect(s.t).toBeLessThan(1000)
  })
})

describe('trace', () => {
  it('keeps every step by default, with step 0 first', () => {
    const t = trace(counter(), { start: 0 }, 5, { record: { n: (s) => s.n } })
    expect(Array.from(t.index)).toEqual([0, 1, 2, 3, 4, 5])
    expect(t.steps.map((s) => s.n)).toEqual([0, 1, 2, 3, 4, 5])
    expect(Array.from(t.series.n.data)).toEqual([0, 1, 2, 3, 4, 5])
    expect(t.series.n.shape).toEqual([6])
    expect(t.meta).toMatchObject({ algorithm: 'counter', stopped: 'limit', steps: 5, every: 1 })
  })

  it('keeps steps divisible by `every` and always the final state', () => {
    const t = trace(counter(), { start: 0 }, 12, { every: 5, record: { n: (s) => s.n } })
    expect(Array.from(t.index)).toEqual([0, 5, 10, 12])
    expect(Array.from(t.series.n.data)).toEqual([0, 5, 10, 12])
  })

  it('gives the same trace for the same seed and a different one for another seed', () => {
    const record = { x: (s: WalkState) => s.x }
    const a = trace(randomWalk, { x0: 0 }, 200, { record, stream: toyStream('7') })
    const b = trace(randomWalk, { x0: 0 }, 200, { record, stream: toyStream('7') })
    const c = trace(randomWalk, { x0: 0 }, 200, { record, stream: toyStream('8') })
    expect(Array.from(a.series.x.data)).toEqual(Array.from(b.series.x.data))
    expect(Array.from(a.series.x.data)).not.toEqual(Array.from(c.series.x.data))
    expect(a.meta.key).toEqual(toyStream('7').key)
  })

  it('stacks recorded numbers, arrays, typed arrays, nested arrays and tensors', () => {
    const t = trace(counter(), { start: 1 }, 3, {
      record: {
        scalar: (s) => s.n,
        vector: (s) => [s.n, 2 * s.n],
        typed: (s) => Float64Array.of(s.n, -s.n, 0),
        matrix: (s) => [
          [s.n, 1],
          [2, s.n * s.n],
        ],
        // A transposed 2×3 view of [[0,1,2],[3,4,5]] + n, read through its strides.
        tensor: (s) =>
          transpose(
            add(
              tensor([
                [0, 1, 2],
                [3, 4, 5],
              ]),
              s.n,
            ),
          ),
      },
    })
    expect(t.series.scalar.shape).toEqual([4])
    expect(t.series.vector.shape).toEqual([4, 2])
    expect(Array.from(t.series.vector.data)).toEqual([1, 2, 2, 4, 3, 6, 4, 8])
    expect(t.series.typed.shape).toEqual([4, 3])
    expect(t.series.matrix.shape).toEqual([4, 2, 2])
    expect(t.series.matrix.strides).toEqual([4, 2, 1])
    expect(Array.from(t.series.matrix.data.subarray(12, 16))).toEqual([4, 1, 2, 16])
    expect(t.series.tensor.shape).toEqual([4, 3, 2])
    expect(Array.from(t.series.tensor.data.subarray(0, 6))).toEqual([1, 4, 2, 5, 3, 6])
    expect(t.series.scalar.dtype).toBe('float64')
    expect(t.series.scalar.offset).toBe(0)
    expect(isContiguous(t.series.tensor)).toBe(true)
  })

  it('rejects ragged or shape-changing recordings', () => {
    expect(() => trace(counter(), { start: 0 }, 2, { record: { r: () => [[1, 2], [3]] } })).toThrow(/ragged/)
    expect(() => trace(counter(), { start: 0 }, 3, { record: { r: (s) => new Array(s.n + 1).fill(0) } })).toThrow(
      /changed shape/,
    )
  })

  it('stops early on done and keeps the final state', () => {
    const t = trace(counter(7), { start: 0 }, 100, { every: 5 })
    expect(t.meta.stopped).toBe('done')
    expect(t.meta.steps).toBe(7)
    expect(Array.from(t.index)).toEqual([0, 5, 7])
    expect(t.steps.at(-1)).toEqual({ n: 7, t: 7 })
    expect(t.timing.stepMs.length).toBe(7)
  })

  it('stops when the initial state is already done', () => {
    const t = trace(counter(0), { start: 0 }, 10)
    expect(t.meta).toMatchObject({ stopped: 'done', steps: 0 })
    expect(Array.from(t.index)).toEqual([0])
  })

  it('detects divergence from a state flag', () => {
    const t = trace(gradientDescent(), { a: [1, 2], x0: [1, 1], lr: 5 }, 1000, { record: { loss: (s) => s.loss } })
    expect(t.meta.stopped).toBe('diverged')
    expect(t.steps.at(-1)!.diverged).toBe(true)
    expect(t.meta.steps).toBeLessThan(1000)
  })

  it('detects divergence from a non-finite recording', () => {
    const opts = { a: [1], x0: [1], lr: 5 }
    const t = trace(gradientDescent({ flag: false }), opts, 2000, { record: { loss: (s) => s.loss } })
    expect(t.meta.stopped).toBe('diverged')
    expect(Number.isFinite(t.series.loss.data.at(-1)!)).toBe(false)
    expect(Array.from(t.series.loss.data.subarray(0, -1)).every(Number.isFinite)).toBe(true)
    const kept = trace(gradientDescent({ flag: false }), opts, 2000, {
      record: { loss: (s) => s.loss },
      stopOnNonFinite: false,
    })
    expect(kept.meta.stopped).toBe('limit')
  })

  it('converges and reports done on a well-conditioned quadratic', () => {
    const t = trace(gradientDescent(), { a: [1, 3], x0: [2, -1], lr: 0.3 }, 1000, { record: { x: (s) => s.x } })
    expect(t.meta.stopped).toBe('done')
    expect(t.series.x.shape).toEqual([t.index.length, 2])
  })

  it('records timing: per-step times, monotone elapsed times, totals and phases', () => {
    const t = trace(gradientDescent(), { a: [1, 3], x0: [2, -1], lr: 0.1 }, 50, {
      every: 4,
      record: { loss: (s) => s.loss },
    })
    expect(t.timing.stepMs.length).toBe(t.meta.steps)
    expect(t.timing.elapsedMs.length).toBe(t.index.length)
    for (let k = 1; k < t.timing.elapsedMs.length; k++)
      expect(t.timing.elapsedMs[k]).toBeGreaterThanOrEqual(t.timing.elapsedMs[k - 1])
    expect(t.timing.stepMs.every((ms) => ms >= 0)).toBe(true)
    expect(t.timing.totalMs).toBeCloseTo(
      t.timing.stepMs.reduce((a, b) => a + b, 0),
      9,
    )
    expect(t.timing.perSecond).toBeGreaterThan(0)
    expect(Object.keys(t.timing.phases).sort()).toEqual(['gradient', 'init', 'loss', 'record'])
    // Phases inside a step cannot exceed the step time.
    expect(t.timing.phases.gradient + t.timing.phases.loss).toBeLessThanOrEqual(t.timing.totalMs + 1e-9)
  })

  it('profile outside a trace just calls the function', () => {
    expect(profile('x', () => 42)).toBe(42)
  })
})

describe('seek', () => {
  const opts = { x0: 0 }
  const stream = () => toyStream('seek')

  it('equals run(i) from scratch', () => {
    for (const i of [0, 1, 17, 64]) {
      expect(seek(randomWalk, opts, i, { stream: stream() }).x).toBe(run(randomWalk, opts, i, { stream: stream() }).x)
    }
  })

  it('equals run(i) from a trace’s checkpoints, and starts from the nearest one', () => {
    let steps = 0
    const counted: Algorithm<{ x0: number }, WalkState> = {
      ...randomWalk,
      step: (s, ctx) => {
        steps++
        return randomWalk.step(s, ctx)
      },
    }
    const t = trace(counted, opts, 100, { every: 10, checkpointEvery: 7, stream: stream() })
    expect(t.checkpoints.index).toEqual([0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98])
    for (const i of [0, 5, 13, 50, 99, 100, 130]) {
      steps = 0
      const s = seek(counted, opts, i, { checkpoints: t })
      expect(s.x).toBe(run(randomWalk, opts, i, { stream: stream() }).x)
      expect(s.t).toBe(i)
      // Kept steps (every 10) and checkpoints (every 7) together bound the work.
      if (i <= 100) expect(steps).toBeLessThan(7)
    }
    // Plain checkpoints work too.
    expect(seek(randomWalk, opts, 30, { checkpoints: t.checkpoints, stream: stream() }).x).toBe(t.steps[3].x)
  })

  it('stops at done like run', () => {
    expect(seek(counter(7), { start: 0 }, 20)).toEqual({ n: 7, t: 7 })
  })
})

describe('extend', () => {
  const opts = { x0: 0 }
  const record = { x: (s: WalkState) => s.x, pair: (s: WalkState) => [s.x, s.t] }

  it('equals a longer trace, with the extra final row replaced', () => {
    for (const [n, m, every] of [
      [12, 8, 5],
      [10, 10, 5],
      [7, 1, 3],
      [0, 9, 2],
    ]) {
      const longer = trace(randomWalk, opts, n + m, { every, record, checkpointEvery: 4, stream: toyStream('e') })
      const short = trace(randomWalk, opts, n, { every, record, checkpointEvery: 4, stream: toyStream('e') })
      const extended = extend(short, randomWalk, m)
      expect(plain(extended)).toEqual(plain(longer))
      expect(extended.timing.stepMs.length).toBe(n + m)
      expect(extended.timing.elapsedMs.length).toBe(extended.index.length)
      // Extending in pieces gives the same result.
      expect(plain(extend(extend(short, randomWalk, 1), randomWalk, m - 1))).toEqual(plain(longer))
    }
  })

  it("equals a longer trace with keep: 'none' and with checkpoints (review foundation 20)", () => {
    for (const keep of ['none', 'checkpoints', 'all'] as const) {
      const longer = trace(randomWalk, opts, 17, { every: 3, record, keep, checkpointEvery: 5, stream: toyStream('k') })
      const short = trace(randomWalk, opts, 9, { every: 3, record, keep, checkpointEvery: 5, stream: toyStream('k') })
      const extended = extend(short, randomWalk, 8)
      expect(plain(extended)).toEqual(plain(longer))
    }
  })

  it('accepts recorders explicitly and rejects different ones', () => {
    const short = trace(randomWalk, opts, 5, { record, stream: toyStream('e') })
    const copy = { ...short } // a copy made elsewhere: its recorders are not remembered
    expect(extend(copy, randomWalk, 5, { record }).series.x.shape).toEqual([11])
    expect(() => extend(copy, randomWalk, 5)).toThrow(/same recorders/)
  })

  it('leaves a trace that stopped early unchanged', () => {
    const t = trace(counter(3), { start: 0 }, 10)
    expect(extend(t, counter(3), 10)).toBe(t)
  })
})

describe('live', () => {
  it('yields the same states as a trace and ends on done', () => {
    const states = [...live(counter(4), { start: 0 })]
    expect(states.map((s) => s.state.n)).toEqual([0, 1, 2, 3, 4])
    expect(states.at(-1)!.stopped).toBe('done')
    const walk = live(randomWalk, { x0: 0 }, { stream: toyStream('l') })
    const t = trace(randomWalk, { x0: 0 }, 20, { stream: toyStream('l') })
    for (let k = 0; k <= 20; k++) expect(walk.next().value!.state.x).toBe(t.steps[k].x)
  })
})

describe('timeSliced', () => {
  const immediate = (resume: () => void) => resume()

  it('yields growing partial traces and ends with the full trace', async () => {
    const record = { x: (s: WalkState) => s.x }
    const full = trace(randomWalk, { x0: 0 }, 50, { every: 3, record, stream: toyStream('t') })
    const partials: Trace<WalkState>[] = []
    for await (const p of timeSliced(randomWalk, { x0: 0 }, 50, 0, {
      every: 3,
      record,
      stream: toyStream('t'),
      schedule: immediate,
    }))
      partials.push(p)
    // A zero budget still makes progress: one step per slice.
    expect(partials.length).toBe(50)
    expect(partials.map((p) => p.meta.steps)).toEqual(Array.from({ length: 50 }, (_, k) => k + 1))
    expect(plain(partials.at(-1)!)).toEqual(plain(full))
    // Each partial is a valid trace that extends to the full one.
    expect(plain(extend(partials[9], randomWalk, 40))).toEqual(plain(full))
  })

  it('stops at done and can be abandoned', async () => {
    const seen: number[] = []
    for await (const p of timeSliced(counter(5), { start: 0 }, 100, 1000, { schedule: immediate }))
      seen.push(p.meta.steps)
    expect(seen).toEqual([5])
    let count = 0
    for await (const _ of timeSliced(counter(), { start: 0 }, 100, 0, { schedule: immediate })) if (++count === 3) break
    expect(count).toBe(3)
  })
})

describe('decimate', () => {
  it('thins kept steps evenly, keeping the first and last', () => {
    const t = trace(counter(), { start: 0 }, 100, { record: { n: (s) => s.n, v: (s) => [s.n, -s.n] } })
    const d = decimate(t, 11)
    expect(Array.from(d.index)).toEqual([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    expect(Array.from(d.series.n.data)).toEqual(Array.from(d.index))
    expect(d.series.v.shape).toEqual([11, 2])
    expect(Array.from(d.series.v.data.subarray(2, 4))).toEqual([10, -10])
    expect(d.timing.elapsedMs.length).toBe(11)
    expect(decimate(t, 1000)).toBe(t)
  })
})

describe('seriesComponents', () => {
  it('splits a series into one line per component', () => {
    const t = trace(counter(), { start: 0 }, 2, {
      record: {
        s: (s) => s.n,
        m: (s) => [
          [s.n, 1],
          [2, 3],
        ],
      },
    })
    expect(seriesComponents(t.series.s)).toEqual([{ label: '', values: Float64Array.of(0, 1, 2) }])
    const m = seriesComponents(t.series.m)
    expect(m.map((c) => c.label)).toEqual(['[0, 0]', '[0, 1]', '[1, 0]', '[1, 1]'])
    expect(Array.from(m[0].values)).toEqual([0, 1, 2])
  })

  it('reads a strided tensor through its strides', () => {
    // Series of shape [3, 2] stored transposed: kept steps along axis 0 of a column-major layout.
    const series = transpose(
      tensor([
        [0, 1, 2],
        [10, 11, 12],
      ]),
    )
    const c = seriesComponents(series)
    expect(Array.from(c[0].values)).toEqual([0, 1, 2])
    expect(Array.from(c[1].values)).toEqual([10, 11, 12])
    expect(toFlat(series)).toEqual([0, 10, 1, 11, 2, 12])
  })
})

describe('divergence from recordings', () => {
  // A counter whose `rate` is undefined (NaN) at step 0, then finite, then NaN again at step 3.
  const alg = {
    name: 'nan-then-finite',
    init: () => ({ t: 0, rate: NaN }),
    step: (s: { t: number; rate: number }) => ({ t: s.t + 1, rate: s.t + 1 === 3 ? NaN : 1 / (s.t + 1) }),
  }
  it('treats a NaN before any finite value as not yet defined', () => {
    const tr = trace(alg, {}, 2, { record: { rate: (s) => s.rate } })
    expect(tr.meta.stopped).toBe('limit')
    expect(tr.meta.steps).toBe(2)
  })
  it('stops as diverged when a finite series becomes NaN', () => {
    const tr = trace(alg, {}, 10, { record: { rate: (s) => s.rate } })
    expect(tr.meta.stopped).toBe('diverged')
    expect(tr.meta.steps).toBe(3)
  })
  it('stops as diverged on an infinity, even at step 0', () => {
    const inf = { name: 'inf', init: () => ({ v: Infinity, t: 0 }), step: (s: { v: number; t: number }) => s }
    expect(trace(inf, {}, 5, { record: { v: (s) => s.v } }).meta.stopped).toBe('diverged')
  })
})

describe('the protocol helper', () => {
  it('passes for the toy algorithms', () => {
    checkProtocol(counter(), { start: 2 }, { steps: 10, record: { n: (s) => s.n } })
    checkProtocol(randomWalk, { x0: 0 }, { steps: 30, random: true, record: { x: (s) => s.x } })
    checkProtocol(gradientDescent(), { a: [1, 3], x0: [2, -1], lr: 0.1 }, { steps: 20 })
  })
  it('fails for a state that keeps a stream it advances (the pre-protocol pattern)', () => {
    const impure: Algorithm<undefined, { t: number; x: number; s: Stream }> = {
      name: 'impure',
      init: (_start, s) => ({ t: 0, x: 0, s }),
      step: (st) => ({ ...st, t: st.t + 1, x: uniform(st.s) }),
    }
    expect(() => checkProtocol(impure, undefined, { steps: 6, random: true })).toThrow()
  })
})

describe('runner consistency (review 2026-10-01)', () => {
  it('timeSliced agrees with trace when only a provisional row has been finite', async () => {
    // Kept rows (even steps) record NaN, off-grid rows 1: a snapshot after an odd step shows a finite provisional row,
    // which used to mark the series finite, so the NaN at the next kept step read as divergence.
    const record = { v: (s: WalkState) => (s.t % 2 ? 1 : NaN) }
    const full = trace(randomWalk, { x0: 0 }, 4, { every: 2, record })
    expect(full.meta.stopped).toBe('limit')
    let last: Trace<WalkState> | undefined
    for await (const p of timeSliced(randomWalk, { x0: 0 }, 4, 0, { every: 2, record, schedule: (r) => r() })) last = p
    expect(last!.meta.stopped).toBe('limit')
    expect(last!.meta.steps).toBe(4)
  })

  it('seek refuses a stream other than the trace’s, whose states it would mix with', () => {
    const t = trace(randomWalk, { x0: 0 }, 10, { checkpointEvery: 5, stream: toyStream('a') })
    expect(() => seek(randomWalk, { x0: 0 }, 7, { checkpoints: t, stream: toyStream('b') })).toThrow(/not the trace/)
    expect(seek(randomWalk, { x0: 0 }, 7, { checkpoints: t, stream: toyStream('a') }).x).toBe(
      run(randomWalk, { x0: 0 }, 7, { stream: toyStream('a') }).x,
    )
  })
})

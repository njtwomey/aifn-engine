import { describe, expect, test } from 'vitest'
import {
  checkProgram,
  CORE_NAMESPACES,
  corePrelude,
  lookup,
  members,
  makePrelude,
  qualified,
  runProgram,
  signature,
  withPrelude,
  type PreludeEntry,
} from 'aifn-compute/interpreter'
import { allclose, registry, tensor, toArray, type Tensor } from 'aifn-compute/foundation/tensor'
import { cholesky, det, eigh, inverse, solve, svd } from 'aifn-compute/numerics/linalg'

const value = (source: string, options = {}) => {
  const r = runProgram(source, options)
  if (!r.ok) throw new Error(`${r.error.name}: ${r.error.message}`)
  return r.value
}

const MAKE = `seed(7)

function make(n = 200, noise = 0.3) {
  const x = array.linspace(0, 6, n)
  const e = random.normal(n)
  const y = x.map((v, i) => 1.5 * v + 2 * math.sin(v) + noise * e[i])
  return [x, y]
}
`

describe('prelude', () => {
  test('math is generated from the primitive registry and records each source module', () => {
    for (const p of registry.list().filter((q) => q.kind === 'elementwise' && q.module === 'foundation/tensor')) {
      if (['complex', 'expj', 'conj', 'realPart', 'imagPart', 'complexAbs', 'angle'].includes(p.name)) continue
      expect(lookup(corePrelude, `math.${p.name}`)?.source).toBe('aifn-compute/foundation/tensor')
    }
    expect(lookup(corePrelude, 'math.erf')?.source).toBe('aifn-compute/numerics/special')
    expect(lookup(corePrelude, 'math.relu')?.source).toBe('aifn-compute/nn/functional')
    expect(lookup(corePrelude, 'linalg.solve')?.source).toBe('aifn-compute/numerics/linalg')
    expect(lookup(corePrelude, 'math.conj')).toBeUndefined()
  })

  test('entries are unique, documented, in declared namespaces, with qualified signatures', () => {
    const names = corePrelude.entries.map(qualified)
    expect(new Set(names).size).toBe(names.length)
    const spaces = new Set(CORE_NAMESPACES.map((n) => n.name))
    for (const e of corePrelude.entries) {
      expect(e.namespace === null || spaces.has(e.namespace)).toBe(true)
      expect(e.doc.length).toBeGreaterThan(3)
      expect(e.source.startsWith('aifn-compute')).toBe(true)
    }
    expect(signature(lookup(corePrelude, 'random.normal')!)).toBe('random.normal(n, mean = 0, sd = 1)')
    expect(members(corePrelude, null).map((e) => e.name)).toEqual(['print', 'seed'])
    for (const n of ['math', 'array', 'random', 'stats', 'linalg', 'signal'])
      expect(members(corePrelude, n).length).toBeGreaterThan(2)
  })

  test('withPrelude adds namespaces and replaces entries; undeclared namespaces are refused', () => {
    const twice: PreludeEntry = {
      namespace: 'mine',
      name: 'twice',
      params: [{ name: 'x' }],
      doc: 'Twice x.',
      source: 'test',
      impl: (_c, x) => 2 * (x as number),
    }
    const sin: PreludeEntry = { ...twice, namespace: 'math', name: 'sin', impl: () => 'mine' }
    const p = withPrelude(corePrelude, makePrelude([{ name: 'mine', doc: 'Mine.', source: 'test' }], [twice, sin]))
    expect(p.entries.length).toBe(corePrelude.entries.length + 1)
    expect(value('return [mine.twice(4), math.sin(1)]', { prelude: p })).toEqual([8, 'mine'])
    expect(() => withPrelude(corePrelude, makePrelude([], [twice]))).toThrow(/undeclared namespace/)
  })
})

describe('namespaces', () => {
  test('names resolve only through their namespace; seed, print, Math and console are top-level', () => {
    expect(value('return typeof sin')).toBe('undefined')
    expect(
      value('return [typeof math.sin, typeof random.normal, typeof seed, typeof print, typeof console.log]'),
    ).toEqual(['function', 'function', 'function', 'function', 'function'])
    const r = runProgram('return sin(1)')
    expect(r.ok ? null : r.error.name).toBe('ReferenceError')
  })

  test('math maps over numbers, arrays and matrices, and Math.* is math.*', () => {
    expect(value('return math.sin(0)')).toBe(0)
    expect(value('return math.exp([0, 1])')).toEqual([1, Math.E])
    expect(value('return math.log1p([[0], [Math.E - 1]])')).toEqual([[0], [1]])
    expect(value('return Math.sqrt([4, 9])')).toEqual([2, 3])
    expect(value('return Math.hypot(3, 4)')).toBe(5)
    expect(value('return math.add(math.mul(2, [1, 2]), [[10], [20]])')).toEqual([
      [12, 14],
      [22, 24],
    ])
  })

  test('array and stats', () => {
    expect(value('return array.linspace(0, 1, 3).map((v) => 2 * v)')).toEqual([0, 1, 2])
    expect(value('return array.stack([[1, 2], [3, 4]], 1)')).toEqual([
      [1, 3],
      [2, 4],
    ])
    expect(value('return [array.sum([1, 2, 3]), array.max([[1, 5], [3, 2]], 0)]')).toEqual([6, [3, 5]])
    expect(
      value(
        'return [stats.mean([1, 2, 3]), stats.var([1, 3]), stats.median([3, 1, 2]), stats.quantile([0, 10], 0.25)]',
      ),
    ).toEqual([2, 1, 2, 2.5])
    const h = value('return stats.histogram([0, 1, 1, 2], 2)') as { counts: number[] }
    expect(h.counts).toEqual([1, 3])
  })

  test('linalg wrappers agree with aifn', () => {
    const A = [
      [4, 1, 0.5],
      [1, 3, 0.2],
      [0.5, 0.2, 2],
    ]
    const b = [1, 2, 3]
    const At = tensor(A)
    const close = (x: unknown, y: Tensor) => expect(allclose(tensor(x as number[]), y)).toBe(true)
    const src = `const A = ${JSON.stringify(A)}\n`
    close(value(`${src}return linalg.solve(A, ${JSON.stringify(b)})`), solve(At, tensor(b)))
    close(value(`${src}return linalg.inv(A)`), inverse(At))
    expect(value(`${src}return linalg.det(A)`)).toBeCloseTo(det(At) as number, 12)
    close(value(`${src}return linalg.cholesky(A)`), cholesky(At).L)
    const e = value(`${src}return linalg.eigh(A)`) as { values: number[]; vectors: number[][] }
    close(e.values, eigh(At).values)
    close(e.vectors, eigh(At).vectors)
    close((value(`${src}return linalg.svd(A)`) as { S: number[] }).S, svd(At).S)
    expect(value('return linalg.matmul([[1, 2], [3, 4]], [1, 1])')).toEqual([3, 7])
    expect(value('return [linalg.dot([1, 2], [3, 4]), linalg.norm([3, 4])]')).toEqual([11, 5])
    expect(value('return linalg.transpose([[1, 2]])')).toEqual(toArray(tensor([[1], [2]])))
    expect(value('return linalg.outer([1, 2], [1, 0])')).toEqual([
      [1, 0],
      [2, 0],
    ])
  })

  test('signal', () => {
    const s = value('return signal.spectrum(array.range(8).map((k) => Math.cos((2 * Math.PI * k) / 4)))') as number[]
    expect(s.map((v) => Math.round(v * 1e9) / 1e9)).toEqual([0, 0, 4, 0, 0])
    expect(value('return signal.convolve([1, 2], [1, 1])')).toEqual([1, 3, 2])
  })
})

describe('run', () => {
  test('the entry function is called when the program returns nothing', () => {
    const [x, y] = value(MAKE) as number[][]
    expect(x).toHaveLength(200)
    expect(y).toHaveLength(200)
    expect((value(MAKE, { args: [5] }) as number[][])[0]).toEqual([0, 1.5, 3, 4.5, 6])
    expect(value('function make() { return 1 }\nreturn 2')).toBe(2)
    expect(value('const x = 1', { entry: null })).toBeUndefined()
  })

  test('print and console.log write to the output', () => {
    const r = runProgram('print("a", 1.23456789, [1, 2])\nconsole.log("b")')
    expect(r.output).toEqual(['a 1.23457 [1,2]', 'b'])
  })
})

describe('seeds', () => {
  test('the same program and seed give the same draws; another seed differs', () => {
    expect(value(MAKE)).toEqual(value(MAKE))
    expect(value(MAKE, { seed: 1 })).toEqual(value(MAKE, { seed: 1 }))
    expect(value(MAKE, { seed: 1 })).not.toEqual(value(MAKE))
    expect(value(MAKE.replace('seed(7)', 'seed(8)'))).not.toEqual(value(MAKE))
  })

  test('draws are keyed by call order: editing a line without draws leaves them unchanged', () => {
    const edited = MAKE.replace('1.5 * v', '3 * v').replace('linspace(0, 6, n)', 'linspace(0, 10, n)')
    const draw = (src: string) => value(src.replace('return [x, y]', 'return e'))
    expect(draw(edited)).toEqual(draw(MAKE))
  })

  test('seed() restarts the draws, Math.random is seeded, and the draws have their ranges', () => {
    const [a, b] = value('seed(1)\nconst a = random.normal(3)\nseed(1)\nreturn [a, random.normal(3)]') as number[][]
    expect(a).toEqual(b)
    expect(value('return Math.random()')).toBe(value('return Math.random()'))
    const [z, u, c, k] = value(
      'return [random.normal(), random.uniform(2, 5, 6), random.bernoulli([0, 1]), random.integers(3, 4)]',
    ) as [number, number[], number[], number[]]
    expect(typeof z).toBe('number')
    expect(u.every((x) => x >= 5 && x < 6)).toBe(true)
    expect(c).toEqual([0, 1])
    expect(k.every((x) => x >= 0 && x < 3)).toBe(true)
    expect([...(value('return random.shuffle([1, 2, 3, 4, 5])') as number[])].sort()).toEqual([1, 2, 3, 4, 5])
  })
})

describe('errors', () => {
  test('runtime errors carry the line and column in the program', () => {
    const r = runProgram('const a = 1\nconst b = foo(a)\n')
    expect(r.ok ? null : r.error).toMatchObject({ name: 'ReferenceError', line: 2, column: 11 })
    const inside = runProgram('function make() {\n  return math.sin([1, "x"])\n}')
    expect(inside.ok ? null : inside.error).toMatchObject({ name: 'TypeError', line: 2 })
  })

  test('syntax errors are reported by runProgram and checkProgram', () => {
    const r = runProgram('const = 3')
    expect(r.ok ? null : r.error.name).toBe('SyntaxError')
    expect(checkProgram('const = 3')?.name).toBe('SyntaxError')
    expect(checkProgram(MAKE)).toBeNull()
    expect(runProgram('', { entry: 'a b' }).ok).toBe(false)
  })
})

describe('math.clip', () => {
  test('limits numbers and arrays to [lo, hi], with array bounds', () => {
    expect(value('return math.clip(5, 0, 1)')).toBe(1)
    expect(value('return math.clip([-2, 0.5, 3], 0, 1)')).toEqual([0, 0.5, 1])
    expect(value('return math.clip([1, 5], [0, 0], [2, 4])')).toEqual([1, 4])
  })
})

describe('prelude names can be shadowed (review models, interpreter)', () => {
  test('a program may declare a prelude name, and the doc example runs', () => {
    const r = runProgram('const stats = 3\nlet print = 4\nreturn stats + print')
    expect(r.ok && r.value).toBe(7)
    const ex = runProgram('seed(7)\nfunction make(n = 5) { return random.normal(n) }')
    expect(ex.ok).toBe(true)
    // Errors still report the program's own line.
    const bad = runProgram('const a = 1\nnull.x')
    expect(!bad.ok && bad.error.line).toBe(2)
  })
})

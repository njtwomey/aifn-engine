/** Synthetic molecules (aifn-methods/data/synthetic), against the site's self-check (small-molecules). */
import { describe, expect, it } from 'vitest'
import {
  atomContribution,
  atomIdentifiers,
  countFingerprints,
  moleculeGraph,
  smallMolecules,
  type Molecule,
} from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'
import { toRows } from 'aifn-compute/foundation/tensor'

const VALENCE = { C: 4, N: 3, O: 2 } as const
const mols = smallMolecules(stream('molecules/data'), { count: 180 })

/** The note's diols, built by hand: a six-ring with hydroxyls on atoms 0 and `second`. */
function diol(second: number): Molecule {
  const atoms = ['C', 'C', 'C', 'C', 'C', 'C', 'O', 'O'] as const
  const bonds = [
    [0, 1, 1],
    [1, 2, 1],
    [2, 3, 1],
    [3, 4, 1],
    [4, 5, 1],
    [5, 0, 1],
    [0, 6, 1],
    [second, 7, 1],
  ] as [number, number, 1][]
  const hydrogens = atoms.map((el, v) => VALENCE[el] - bonds.filter(([a, b]) => a === v || b === v).length)
  return {
    scaffold: 'ring6',
    atoms: [...atoms],
    bonds,
    hydrogens,
    ring: atoms.map((_, v) => v < 6),
    double: atoms.map(() => false),
    pos: atoms.map(() => [0, 0] as const),
    contributions: atoms.map((el, v) => atomContribution(el, hydrogens[v], false)),
    property: 0,
  }
}

describe('smallMolecules', () => {
  it('hydrogens are the valence minus the bond orders, never negative', () => {
    for (const m of mols) {
      const used = m.atoms.map(() => 0)
      for (const [a, b, o] of m.bonds) {
        used[a] += o
        used[b] += o
      }
      m.atoms.forEach((el, v) => expect(m.hydrogens[v]).toBe(VALENCE[el] - used[v]))
      expect(m.hydrogens.every((h) => h >= 0)).toBe(true)
    }
  })
  it('contributions are the atom terms plus polar-pair shares (multiples of 0.25 on N and O only); small noise', () => {
    for (const m of mols) {
      m.contributions.forEach((c, v) => {
        const extra = c - atomContribution(m.atoms[v], m.hydrogens[v], m.double[v])
        expect(Math.abs(extra / 0.25 - Math.round(extra / 0.25))).toBeLessThan(1e-9)
        if (m.atoms[v] === 'C') expect(Math.abs(extra)).toBeLessThan(1e-12)
      })
      expect(Math.abs(m.property - m.contributions.reduce((s, x) => s + x, 0))).toBeLessThan(0.3)
    }
  })
  it('molecule i does not change when more are drawn', () => {
    expect(smallMolecules(stream('molecules/data'), { count: 3 })[2]).toEqual(mols[2])
  })
})

describe('fingerprints', () => {
  const [d12, d13] = [diol(1), diol(2)]
  it('1,2- and 1,3-cyclohexanediol: atom terms 0.30 each, equal at radius 0, different at radius 1', () => {
    expect(d12.contributions.reduce((s, x) => s + x, 0)).toBeCloseTo(0.3, 12)
    expect(d13.contributions.reduce((s, x) => s + x, 0)).toBeCloseTo(0.3, 12)
    const [a0, b0] = toRows(countFingerprints([d12, d13], 0).counts) as number[][]
    expect(a0).toEqual(b0)
    const [a1, b1] = toRows(countFingerprints([d12, d13], 1).counts) as number[][]
    expect(a1).not.toEqual(b1)
  })
  it('one identifier per atom per round', () => {
    expect(atomIdentifiers(d12, 2).map((r) => r.length)).toEqual([8, 8, 8])
  })
})

describe('moleculeGraph', () => {
  it('nodes and edges are atoms and bonds; feature rows sum to 2 + ring + double', () => {
    const m = mols[0]
    const g = moleculeGraph(m)
    expect([g.graph.nodes, g.graph.edges.length]).toEqual([m.atoms.length, m.bonds.length])
    expect(g.graph.edges.map((e) => e.weight)).toEqual(m.bonds.map((b) => b[2]))
    ;(toRows(g.x) as number[][]).forEach((row, v) =>
      expect(row.reduce((s, x) => s + x, 0)).toBe(2 + (m.ring[v] ? 1 : 0) + (m.double[v] ? 1 : 0)),
    )
  })
})

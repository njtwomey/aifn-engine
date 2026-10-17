/**
 * Synthetic molecules with a property whose parts are known, for graph regression. Each molecule is a carbon scaffold,
 * a six-ring, a five-ring or a four-carbon chain, with one to four substituent groups (methyl, ethyl, hydroxyl,
 * methoxy, amine, carbonyl) on distinct scaffold atoms. Atoms are carbon, nitrogen and oxygen with valences 4, 3 and 2;
 * hydrogens are implicit, the valence minus the bond orders.
 *
 * The property imitates an atom-additive partition coefficient in the style of Wildman and Crippen (1999), "Prediction
 * of physicochemical parameters by atomic contributions", J. Chem. Inf. Comput. Sci. 39(5), with invented values:
 * every heavy atom contributes a value fixed by its element, its hydrogen count and whether it carries a double bond
 * (`atomContribution`), and every pair of polar atoms (N or O) exactly three bonds apart adds `POLAR_PAIR` more, a
 * stand-in for an intramolecular hydrogen bond, split equally between the two atoms. Measurement noise of standard
 * deviation `noise` is added to the total. The per-atom contributions are returned, so a figure can draw where the
 * property comes from.
 *
 * `countFingerprints` is an unhashed count version of the extended-connectivity (Morgan) fingerprint of Rogers and
 * Hahn (2010), "Extended-connectivity fingerprints", J. Chem. Inf. Model. 50(5): round 0 names each atom by element,
 * hydrogen count, ring membership and double bond; round $r$ appends the sorted multiset of (bond order, neighbour
 * name at round $r - 1$), Weisfeiler-Leman refinement with bond orders as edge labels, kept as readable strings; every
 * atom contributes one identifier per round up to the radius, and the fingerprint counts identifiers over a vocabulary
 * learned from a set of molecules (no hashing or folding, and duplicate substructures are counted, not removed).
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { child, integers, normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { fromEdges, type Graph } from 'aifn-compute/graph'
import { hopDistances } from 'aifn-compute/graph/traversal'

/** A heavy atom's element: carbon, nitrogen or oxygen. */
export type AtomElement = 'C' | 'N' | 'O'

/** A molecule's carbon scaffold: a six-ring, a five-ring or a four-carbon chain. */
export type MoleculeScaffold = 'ring6' | 'ring5' | 'chain'

/** The scaffolds, in the order `smallMolecules` draws them from. */
export const SCAFFOLDS: readonly MoleculeScaffold[] = ['ring6', 'ring5', 'chain']

/** The valence of each element. */
const VALENCE: Record<AtomElement, number> = { C: 4, N: 3, O: 2 }

/** A bond: two atom indices and the bond order, 1 or 2. */
export type MoleculeBond = readonly [number, number, 1 | 2]

/** A synthetic molecule. */
export interface Molecule {
  /** Its scaffold. */
  readonly scaffold: MoleculeScaffold
  /** The element of each heavy atom; the scaffold's atoms come first. */
  readonly atoms: readonly AtomElement[]
  /** The bonds. */
  readonly bonds: readonly MoleculeBond[]
  /** The implicit hydrogens of each atom. */
  readonly hydrogens: readonly number[]
  /** Whether each atom lies on the scaffold ring. */
  readonly ring: readonly boolean[]
  /** Whether each atom carries a double bond. */
  readonly double: readonly boolean[]
  /** Drawing positions, bond length about 1. */
  readonly pos: readonly (readonly [number, number])[]
  /** Each atom's share of the noise-free property. */
  readonly contributions: readonly number[]
  /** The property: the sum of the contributions plus noise. */
  readonly property: number
}

/**
 * The contribution of an atom to the property, by element, hydrogen count and double bond: invented values with
 * Crippen-like signs (carbon positive and growing with its hydrogens, nitrogen and oxygen negative).
 *
 * @param element The element.
 * @param hydrogens The number of implicit hydrogens.
 * @param double Whether the atom carries a double bond.
 * @returns The contribution.
 *
 * @example A methyl carbon, an amine nitrogen and a hydroxyl oxygen
 * print(atomContribution('C', 3, false), atomContribution('N', 2, false), atomContribution('O', 1, false))
 */
export function atomContribution(element: AtomElement, hydrogens: number, double: boolean): number {
  if (element === 'C') return [0.1, 0.25, 0.4, 0.55][hydrogens] ?? 0
  if (element === 'N') return [-0.5, -0.75, -1.0][hydrogens] ?? 0
  return hydrogens === 1 ? -0.9 : double ? -0.6 : -0.35
}

/** The extra property of each pair of polar atoms three bonds apart, split equally between them. */
export const POLAR_PAIR = 0.5

/** A substituent group. */
type Group = {
  /** Its atoms, in order from the one bonded to the scaffold. */
  atoms: AtomElement[]
  /** The order of the bond to the scaffold. */
  first: 1 | 2
}
/** The substituent groups by name. */
const GROUPS: Record<string, Group> = {
  methyl: { atoms: ['C'], first: 1 },
  ethyl: { atoms: ['C', 'C'], first: 1 },
  hydroxyl: { atoms: ['O'], first: 1 },
  methoxy: { atoms: ['O', 'C'], first: 1 },
  amine: { atoms: ['N'], first: 1 },
  carbonyl: { atoms: ['O'], first: 2 },
}

/** The names of the substituent groups, in the order they are drawn from. */
export const MOLECULE_GROUPS: readonly string[] = Object.keys(GROUPS)

/**
 * One molecule on a given scaffold, its groups drawn from the stream. Throws `DomainError` if a valence were exceeded
 * (it cannot be with these groups).
 *
 * @param s The molecule's stream, advanced by the draws: the number of groups, a shuffle of the sites, a group per
 *   site, then the noise.
 * @param scaffold The scaffold.
 * @param maxGroups The most substituent groups.
 * @param noise The standard deviation of the measurement noise.
 * @returns The molecule.
 */
function molecule(s: Stream, scaffold: MoleculeScaffold, maxGroups: number, noise: number): Molecule {
  const size = scaffold === 'ring6' ? 6 : scaffold === 'ring5' ? 5 : 4
  const ringScaffold = scaffold !== 'chain'
  const atoms: AtomElement[] = new Array<AtomElement>(size).fill('C')
  const bonds: [number, number, 1 | 2][] = []
  for (let i = 0; i + 1 < size; i++) bonds.push([i, i + 1, 1])
  if (ringScaffold) bonds.push([size - 1, 0, 1])
  // Scaffold drawing: a regular polygon, or a zig-zag chain.
  const pos: [number, number][] = ringScaffold
    ? Array.from({ length: size }, (_, i) => {
        const a = Math.PI / 2 - (2 * Math.PI * i) / size
        const r = 0.5 / Math.sin(Math.PI / size)
        return [r * Math.cos(a), r * Math.sin(a)] as [number, number]
      })
    : Array.from(
        { length: size },
        (_, i) => [(i - (size - 1) / 2) * 0.87, i % 2 === 0 ? -0.25 : 0.25] as [number, number],
      )
  const groups = 1 + integers(s, maxGroups)
  const sites = Array.from({ length: size }, (_, i) => i)
  for (let i = sites.length - 1; i > 0; i--) {
    const j = integers(s, i + 1)
    ;[sites[i], sites[j]] = [sites[j], sites[i]]
  }
  for (const site of sites.slice(0, Math.min(groups, size))) {
    const g = GROUPS[MOLECULE_GROUPS[integers(s, MOLECULE_GROUPS.length)]]
    // Outward direction from the scaffold's centre (for a chain: up or down, alternating).
    const [px, py] = pos[site]
    const len = Math.hypot(px, py)
    const out: [number, number] = ringScaffold ? [px / len, py / len] : [0, site % 2 === 0 ? -1 : 1]
    let previous = site
    g.atoms.forEach((el, k) => {
      const v = atoms.length
      atoms.push(el)
      bonds.push([previous, v, k === 0 ? g.first : 1])
      const wiggle = k === 0 ? 0 : 0.5 * (site % 2 === 0 ? 1 : -1)
      pos.push([px + out[0] * (k + 1) - out[1] * wiggle * k, py + out[1] * (k + 1) + out[0] * wiggle * k])
      previous = v
    })
  }
  const n = atoms.length
  const used = new Array<number>(n).fill(0)
  const double = new Array<boolean>(n).fill(false)
  for (const [a, b, o] of bonds) {
    used[a] += o
    used[b] += o
    if (o === 2) double[a] = double[b] = true
  }
  const hydrogens = atoms.map((el, v) => VALENCE[el] - used[v])
  if (hydrogens.some((h) => h < 0)) throw new DomainError('smallMolecules', 'smallMolecules: valence exceeded')
  const ring = atoms.map((_, v) => ringScaffold && v < size)
  const contributions = atoms.map((el, v) => atomContribution(el, hydrogens[v], double[v]))
  const d = toFlat(
    hopDistances(
      fromEdges(
        n,
        bonds.map(([a, b]) => [a, b] as const),
        { directed: false },
      ),
    ),
  )
  for (let u = 0; u < n; u++)
    for (let v = u + 1; v < n; v++)
      if (atoms[u] !== 'C' && atoms[v] !== 'C' && d[u * n + v] === 3) {
        contributions[u] += POLAR_PAIR / 2
        contributions[v] += POLAR_PAIR / 2
      }
  const total = contributions.reduce((a, b) => a + b, 0)
  return { scaffold, atoms, bonds, hydrogens, ring, double, pos, contributions, property: total + noise * normal(s) }
}

/** Options of `smallMolecules`. */
export interface SmallMoleculeOptions {
  /** How many molecules. */
  count: number
  /** The scaffolds to draw from, uniformly (default all three). */
  scaffolds?: readonly MoleculeScaffold[]
  /** At most this many substituent groups, at least one (default 4). */
  maxGroups?: number
  /** The standard deviation of the measurement noise (default 0.05). */
  noise?: number
}

/**
 * Draw synthetic molecules (see the file comment). Throws `DomainError` for a count that is not a count, no scaffolds
 * or fewer than one group.
 *
 * @param s The random stream; molecule $i$ draws from its child $i$ (the scaffold, then the groups and the noise), so
 *   the first molecules do not change when more are drawn.
 * @param options The number of molecules, the scaffolds, the most groups and the noise.
 * @returns The molecules.
 *
 * @example Three molecules: scaffold, atoms and property
 * for (const m of smallMolecules(stream(0), { count: 3 })) print(m.scaffold, m.atoms.join(''), m.property)
 */
export function smallMolecules(s: Stream, options: SmallMoleculeOptions): Molecule[] {
  const { count, scaffolds = SCAFFOLDS, maxGroups = 4, noise = 0.05 } = options
  const counts = Number.isInteger(count) && count >= 0 && Number.isInteger(maxGroups) && maxGroups >= 1
  if (!counts || scaffolds.length === 0)
    throw new DomainError('smallMolecules', 'smallMolecules: need a count, a scaffold and at least one group')
  return Array.from({ length: count }, (_, i) => {
    const si = child(s, i)
    const scaffold = scaffolds[Math.floor(uniform(si) * scaffolds.length)]
    return molecule(si, scaffold, maxGroups, noise)
  })
}

/** The width of an atom's feature row: element (3), hydrogen count 0 to 3 (4), ring, double bond. */
export const ATOM_FEATURES = 9

/**
 * The one-hot feature row of an atom.
 *
 * @param m The molecule.
 * @param v The atom.
 * @param out The array to write the `ATOM_FEATURES` entries into, at `offset`.
 * @param offset Where the row starts in `out`.
 */
function atomRow(m: Molecule, v: number, out: Float64Array, offset: number): void {
  out[offset + ['C', 'N', 'O'].indexOf(m.atoms[v])] = 1
  out[offset + 3 + Math.min(3, m.hydrogens[v])] = 1
  out[offset + 7] = m.ring[v] ? 1 : 0
  out[offset + 8] = m.double[v] ? 1 : 0
}

/**
 * A molecule as an undirected graph with the bond order as edge weight, and its atom-feature matrix (one row of
 * `ATOM_FEATURES` per atom: element, hydrogen count, ring, double bond).
 *
 * @param m The molecule.
 * @returns The `graph` and the features `x` ($n \times 9$).
 *
 * @example The graph of the first molecule
 * const { graph, x } = moleculeGraph(smallMolecules(stream(0), { count: 1 })[0])
 * print('atoms:', graph.nodes, ' bonds:', graph.edges.length, ' x:', x.shape)
 */
export function moleculeGraph(m: Molecule): { graph: Graph; x: Tensor } {
  const n = m.atoms.length
  const x = new Float64Array(n * ATOM_FEATURES)
  for (let v = 0; v < n; v++) atomRow(m, v, x, v * ATOM_FEATURES)
  return {
    graph: fromEdges(
      n,
      m.bonds.map(([from, to, order]) => [from, to, order] as const),
      { directed: false },
    ),
    x: fromData(x, [n, ATOM_FEATURES]),
  }
}

/**
 * Each atom's identifier at rounds 0 to `radius` (see the file comment).
 *
 * @param m The molecule.
 * @param radius The last round.
 * @returns `ids[r][v]`, atom $v$'s identifier at round $r$.
 *
 * @example Round 0 and 1 identifiers of a molecule's first atom
 * const m = smallMolecules(stream(0), { count: 1 })[0]
 * print(atomIdentifiers(m, 1).map((round) => round[0]))
 */
export function atomIdentifiers(m: Molecule, radius: number): string[][] {
  const n = m.atoms.length
  const near: [number, number][][] = Array.from({ length: n }, () => [])
  for (const [a, b, o] of m.bonds) {
    near[a].push([b, o])
    near[b].push([a, o])
  }
  let current = m.atoms.map((el, v) => `${el}H${m.hydrogens[v]}${m.ring[v] ? 'r' : ''}${m.double[v] ? '=' : ''}`)
  const rounds = [current]
  for (let r = 1; r <= radius; r++) {
    const prev = current
    current = prev.map(
      (id, v) =>
        `${id}(${near[v]
          .map(([u, o]) => `${o}${prev[u]}`)
          .sort()
          .join(',')})`,
    )
    rounds.push(current)
  }
  return rounds
}

/**
 * Count fingerprints of molecules over a vocabulary: the identifiers of every round up to `radius` seen in
 * `vocabularyFrom` (default the molecules themselves), sorted. Identifiers outside the vocabulary are dropped.
 *
 * @param mols The molecules to fingerprint.
 * @param radius The last round of identifiers.
 * @param vocabularyFrom The molecules the vocabulary is learned from, e.g. a training set.
 * @returns The `vocabulary` and the counts, an $m \times$ (vocabulary size) float64 matrix.
 *
 * @example Radius-1 fingerprints of five molecules
 * const { vocabulary, counts } = countFingerprints(smallMolecules(stream(0), { count: 5 }), 1)
 * print('identifiers:', vocabulary.length, ' counts:', counts.shape)
 */
export function countFingerprints(
  mols: readonly Molecule[],
  radius: number,
  vocabularyFrom: readonly Molecule[] = mols,
): { vocabulary: string[]; counts: Tensor } {
  const ids = (m: Molecule) => atomIdentifiers(m, radius).flat()
  const vocabulary = [...new Set(vocabularyFrom.flatMap(ids))].sort()
  const index = new Map(vocabulary.map((id, k) => [id, k]))
  const V = vocabulary.length
  const counts = new Float64Array(mols.length * V)
  mols.forEach((m, i) => {
    for (const id of ids(m)) {
      const k = index.get(id)
      if (k !== undefined) counts[i * V + k]++
    }
  })
  return { vocabulary, counts: fromData(counts, [mols.length, V]) }
}

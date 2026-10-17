/**
 * Small fixed knowledge graphs for knowledge-graph embedding figures: entity names, relation names and true triples
 * (head, relation, tail) as indices into them.
 *
 * `familyKnowledgeGraph` is a three-generation family of nine people with five relations that between them show every
 * relation pattern the models are compared on: `spouse_of` and `sibling_of` are symmetric, `child_of` is the inverse
 * of `parent_of`, `grandparent_of` is the composition of `parent_of` with itself, and `parent_of` is one-to-many.
 * `relationPatternGraph` isolates one pattern in a graph of eight or nine entities, so a two-dimensional model can be
 * fitted to it and inspected. The patterns are those of Sun, Deng, Nie and Tang (2019), "RotatE: knowledge graph
 * embedding by relational rotation in complex space", ICLR, §3.1 (symmetry, antisymmetry, inversion, composition),
 * plus the one-to-many mapping property of Wang, Zhang, Feng and Chen (2014), "Knowledge graph embedding by translating
 * on hyperplanes", AAAI, §1. The family itself is made up.
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { definer } from 'aifn-compute/foundation/registry'

/** A fact (head, relation, tail) as indices into a knowledge graph's entity and relation names. */
export type IndexTriple = readonly [number, number, number]

/** A knowledge graph: entity names, relation names and the true triples. */
export interface KnowledgeGraph {
  /** The entity names; entity $e$ is `entities[e]`. */
  readonly entities: readonly string[]
  /** The relation names; relation $r$ is `relations[r]`. */
  readonly relations: readonly string[]
  /** The true facts, each (head, relation, tail) as indices. */
  readonly triples: readonly IndexTriple[]
}

const FAMILY = ['Arthur', 'Beth', 'Carl', 'Dana', 'Eve', 'Frank', 'Gina', 'Hugo', 'Iris'] as const
const FAMILY_RELATIONS = ['parent_of', 'child_of', 'spouse_of', 'sibling_of', 'grandparent_of'] as const

/**
 * The family knowledge graph: Arthur and Beth are married with children Carl and Dana; Carl is married to Eve, with
 * children Gina and Hugo; Dana is married to Frank, with child Iris. 36 triples: 10 `parent_of`, their 10 inverses
 * `child_of`, 6 `spouse_of` and 4 `sibling_of` (both directions), and 6 `grandparent_of`.
 *
 * @returns The graph: 9 entities, 5 relations and 36 triples, grouped by relation in that order.
 *
 * @example The facts about Carl
 * const kg = familyKnowledgeGraph()
 * const name = ([h, r, t]) => `${kg.entities[h]} ${kg.relations[r]} ${kg.entities[t]}`
 * print(kg.triples.length, 'triples')
 * for (const f of kg.triples.filter(([h]) => kg.entities[h] === 'Carl')) print(name(f))
 */
export function familyKnowledgeGraph(): KnowledgeGraph {
  const [A, B, C, D, E, F, G, H, I] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
  const [parent, child, spouse, sibling, grandparent] = [0, 1, 2, 3, 4]
  const parents: [number, number][] = [
    [A, C],
    [A, D],
    [B, C],
    [B, D],
    [C, G],
    [C, H],
    [E, G],
    [E, H],
    [D, I],
    [F, I],
  ]
  const couples: [number, number][] = [
    [A, B],
    [C, E],
    [D, F],
  ]
  const siblings: [number, number][] = [
    [C, D],
    [G, H],
  ]
  const grandparents: [number, number][] = [
    [A, G],
    [A, H],
    [A, I],
    [B, G],
    [B, H],
    [B, I],
  ]
  const triples: IndexTriple[] = [
    ...parents.map(([p, c]) => [p, parent, c] as const),
    ...parents.map(([p, c]) => [c, child, p] as const),
    ...couples.flatMap(([a, b]) => [[a, spouse, b] as const, [b, spouse, a] as const]),
    ...siblings.flatMap(([a, b]) => [[a, sibling, b] as const, [b, sibling, a] as const]),
    ...grandparents.map(([g, c]) => [g, grandparent, c] as const),
  ]
  return { entities: FAMILY, relations: FAMILY_RELATIONS, triples }
}

/** A relation pattern that `relationPatternGraph` isolates. */
export type RelationPattern = 'symmetric' | 'inverse' | 'composition' | 'one-to-many'

/**
 * A graph with one relation pattern:
 * - `symmetric`: four couples and `married_to` in both directions (8 triples);
 * - `inverse`: four parent-child pairs with `parent_of` and its inverse `child_of` (8 triples);
 * - `composition`: three people, their birth cities and those cities' countries, with `born_in`, `city_in` and
 *   `citizen_of`, the composition of the first two (9 triples);
 * - `one-to-many`: two parents with three children each and `parent_of` (6 triples).
 *
 * Throws `DomainError` for any other pattern.
 *
 * @param pattern The pattern.
 * @returns The graph.
 *
 * @example Composition: born in a city, the city in a country, so a citizen of the country
 * const kg = relationPatternGraph('composition')
 * for (const [h, r, t] of kg.triples) print(kg.entities[h], kg.relations[r], kg.entities[t])
 */
export function relationPatternGraph(pattern: RelationPattern): KnowledgeGraph {
  if (pattern === 'symmetric') {
    const entities = ['Ann', 'Ben', 'Cat', 'Dan', 'Eva', 'Fred', 'Gail', 'Hal']
    const triples: IndexTriple[] = [0, 2, 4, 6].flatMap((a) => [[a, 0, a + 1] as const, [a + 1, 0, a] as const])
    return { entities, relations: ['married_to'], triples }
  }
  if (pattern === 'inverse') {
    const entities = ['Ann', 'Ben', 'Cat', 'Dan', 'Eva', 'Fred', 'Gail', 'Hal']
    const triples: IndexTriple[] = [0, 1, 2, 3].flatMap((p) => [[p, 0, p + 4] as const, [p + 4, 1, p] as const])
    return { entities, relations: ['parent_of', 'child_of'], triples }
  }
  if (pattern === 'composition') {
    const entities = ['Ann', 'Ben', 'Cat', 'Lyon', 'Porto', 'Turin', 'France', 'Portugal', 'Italy']
    const triples: IndexTriple[] = [0, 1, 2].flatMap((p) => [
      [p, 0, p + 3] as const,
      [p + 3, 1, p + 6] as const,
      [p, 2, p + 6] as const,
    ])
    return { entities, relations: ['born_in', 'city_in', 'citizen_of'], triples }
  }
  if (pattern !== 'one-to-many')
    throw new DomainError('relationPatternGraph', `relationPatternGraph: unknown pattern '${String(pattern)}'`)
  const entities = ['Ann', 'Ben', 'Cat', 'Dan', 'Eva', 'Fred', 'Gail', 'Hal']
  const triples: IndexTriple[] = [
    [0, 0, 2],
    [0, 0, 3],
    [0, 0, 4],
    [1, 0, 5],
    [1, 0, 6],
    [1, 0, 7],
  ]
  return { entities, relations: ['parent_of'], triples }
}

const fn = definer<FunctionInfo>('function', 'data/synthetic')
const KG = [
  'knowledge-graph-embeddings',
  'translational-and-rotational-knowledge-graph-models',
  'bilinear-and-tensor-factorisation-knowledge-graph-models',
]

fn(
  {
    key: 'familyKnowledgeGraph',
    name: 'Family knowledge graph',
    summary: 'Nine people in three generations with symmetric, inverse, composed and one-to-many relations.',
    role: 'construction',
    notes: KG,
    cite: ['sun2019rotate'],
  },
  familyKnowledgeGraph,
)
fn(
  {
    key: 'relationPatternGraph',
    name: 'Relation-pattern knowledge graph',
    summary: 'A tiny knowledge graph that shows one relation pattern: symmetry, inversion, composition or one-to-many.',
    role: 'construction',
    notes: KG,
    cite: ['sun2019rotate', 'bordes2013transe'],
  },
  relationPatternGraph,
)

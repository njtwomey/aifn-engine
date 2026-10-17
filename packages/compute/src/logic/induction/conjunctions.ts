/**
 * Conjunctions of attribute tests, the hypothesis language of classical concept learning, with their generality order
 * and its refinement operators.
 *
 * Mitchell (1982), "Generalization as search"; Mitchell (1997), "Machine Learning", ch. 2. An instance gives every
 * attribute one of its values. A conjunction gives each attribute either a required value or `ANY` (Mitchell's "?");
 * the empty hypothesis (`null`, Mitchell's $\langle \emptyset, \dots, \emptyset \rangle$, which covers nothing) is the
 * bottom of the order. $g \succeq h$ ($g$ at least as general as $h$) when $g$ covers every instance $h$ covers. The
 * downward refinement operator $\rho_\downarrow$ sets one "?" to a value, the upward one $\rho_\uparrow$ sets one value
 * to "?", and the least general generalisation of a hypothesis and an instance is Find-S's update: the propositional
 * case of the lgg of clauses (`clauseLgg`). Mitchell writes each empty attribute separately; here every hypothesis
 * that covers nothing is the single `null`, so attributes with $k_i$ values give a language of
 * $1 + \prod_i (k_i + 1)$ conjunctions.
 *
 * Attributes and values are indices: an `AttributeSpace` names them. Enumeration is exhaustive, for teaching sizes.
 */

/** "Any value": the attribute is not tested. */
export const ANY = -1

/** The attributes: their names and, for each, the names of its values. */
export interface AttributeSpace {
  /** The attributes' names. */
  readonly names: readonly string[]
  /** For each attribute, the names of its values; an instance or a conjunction refers to them by index. */
  readonly values: readonly (readonly string[])[]
}

/** An instance: the index of its value for each attribute. */
export type AttributeInstance = readonly number[]

/** A conjunction: per attribute a required value index or `ANY`; `null` is the empty hypothesis (covers nothing). */
export type Conjunction = readonly number[] | null

/** An instance with its label. */
export interface LabelledInstance {
  /** The instance. */
  readonly x: AttributeInstance
  /** Whether it is a positive example of the concept. */
  readonly positive: boolean
}

/**
 * True when the conjunction covers (classifies as positive) the instance.
 *
 * @param h The conjunction.
 * @param x The instance.
 * @returns Whether every attribute `h` tests has `x`'s value (false for the empty hypothesis).
 *
 * @example Sunny days, whatever the wind
 * print(covers([0, ANY], [0, 1]), covers([0, ANY], [1, 1]), covers(null, [0, 1]))
 */
export function covers(h: Conjunction, x: AttributeInstance): boolean {
  if (h === null) return false
  for (let i = 0; i < h.length; i++) if (h[i] !== ANY && h[i] !== x[i]) return false
  return true
}

/**
 * True when `g` is at least as general as `h` ($g \succeq h$): every instance `h` covers, `g` covers.
 *
 * @param g The candidate more general conjunction.
 * @param h The candidate more specific conjunction.
 * @returns Whether $g \succeq h$; the empty hypothesis is below everything.
 *
 * @example The order is partial
 * print(moreGeneralOrEqual([0, ANY], [0, 1]), moreGeneralOrEqual([0, ANY], [ANY, 1]), moreGeneralOrEqual([0, 1], null))
 */
export function moreGeneralOrEqual(g: Conjunction, h: Conjunction): boolean {
  if (h === null) return true
  if (g === null) return false
  for (let i = 0; i < g.length; i++) if (g[i] !== ANY && g[i] !== h[i]) return false
  return true
}

/**
 * True when `g` is strictly more general than `h` ($g \succeq h$ and not $h \succeq g$).
 *
 * @param g The candidate more general conjunction.
 * @param h The candidate more specific conjunction.
 * @returns Whether $g \succ h$.
 *
 * @example Not strictly more general than itself
 * print(strictlyMoreGeneral([0, ANY], [0, 1]), strictlyMoreGeneral([0, ANY], [0, ANY]))
 */
export function strictlyMoreGeneral(g: Conjunction, h: Conjunction): boolean {
  return moreGeneralOrEqual(g, h) && !moreGeneralOrEqual(h, g)
}

/**
 * True when the two conjunctions are the same hypothesis.
 *
 * @param a The first conjunction.
 * @param b The second conjunction.
 * @returns Whether they test the same values (or are both empty).
 *
 * @example Equality of hypotheses, not of arrays
 * print(sameConjunction([0, ANY], [0, ANY]), sameConjunction(null, null), sameConjunction([0, ANY], null))
 */
export function sameConjunction(a: Conjunction, b: Conjunction): boolean {
  if (a === null || b === null) return a === b
  return a.length === b.length && a.every((v, i) => v === b[i])
}

/**
 * The most general conjunction $\langle ?, \dots, ? \rangle$, which covers every instance.
 *
 * @param space The attributes.
 * @returns A conjunction of `ANY` for every attribute.
 *
 * @example The top of the order
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * print(conjunctionText(mostGeneral(space), space))
 */
export function mostGeneral(space: AttributeSpace): Conjunction {
  return space.names.map(() => ANY)
}

/**
 * The most specific conjunction that covers the instance: every attribute set to the instance's value.
 *
 * @param x The instance.
 * @returns A copy of `x` as a conjunction.
 *
 * @example An instance as a hypothesis
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * print(conjunctionText(fromInstance([1, 0]), space))
 */
export function fromInstance(x: AttributeInstance): Conjunction {
  return x.slice()
}

/**
 * The least general generalisation of a conjunction and an instance: the most specific conjunction that is at least
 * as general as `h` and covers `x`. Attributes where `h` disagrees with `x` become `ANY`; the empty hypothesis
 * generalises to the instance itself. Unique for conjunctions, so this is Find-S's update.
 *
 * @param h The conjunction to generalise.
 * @param x The instance it must cover.
 * @returns The generalisation.
 *
 * @example Two sunny days with different winds
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * const h = generaliseToCover(generaliseToCover(null, [0, 0]), [0, 1])
 * print(conjunctionText(h, space))
 */
export function generaliseToCover(h: Conjunction, x: AttributeInstance): Conjunction {
  if (h === null) return fromInstance(x)
  return h.map((v, i) => (v === ANY || v === x[i] ? v : ANY))
}

/**
 * The downward refinement operator $\rho_\downarrow$: the conjunctions one step more specific than `h`, each with one
 * `ANY` set to a value. A conjunction with no `ANY` refines to the empty hypothesis; the empty hypothesis has no
 * refinements.
 *
 * @param h The conjunction to refine.
 * @param space The attributes, for their values.
 * @returns The one-step specialisations, attribute by attribute and value by value.
 *
 * @example The children of the top
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * print(specialisations(mostGeneral(space), space).map((h) => conjunctionText(h, space)))
 */
export function specialisations(h: Conjunction, space: AttributeSpace): Conjunction[] {
  if (h === null) return []
  const out: Conjunction[] = []
  h.forEach((v, i) => {
    if (v !== ANY) return
    space.values[i].forEach((_, value) => out.push(h.map((w, j) => (j === i ? value : w))))
  })
  return out.length ? out : [null]
}

/**
 * The upward refinement operator $\rho_\uparrow$: the conjunctions one step more general than `h`, each with one
 * tested attribute set to `ANY`. The empty hypothesis generalises to every fully specified conjunction (one per
 * instance); the most general conjunction has none.
 *
 * @param h The conjunction to generalise.
 * @param space The attributes, for the instances above the empty hypothesis.
 * @returns The one-step generalisations.
 *
 * @example The parents of a fully specified conjunction
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * print(generalisations([0, 1], space).map((h) => conjunctionText(h, space)))
 * print('above the empty hypothesis:', generalisations(null, space).length)
 */
export function generalisations(h: Conjunction, space: AttributeSpace): Conjunction[] {
  if (h === null) return allInstances(space).map(fromInstance)
  const out: Conjunction[] = []
  h.forEach((v, i) => {
    if (v !== ANY) out.push(h.map((w, j) => (j === i ? ANY : w)))
  })
  return out
}

/**
 * The minimal specialisations of `h` that exclude the negative instance `x`: each `ANY` set to a value other than
 * `x`'s. A conjunction that already excludes `x` is returned alone. Candidate elimination's update of $G$.
 *
 * @param h The conjunction to specialise.
 * @param x The negative instance to exclude.
 * @param space The attributes, for their values.
 * @returns The minimal specialisations that do not cover `x` (empty when `h` tests every attribute and covers `x`).
 *
 * @example Exclude a rainy, windy day
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * print(specialiseToExclude(mostGeneral(space), [1, 1], space).map((h) => conjunctionText(h, space)))
 */
export function specialiseToExclude(h: Conjunction, x: AttributeInstance, space: AttributeSpace): Conjunction[] {
  if (!covers(h, x)) return [h]
  const g = h as readonly number[]
  const out: Conjunction[] = []
  g.forEach((v, i) => {
    if (v !== ANY) return
    space.values[i].forEach((_, value) => {
      if (value !== x[i]) out.push(g.map((w, j) => (j === i ? value : w)))
    })
  })
  return out
}

/**
 * Every instance of the space, the last attribute varying fastest.
 *
 * @param space The attributes.
 * @returns The $\prod_i k_i$ instances.
 *
 * @example Four instances of two binary attributes
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * print(allInstances(space).map((x) => instanceText(x, space)))
 */
export function allInstances(space: AttributeSpace): AttributeInstance[] {
  let out: number[][] = [[]]
  for (const vals of space.values) out = out.flatMap((p) => vals.map((_, v) => [...p, v]))
  return out
}

/**
 * Every conjunction of the language, ordered from the most general down: by number of tested attributes, then by
 * value (`ANY` first). The empty hypothesis comes last when `includeEmpty` is true.
 *
 * @param space The attributes.
 * @param includeEmpty Whether to append the empty hypothesis `null`.
 * @returns The $\prod_i (k_i + 1)$ conjunctions (and the empty one).
 *
 * @example Nine conjunctions and the empty one
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * print(allConjunctions(space).map((h) => conjunctionText(h, space)))
 */
export function allConjunctions(space: AttributeSpace, includeEmpty = true): Conjunction[] {
  let out: number[][] = [[]]
  for (const vals of space.values) out = out.flatMap((p) => [ANY, ...vals.map((_, v) => v)].map((v) => [...p, v]))
  const tested = (h: readonly number[]) => h.filter((v) => v !== ANY).length
  const sorted: Conjunction[] = out
    .map((h, k) => ({ h, k }))
    .sort((a, b) => tested(a.h) - tested(b.h) || a.k - b.k)
    .map((e) => e.h)
  return includeEmpty ? [...sorted, null] : sorted
}

/**
 * The number of attributes a conjunction tests (its depth below $\langle ?, \dots, ? \rangle$); the empty hypothesis
 * counts one more than the number of attributes.
 *
 * @param h The conjunction.
 * @param space The attributes.
 * @returns The level of `h` in the Hasse diagram, 0 at the top.
 *
 * @example Levels from the top to the bottom
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * print([mostGeneral(space), [0, ANY], [0, 1], null].map((h) => conjunctionLevel(h, space)))
 */
export function conjunctionLevel(h: Conjunction, space: AttributeSpace): number {
  return h === null ? space.names.length + 1 : h.filter((v) => v !== ANY).length
}

/**
 * The covering relation of the generality order among `hypotheses`: pairs $(i, j)$ where `hypotheses[j]` is an
 * immediate specialisation of `hypotheses[i]` ($j \in \rho_\downarrow(i)$). These are the edges of the Hasse diagram.
 *
 * @param hypotheses The conjunctions, as `allConjunctions` lists them or any subset.
 * @param space The attributes.
 * @returns The edges as index pairs, the more general first.
 *
 * @example The Hasse diagram of two binary attributes
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * const hs = allConjunctions(space)
 * print(hs.length, 'hypotheses,', hasseEdges(hs, space).length, 'edges')
 */
export function hasseEdges(hypotheses: readonly Conjunction[], space: AttributeSpace): [number, number][] {
  const key = (h: Conjunction) => (h === null ? '⊥' : h.join(','))
  const index = new Map(hypotheses.map((h, i) => [key(h), i]))
  const edges: [number, number][] = []
  hypotheses.forEach((h, i) => {
    for (const s of specialisations(h, space)) {
      const j = index.get(key(s))
      if (j !== undefined) edges.push([i, j])
    }
  })
  return edges
}

/**
 * The positive and negative examples (by index) a conjunction covers.
 *
 * @param h The conjunction.
 * @param examples The labelled examples.
 * @returns The indices of the covered positives and of the covered negatives, in order.
 *
 * @example Coverage of a hypothesis
 * const ex = [{ x: [0, 0], positive: true }, { x: [0, 1], positive: false }, { x: [1, 0], positive: true }]
 * print(conjunctionCoverage([0, ANY], ex))
 */
export function conjunctionCoverage(
  h: Conjunction,
  examples: readonly LabelledInstance[],
): { positives: number[]; negatives: number[] } {
  const positives: number[] = []
  const negatives: number[] = []
  examples.forEach((e, i) => {
    if (covers(h, e.x)) (e.positive ? positives : negatives).push(i)
  })
  return { positives, negatives }
}

/**
 * True when the conjunction covers every positive and no negative example.
 *
 * @param h The conjunction.
 * @param examples The labelled examples.
 * @returns Whether `h` is consistent with them.
 *
 * @example Weak wind explains the examples; sunny does not
 * const ex = [{ x: [0, 0], positive: true }, { x: [0, 1], positive: false }, { x: [1, 0], positive: true }]
 * print(consistent([ANY, 0], ex), consistent([0, ANY], ex))
 */
export function consistent(h: Conjunction, examples: readonly LabelledInstance[]): boolean {
  return examples.every((e) => covers(h, e.x) === e.positive)
}

/**
 * The conjunction as text: `tuple` gives Mitchell's $\langle \text{sunny}, ?, \dots \rangle$
 * ($\langle \emptyset \rangle$ for the empty hypothesis); `logic` gives `sky = sunny ∧ wind = strong` (`true` for the
 * most general, `false` for the empty one); `tex` gives the same as KaTeX.
 *
 * @param h The conjunction.
 * @param space The attributes, for the names.
 * @param style Which notation.
 * @returns The text.
 *
 * @example Three notations
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * print(conjunctionText([0, ANY], space), '|', conjunctionText([0, 1], space, 'logic'))
 * print(conjunctionText([0, 1], space, 'tex'))
 */
export function conjunctionText(
  h: Conjunction,
  space: AttributeSpace,
  style: 'tuple' | 'logic' | 'tex' = 'tuple',
): string {
  if (style === 'tuple') {
    if (h === null) return '⟨∅⟩'
    return `⟨${h.map((v, i) => (v === ANY ? '?' : space.values[i][v])).join(', ')}⟩`
  }
  if (h === null) return style === 'tex' ? '\\bot' : 'false'
  const tests = h.flatMap((v, i) => (v === ANY ? [] : [[space.names[i], space.values[i][v]] as const]))
  if (tests.length === 0) return style === 'tex' ? '\\top' : 'true'
  if (style === 'logic') return tests.map(([a, v]) => `${a} = ${v}`).join(' ∧ ')
  return tests.map(([a, v]) => `\\text{${a}} = \\text{${v}}`).join(' \\land ')
}

/**
 * An instance as text, $\langle \text{sunny}, \text{warm}, \dots \rangle$.
 *
 * @param x The instance.
 * @param space The attributes, for the value names.
 * @returns The text.
 *
 * @example An instance
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * print(instanceText([1, 0], space))
 */
export function instanceText(x: AttributeInstance, space: AttributeSpace): string {
  return `⟨${x.map((v, i) => space.values[i][v]).join(', ')}⟩`
}

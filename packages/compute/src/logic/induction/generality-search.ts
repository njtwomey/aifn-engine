/**
 * The two directions in which a rule learner searches the generality order of conjunctions: top-down by
 * specialisation, bottom-up by least general generalisation.
 *
 * Mitchell (1982), "Generalization as search"; De Raedt (2008), "Logical and Relational Learning", ch. 3. Top-down
 * (general to specific): start at the most general conjunction $\langle ?, \dots, ? \rangle$ and, while the current
 * hypothesis covers a negative example, move to the refinement in $\rho_\downarrow$ (one more attribute test) with the
 * best score; the default score is $p - n$, positives minus negatives covered, ties broken by more positives and then
 * by the refinement order. This is the hill-climbing step of rule growers such as CN2 and FOIL with a simpler score
 * (beam search over a refinement operator in general is `refinementSearch` of `aifn-compute/optim/search`). Bottom-up
 * (specific to general): start at the empty hypothesis and replace it by its least general generalisation with each
 * positive example in turn (Find-S), the propositional case of the lgg. Both return every state visited, for a
 * stepped figure.
 */
import {
  conjunctionCoverage,
  generaliseToCover,
  mostGeneral,
  specialisations,
  type AttributeSpace,
  type Conjunction,
  type LabelledInstance,
} from './conjunctions'

/** A refinement scored by `topDownSearch`. */
export interface ScoredRefinement {
  /** The refinement. */
  readonly hypothesis: Conjunction
  /** The positives it covers. */
  readonly p: number
  /** The negatives it covers. */
  readonly n: number
  /** Its score. */
  readonly score: number
}

/** One state of a search: the hypothesis, the examples it covers, and the candidates considered to reach the next. */
export interface SearchStep {
  /** The current hypothesis. */
  readonly hypothesis: Conjunction
  /** The positive examples it covers (indices into the examples). */
  readonly positives: readonly number[]
  /** The negative examples it covers (indices into the examples). */
  readonly negatives: readonly number[]
  /** The refinements scored at this state (top-down only; empty at the last state), with their counts and scores. */
  readonly candidates: readonly ScoredRefinement[]
  /** The example that caused this step (bottom-up only), or null. */
  readonly example: number | null
  /** What the step did, in words. */
  readonly note: string
}

/** Options of `topDownSearch`. */
export interface TopDownOptions {
  /** The score of a refinement from its positives $p$ and negatives $n$ covered (default $p - n$). */
  score?: (p: number, n: number) => number
  /** Most specialisation steps (default the number of attributes, the depth of the order). */
  maxSteps?: number
}

/**
 * Greedy general-to-specific search: from $\langle ?, \dots, ? \rangle$, take the best-scoring one-step
 * specialisation until no negative is covered, no refinement keeps a positive, or the step limit is reached.
 *
 * @param space The attributes.
 * @param examples The labelled examples.
 * @param options The score and the step limit (see `TopDownOptions`).
 * @returns The states visited, from $\langle ?, \dots, ? \rangle$ to the hypothesis returned (the last state).
 *
 * @example Specialise until no negative is covered
 * const space = { names: ['a', 'b', 'c'], values: [['0', '1'], ['0', '1'], ['0', '1']] }
 * const ex = [
 *   { x: [1, 0, 0], positive: true },
 *   { x: [1, 1, 0], positive: true },
 *   { x: [0, 1, 0], positive: false },
 * ]
 * for (const s of topDownSearch(space, ex)) print(conjunctionText(s.hypothesis, space, 'logic'), '|', s.note)
 */
export function topDownSearch(
  space: AttributeSpace,
  examples: readonly LabelledInstance[],
  options: TopDownOptions = {},
): SearchStep[] {
  const score = options.score ?? ((p: number, n: number) => p - n)
  const maxSteps = options.maxSteps ?? space.names.length
  let h: Conjunction = mostGeneral(space)
  const steps: SearchStep[] = []
  for (let k = 0; ; k++) {
    const cov = conjunctionCoverage(h, examples)
    if (cov.negatives.length === 0) {
      steps.push({
        hypothesis: h,
        ...cov,
        candidates: [],
        example: null,
        note: `covers ${cov.positives.length} positives and no negative: stop`,
      })
      return steps
    }
    if (k >= maxSteps) {
      steps.push({ hypothesis: h, ...cov, candidates: [], example: null, note: 'step limit reached' })
      return steps
    }
    const candidates = specialisations(h, space)
      .map((r) => {
        const c = conjunctionCoverage(r, examples)
        return {
          hypothesis: r,
          p: c.positives.length,
          n: c.negatives.length,
          score: score(c.positives.length, c.negatives.length),
        }
      })
      .filter((c) => c.p > 0)
    if (candidates.length === 0) {
      steps.push({
        hypothesis: h,
        ...cov,
        candidates: [],
        example: null,
        note: 'every refinement loses all positives: stop',
      })
      return steps
    }
    const best = candidates.reduce((a, b) => (b.score > a.score || (b.score === a.score && b.p > a.p) ? b : a))
    steps.push({
      hypothesis: h,
      ...cov,
      candidates,
      example: null,
      note: `covers ${cov.negatives.length} negative${cov.negatives.length === 1 ? '' : 's'}: specialise to the best of ${candidates.length} refinements`,
    })
    h = best.hypothesis
  }
}

/**
 * Bottom-up search by least general generalisation: from the empty hypothesis, generalise just enough to cover each
 * positive example in turn (Find-S). Negatives are not used to move; each state reports the negatives covered.
 *
 * @param examples The labelled examples; positives are taken in order.
 * @returns The states: the empty hypothesis, then one per positive example that forced a change.
 *
 * @example Generalise from the positives
 * const space = { names: ['a', 'b', 'c'], values: [['0', '1'], ['0', '1'], ['0', '1']] }
 * const ex = [
 *   { x: [1, 0, 0], positive: true },
 *   { x: [1, 1, 0], positive: true },
 *   { x: [0, 1, 0], positive: false },
 * ]
 * for (const s of bottomUpSearch(ex)) print(conjunctionText(s.hypothesis, space, 'logic'), '|', s.note)
 */
export function bottomUpSearch(examples: readonly LabelledInstance[]): SearchStep[] {
  let h: Conjunction = null
  const steps: SearchStep[] = [
    {
      hypothesis: null,
      positives: [],
      negatives: [],
      candidates: [],
      example: null,
      note: 'start from the empty hypothesis',
    },
  ]
  examples.forEach((e, i) => {
    if (!e.positive) return
    const next = generaliseToCover(h, e.x)
    if (h !== null && next !== null && next.every((v, j) => v === h![j])) return
    h = next
    const cov = conjunctionCoverage(h, examples)
    steps.push({
      hypothesis: h,
      ...cov,
      candidates: [],
      example: i,
      note: `generalise just enough to cover positive example ${i + 1}`,
    })
  })
  return steps
}

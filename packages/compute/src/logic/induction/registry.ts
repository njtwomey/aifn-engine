/** The algorithms and functions of `aifn-compute/logic/induction`. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as conjunctions from './conjunctions'
import * as failures from './failures'
import * as foil from './foil'
import * as generalise from './generalise'
import * as generalitySearch from './generality-search'
import * as metagol from './metagol'
import * as progol from './progol'
import * as propositionalise from './propositionalise'
import * as ruleMining from './rule-mining'
import * as versionSpace from './version-space'

const algorithm = definer<AlgorithmInfo>('algorithm', 'logic/induction')
const fn = definer<FunctionInfo>('function', 'logic/induction')

algorithm(
  {
    key: 'foilSteps',
    name: 'FOIL (sequential covering)',
    summary:
      "Quinlan's FOIL: grow one clause at a time by the literal of highest FOIL gain until it covers no negatives, remove the positives it covers, repeat.",
    problem: 'logic-program',
    tags: ['inductive logic programming', 'rule learning', 'sequential covering'],
    state: { iterate: 'current', flags: ['terminated'] },
  },
  foil.foilSteps,
)
fn(
  {
    key: 'foil',
    name: 'FOIL to the end',
    role: 'fit',
    summary: 'The clauses FOIL learns for a target relation from background facts and examples, as Prolog text.',
  },
  foil.foil,
)
fn(
  {
    key: 'foilProblem',
    name: 'FOIL problem',
    role: 'construction',
    summary:
      'Background relations and positive and negative examples with constants interned; closed-world negatives when none are given.',
  },
  foil.foilProblem,
)
fn(
  {
    key: 'foilRefinements',
    name: 'FOIL refinement operator',
    role: 'transform',
    summary: 'Every literal FOIL can add to a clause, each with the extended tuples, counts and FOIL gain.',
  },
  foil.foilRefinements,
)
fn({ key: 'foilGain', name: 'FOIL gain', role: 'estimator' }, foil.foilGain)
fn(
  {
    key: 'foilCoverage',
    name: 'Examples a clause covers',
    role: 'property',
    summary: 'The positive and negative examples a clause body covers, by backtracking over bindings of its variables.',
  },
  foil.foilCoverage,
)
fn({ key: 'foilSimplify', name: 'Simplify a learned clause', role: 'transform' }, foil.foilSimplify)
fn(
  {
    key: 'thetaSubsumption',
    name: 'θ-subsumption',
    role: 'solver',
    summary: 'A substitution θ with Cθ ⊆ D (C at least as general as D), or null; backtracking over literal matches.',
    notes: ['theta-subsumption'],
    cite: ['plotkin1972thesis', 'gottlob1985'],
  },
  generalise.thetaSubsumption,
)
fn(
  { key: 'thetaSubsumes', name: 'Is θ-subsumed', role: 'property', notes: ['theta-subsumption'] },
  generalise.thetaSubsumes,
)
fn(
  {
    key: 'termLgg',
    name: 'Least general generalisation of terms',
    role: 'transform',
    summary: "Plotkin's anti-unification: differing subterm pairs become variables, the same pair the same variable.",
    notes: ['least-general-generalisation'],
    cite: ['plotkin1972thesis'],
  },
  generalise.termLgg,
)
fn(
  {
    key: 'clauseLgg',
    name: 'Least general generalisation of clauses',
    role: 'transform',
    summary: 'The lgg of two clauses: heads and every pair of same-predicate body literals, with one variable table.',
    notes: ['least-general-generalisation'],
    cite: ['plotkin1972thesis'],
  },
  generalise.clauseLgg,
)
fn({ key: 'reduceClause', name: 'Reduce a clause', role: 'transform' }, generalise.reduceClause)

// ── Generality traced ──

fn(
  {
    key: 'thetaSubsumptionTrace',
    name: 'θ-subsumption step by step',
    tex: 'C\\theta \\subseteq D',
    role: 'solver',
    summary:
      'The backtracking search for θ with Cθ ⊆ D, one match attempt per state (match, clash, backtrack), the literals of C in written or most-constrained order.',
    notes: ['theta-subsumption'],
    cite: ['plotkin1972thesis', 'gottlob1985'],
  },
  generalise.thetaSubsumptionTrace,
)
fn(
  {
    key: 'termLggTraced',
    name: 'Least general generalisation, comparison by comparison',
    role: 'transform',
    summary:
      'Anti-unification of two terms with every comparison recorded (kept, descended into, new or reused variable) and a table of pairs shared across terms.',
    notes: ['least-general-generalisation'],
    cite: ['plotkin1972thesis'],
  },
  generalise.termLggTraced,
)
fn(
  {
    key: 'clauseLggTrace',
    name: 'Clause lgg, pair by pair',
    role: 'transform',
    summary: 'The lgg of two clauses one literal pair at a time, with the table of pairs after each.',
    notes: ['least-general-generalisation'],
    cite: ['plotkin1972thesis'],
  },
  generalise.clauseLggTrace,
)
fn(
  {
    key: 'relativeLgg',
    name: 'Relative least general generalisation (Golem)',
    role: 'fit',
    summary:
      'The lgg of e1 ← B and e2 ← B for ground background facts B, without ground and head-unlinked literals, then reduced.',
    notes: ['least-general-generalisation'],
    cite: ['muggleton1990golem', 'plotkin1972thesis'],
  },
  generalise.relativeLgg,
)
fn(
  {
    key: 'headLinked',
    name: 'Head-linked literals',
    role: 'property',
    summary: 'The body literals joined to the head by a chain of shared variables, and the others.',
    notes: ['least-general-generalisation'],
  },
  generalise.headLinked,
)

// ── Inverse entailment, meta-interpretive learning, learning from failures ──

fn(
  {
    key: 'bottomClause',
    name: 'Bottom clause',
    tex: '\\bot(e)',
    role: 'construction',
    summary:
      'The most specific clause, within mode declarations and a depth, that with ground background facts entails an example (Progol saturation).',
    notes: ['inverse-entailment-and-progol'],
    cite: ['muggleton1995progol', 'srinivasan2001aleph'],
  },
  progol.bottomClause,
)
fn(
  {
    key: 'bottomClauseSearch',
    name: 'Search below a bottom clause',
    role: 'fit',
    summary:
      'Breadth-first search of the clauses whose body is a subset of a bottom clause, scored by compression P − N − L + 1, with coverage by SLD resolution.',
    notes: ['inverse-entailment-and-progol'],
    cite: ['muggleton1995progol', 'srinivasan2001aleph'],
  },
  progol.bottomClauseSearch,
)
fn(
  {
    key: 'metagol',
    name: 'Meta-interpretive learning (Metagol)',
    role: 'fit',
    summary:
      'Dyadic Datalog programs learned by proving the examples with metarules, inventing predicates, under iterative deepening on the number of clauses; every event recorded.',
    notes: ['meta-interpretive-learning', 'predicate-invention'],
    cite: ['muggleton2015mil'],
  },
  metagol.metagol,
)
fn(
  {
    key: 'learnFromFailures',
    name: 'Learning from failures (Popper)',
    role: 'fit',
    summary:
      'Generate the smallest program no constraint prunes, test it, and prune the generalisations, specialisations or separable programs its failure rules out.',
    notes: ['learning-from-failures'],
    cite: ['cropper2021popper'],
  },
  failures.learnFromFailures,
)

// ── Relational data ──

fn(
  {
    key: 'mineClosedRules',
    name: 'Closed rule mining (AMIE)',
    role: 'fit',
    summary:
      'Every closed rule of at most three atoms over a knowledge graph, with support, head coverage, standard and PCA confidence.',
    notes: ['knowledge-graph-rule-mining'],
    cite: ['galarraga2013amie'],
  },
  ruleMining.mineClosedRules,
)
fn(
  {
    key: 'predictLinks',
    name: 'Link prediction by maximum aggregation',
    role: 'transform',
    summary: 'The facts mined rules predict, each scored by the best rule predicting it (AnyBURL).',
    notes: ['knowledge-graph-rule-mining'],
    cite: ['meilicke2019anyburl'],
  },
  ruleMining.predictLinks,
)
fn(
  {
    key: 'existentialFeatures',
    name: 'Existential features (RSD)',
    role: 'transform',
    summary:
      'Boolean features of individuals: some row of a child table satisfies a conjunction of attribute tests; irrelevant ones filtered.',
    notes: ['propositionalisation-and-relational-learning'],
    cite: ['zelezny2006rsd', 'kramer2001propositionalisation'],
  },
  propositionalise.existentialFeatures,
)
fn(
  {
    key: 'aggregateFeatures',
    name: 'Aggregate features (RELAGGS)',
    role: 'transform',
    summary: 'Per child table, the count of rows of each individual and the count with each attribute value.',
    notes: ['propositionalisation-and-relational-learning'],
    cite: ['krogel2001relaggs', 'kramer2001propositionalisation'],
  },
  propositionalise.aggregateFeatures,
)
fn(
  {
    key: 'wordify',
    name: 'Wordification',
    role: 'transform',
    summary:
      'Each individual as a document of table_attribute_value words and their conjunctions within a row, weighted by tf, tf-idf or presence.',
    notes: ['propositionalisation-and-relational-learning'],
    cite: ['perovsek2015wordification'],
  },
  propositionalise.wordify,
)
fn(
  {
    key: 'attributeFeatures',
    name: "One-hot features of individuals' attributes",
    role: 'transform',
    notes: ['propositionalisation-and-relational-learning'],
  },
  propositionalise.attributeFeatures,
)
fn(
  {
    key: 'featureTable',
    name: 'Flat feature table',
    role: 'transform',
    notes: ['propositionalisation-and-relational-learning'],
  },
  propositionalise.featureTable,
)

// ── Concept learning over conjunctions ──

algorithm(
  {
    key: 'candidateElimination',
    name: 'Candidate elimination (version spaces)',
    summary:
      "Mitchell's candidate elimination: one example per step, the specific boundary S generalised by positives and the general boundary G specialised by negatives.",
    problem: 'table',
    tags: ['concept learning', 'version space'],
    state: { flags: ['terminated'] },
    notes: ['concept-learning-and-version-spaces'],
    cite: ['mitchell1982', 'mitchell1997'],
  },
  versionSpace.candidateElimination,
)
fn(
  {
    key: 'findS',
    name: 'Find-S',
    role: 'fit',
    summary: 'The most specific conjunction covering every positive example, generalising one positive at a time.',
    notes: ['concept-learning-and-version-spaces'],
    cite: ['mitchell1997'],
  },
  versionSpace.findS,
)
fn(
  {
    key: 'versionSpaceMembers',
    name: 'Members of a version space',
    role: 'property',
    summary: 'Every conjunction between the boundaries S and G.',
    notes: ['concept-learning-and-version-spaces'],
    cite: ['mitchell1982'],
  },
  versionSpace.versionSpaceMembers,
)
fn(
  {
    key: 'versionSpaceVote',
    name: 'Version-space vote',
    role: 'inference',
    summary: 'How many members of a version space classify an instance as positive and as negative, and the verdict.',
    notes: ['concept-learning-and-version-spaces'],
  },
  versionSpace.versionSpaceVote,
)
fn(
  {
    key: 'hasseEdges',
    name: 'Hasse diagram of the generality order',
    role: 'construction',
    summary: 'The covering pairs of the more-general-than order among conjunctions: one-step specialisations.',
    notes: ['generality-ordering-and-refinement'],
    cite: ['mitchell1982'],
  },
  conjunctions.hasseEdges,
)
fn(
  {
    key: 'specialisations',
    name: 'Downward refinement of a conjunction',
    tex: '\\rho_\\downarrow',
    role: 'transform',
    notes: ['generality-ordering-and-refinement'],
  },
  conjunctions.specialisations,
)
fn(
  {
    key: 'generalisations',
    name: 'Upward refinement of a conjunction',
    tex: '\\rho_\\uparrow',
    role: 'transform',
    notes: ['generality-ordering-and-refinement'],
  },
  conjunctions.generalisations,
)
fn(
  {
    key: 'topDownSearch',
    name: 'Top-down search of the generality order',
    role: 'solver',
    summary:
      'Hill climbing from the most general conjunction by the best one-step specialisation until no negative is covered.',
    notes: ['generality-ordering-and-refinement'],
    cite: ['mitchell1982'],
  },
  generalitySearch.topDownSearch,
)
fn(
  {
    key: 'bottomUpSearch',
    name: 'Bottom-up search by least general generalisation',
    role: 'solver',
    summary: 'From the empty hypothesis, the least generalisation covering each positive example in turn.',
    notes: ['generality-ordering-and-refinement'],
    cite: ['mitchell1982'],
  },
  generalitySearch.bottomUpSearch,
)

/** The step-through algorithms of the module, keyed by factory name. */
export const inductionAlgorithms = entries<AlgorithmInfo>('algorithm', foil, versionSpace) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
>

/** The functions of the module, keyed by name. */
export const inductionFunctions = entries<FunctionInfo>(
  'function',
  foil,
  generalise,
  conjunctions,
  versionSpace,
  generalitySearch,
  progol,
  metagol,
  failures,
  ruleMining,
  propositionalise,
) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

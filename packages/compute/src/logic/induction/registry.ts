/** The algorithms and functions of `aifn-compute/logic/induction`. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as foil from './foil'
import * as generalise from './generalise'

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
  },
  generalise.thetaSubsumption,
)
fn({ key: 'thetaSubsumes', name: 'Is θ-subsumed', role: 'property' }, generalise.thetaSubsumes)
fn(
  {
    key: 'termLgg',
    name: 'Least general generalisation of terms',
    role: 'transform',
    summary: "Plotkin's anti-unification: differing subterm pairs become variables, the same pair the same variable.",
  },
  generalise.termLgg,
)
fn(
  {
    key: 'clauseLgg',
    name: 'Least general generalisation of clauses',
    role: 'transform',
    summary: 'The lgg of two clauses: heads and every pair of same-predicate body literals, with one variable table.',
  },
  generalise.clauseLgg,
)
fn({ key: 'reduceClause', name: 'Reduce a clause', role: 'transform' }, generalise.reduceClause)

/** The step-through algorithms of the module, keyed by factory name. */
export const inductionAlgorithms = entries<AlgorithmInfo>('algorithm', foil) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
>

/** The functions of the module, keyed by name. */
export const inductionFunctions = entries<FunctionInfo>('function', foil, generalise) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>

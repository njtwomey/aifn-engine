/**
 * The functions of `aifn-compute/learning/rules`: the rule learners (sequential covering, RIPPER, C4.5rules, Bayesian
 * rule lists, CBA and CMAR, CORELS, Boolean rules by column generation), the rule quality measures and the pieces the
 * learners share (conditions, antecedents, RuleFit's terms), each with the site notes that use it.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as associative from './associative'
import * as booleanRules from './boolean-rules'
import * as brl from './brl'
import * as c45 from './c45'
import * as conditions from './conditions'
import * as corels from './corels'
import * as covering from './covering'
import * as quality from './quality'
import * as ripper from './ripper'
import * as rulefit from './rulefit'

const fn = definer<FunctionInfo>('function', 'learning/rules')

const RULE_NOTES = [
  'decision-lists-and-rule-sets',
  'sequential-covering',
  'ripper',
  'rule-quality-measures',
  'rules-from-decision-trees',
  'bayesian-rule-lists',
  'association-rule-classification',
  'certifiably-optimal-rule-lists',
  'rule-ensembles-and-rulefit',
]

fn(
  {
    key: 'thresholdConditions',
    name: 'Threshold conditions',
    summary: 'The conditions x ≤ q and x > q at the inner quantile edges of each numeric feature.',
    role: 'construction',
    notes: RULE_NOTES,
  },
  conditions.thresholdConditions,
)
fn(
  {
    key: 'nominalConditions',
    name: 'Nominal conditions',
    summary: 'The equality tests x = v of integer-coded nominal features, as intervals of width 1.',
    role: 'construction',
    notes: RULE_NOTES,
  },
  conditions.nominalConditions,
)
fn(
  {
    key: 'mineAntecedents',
    name: 'Frequent antecedents (Apriori)',
    summary: 'Every conjunction of conditions with at least the given support, mined level by level as in Apriori.',
    role: 'estimator',
    notes: [
      'bayesian-rule-lists',
      'association-rule-classification',
      'certifiably-optimal-rule-lists',
      'rule-quality-measures',
    ],
    cite: ['letham2015', 'liu1998cba'],
  },
  conditions.mineAntecedents,
)
fn(
  {
    key: 'applyClassRules',
    name: 'Apply a decision list',
    summary: 'The class of the first rule a row meets, else the default class.',
    role: 'transform',
    notes: ['rules-from-decision-trees', 'decision-lists-and-rule-sets'],
    cite: ['rivest1987'],
  },
  conditions.applyClassRules,
)
fn(
  {
    key: 'ruleQuality',
    name: 'Rule quality measures',
    summary:
      'Coverage, precision, Laplace, m-estimate, rule accuracy, WRAcc, FOIL gain, the CN2 likelihood ratio and the RIPPER pruning value of a rule’s counts.',
    role: 'estimator',
    notes: ['rule-quality-measures', 'sequential-covering', 'ripper'],
    cite: ['furnkranz2005', 'lavrac1999', 'clark1991', 'quinlan1990', 'clark1989', 'cohen1995'],
  },
  quality.ruleQuality,
)
fn(
  {
    key: 'laplace',
    name: 'Laplace estimate',
    tex: '\\frac{p + 1}{p + n + k}',
    summary: 'Precision with one pseudo-count per class.',
    role: 'estimator',
    notes: ['rule-quality-measures', 'sequential-covering'],
    cite: ['clark1991'],
  },
  quality.laplace,
)
fn(
  {
    key: 'mEstimate',
    name: 'm-estimate',
    tex: '\\frac{p + m P/(P + N)}{p + n + m}',
    summary: 'Precision shrunk towards the class prior by m pseudo-counts.',
    role: 'estimator',
    notes: ['rule-quality-measures'],
    cite: ['furnkranz2005'],
  },
  quality.mEstimate,
)
fn(
  {
    key: 'likelihoodRatio',
    name: 'CN2 likelihood ratio',
    summary: 'CN2’s significance statistic of a rule’s covered class counts against those the prior expects.',
    role: 'test',
    notes: ['sequential-covering', 'rule-quality-measures'],
    cite: ['clark1989'],
  },
  quality.likelihoodRatio,
)
fn(
  {
    key: 'sequentialCovering',
    name: 'Sequential covering (CN2)',
    summary:
      'Separate and conquer: learn the best rule by beam search under a quality measure, remove the rows it covers, repeat; with a full trace.',
    role: 'fit',
    notes: ['sequential-covering', 'decision-lists-and-rule-sets'],
    cite: ['clark1989', 'clark1991', 'rivest1987'],
  },
  covering.sequentialCovering,
)
fn(
  {
    key: 'applyCoveringRules',
    name: 'Apply covering rules',
    summary: 'Classify rows by learned covering rules, as a decision list or by CN2’s unordered vote.',
    role: 'transform',
    notes: ['decision-lists-and-rule-sets', 'sequential-covering'],
    cite: ['clark1991'],
  },
  covering.applyCoveringRules,
)
fn(
  {
    key: 'ripper',
    name: 'RIPPER',
    summary:
      'IREP* with FOIL-gain growth, reduced-error pruning and a description-length stop, then RIPPERk optimisation passes; class by class.',
    role: 'fit',
    notes: ['ripper'],
    cite: ['cohen1995'],
    random: true,
  },
  ripper.ripper,
)
fn(
  {
    key: 'ripperDescriptionLength',
    name: 'RIPPER description length',
    summary: 'The bits of a two-class rule set and of its exceptions, as RIPPER and JRip encode them.',
    role: 'estimator',
    notes: ['ripper'],
    cite: ['cohen1995'],
  },
  ripper.ripperDescriptionLength,
)
fn(
  {
    key: 'pessimisticError',
    name: 'Pessimistic error',
    tex: 'U_{CF}(E, N)',
    summary: 'C4.5’s upper confidence limit on a rule’s error rate: the exact binomial (Clopper-Pearson) bound.',
    role: 'estimator',
    notes: ['rules-from-decision-trees'],
    cite: ['quinlan1993'],
  },
  c45.pessimisticError,
)
fn(
  {
    key: 'generaliseRule',
    name: 'Generalise a rule (C4.5rules)',
    summary: 'Delete conditions one at a time while the pessimistic error does not rise.',
    role: 'transform',
    notes: ['rules-from-decision-trees'],
    cite: ['quinlan1987', 'quinlan1993'],
  },
  c45.generaliseRule,
)
fn(
  {
    key: 'c45Rules',
    name: 'C4.5rules',
    summary:
      'A tree’s path rules generalised by pessimistic error, chosen per class by description length and ordered by false positives.',
    role: 'fit',
    notes: ['rules-from-decision-trees'],
    cite: ['quinlan1987', 'quinlan1993'],
  },
  c45.c45Rules,
)
fn(
  {
    key: 'ruleListLogPosterior',
    name: 'Bayesian rule list posterior',
    summary:
      'The log prior (truncated Poisson length and cardinalities) and Dirichlet-multinomial log likelihood of a decision list.',
    role: 'inference',
    notes: ['bayesian-rule-lists'],
    cite: ['letham2015'],
  },
  brl.ruleListLogPosterior,
)
fn(
  {
    key: 'bayesianRuleListTrace',
    name: 'Bayesian rule list sampler',
    summary: 'Metropolis-Hastings over decision lists with move, add and remove proposals; every state of the chain.',
    role: 'simulation',
    notes: ['bayesian-rule-lists'],
    cite: ['letham2015', 'yang2017sbrl'],
    random: true,
  },
  brl.bayesianRuleListTrace,
)
fn(
  {
    key: 'classAssociationRules',
    name: 'Class association rules',
    summary:
      'The CARs of frequent condition sets (best class, support and confidence thresholds), in CBA’s precedence.',
    role: 'estimator',
    notes: ['association-rule-classification'],
    cite: ['liu1998cba'],
  },
  associative.classAssociationRules,
)
fn(
  {
    key: 'cbaBuild',
    name: 'CBA classifier builder (M1)',
    summary: 'Select ranked rules by database coverage and cut the list where the training errors are fewest.',
    role: 'fit',
    notes: ['association-rule-classification'],
    cite: ['liu1998cba'],
  },
  associative.cbaBuild,
)
fn(
  {
    key: 'cmarBuild',
    name: 'CMAR',
    summary: 'Prune ranked rules by generality, a χ² test and δ-coverage; predict by weighted χ².',
    role: 'fit',
    notes: ['association-rule-classification'],
    cite: ['li2001cmar'],
  },
  associative.cmarBuild,
)
fn(
  {
    key: 'ruleChiSquare',
    name: 'χ² of a rule and its maximum',
    tex: '\\chi^2, \\max\\chi^2',
    summary: 'The 2 × 2 χ² of a rule against its class, and the largest value its margins allow.',
    role: 'test',
    notes: ['association-rule-classification'],
    cite: ['li2001cmar'],
  },
  associative.ruleChiSquare,
)
fn(
  {
    key: 'corels',
    name: 'CORELS',
    summary:
      'Branch and bound over rule-list prefixes for the least regularised risk, with CORELS’s bounds and an optimality certificate.',
    role: 'solver',
    notes: ['certifiably-optimal-rule-lists'],
    cite: ['angelino2018', 'angelino2017'],
  },
  corels.corels,
)
fn(
  {
    key: 'booleanRuleColumnGeneration',
    name: 'Boolean rules by column generation',
    summary:
      'A DNF rule set of least Hamming loss within a complexity budget: column generation on the LP relaxation, then the restricted integer program.',
    role: 'solver',
    notes: ['boolean-rule-learning'],
    cite: ['dash2018brcg'],
  },
  booleanRules.booleanRuleColumnGeneration,
)
fn(
  {
    key: 'ensembleRules',
    name: 'Rules of a tree ensemble',
    summary: 'The conjunctive rule of every non-root node of every tree, 2(t − 1) per tree of t leaves.',
    role: 'transform',
    notes: ['rule-ensembles-and-rulefit'],
    cite: ['friedman2008rulefit'],
  },
  rulefit.ensembleRules,
)
fn(
  {
    key: 'ruleFeatures',
    name: 'Rule columns',
    summary: 'Rules as 0/1 columns on the training rows, one per distinct coverage, constant ones dropped.',
    role: 'transform',
    notes: ['rule-ensembles-and-rulefit'],
    cite: ['friedman2008rulefit'],
  },
  rulefit.ruleFeatures,
)
fn(
  {
    key: 'linearTerms',
    name: 'RuleFit linear terms',
    summary: 'Winsorising limits of each input and the scale that gives the winsorised input standard deviation 0.4.',
    role: 'estimator',
    notes: ['rule-ensembles-and-rulefit'],
    cite: ['friedman2008rulefit'],
  },
  rulefit.linearTerms,
)

/** The functions of the module, keyed by name. */
export const ruleFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>(
    'function',
    conditions,
    quality,
    covering,
    ripper,
    c45,
    brl,
    associative,
    corels,
    booleanRules,
    rulefit,
  ) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

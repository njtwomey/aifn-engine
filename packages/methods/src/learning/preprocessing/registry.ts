/**
 * The registry entries of the functions of `aifn-methods/learning/preprocessing` besides its transformers (which
 * `defineModel` registers beside their definitions): the fitting helpers, cross-fitted target encoding and the
 * resamplers for imbalanced classes, with their notes and citations.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as encoding from './encoding'
import * as imbalanced from './imbalanced'
import * as transformer from './transformer'

const fn = definer<FunctionInfo>('function', 'learning/preprocessing')

fn(
  { key: 'fitTransform', name: 'Fit and transform', role: 'transform', notes: ['feature-scaling', 'data-leakage'] },
  transformer.fitTransform,
)
fn({ key: 'checkColumns', name: 'Check columns', role: 'property' }, transformer.checkColumns)
fn(
  {
    key: 'targetEncodeCrossFit',
    name: 'Cross-fitted target encoding',
    summary: 'Target means per category computed out of fold, so a row never sees its own label.',
    role: 'transform',
    notes: ['categorical-encoding', 'data-leakage'],
    cite: ['micci2001'],
  },
  encoding.targetEncodeCrossFit,
)

const IMB = ['class-imbalance', 'random-over-and-undersampling']
const SMOTE = ['smote-and-variants', 'class-imbalance']
fn(
  {
    key: 'randomOverSample',
    name: 'Random over-sampling',
    summary: 'Copies of minority rows drawn with replacement until the classes are balanced.',
    role: 'transform',
    random: true,
    notes: IMB,
  },
  imbalanced.randomOverSample,
)
fn(
  {
    key: 'randomUnderSample',
    name: 'Random under-sampling',
    summary: 'Every class cut to the smallest class’s count, without replacement.',
    role: 'transform',
    random: true,
    notes: IMB,
  },
  imbalanced.randomUnderSample,
)
fn(
  {
    key: 'smote',
    name: 'SMOTE',
    summary: 'Synthetic minority rows on segments between a minority row and one of its k nearest minority neighbours.',
    role: 'transform',
    random: true,
    notes: [...SMOTE, 'does-smote-help'],
    cite: ['chawla2002'],
  },
  imbalanced.smote,
)
fn(
  {
    key: 'borderStatus',
    name: 'Borderline status',
    summary: 'Safe, danger or noise by the share of other classes among a row’s m nearest neighbours.',
    role: 'property',
    notes: SMOTE,
    cite: ['han2005'],
  },
  imbalanced.borderStatus,
)
fn(
  {
    key: 'borderlineSmote',
    name: 'Borderline-SMOTE',
    summary: 'SMOTE from the minority rows in danger only: near the class boundary but not surrounded.',
    role: 'transform',
    random: true,
    notes: SMOTE,
    cite: ['han2005'],
  },
  imbalanced.borderlineSmote,
)
fn(
  {
    key: 'adasynWeights',
    name: 'ADASYN allocation',
    summary: 'Synthetic rows per minority row in proportion to the other classes among its k nearest neighbours.',
    role: 'property',
    notes: SMOTE,
    cite: ['he2008'],
  },
  imbalanced.adasynWeights,
)
fn(
  {
    key: 'adasyn',
    name: 'ADASYN',
    summary: 'Adaptive synthetic sampling: more synthetic rows where the minority class is hardest to learn.',
    role: 'transform',
    random: true,
    notes: SMOTE,
    cite: ['he2008'],
  },
  imbalanced.adasyn,
)
fn(
  {
    key: 'tomekLinks',
    name: 'Tomek links',
    summary: 'Pairs of rows of different classes that are each other’s nearest neighbour.',
    role: 'property',
    notes: IMB,
  },
  imbalanced.tomekLinks,
)
fn(
  {
    key: 'removeTomekLinks',
    name: 'Tomek-link cleaning',
    summary: 'Remove the non-minority member of every Tomek link, clearing the class boundary.',
    role: 'transform',
    notes: IMB,
  },
  imbalanced.removeTomekLinks,
)

/** The functions of the module, keyed by name. */
export const preprocessingFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', transformer, encoding, imbalanced) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >

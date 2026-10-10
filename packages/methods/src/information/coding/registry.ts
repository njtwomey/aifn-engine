/**
 * The registry of `aifn-methods/information/coding`: Huffman's algorithm as a traceable algorithm, and the source
 * codes, encoders, code properties and code-size bounds as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as coding from './coding'

/** A table of the module's registry entries, keyed by name. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const fn = definer<FunctionInfo>('function', 'information/coding')
const SRC = ['source-coding-theorem', 'entropy']
const TREE = [...SRC, 'hierarchical-softmax']

definer<AlgorithmInfo>('algorithm', 'information/coding')(
  {
    key: 'huffmanSteps',
    name: 'Huffman coding',
    summary:
      'Pop the D least-weight roots off a priority queue, join them under a new node and push it back, until one tree remains; binary or D-ary, with a tie-breaking rule.',
    problem: 'graph',
    state: { iterate: 'nodes', flags: [] },
    notes: TREE,
    cite: ['huffman1952'],
  },
  coding.huffmanSteps,
)
fn(
  {
    key: 'huffmanTree',
    name: 'Huffman tree',
    role: 'construction',
    returns: 'tree',
    notes: TREE,
    cite: ['huffman1952'],
  },
  coding.huffmanTree,
)
fn(
  { key: 'huffmanCode', name: 'Huffman code', role: 'construction', notes: SRC, cite: ['huffman1952'] },
  coding.huffmanCode,
)
fn(
  {
    key: 'huffmanDummies',
    name: 'Huffman dummy symbols',
    summary: 'Zero-weight leaves a D-ary Huffman code needs so that every merge takes D nodes.',
    role: 'property',
    notes: SRC,
  },
  coding.huffmanDummies,
)
fn(
  {
    key: 'canonicalCode',
    name: 'Canonical prefix code',
    summary: 'Codewords from lengths alone: consecutive numbers in (length, symbol) order.',
    role: 'construction',
    notes: SRC,
  },
  coding.canonicalCode,
)
fn(
  {
    key: 'sourceExtension',
    name: 'Source extension (block coding)',
    summary: 'The K^n probabilities of blocks of n i.i.d. symbols.',
    role: 'construction',
    notes: SRC,
    cite: ['cover2006'],
  },
  coding.sourceExtension,
)
fn({ key: 'prefixEncode', name: 'Prefix encoding', role: 'transform', notes: SRC }, coding.prefixEncode)
fn({ key: 'prefixDecode', name: 'Prefix decoding by tree walk', role: 'transform', notes: SRC }, coding.prefixDecode)
fn(
  {
    key: 'shannonCode',
    name: 'Shannon code',
    summary: 'Codeword lengths ⌈−log₂ p⌉.',
    role: 'construction',
    notes: SRC,
    cite: ['shannon1948'],
  },
  coding.shannonCode,
)
fn({ key: 'shannonFanoCode', name: 'Shannon–Fano code', role: 'construction', notes: SRC }, coding.shannonFanoCode)
fn(
  { key: 'arithmeticInterval', name: 'Arithmetic-coding interval', role: 'transform', notes: SRC },
  coding.arithmeticInterval,
)
fn(
  {
    key: 'kraftSum',
    name: 'Kraft sum',
    tex: '\\sum_i 2^{-\\ell_i}',
    role: 'property',
    notes: SRC,
    cite: ['cover2006'],
  },
  coding.kraftSum,
)
fn(
  { key: 'hammingDistance', name: 'Hamming distance', role: 'property', cite: ['hamming1950'] },
  coding.hammingDistance,
)
fn({ key: 'hammingWeight', name: 'Hamming weight', role: 'property' }, coding.hammingWeight)
fn({ key: 'minimumDistance', name: 'Minimum distance of a code', role: 'property' }, coding.minimumDistance)
fn(
  { key: 'hammingBound', name: 'Hamming (sphere-packing) bound', role: 'property', cite: ['hamming1950'] },
  coding.hammingBound,
)
fn({ key: 'singletonBound', name: 'Singleton bound', role: 'property' }, coding.singletonBound)
fn({ key: 'plotkinBound', name: 'Plotkin bound', role: 'property' }, coding.plotkinBound)

/** The algorithms of the module, keyed by factory name. */
export const codingAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  coding,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const codingFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', coding) as Table<FunctionInfo>

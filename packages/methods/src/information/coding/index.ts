/**
 * `aifn-methods/information/coding`: source coding (Huffman codes as a step-through algorithm, canonical codes, source
 * extensions, prefix encoding and decoding, Shannon–Fano, Shannon and arithmetic coding) and error-correcting code
 * bounds.
 */

export {
  arithmeticInterval,
  canonicalCode,
  hammingBound,
  hammingDistance,
  hammingWeight,
  huffmanCode,
  huffmanDummies,
  huffmanSteps,
  huffmanTree,
  kraftSum,
  minimumDistance,
  plotkinBound,
  prefixDecode,
  prefixEncode,
  shannonCode,
  shannonFanoCode,
  singletonBound,
  sourceExtension,
  type ArithmeticInterval,
  type Decoded,
  type Encoded,
  type HuffmanEdgeData,
  type HuffmanNode,
  type HuffmanNodeData,
  type HuffmanOptions,
  type HuffmanState,
  type HuffmanTies,
  type HuffmanTree,
  type PrefixCode,
  type Word,
} from './coding'
export { codingAlgorithms, codingFunctions } from './registry'

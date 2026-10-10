/**
 * `aifn-methods/information/coding`: source codes for discrete memoryless sources, and the distances and size bounds
 * of error-correcting codes.
 *
 * - Huffman codes, optimal among prefix codes: `huffmanCode` (binary or $D$-ary, with a tie rule that sets the spread
 *   of the lengths), `huffmanSteps` (the algorithm one merge at a time, traceable), `huffmanTree` (its code tree) and
 *   `huffmanDummies` (the padding a $D$-ary code needs).
 * - Other codes: `shannonFanoCode` (Fano's splitting), `shannonCode` (lengths $\lceil -\log_2 p_k \rceil$) and
 *   `canonicalCode` (codewords from lengths alone); each `PrefixCode` reports its expected length beside the entropy.
 * - Using a code: `prefixEncode`, `prefixDecode` (by walking the tree), `sourceExtension` (blocks of $n$ symbols, whose
 *   codes approach the entropy), `kraftSum` (whether lengths admit a prefix code) and `arithmeticInterval` (the
 *   interval of a message).
 * - Error-correcting codes: `hammingDistance`, `hammingWeight`, `minimumDistance`, and bounds on the number of
 *   codewords $A_q(n, d)$: `hammingBound` (sphere packing), `singletonBound` and `plotkinBound`.
 * - The registry: `codingAlgorithms` and `codingFunctions`.
 *
 * Conventions: symbols are indices $0, \dots, K - 1$; probabilities may be counts and are normalised; codewords are
 * strings of the digits $0, \dots, D - 1$ (0–9 then a–z, so $D \le 36$), and lengths are in $D$-ary digits (bits for
 * $D = 2$). Bad input throws `DomainError`, and words of different lengths throw `ShapeError`.
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

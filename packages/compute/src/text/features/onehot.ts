/**
 * One-hot encoding of a token sequence: column $j$ of the vocabulary $\times$ positions matrix is the indicator vector
 * $\evec_{\mathrm{id}(t_j)}$. Every word is a standard basis vector, so any two distinct words are orthogonal (cosine
 * 0) at Euclidean distance $\sqrt{2}$: the encoding carries identity and nothing else, which is what distributional
 * representations improve on.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { tokenId, type Vocabulary } from 'aifn-compute/text/vocabulary'

/**
 * The one-hot matrix of a token sequence (float64 [V, n]): entry $(\mathrm{id}(t_j), j)$ is 1 and every other entry 0.
 * A token outside the vocabulary takes the unknown id when the vocabulary has one; otherwise it is an error
 * (`DomainError`), or its column is left zero with `onUnknown: 'zero'`.
 *
 * @param tokens The token sequence $t_0, \dots, t_{n-1}$, one column each.
 * @param vocabulary The vocabulary, one row per id.
 * @param options What to do with an out-of-vocabulary token when the vocabulary has no unknown id.
 * @param options.onUnknown `'error'` (the default) throws; `'zero'` leaves the token's column zero.
 * @returns The one-hot matrix, vocabulary $\times$ positions.
 *
 * @example One column per position
 * const v = bagOfWords([['the', 'cat', 'sat', 'mat']]).vocabulary
 * print('rows', v.tokens)
 * print(oneHotTokens(['the', 'cat', 'sat'], v))
 * print('unknown left zero', oneHotTokens(['the', 'dog'], v, { onUnknown: 'zero' }))
 */
export function oneHotTokens(
  tokens: readonly string[],
  vocabulary: Vocabulary,
  options: { onUnknown?: 'error' | 'zero' } = {},
): Tensor {
  const V = vocabulary.tokens.length
  const n = tokens.length
  const out = new Float64Array(V * n)
  tokens.forEach((t, j) => {
    let id = tokenId(vocabulary, t)
    if (id < 0) id = vocabulary.unknown
    if (id < 0) {
      if (options.onUnknown === 'zero') return
      throw new DomainError(
        'oneHotTokens',
        `oneHotTokens: '${t}' is not in the vocabulary and there is no unknown token`,
      )
    }
    out[id * n + j] = 1
  })
  return fromData(out, [V, n])
}

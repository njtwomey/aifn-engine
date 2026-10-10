/**
 * `aifn-methods/information`: information theory applied: channel capacity and rate–distortion, source and
 * error-correcting codes, and KL projection.
 *
 * - `information/channels`: the capacity $C = \max_p I(X; Y)$ of a discrete memoryless channel and the
 *   rate–distortion function $R(D)$ of a discrete source, by the Blahut–Arimoto algorithms (traceable, with bounds).
 * - `information/coding`: source codes (Huffman, binary or $D$-ary and step by step; Shannon–Fano; Shannon;
 *   canonical; arithmetic-coding intervals) with their expected lengths against the entropy, prefix encoding and
 *   decoding, and the distances and size bounds of error-correcting codes (Hamming, Singleton, Plotkin).
 * - `information/projection`: the moment and information projections of a univariate distribution onto the normal
 *   family, $\argmin_q \KL(p \,\Vert\, q)$ and $\argmin_q \KL(q \,\Vert\, p)$.
 *
 * The measures themselves (entropy, mutual information, divergences) are in `aifn-compute/probability/information`;
 * Cover and Thomas (2006), "Elements of Information Theory", covers the module's ground.
 */

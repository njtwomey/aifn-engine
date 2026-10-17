/**
 * Rule mining on knowledge graphs: closed Horn rules over binary relations scored as AMIE scores them, and link
 * prediction from the mined rules.
 *
 * Galárraga, Teflioudi, Hose and Suchanek (2013), "AMIE: Association Rule Mining under Incomplete Evidence in
 * Ontological Knowledge Bases", WWW, §§3 and 4; link prediction by maximum aggregation as Meilicke et al. (2019),
 * "Anytime Bottom-Up Rule Learning for Knowledge Graph Completion", IJCAI, §5. A rule has a head $r(x, y)$ and a body
 * of one atom, $s(x, y)$ or $s(y, x)$, or of two atoms joined by a fresh variable $z$: $s(x, z)$ or $s(z, x)$ with
 * $t(z, y)$ or $t(y, z)$. Every variable occurs twice, so the rule is closed. Of the pairs $(x, y)$ the body yields,
 * the support is the number that are facts of $r$; head coverage divides it by the number of facts of $r$; standard
 * confidence by the number of body pairs; and confidence under the partial completeness assumption (PCA confidence)
 * only by the body pairs whose $x$ has some fact of $r$, since for an $x$ with none the missing facts are unknown
 * rather than false.
 *
 * Differences from AMIE: the candidates are enumerated exhaustively rather than by AMIE's refinement operators, only
 * rules of at most three atoms are mined (AMIE's default length), and body pairs with $x = y$ are dropped. Entity
 * names may hold any character (pairs are keyed with a separator no name contains).
 */

/** A fact (subject, relation, object). */
export type Triple = readonly [string, string, string]

/** A body atom: a relation and its two arguments among the variables $x$, $y$, $z$. */
export interface RuleAtom {
  /** The relation. */
  readonly relation: string
  /** Its arguments, in order. */
  readonly args: readonly ['x' | 'y' | 'z', 'x' | 'y' | 'z']
}

/** A closed rule with its scores and the pairs it yields. */
export interface MinedRule {
  /** The head relation; the head is $r(x, y)$. */
  readonly head: string
  /** The body atoms. */
  readonly body: readonly RuleAtom[]
  /** The rule as text, such as `livesIn(x, y) ⇐ marriedTo(x, z), livesIn(z, y)`. */
  readonly text: string
  /** Body pairs that are facts of the head relation. */
  readonly support: number
  /** Facts of the head relation. */
  readonly headSize: number
  /** Support over the facts of the head relation. */
  readonly headCoverage: number
  /** Distinct body pairs $(x, y)$ with $x \ne y$. */
  readonly bodySize: number
  /** Standard confidence: support over the body pairs. */
  readonly confidence: number
  /** Body pairs whose $x$ has at least one fact of the head relation. */
  readonly pcaBodySize: number
  /** PCA confidence: support over `pcaBodySize`. */
  readonly pcaConfidence: number
  /** Body pairs that are facts of the head relation. */
  readonly supporting: readonly (readonly [string, string])[]
  /** Body pairs that are not: the rule's predictions. */
  readonly predictions: readonly (readonly [string, string])[]
}

/** Options of `mineClosedRules`. */
export interface MiningOptions {
  /** Least support (default 2). */
  minSupport?: number
  /** Least head coverage (default 0.01, AMIE's default). */
  minHeadCoverage?: number
  /** Least PCA confidence (default 0.1, AMIE's default). */
  minPcaConfidence?: number
}

/** The separator of a pair's key; no entity name holds it. */
const SEP = '\u0000'

/**
 * The key of a pair.
 *
 * @param a The first entity.
 * @param b The second entity.
 * @returns The key.
 */
const pairKey = (a: string, b: string) => `${a}${SEP}${b}`

/**
 * The pair of a key.
 *
 * @param k The key, as `pairKey` makes it.
 * @returns The two entities.
 */
const splitKey = (k: string) => k.split(SEP) as [string, string]

/**
 * A body atom as text, such as `livesIn(z, y)`.
 *
 * @param a The atom.
 * @returns The text.
 */
const atomText = (a: RuleAtom) => `${a.relation}(${a.args[0]}, ${a.args[1]})`

/**
 * Every closed rule of at most three atoms whose scores pass the thresholds, best first: by PCA confidence, then
 * support, then text. Repeated triples count once.
 *
 * @param triples The knowledge graph.
 * @param options The thresholds (see `MiningOptions`).
 * @returns The rules with their scores, supporting pairs and predictions.
 *
 * @example Mine a tiny graph
 * const kg = [
 *   ['ann', 'marriedTo', 'ben'], ['ben', 'marriedTo', 'ann'], ['cal', 'marriedTo', 'dia'],
 *   ['ann', 'livesIn', 'paris'], ['ben', 'livesIn', 'paris'], ['cal', 'livesIn', 'rome'],
 * ]
 * for (const r of mineClosedRules(kg, { minSupport: 1 })) print(r.text, r.support, r.pcaConfidence)
 */
export function mineClosedRules(triples: readonly Triple[], options: MiningOptions = {}): MinedRule[] {
  const minSupport = options.minSupport ?? 2
  const minHc = options.minHeadCoverage ?? 0.01
  const minPca = options.minPcaConfidence ?? 0.1
  const relations = [...new Set(triples.map((t) => t[1]))]
  const facts = new Map<string, Set<string>>()
  const out = new Map<string, Map<string, string[]>>()
  const inn = new Map<string, Map<string, string[]>>()
  for (const r of relations) {
    facts.set(r, new Set())
    out.set(r, new Map())
    inn.set(r, new Map())
  }
  for (const [s, r, o] of triples) {
    const key = pairKey(s, o)
    if (facts.get(r)!.has(key)) continue
    facts.get(r)!.add(key)
    const om = out.get(r)!
    om.set(s, [...(om.get(s) ?? []), o])
    const im = inn.get(r)!
    im.set(o, [...(im.get(o) ?? []), s])
  }
  const pairs = (r: string): [string, string][] => [...facts.get(r)!].map(splitKey)

  // The objects of s(from, ·) (forward) or the subjects of s(·, from) (backward).
  const step = (r: string, from: string, forward: boolean): string[] =>
    (forward ? out.get(r)!.get(from) : inn.get(r)!.get(from)) ?? []

  const bodies: RuleAtom[][] = []
  for (const s of relations) {
    bodies.push([{ relation: s, args: ['x', 'y'] }])
    bodies.push([{ relation: s, args: ['y', 'x'] }])
    for (const t of relations)
      for (const a of [true, false])
        for (const b of [true, false])
          bodies.push([
            { relation: s, args: a ? ['x', 'z'] : ['z', 'x'] },
            { relation: t, args: b ? ['z', 'y'] : ['y', 'z'] },
          ])
  }

  const bodyPairs = (body: RuleAtom[]): Set<string> => {
    const res = new Set<string>()
    if (body.length === 1) {
      for (const [a, b] of pairs(body[0].relation)) res.add(body[0].args[0] === 'x' ? pairKey(a, b) : pairKey(b, a))
      return res
    }
    const [first, second] = body
    const xs = new Set(pairs(first.relation).map(([a, b]) => (first.args[0] === 'x' ? a : b)))
    for (const x of xs)
      for (const z of step(first.relation, x, first.args[0] === 'x'))
        for (const y of step(second.relation, z, second.args[0] === 'z')) res.add(pairKey(x, y))
    return res
  }

  const rules: MinedRule[] = []
  for (const head of relations) {
    const headFacts = facts.get(head)!
    const subjects = new Set(pairs(head).map(([a]) => a))
    for (const body of bodies) {
      // Skip the trivial rule r(x, y) ⇐ r(x, y).
      if (body.length === 1 && body[0].relation === head && body[0].args[0] === 'x') continue
      const bp = [...bodyPairs(body)].filter((k) => {
        const [x, y] = splitKey(k)
        return x !== y
      })
      if (bp.length === 0) continue
      const supporting = bp.filter((k) => headFacts.has(k))
      const support = supporting.length
      if (support < minSupport) continue
      const headCoverage = support / headFacts.size
      if (headCoverage < minHc) continue
      const pcaBody = bp.filter((k) => subjects.has(splitKey(k)[0])).length
      const pcaConfidence = support / pcaBody
      if (pcaConfidence < minPca) continue
      rules.push({
        head,
        body,
        text: `${head}(x, y) ⇐ ${body.map(atomText).join(', ')}`,
        support,
        headSize: headFacts.size,
        headCoverage,
        bodySize: bp.length,
        confidence: support / bp.length,
        pcaBodySize: pcaBody,
        pcaConfidence,
        supporting: supporting.map(splitKey),
        predictions: bp.filter((k) => !headFacts.has(k)).map(splitKey),
      })
    }
  }
  rules.sort((a, b) => b.pcaConfidence - a.pcaConfidence || b.support - a.support || a.text.localeCompare(b.text))
  return rules
}

/** A predicted fact with its score and the rule that gave it. */
export interface LinkPrediction {
  /** The subject. */
  readonly subject: string
  /** The relation. */
  readonly relation: string
  /** The object. */
  readonly object: string
  /** The highest PCA confidence among the rules that predict it (maximum aggregation). */
  readonly score: number
  /** The text of the rule with that confidence. */
  readonly rule: string
  /** The texts of every rule that predicts it. */
  readonly rules: readonly string[]
}

/**
 * Facts the rules predict that are not in the graph, each scored by the best rule predicting it (AnyBURL's maximum
 * aggregation), best first: by score, then subject, then object.
 *
 * @param rules Mined rules, as `mineClosedRules` returns them.
 * @returns The predictions.
 *
 * @example Predict a missing home
 * const kg = [
 *   ['ann', 'marriedTo', 'ben'], ['ben', 'marriedTo', 'ann'], ['cal', 'marriedTo', 'dia'], ['dia', 'marriedTo', 'cal'],
 *   ['ann', 'livesIn', 'paris'], ['ben', 'livesIn', 'paris'], ['cal', 'livesIn', 'rome'],
 * ]
 * const best = predictLinks(mineClosedRules(kg, { minSupport: 1 }))[0]
 * print(best.subject, best.relation, best.object, 'score', best.score, 'by', best.rule)
 */
export function predictLinks(rules: readonly MinedRule[]): LinkPrediction[] {
  const byFact = new Map<string, { s: string; r: string; o: string; best: number; rule: string; rules: string[] }>()
  for (const rule of rules)
    for (const [s, o] of rule.predictions) {
      const key = `${s}${SEP}${rule.head}${SEP}${o}`
      const cur = byFact.get(key)
      if (!cur) byFact.set(key, { s, r: rule.head, o, best: rule.pcaConfidence, rule: rule.text, rules: [rule.text] })
      else {
        cur.rules.push(rule.text)
        if (rule.pcaConfidence > cur.best) {
          cur.best = rule.pcaConfidence
          cur.rule = rule.text
        }
      }
    }
  return [...byFact.values()]
    .map((p) => ({ subject: p.s, relation: p.r, object: p.o, score: p.best, rule: p.rule, rules: p.rules }))
    .sort((a, b) => b.score - a.score || a.subject.localeCompare(b.subject) || a.object.localeCompare(b.object))
}

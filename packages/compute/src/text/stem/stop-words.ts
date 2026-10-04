/**
 * English stop-word lists, kept as published so that results match the libraries that use them:
 *
 * - `nltk`: NLTK's English list (198 words in NLTK 3.10), derived from the Snowball stop list, with clitic
 *   fragments ("don", "t") left by splitting contractions;
 * - `scikitLearn`: scikit-learn's `ENGLISH_STOP_WORDS` (318 words), from the Glasgow Information Retrieval Group.
 */

const split = (s: string): readonly string[] => Object.freeze(s.split(' '))

/** The stop-word lists by name. */
export const STOP_WORDS = Object.freeze({
  nltk: split(
    "a about above after again against ain all am an and any are aren aren't as at be because been before " +
      "being below between both but by can couldn couldn't d did didn didn't do does doesn doesn't doing don " +
      "don't down during each few for from further had hadn hadn't has hasn hasn't have haven haven't having he " +
      "he'd he'll her here hers herself he's him himself his how i i'd if i'll i'm in into is isn isn't it it'd " +
      "it'll it's its itself i've just ll m ma me mightn mightn't more most mustn mustn't my myself needn " +
      "needn't no nor not now o of off on once only or other our ours ourselves out over own re s same shan " +
      "shan't she she'd she'll she's should shouldn shouldn't should've so some such t than that that'll the " +
      "their theirs them themselves then there these they they'd they'll they're they've this those through to " +
      "too under until up ve very was wasn wasn't we we'd we'll we're were weren weren't we've what when where " +
      "which while who whom why will with won won't wouldn wouldn't y you you'd you'll your you're yours " +
      "yourself yourselves you've",
  ),
  scikitLearn: split(
    'a about above across after afterwards again against all almost alone along already also although always ' +
      'am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as ' +
      'at back be became because become becomes becoming been before beforehand behind being below beside ' +
      'besides between beyond bill both bottom but by call can cannot cant co con could couldnt cry de describe ' +
      'detail do done down due during each eg eight either eleven else elsewhere empty enough etc even ever ' +
      'every everyone everything everywhere except few fifteen fifty fill find fire first five for former ' +
      'formerly forty found four from front full further get give go had has hasnt have he hence her here ' +
      'hereafter hereby herein hereupon hers herself him himself his how however hundred i ie if in inc indeed ' +
      'interest into is it its itself keep last latter latterly least less ltd made many may me meanwhile might ' +
      'mill mine more moreover most mostly move much must my myself name namely neither never nevertheless next ' +
      'nine no nobody none noone nor not nothing now nowhere of off often on once one only onto or other others ' +
      'otherwise our ours ourselves out over own part per perhaps please put rather re same see seem seemed ' +
      'seeming seems serious several she should show side since sincere six sixty so some somehow someone ' +
      'something sometime sometimes somewhere still such system take ten than that the their them themselves ' +
      'then thence there thereafter thereby therefore therein thereupon these they thick thin third this those ' +
      'though three through throughout thru thus to together too top toward towards twelve twenty two un under ' +
      'until up upon us very via was we well were what whatever when whence whenever where whereafter whereas ' +
      'whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will ' +
      'with within without would yet you your yours yourself yourselves',
  ),
})

/** A named stop-word list. */
export type StopList = keyof typeof STOP_WORDS

/**
 * The tokens not in the stop list, compared after lower-casing. `list` is a named list (default `nltk`) or any list of
 * words.
 */
export function removeStopWords(tokens: readonly string[], list: StopList | readonly string[] = 'nltk'): string[] {
  const stop = new Set(typeof list === 'string' ? STOP_WORDS[list] : list)
  return tokens.filter((t) => !stop.has(t.toLowerCase()))
}

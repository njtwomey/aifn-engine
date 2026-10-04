/**
 * Prolog highlighting for `CodeEditor` (a CodeMirror stream language): comments, quoted atoms, numbers, variables,
 * functors (a name before `(`), control (`:-`, `?-`, `!`, `is`, `\+`) and operators.
 */
import { StreamLanguage, type StringStream } from '@codemirror/language'

type State = { comment: boolean }

const CONTROL = new Set([':-', '?-', '!', 'is', '\\+', '->', ';', 'not', 'findall', 'fail', 'true'])

export const prologLanguage = StreamLanguage.define<State>({
  name: 'prolog',
  startState: () => ({ comment: false }),
  token(stream: StringStream, state: State): string | null {
    if (state.comment) {
      if (stream.skipTo('*/')) {
        stream.next()
        stream.next()
        state.comment = false
      } else stream.skipToEnd()
      return 'comment'
    }
    if (stream.eatSpace()) return null
    if (stream.match('%')) {
      stream.skipToEnd()
      return 'comment'
    }
    if (stream.match('/*')) {
      state.comment = true
      return 'comment'
    }
    if (stream.match(/^'(?:[^'\\]|\\.|'')*'?/)) return 'string'
    if (stream.match(/^"(?:[^"\\]|\\.)*"?/)) return 'string'
    if (stream.match(/^\d+(\.\d+)?/)) return 'number'
    if (stream.match(/^[A-Z_][A-Za-z0-9_]*/)) return 'typeName'
    const word = stream.match(/^[a-z][A-Za-z0-9_]*/) as RegExpMatchArray | null
    if (word) {
      if (CONTROL.has(word[0])) return 'keyword'
      return stream.peek() === '(' ? 'propertyName' : 'atom'
    }
    const symbol = stream.match(/^[+\-*/\\^<>=~:.?@#&$]+/) as RegExpMatchArray | null
    if (symbol) return CONTROL.has(symbol[0]) ? 'keyword' : 'operator'
    const c = stream.next()
    if (c === '!') return 'keyword'
    return c && '()[]|,'.includes(c) ? 'punctuation' : null
  },
  languageData: { commentTokens: { line: '%', block: { open: '/*', close: '*/' } } },
})

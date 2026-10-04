/**
 * `make shots` and `make thumbs`: screenshot the examples' pages and their figures in headless Chrome, so the pixels
 * can be checked, and rebuild the gallery thumbnails.
 *
 *   node scripts/screenshot.ts                                    # every page, light and dark
 *   node scripts/screenshot.ts --only lines controls/sliders
 *   node scripts/screenshot.ts --only 'lines/line-chart#some-figure' --theme dark --mobile
 *   node scripts/screenshot.ts --only lines/x --act 'some-figure:drag 0.2 0.5 0.7 0.4'
 *   node scripts/screenshot.ts --url http://localhost:5192     # reuse a running dev server (make examples)
 *
 * --thumbs also saves each page's first chart area as examples/public/thumbs/<path>-<theme>.png.
 * Options: --only <section | section/slug | section/slug#figure-id> ...; --theme light|dark|both (default both);
 * --width 1440 --height 900 --dpr 2; --mobile (adds a 390 × 844 pass); --port 5191 (first port tried for its own Vite);
 * --timeout 15000 (ms to wait for a page to settle); --act '<figure-id>:drag x0 y0 x1 y1' or '<figure-id>:click x y'
 * (repeatable; coordinates are fractions of the figure's chart area, applied on every page that has that figure), or
 * '<figure-id>:press <button aria-label> [times]' (e.g. 'press First', 'press Next 20' or 'press Play' on a Player),
 * or '<figure-id>:select <option label>' (opens each dropdown of the figure until one offers that option, and picks it),
 * or '<figure-id>:hover x y' (moves the pointer there and leaves it: the figure's and the page's captures show the hover
 * state, e.g. a tooltip or a hover guide), or '<figure-id>:click node <id | label>' / 'hover node <id | label>' (a Diagram
 * or TreeView node of the figure, by node id (a TreeView's node index), else its label exactly, else the first label containing the text; e.g.
 * 'click node 7' pins a tree node). The last hover act holds for the captures; a later click or drag ends it;
 * --restart-every 40 (pages per Chrome; a hung protocol call also restarts it and retries the page once);
 * --profile (each --act drag becomes 60 pointer moves at display rate, measured: input-to-paint, frame times, dropped
 * frames, long tasks, script/layout/style time, ECharts setOption calls and the heaviest functions by CPU self time,
 * written to profile.json with a one-line summary in profile.txt).
 *
 * Output, overwritten per page: .scratch/shots/<section>/<slug>/<theme>[-mobile]/{page.png,<figure-id>.png,
 * console.txt}, and .scratch/shots/index.md listing every file of the run with its page URL. A page fails on
 * console errors, uncaught exceptions, a figure that failed to render, a KaTeX error, or a chart with zero size; the exit
 * code is 1 if any page failed.
 *
 * How it works: one headless Chrome and one tab, driven over the DevTools protocol with Node's global WebSocket (no
 * dependency). Each theme and viewport loads the app once (the theme is written to localStorage before the app starts);
 * pages are then opened as the sidebar opens them (pushState + popstate). A page has settled when fonts are loaded,
 * every chart carries `data-chart-ready` (set by `EChart` on ECharts' 'finished' event, cleared on each redraw) and the
 * DOM and chart sizes are unchanged over consecutive frames, and no element is `aria-busy` (a figure waiting on a worker). Chrome and Vite are always torn down, on error and Ctrl-C
 * too; Chrome is killed with SIGKILL.
 */
import { spawn, type ChildProcess } from 'node:child_process'
import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import type { ViteDevServer } from 'vite'

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const REPO = path.resolve(import.meta.dirname, '..')
const THEME_KEY = 'aifn-lab:theme'

// ---------------------------------------------------------------------------------------------------------------------
// Options

/** `node`: a Diagram or TreeView node of the figure (by node id or label) instead of a point given as fractions. */
type Act = {
  figure: string
  kind: 'click' | 'drag' | 'press' | 'select' | 'hover'
  at: number[]
  button?: string
  node?: string
}
type Theme = 'light' | 'dark'

const opts = {
  only: [] as string[],
  themes: ['light', 'dark'] as Theme[],
  width: 1440,
  height: 900,
  dpr: 2,
  mobile: false,
  port: 5191,
  url: undefined as string | undefined,
  timeout: 15_000,
  acts: [] as Act[],
  profile: false,
  sliders: true,
  /** Pages per Chrome: a long run restarts Chrome this often (a single Chrome stalled after ~110 pages). */
  restartEvery: 40,
  /** `--thumbs` (examples): also save each page's first chart area, shrunk, as its gallery thumbnail. */
  thumbs: false,
}

function parseAct(spec: string): Act {
  // '<figure-id>:press <button label> [times]': clicks a button by its aria-label (e.g. a Player's Next or Play).
  const press = /^([^:]+):\s*press\s+(.+?)(?:\s+(\d+))?$/.exec(spec.trim())
  if (press) return { figure: press[1].trim(), kind: 'press', at: [Number(press[3] ?? 1)], button: press[2].trim() }
  // '<figure-id>:select <option label>': picks that option in whichever of the figure's dropdowns offers it.
  const choose = /^([^:]+):\s*select\s+(.+)$/.exec(spec.trim())
  if (choose) return { figure: choose[1].trim(), kind: 'select', at: [], button: choose[2].trim() }
  // '<figure-id>:click node <id | label>' or 'hover node …': a Diagram or TreeView node.
  const node = /^([^:]+):\s*(click|hover)\s+node\s+(.+)$/.exec(spec.trim())
  if (node) return { figure: node[1].trim(), kind: node[2] as Act['kind'], at: [], node: node[3].trim() }
  const m = /^([^:]+):\s*(click|drag|hover)\s+(.+)$/.exec(spec.trim())
  const at = m
    ? m[3]
        .trim()
        .split(/[\s,]+/)
        .map(Number)
    : []
  const want = m?.[2] === 'drag' ? 4 : 2
  if (!m || at.length !== want || at.some((v) => !Number.isFinite(v)))
    throw new Error(
      `bad --act '${spec}': use '<figure-id>:drag x0 y0 x1 y1', '<figure-id>:click x y', '<figure-id>:hover x y' ` +
        `(fractions), '<figure-id>:click node <id|label>' or '<figure-id>:hover node <id|label>'`,
    )
  return { figure: m[1].trim(), kind: m[2] as Act['kind'], at }
}

const argv = process.argv.slice(2)
for (let i = 0; i < argv.length; i++) {
  const a = argv[i]
  const next = () => {
    const v = argv[++i]
    if (v === undefined) throw new Error(`${a} needs a value`)
    return v
  }
  if (a === '--only') while (argv[i + 1] !== undefined && !argv[i + 1].startsWith('--')) opts.only.push(argv[++i])
  else if (a === '--theme') {
    const t = next()
    if (t !== 'light' && t !== 'dark' && t !== 'both') throw new Error('--theme is light, dark or both')
    opts.themes = t === 'both' ? ['light', 'dark'] : [t]
  } else if (a === '--width') opts.width = Number(next())
  else if (a === '--height') opts.height = Number(next())
  else if (a === '--dpr') opts.dpr = Number(next())
  else if (a === '--mobile') opts.mobile = true
  else if (a === '--port') opts.port = Number(next())
  else if (a === '--url') opts.url = next().replace(/\/+$/, '')
  else if (a === '--timeout') opts.timeout = Number(next())
  else if (a === '--act') opts.acts.push(parseAct(next()))
  else if (a === '--profile') opts.profile = true
  else if (a === '--sliders') opts.sliders = true
  else if (a === '--no-sliders') opts.sliders = false
  else if (a === '--restart-every') opts.restartEvery = Math.max(1, Number(next()))
  else if (a === '--thumbs') opts.thumbs = true
  else if (a === '--help' || a === '-h') {
    console.log(
      'usage: node scripts/screenshot.ts [--only <section|section/slug|path#figure> ...] [--theme light|dark|both]',
    )
    console.log(
      '  [--width 1440] [--height 900] [--dpr 2] [--mobile] [--url http://...] [--port 5191] [--timeout ms] [--profile]',
    )
    console.log(
      "  [--act '<fig>:drag x0 y0 x1 y1' | '<fig>:click x y' | '<fig>:hover x y' | '<fig>:click node <id|label>'",
    )
    console.log("         | '<fig>:hover node <id|label>' | '<fig>:press <button> [n]' | '<fig>:select <option>'] ...")
    console.log(
      '  hover acts leave the pointer in place for the captures; x y are fractions of the chart area (see the file header)',
    )
    process.exit(0)
  } else opts.only.push(a)
}

// ---------------------------------------------------------------------------------------------------------------------
// Teardown: whatever happens, Chrome and Vite go.

let chrome: ChildProcess | undefined
let vite: ViteDevServer | undefined
let profile: string | undefined

async function teardown() {
  if (chrome && chrome.exitCode === null) chrome.kill('SIGKILL')
  chrome = undefined
  const server = vite
  vite = undefined
  if (server) await Promise.race([server.close(), new Promise((r) => setTimeout(r, 3000))])
  if (profile) rmSync(profile, { recursive: true, force: true })
  profile = undefined
}
process.on('exit', () => {
  if (chrome && chrome.exitCode === null) chrome.kill('SIGKILL')
})
for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const)
  process.on(signal, () => {
    console.error(`\n${signal}: stopping Chrome and Vite`)
    void teardown().finally(() => process.exit(130))
  })

// ---------------------------------------------------------------------------------------------------------------------
// A minimal DevTools protocol client over one browser WebSocket, with flattened page sessions.

type Params = Record<string, unknown>
type StackFrame = { functionName: string; url: string; lineNumber: number }
type Listener = (method: string, params: Params) => void

class Cdp {
  private id = 0
  private pending = new Map<number, { resolve: (v: Params) => void; reject: (e: Error) => void; method: string }>()
  private listeners = new Set<Listener>()
  private ws: WebSocket
  session?: string

  private constructor(ws: WebSocket) {
    this.ws = ws
    ws.addEventListener('message', (event) => {
      const msg = JSON.parse(String(event.data)) as {
        id?: number
        result?: Params
        error?: { message: string }
        method?: string
        params?: Params
      }
      if (msg.id !== undefined) {
        const p = this.pending.get(msg.id)
        this.pending.delete(msg.id)
        if (msg.error) p?.reject(new Error(`${p.method}: ${msg.error.message}`))
        else p?.resolve(msg.result ?? {})
      } else if (msg.method) for (const l of this.listeners) l(msg.method, msg.params ?? {})
    })
    ws.addEventListener('close', () => {
      for (const p of this.pending.values()) p.reject(new Error(`${p.method}: the browser connection closed`))
      this.pending.clear()
    })
  }

  static connect(url: string): Promise<Cdp> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url)
      ws.addEventListener('open', () => resolve(new Cdp(ws)))
      ws.addEventListener('error', () => reject(new Error(`cannot connect to ${url}`)))
    })
  }

  /** A command to the page session (or the browser, before one is attached). */
  send<T = Params>(method: string, params: Params = {}, timeout = 60_000): Promise<T> {
    const id = ++this.id
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`${method}: no reply after ${timeout} ms`))
      }, timeout)
      this.pending.set(id, {
        method,
        resolve: (v) => (clearTimeout(timer), resolve(v as T)),
        reject: (e) => (clearTimeout(timer), reject(e)),
      })
      this.ws.send(JSON.stringify({ id, method, params, ...(this.session ? { sessionId: this.session } : {}) }))
    })
  }

  on(l: Listener) {
    this.listeners.add(l)
    return () => this.listeners.delete(l)
  }

  /** Evaluates an expression in the page and returns its (awaited) value. */
  async eval<T>(expression: string, timeout = 60_000): Promise<T> {
    const r = await this.send<{ result: { value?: T }; exceptionDetails?: { text: string; exception?: Params } }>(
      'Runtime.evaluate',
      { expression, awaitPromise: true, returnByValue: true },
      timeout,
    )
    if (r.exceptionDetails)
      throw new Error(`in page: ${String(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text)}`)
    return r.result.value as T
  }

  close() {
    this.ws.close()
  }
}

function launchChrome(): Promise<string> {
  profile = mkdtempSync(path.join(tmpdir(), 'shots-'))
  const proc = spawn(
    CHROME,
    [
      '--headless=new',
      '--remote-debugging-port=0',
      `--user-data-dir=${profile}`,
      '--hide-scrollbars',
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-extensions',
      '--mute-audio',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows',
      'about:blank',
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  )
  chrome = proc
  return new Promise((resolve, reject) => {
    let err = ''
    const timer = setTimeout(() => reject(new Error('Chrome did not open its DevTools port within 20 s')), 20_000)
    proc.stderr!.on('data', (chunk: Buffer) => {
      err += chunk.toString()
      const m = /DevTools listening on (ws:\/\/\S+)/.exec(err)
      if (m) {
        clearTimeout(timer)
        resolve(m[1])
      }
    })
    proc.on('exit', (code) => {
      clearTimeout(timer)
      reject(new Error(`Chrome exited (${code}) before it was ready`))
    })
  })
}

// ---------------------------------------------------------------------------------------------------------------------
// In-page helpers (run with Runtime.evaluate).

/** Resolves once fonts are loaded, every chart is ready and the DOM and chart sizes hold still for three checks. */
const SETTLE = (timeout: number) => `(async () => {
  const frame = () => new Promise((r) => requestAnimationFrame(() => r()))
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
  const t0 = performance.now()
  await document.fonts.ready
  let last = '', stable = 0, pending = 0
  while (performance.now() - t0 < ${timeout}) {
    await frame(); await frame()
    const charts = [...document.querySelectorAll('[_echarts_instance_]')]
    // A figure waiting on a worker marks itself aria-busy; it counts as pending until the answer lands.
    pending = charts.filter((c) => !c.hasAttribute('data-chart-ready')).length +
      document.querySelectorAll('[aria-busy="true"]').length
    const main = document.querySelector('main')
    const sig = document.getElementsByTagName('*').length + '|' + (main ? main.scrollHeight : 0) + '|' +
      charts.map((c) => c.clientWidth + 'x' + c.clientHeight).join(',')
    if (pending === 0 && sig === last) { if (++stable >= 2) return { ok: true, ms: performance.now() - t0 } }
    else stable = 0
    last = sig
    await sleep(60)
  }
  // Which figures hold the charts that never settled, so the warning names them.
  const where = [...document.querySelectorAll('[_echarts_instance_]:not([data-chart-ready])')].map((c) => {
    const f = c.closest('section[data-figure-id]')
    return f ? f.getAttribute('data-figure-id') : '?'
  })
  return { ok: false, pending, where: [...new Set(where)], ms: performance.now() - t0 }
})()`

type Inspect = {
  figures: { id: string; zero: number; charts: number }[]
  failed: number
  katex: number
  scrollHeight: number
  title: string
}

const INSPECT = `(() => {
  const figures = [...document.querySelectorAll('main section[data-figure-id]')].map((f) => {
    const charts = [...f.querySelectorAll('[_echarts_instance_]')]
    const zero = charts.filter((c) => {
      const r = c.getBoundingClientRect()
      const surface = c.querySelector('canvas, svg')
      const s = surface ? surface.getBoundingClientRect() : { width: 0, height: 0 }
      return r.width < 1 || r.height < 1 || s.width < 1 || s.height < 1
    }).length
    return { id: f.getAttribute('data-figure-id'), zero, charts: charts.length }
  })
  const main = document.querySelector('main')
  return {
    figures,
    failed: [...document.querySelectorAll('main [role=alert]')].filter((e) => /failed to render/.test(e.textContent)).length,
    katex: document.querySelectorAll('main .katex-error').length,
    scrollHeight: main ? main.getBoundingClientRect().top + main.scrollHeight : document.documentElement.scrollHeight,
    title: document.querySelector('main h1')?.textContent ?? '',
  }
})()`

type ProfileRaw = {
  frames: number[]
  moves: { t: number; paint: number }[]
  longtasks: { start: number; duration: number }[]
  stats: { setOption: number; full: number }
}
type Profile = { summary: string } & Record<string, unknown>

type CpuProfile = {
  nodes: { id: number; callFrame: { functionName: string; url: string; lineNumber: number } }[]
  samples: number[]
  timeDeltas: number[]
}

/** The `top` functions by sampled self time (ms) in a CPU profile, with their source location. */
function selfTime(cpu: CpuProfile, top: number) {
  const byNode = new Map(cpu.nodes.map((n) => [n.id, n.callFrame]))
  const total = new Map<string, number>()
  cpu.samples.forEach((id, i) => {
    const f = byNode.get(id)
    if (!f) return
    const url = f.url.replace(/^https?:\/\/[^/]+/, '').replace(/\?.*$/, '')
    const key = `${f.functionName || '(anonymous)'} ${url ? `${url}:${f.lineNumber + 1}` : ''}`.trim()
    total.set(key, (total.get(key) ?? 0) + (cpu.timeDeltas[i + 1] ?? 0) / 1000)
  })
  return [...total]
    .filter(([k]) => !/^\((idle|program)\)/.test(k))
    .sort((a, b) => b[1] - a[1])
    .slice(0, top)
    .map(([fn, ms]) => ({ fn, ms: Math.round(ms * 10) / 10 }))
}

const quantile = (xs: number[], q: number) => {
  if (!xs.length) return 0
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.min(s.length - 1, Math.floor(q * s.length))]
}

/** Starts the in-page recorders of `--profile`; PROFILE_STOP stops them and returns what they saw. */
const PROFILE_START = `(() => {
  const s = window.__labStats ?? { setOption: 0, full: 0 }
  const P = (window.__labProfile = { frames: [], moves: [], longtasks: [], stats0: { ...s }, stop: false })
  const loop = (t) => { P.frames.push(t); if (!P.stop) requestAnimationFrame(loop) }
  requestAnimationFrame(loop)
  try {
    P.obs = new PerformanceObserver((l) => { for (const e of l.getEntries()) P.longtasks.push({ start: e.startTime, duration: e.duration }) })
    P.obs.observe({ type: 'longtask' })
  } catch {}
  P.onMove = (e) => {
    if (!e.buttons) return
    const m = { t: e.timeStamp, paint: 0 }
    P.moves.push(m)
    // The frame after the move, then the first task after that frame: the move's effect has been painted.
    requestAnimationFrame(() => { const ch = new MessageChannel(); ch.port1.onmessage = () => { m.paint = performance.now() }; ch.port2.postMessage(0) })
  }
  addEventListener('pointermove', P.onMove, { capture: true })
})()`

const PROFILE_STOP = `(() => {
  const P = window.__labProfile
  P.stop = true
  removeEventListener('pointermove', P.onMove, { capture: true })
  P.obs?.disconnect()
  const s = window.__labStats ?? { setOption: 0, full: 0 }
  const t0 = P.moves.length ? P.moves[0].t : 0
  return {
    frames: P.frames.filter((t) => t >= t0),
    moves: P.moves,
    longtasks: P.longtasks,
    stats: { setOption: s.setOption - P.stats0.setOption, full: s.full - P.stats0.full },
  }
})()`

// ---------------------------------------------------------------------------------------------------------------------
// Selection

type Target = { path: string; figure?: string }

function select(all: readonly string[]): Target[] {
  if (opts.only.length === 0) return all.map((p) => ({ path: p }))
  const out: Target[] = []
  for (const raw of opts.only) {
    const spec = raw.replace(/^\/+|\/+$/g, '')
    const [where, figure] = spec.split('#')
    const hits = all.filter((p) => p === where || p.startsWith(`${where}/`))
    if (hits.length === 0) throw new Error(`--only ${raw}: no page matches (pages look like module/specimen-slug)`)
    for (const p of hits) out.push({ path: p, figure: figure || undefined })
  }
  // One entry per page; a page named in full wins over a figure of it.
  const byPath = new Map<string, Target>()
  for (const t of out) {
    const prev = byPath.get(t.path)
    byPath.set(t.path, prev && (!prev.figure || !t.figure) ? { path: t.path } : t)
  }
  return [...byPath.values()]
}

// ---------------------------------------------------------------------------------------------------------------------
// Main

type Viewport = { width: number; height: number; mobile: boolean; suffix: string }

// The app being shot: the examples gallery.
const APP = { dir: path.join(REPO, 'examples'), start: '/', nav: 'Recipes', out: '.scratch/shots' }
const OUT = path.join(REPO, APP.out)
const THUMBS = path.join(APP.dir, 'public/thumbs')

const viewports: Viewport[] = [{ width: opts.width, height: opts.height, mobile: false, suffix: '' }]
if (opts.mobile) viewports.push({ width: 390, height: 844, mobile: true, suffix: '-mobile' })

const index: string[] = []
let failures = 0
let shots = 0
const t0 = performance.now()

async function run() {
  let base = opts.url
  if (!base) {
    const { createServer } = await import('vite')
    vite = await createServer({
      configFile: path.join(APP.dir, 'vite.config.ts'),
      // Its own port and no HMR: edits made elsewhere during the run must not reload the page mid-capture.
      server: { port: opts.port, strictPort: false, hmr: false, forwardConsole: false },
      logLevel: 'warn',
    })
    await vite.listen()
    base = (vite.resolvedUrls?.local[0] ?? `http://localhost:${opts.port}/`).replace(/\/+$/, '')
    console.log(`examples dev server → ${base}/`)
  } else {
    const up = await fetch(`${base}/`).then(
      (r) => r.ok,
      () => false,
    )
    if (!up) throw new Error(`nothing is serving ${base}/ (start it with make examples, or drop --url)`)
  }

  // One Chrome and one tab at a time; `openBrowser` replaces both (the run restarts Chrome every few dozen pages).
  let cdp!: Cdp
  const openBrowser = async () => {
    cdp?.close()
    if (chrome && chrome.exitCode === null) chrome.kill('SIGKILL')
    if (profile) rmSync(profile, { recursive: true, force: true })
    cdp = await Cdp.connect(await launchChrome())
    // Chrome's own first tab: a tab opened with Target.createTarget sits in the background, where frames are throttled.
    const { targetInfos } = await cdp.send<{ targetInfos: { targetId: string; type: string }[] }>('Target.getTargets')
    const first = targetInfos.find((t) => t.type === 'page')
    const targetId =
      first?.targetId ?? (await cdp.send<{ targetId: string }>('Target.createTarget', { url: 'about:blank' })).targetId
    const { sessionId } = await cdp.send<{ sessionId: string }>('Target.attachToTarget', { targetId, flatten: true })
    cdp.session = sessionId
    await cdp.send('Page.enable')
    await cdp.send('Runtime.enable')
    await cdp.send('Log.enable')
    if (opts.profile) await cdp.send('Performance.enable')
    cdp.on(onConsole)
  }

  // The console of the page being shot: errors and warnings, plus uncaught exceptions and failed requests.
  let log: { level: 'error' | 'warning' | 'info'; text: string }[] = []
  const onConsole: Listener = (method, p) => {
    if (method === 'Runtime.consoleAPICalled') {
      const type = p.type as string
      if (type !== 'error' && type !== 'warning' && type !== 'assert') return
      const args = (p.args as { value?: unknown; description?: string }[]).map((a) =>
        a.value !== undefined ? String(a.value) : (a.description ?? ''),
      )
      // Where it was logged from: the first few frames of the page's own code, not React's or ECharts'.
      const frames = ((p.stackTrace as { callFrames?: StackFrame[] } | undefined)?.callFrames ?? [])
        .filter((f) => !f.url.includes('/node_modules/'))
        .slice(0, 3)
        .map(
          (f) =>
            `    at ${f.functionName || '(anonymous)'} ${f.url.replace(/^https?:\/\/[^/]+/, '')}:${f.lineNumber + 1}`,
        )
      log.push({ level: type === 'warning' ? 'warning' : 'error', text: [args.join(' '), ...frames].join('\n') })
    } else if (method === 'Runtime.exceptionThrown') {
      const d = p.exceptionDetails as { text: string; exception?: { description?: string } }
      log.push({ level: 'error', text: `uncaught: ${d.exception?.description ?? d.text}` })
    } else if (method === 'Log.entryAdded') {
      const e = p.entry as { level: string; text: string; url?: string }
      // The dev server has no favicon; that 404 is not the page's.
      if ((e.level === 'error' || e.level === 'warning') && !e.url?.endsWith('/favicon.ico'))
        log.push({ level: e.level, text: `${e.text}${e.url ? ` (${e.url})` : ''}` })
    }
  }

  const loaded = () =>
    new Promise<void>((resolve) => {
      const off = cdp.on((m) => m === 'Page.loadEventFired' && (off(), resolve()))
    })

  const setViewport = (v: Viewport, height = v.height) =>
    cdp.send('Emulation.setDeviceMetricsOverride', {
      width: v.width,
      height,
      deviceScaleFactor: opts.dpr,
      mobile: v.mobile,
    })

  const settle = async (what: string) => {
    const r = await cdp.eval<{ ok: boolean; pending?: number; where?: string[]; ms: number }>(
      SETTLE(opts.timeout),
      opts.timeout + 10_000,
    )
    if (!r.ok)
      log.push({
        level: 'warning',
        text:
          `shots: ${what} did not settle in ${opts.timeout} ms (${r.pending} charts still animating` +
          `${r.where?.length ? `, in ${r.where.join(', ')}` : ''})`,
      })
    return r
  }

  const capture = async (file: string, clip?: { x: number; y: number; width: number; height: number }) => {
    const r = await cdp.send<{ data: string }>('Page.captureScreenshot', {
      format: 'png',
      ...(clip ? { clip: { ...clip, scale: 1 } } : {}),
    })
    writeFileSync(file, Buffer.from(r.data, 'base64'))
    shots++
  }

  const mouse = (type: string, x: number, y: number, buttons = 0) =>
    cdp.send('Input.dispatchMouseEvent', {
      type,
      x,
      y,
      button: type === 'mouseMoved' && !buttons ? 'none' : 'left',
      buttons,
      clickCount: type === 'mouseMoved' ? 0 : 1,
    })

  /**
   * `--profile`: a drag of 60 pointer moves at display rate (one per 16.7 ms), measured in the page (frame times from a
   * requestAnimationFrame loop, input-to-paint per move as event time to the task after the next frame, long tasks) and
   * by the DevTools Performance domain (script, layout and style time), with ECharts `setOption` calls counted by
   * `window.__labStats` (dev-only, in viz/EChart.tsx).
   */
  const profileDrag = async (act: Act, from: readonly [number, number], to: readonly [number, number]) => {
    const MOVES = 60
    const FRAME = 1000 / 60
    await mouse('mouseMoved', from[0], from[1])
    await mouse('mousePressed', from[0], from[1], 1)
    await cdp.eval(PROFILE_START)
    const metrics = async () =>
      Object.fromEntries(
        (await cdp.send<{ metrics: { name: string; value: number }[] }>('Performance.getMetrics')).metrics.map((m) => [
          m.name,
          m.value,
        ]),
      ) as Record<string, number>
    await cdp.send('Profiler.enable')
    await cdp.send('Profiler.setSamplingInterval', { interval: 200 })
    await cdp.send('Profiler.start')
    const before = await metrics()
    const t0 = performance.now()
    for (let i = 1; i <= MOVES; i++) {
      await mouse('mouseMoved', from[0] + ((to[0] - from[0]) * i) / MOVES, from[1] + ((to[1] - from[1]) * i) / MOVES, 1)
      const wait = t0 + i * FRAME - performance.now()
      if (wait > 0) await new Promise((r) => setTimeout(r, wait))
    }
    const wall = performance.now() - t0
    // Let the last move paint before stopping the recorders, then release (the release redraw is not measured).
    await new Promise((r) => setTimeout(r, 100))
    const raw = await cdp.eval<ProfileRaw>(PROFILE_STOP)
    const after = await metrics()
    const { profile: cpu } = await cdp.send<{ profile: CpuProfile }>('Profiler.stop')
    await cdp.send('Profiler.disable')
    await mouse('mouseReleased', to[0], to[1])
    const d = (name: string) => Math.round(((after[name] ?? 0) - (before[name] ?? 0)) * 1000 * 10) / 10
    const latency = raw.moves.filter((m) => m.paint > 0).map((m) => m.paint - m.t)
    const intervals = raw.frames.slice(1).map((t, i) => t - raw.frames[i])
    const dropped = intervals.reduce((n, dt) => n + Math.max(0, Math.round(dt / FRAME) - 1), 0)
    const calls = raw.stats.setOption
    const r1 = (x: number) => Math.round(x * 10) / 10
    const result = {
      figure: act.figure,
      path: `${act.at.join(' ')} (fractions of the chart area)`,
      movesSent: MOVES,
      movesSeen: raw.moves.length,
      wallMs: r1(wall),
      inputToPaintMs: {
        median: r1(quantile(latency, 0.5)),
        p95: r1(quantile(latency, 0.95)),
        max: r1(Math.max(0, ...latency)),
      },
      frames: {
        count: raw.frames.length,
        medianMs: r1(quantile(intervals, 0.5)),
        p95Ms: r1(quantile(intervals, 0.95)),
        maxMs: r1(Math.max(0, ...intervals)),
        dropped,
      },
      longTasks: {
        count: raw.longtasks.length,
        totalMs: r1(raw.longtasks.reduce((a, t) => a + t.duration, 0)),
        maxMs: r1(Math.max(0, ...raw.longtasks.map((t) => t.duration))),
      },
      scriptingMs: d('ScriptDuration'),
      layoutMs: d('LayoutDuration'),
      styleMs: d('RecalcStyleDuration'),
      taskMs: d('TaskDuration'),
      setOption: { calls, full: raw.stats.full, perMove: r1(calls / Math.max(1, raw.moves.length)) },
      // Where the time went: sampled self time by function, heaviest first.
      selfTime: selfTime(cpu, 15),
      summary: '',
    }
    result.summary =
      `profile ${act.figure}: input-to-paint median ${result.inputToPaintMs.median} ms, p95 ${result.inputToPaintMs.p95} ms; ` +
      `dropped frames ${dropped}/${Math.round(wall / FRAME)}; setOption ${result.setOption.perMove}/move ` +
      `(${calls} calls, ${raw.stats.full} full); long tasks ${result.longTasks.count} (${result.longTasks.totalMs} ms); ` +
      `scripting ${result.scriptingMs} ms over ${Math.round(wall)} ms`
    return result
  }

  /** Scrolls a figure to the top of the page's scroll area and returns its box (and its chart area's). */
  const reveal = (id: string) =>
    cdp.eval<{ x: number; y: number; width: number; height: number; area: DOMRect | null } | null>(`(async () => {
      const f = document.querySelector('main section[data-figure-id="${id.replace(/"/g, '\\"')}"]')
      if (!f) return null
      f.scrollIntoView({ block: 'start', behavior: 'instant' })
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
      const b = f.getBoundingClientRect()
      const sep = f.querySelector('[role=separator][aria-label="Resize figure"]')
      const a = sep ? sep.parentElement.getBoundingClientRect() : null
      return { x: b.x, y: b.y, width: b.width, height: b.height,
        area: a && { x: a.x, y: a.y, width: a.width, height: a.height } }
    })()`)

  /**
   * Where an act points, in window pixels: a Diagram/TreeView node's centre (`act.node`), or fractions of the figure's
   * chart area. `scroll` brings the figure into view first; without it the page is measured where it is (a capture of
   * the full page, the window grown to the content).
   */
  const locate = async (act: Act, scroll: boolean): Promise<readonly [number, number] | null> => {
    const fig = `main section[data-figure-id="${act.figure.replace(/"/g, '\\"')}"]`
    if (act.node !== undefined)
      return cdp.eval<[number, number] | null>(`(async () => {
        const f = document.querySelector(${JSON.stringify(fig)})
        if (!f) return null
        if (${scroll}) {
          f.scrollIntoView({ block: 'start', behavior: 'instant' })
          await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
        }
        const want = ${JSON.stringify(act.node)}
        const nodes = [...f.querySelectorAll('[data-node-id]')]
        const label = (e) => e.getAttribute('data-node-label') ?? ''
        // A TreeView's diagram ids are 't<index>': a bare number matches the node index too.
        const id = (e) => e.getAttribute('data-node-id') ?? ''
        const n = nodes.find((e) => id(e) === want) ??
          (/^\\d+$/.test(want) ? nodes.find((e) => id(e).replace(/^\\D+/, '') === want) : undefined) ??
          nodes.find((e) => label(e) === want) ??
          nodes.find((e) => label(e).toLowerCase().includes(want.toLowerCase()))
        if (!n) return null
        // Diagram nodes move with a transition (a step of a player): wait until the node holds still.
        const frame = () => new Promise((r) => requestAnimationFrame(() => r()))
        const settleRect = async () => {
          let b = n.getBoundingClientRect(), still = 0
          for (let i = 0; i < 120 && still < 3; i++) {
            await frame()
            const c = n.getBoundingClientRect()
            still = c.x === b.x && c.y === b.y && c.width === b.width ? still + 1 : 0
            b = c
          }
          return b
        }
        let b = await settleRect()
        // A node below the window (a tall figure revealed from its top) is scrolled to, or the press would miss it.
        if (b.y < 0 || b.y + b.height > innerHeight || b.x < 0 || b.x + b.width > innerWidth) {
          n.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' })
          b = await settleRect()
        }
        const x = b.x + b.width / 2, y = b.y + b.height / 2
        // What a press there lands on: an element over the node (a label) would swallow it.
        const hit = document.elementFromPoint(x, y)
        if (hit && !n.contains(hit)) console.warn('shots: node ' + want + ' is covered at its centre by ' + hit.tagName + '.' + (hit.getAttribute('class') ?? ''))
        return [x, y]
      })()`)
    const box = scroll
      ? await reveal(act.figure)
      : await cdp.eval<{ area: DOMRect | null; x: number; y: number; width: number; height: number } | null>(`(() => {
          const f = document.querySelector(${JSON.stringify(fig)})
          if (!f) return null
          const b = f.getBoundingClientRect()
          const sep = f.querySelector('[role=separator][aria-label="Resize figure"]')
          const a = sep ? sep.parentElement.getBoundingClientRect() : null
          return { x: b.x, y: b.y, width: b.width, height: b.height,
            area: a && { x: a.x, y: a.y, width: a.width, height: a.height } }
        })()`)
    if (!box) return null
    const area = box.area ?? box
    return [area.x + act.at[0] * area.width, area.y + act.at[1] * area.height]
  }

  /** Load the app in a theme and viewport (the theme is in localStorage before the app starts); returns every page. */
  const openLab = async (v: Viewport, theme: Theme) => {
    await setViewport(v)
    await cdp.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: theme }] })
    const { identifier } = await cdp.send<{ identifier: string }>('Page.addScriptToEvaluateOnNewDocument', {
      source: `try { localStorage.setItem(${JSON.stringify(THEME_KEY)}, ${JSON.stringify(theme)}) } catch {}`,
    })
    log = []
    const onLoad = loaded()
    // The diagrams page draws no charts, so the app is up quickly; the pages to shoot are then opened in place.
    await cdp.send('Page.navigate', { url: `${base}${APP.start}` })
    await Promise.race([onLoad, new Promise((r) => setTimeout(r, 60_000))])
    const up = await cdp.eval<boolean>(
      `new Promise((r) => { const t0 = performance.now(); const t = () => document.querySelector('nav[aria-label="${APP.nav}"] a') ? r(true) : performance.now() - t0 > 30000 ? r(false) : setTimeout(t, 50); t() })`,
    )
    if (!up)
      throw new Error(
        `the app did not render its index within 30 s at ${base}${APP.start}` +
          (log.length ? `; console:\n${log.map((l) => `${l.level.toUpperCase()}  ${l.text}`).join('\n')}` : ''),
      )
    await cdp.send('Page.removeScriptToEvaluateOnNewDocument', { identifier })
    // The registry, as the shell lists it: every recipe under src/recipes.
    const all = await cdp.eval<string[]>(
      `[...document.querySelectorAll('nav[aria-label="${APP.nav}"] a[href]')].map((a) => decodeURIComponent(a.getAttribute('href').slice(1)))`,
    )
    return all
  }
  /** A fresh Chrome with the app loaded where the run was. */
  const restart = async (v: Viewport, theme: Theme) => {
    await openBrowser()
    try {
      await openLab(v, theme)
    } catch (e) {
      // A module mid-edit (another change landing while the run goes on) can fail one load: wait and load again.
      console.log(`      reload failed (${(e as Error).message.split('\n')[0]}); retrying in 5 s`)
      await new Promise((r) => setTimeout(r, 5000))
      await openLab(v, theme)
    }
  }
  await openBrowser()

  for (const v of viewports) {
    for (const theme of opts.themes) {
      const label = `${theme}${v.suffix}`
      const tLoad = performance.now()
      const all = await openLab(v, theme)
      const targets = select(all)
      console.log(
        `\n${label}: ${targets.length} page(s), ${v.width}×${v.height} @${opts.dpr}x (app loaded in ${Math.round(performance.now() - tLoad)} ms)`,
      )

      const shoot = async (target: Target) => {
        const tPage = performance.now()
        const dir = path.join(OUT, ...target.path.split('/'), label)
        rmSync(dir, { recursive: true, force: true })
        mkdirSync(dir, { recursive: true })
        log = []
        const url = `${base}/${target.path}`
        await cdp.eval(`(async () => {
          const key = ${JSON.stringify(target.path)}
          history.pushState(null, '', '/' + key)
          dispatchEvent(new PopStateEvent('popstate'))
          const current = () => document.querySelector('nav[aria-label="${APP.nav}"] a[aria-current="page"]')?.getAttribute('href')
          const t0 = performance.now()
          while (decodeURIComponent((current() ?? '').slice(1)) !== key && performance.now() - t0 < 10000)
            await new Promise((r) => setTimeout(r, 20))
          document.querySelector('main')?.scrollTo({ top: 0, behavior: 'instant' })
        })()`)
        await settle(target.path)

        const profiles: Profile[] = []
        // The last hover act: the pointer goes back to it before each capture that shows its figure.
        let hovering: Act | undefined
        for (const act of opts.acts) {
          const box = await reveal(act.figure)
          if (!box) continue
          if (act.kind !== 'press' && act.kind !== 'select') hovering = undefined
          const area = box.area ?? box
          const at = (fx: number, fy: number) => [area.x + fx * area.width, area.y + fy * area.height] as const
          if (act.kind === 'press') {
            const pressed = await cdp.eval<number>(`(async () => {
              const f = document.querySelector('main section[data-figure-id="${act.figure.replace(/"/g, '\\"')}"]')
              const b = f && f.querySelector('button[aria-label=${JSON.stringify(act.button)}]')
              if (!b) return 0
              for (let i = 0; i < ${act.at[0]}; i++) {
                b.click()
                await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
              }
              return ${act.at[0]}
            })()`)
            if (!pressed) log.push({ level: 'error', text: `shots: no button '${act.button}' in ${act.figure}` })
          } else if (act.kind === 'select') {
            // Real pointer presses (the dropdowns open on pointer events, not on element.click()), one trigger at a time.
            const fig = `main section[data-figure-id="${act.figure.replace(/"/g, '\\"')}"]`
            const triggers = await cdp.eval<{ x: number; y: number }[]>(
              `[...document.querySelectorAll('${fig} [role="combobox"]')].map((b) => { const r = b.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 } })`,
            )
            const want = JSON.stringify(act.button)
            let picked = false
            for (const t of triggers) {
              await mouse('mouseMoved', t.x, t.y)
              await mouse('mousePressed', t.x, t.y, 1)
              await mouse('mouseReleased', t.x, t.y)
              const option = await cdp.eval<{ x: number; y: number } | null>(`new Promise((r) => {
                const t0 = performance.now()
                const look = () => {
                  const o = [...document.querySelectorAll('[role="option"]')].find((e) => e.textContent.trim() === ${want})
                  if (o) { o.scrollIntoView({ block: 'nearest' }); const b = o.getBoundingClientRect(); return r({ x: b.x + b.width / 2, y: b.y + b.height / 2 }) }
                  performance.now() - t0 > 1000 ? r(null) : requestAnimationFrame(look)
                }
                look()
              })`)
              if (option) {
                await mouse('mouseMoved', option.x, option.y)
                await mouse('mousePressed', option.x, option.y, 1)
                await mouse('mouseReleased', option.x, option.y)
                picked = true
                break
              }
              await cdp.send('Input.dispatchKeyEvent', {
                type: 'keyDown',
                key: 'Escape',
                code: 'Escape',
                windowsVirtualKeyCode: 27,
              })
              await cdp.send('Input.dispatchKeyEvent', {
                type: 'keyUp',
                key: 'Escape',
                code: 'Escape',
                windowsVirtualKeyCode: 27,
              })
            }
            if (!picked) log.push({ level: 'error', text: `shots: no option '${act.button}' in ${act.figure}` })
          } else if (act.kind === 'click' || act.kind === 'hover') {
            const point = await locate(act, false)
            if (!point) {
              log.push({ level: 'error', text: `shots: no node '${act.node}' in ${act.figure}` })
              continue
            }
            let [x, y] = point
            await mouse('mouseMoved', x, y)
            // Let the page answer the hover (a re-render) before pressing, as a reader's hand would.
            await new Promise((r) => setTimeout(r, 120))
            if (act.kind === 'click') {
              // Hovering may change the page above the target (a readout that grows): find it again before pressing.
              const again = await locate(act, false)
              if (again && (again[0] !== x || again[1] !== y)) {
                ;[x, y] = again
                await mouse('mouseMoved', x, y)
              }
              await mouse('mousePressed', x, y, 1)
              await new Promise((r) => setTimeout(r, 60))
              await mouse('mouseReleased', x, y)
            } else hovering = act
          } else {
            const [x0, y0] = at(act.at[0], act.at[1])
            const [x1, y1] = at(act.at[2], act.at[3])
            if (opts.profile) {
              const result = await profileDrag(act, [x0, y0], [x1, y1])
              profiles.push(result)
              console.log(`      ${result.summary}`)
            } else {
              await mouse('mouseMoved', x0, y0)
              await mouse('mousePressed', x0, y0, 1)
              for (let i = 1; i <= 12; i++)
                await mouse('mouseMoved', x0 + ((x1 - x0) * i) / 12, y0 + ((y1 - y0) * i) / 12, 1)
              await mouse('mouseReleased', x1, y1)
            }
          }
          log.push({
            level: 'info',
            text: `shots: applied --act ${act.figure}:${act.kind} ${act.button ?? (act.node !== undefined ? `node ${act.node}` : act.at.join(' '))}`,
          })
          await settle(`${target.path} after --act`)
        }
        // Park the pointer on the sidebar's edge, off every chart, so no tooltip shows in the capture; a hover act puts it
        // back on its target before each capture that shows it.
        const park = () => mouse('mouseMoved', 1, v.height - 1)
        const rehover = async (what: string) => {
          const point = hovering && (await locate(hovering, false))
          if (!point) return park()
          await mouse('mouseMoved', point[0], point[1])
          await settle(what)
        }
        await park()

        const info = await cdp.eval<Inspect>(INSPECT)
        const problems: string[] = []
        if (!info.title) problems.push('no page title (the page did not render)')
        if (info.failed) problems.push(`${info.failed} figure(s) failed to render`)
        if (info.katex) problems.push(`${info.katex} KaTeX error(s)`)
        for (const f of info.figures)
          if (f.zero) problems.push(`figure ${f.id}: ${f.zero} of ${f.charts} chart(s) have zero size`)

        const files: string[] = []
        if (profiles.length) {
          const file = path.join(dir, 'profile.json')
          writeFileSync(file, JSON.stringify(profiles.length === 1 ? profiles[0] : profiles, null, 2) + '\n')
          writeFileSync(path.join(dir, 'profile.txt'), profiles.map((p) => p.summary).join('\n') + '\n')
          files.push(file, path.join(dir, 'profile.txt'))
        }
        const figures = target.figure ? info.figures.filter((f) => f.id === target.figure) : info.figures
        if (target.figure && figures.length === 0) problems.push(`no figure #${target.figure} on the page`)
        for (const f of figures) {
          let box = await reveal(f.id)
          if (!box) continue
          const tall = box.height + 16 > v.height
          if (tall) {
            // Taller than the window: grow the window to fit it, capture, and restore.
            await setViewport(v, Math.ceil(box.height + 80))
            await settle(`${target.path}#${f.id}`)
            box = (await reveal(f.id)) ?? box
          }
          if (hovering?.figure === f.id) await rehover(`${target.path}#${f.id} (hover)`)
          else await park()
          const file = path.join(dir, `${f.id}.png`)
          await capture(file, { x: box.x, y: box.y, width: box.width, height: box.height })
          files.push(file)
          // A gallery thumbnail: the page's first figure, its chart area only, at most 480 px a side (macOS sips).
          if (opts.thumbs && v.suffix === '' && f === figures[0] && target.path) {
            const thumb = path.join(THUMBS, `${target.path}-${theme}.png`)
            mkdirSync(path.dirname(thumb), { recursive: true })
            const a = box.area ?? box
            await capture(thumb, { x: a.x, y: a.y, width: a.width, height: a.height })
            execFileSync('sips', ['-Z', '480', thumb], { stdio: 'ignore' })
            files.push(thumb)
          }
          if (tall) {
            await setViewport(v)
            await settle(target.path)
          }
        }

        // The full page: the shell scrolls inside <main>, so the window grows to the content's height for one capture.
        if (!target.figure) {
          await cdp.eval(`document.querySelector('main')?.scrollTo({ top: 0, behavior: 'instant' })`)
          // Figures sized from the window's height (equal-aspect panels) grow with it, so measure again until it fits.
          let height = v.height
          let want = Math.ceil(info.scrollHeight)
          for (let i = 0; i < 4 && want > height && height < 16_000; i++) {
            height = Math.min(want, 16_000)
            await setViewport(v, height)
            await settle(`${target.path} (full page)`)
            want = Math.ceil((await cdp.eval<Inspect>(INSPECT)).scrollHeight)
          }
          if (hovering) await rehover(`${target.path} (full page, hover)`)
          const file = path.join(dir, 'page.png')
          await capture(file)
          files.unshift(file)
          if (height > v.height) await setViewport(v)
        }

        const errors = log.filter((l) => l.level === 'error')
        if (errors.length) problems.push(`${errors.length} console error(s)`)
        // `--sliders` (after the captures, so the images show the defaults): every slider must jump to where its track is clicked. Click each one at 20% and 80% of its
        // track with real mouse events and read its value back; a slider whose value does not move fails the page.
        const sliderProblems: string[] = []
        if (opts.sliders) {
          const count = await cdp.eval<number>(`document.querySelectorAll('main [data-slot="slider"]').length`)
          const disabled = await cdp.eval<number[]>(
            `[...document.querySelectorAll('main [data-slot="slider"]')].flatMap((el, i) => (el.hasAttribute('data-disabled') ? [i] : []))`,
          )
          for (let i = 0; i < count; i++) {
            if (disabled.includes(i)) continue
            const probe = (frac: number) =>
              cdp.eval<{ x: number; y: number; label: string; on: string | null } | null>(`(() => {
                const root = document.querySelectorAll('main [data-slot="slider"]')[${i}]
                if (!root) return null
                root.scrollIntoView({ block: 'center', behavior: 'instant' })
                const track = root.querySelector('[data-slot="slider-track"]') ?? root
                const r = track.getBoundingClientRect()
                const label = root.closest('[class*="flex-col"]')?.querySelector('label')?.textContent ?? 'slider ${i}'
                const x = r.left + ${frac} * r.width
                const y = r.top + r.height / 2
                // What the click lands on, for the failure message: an overlay here would swallow it.
                const hit = document.elementFromPoint(x, y)
                const on = hit && !root.contains(hit) ? (hit.getAttribute('data-slot') ?? hit.tagName.toLowerCase()) : null
                return { x, y, label, on }
              })()`)
            const read = () =>
              cdp.eval<string | null>(`(() => {
                const root = document.querySelectorAll('main [data-slot="slider"]')[${i}]
                const thumb = root?.querySelector('input[type="range"], [role="slider"]')
                return thumb ? (thumb.value ?? thumb.getAttribute('aria-valuenow')) : null
              })()`)
            const values: (string | null)[] = []
            let label = `slider ${i}`
            let covered: string | null = null
            for (const frac of [0.2, 0.8]) {
              const at = await probe(frac)
              if (!at) break
              label = at.label
              covered ??= at.on
              await mouse('mouseMoved', at.x, at.y)
              await mouse('mousePressed', at.x, at.y, 1)
              await mouse('mouseReleased', at.x, at.y)
              await new Promise((r) => setTimeout(r, 120))
              values.push(await read())
            }
            if (values.length === 2 && (values[0] === null || values[0] === values[1]))
              sliderProblems.push(
                `slider "${label.trim()}" did not move when its track was clicked (${values.join(' → ')})${covered ? `; the click landed on ${covered}` : ''}`,
              )
          }
          if (count) log.push({ level: 'info', text: `shots: clicked ${count} slider(s)` })
        }

        problems.push(...sliderProblems)

        const consoleFile = path.join(dir, 'console.txt')
        writeFileSync(
          consoleFile,
          log.length
            ? log.map((l) => `${l.level.toUpperCase()}  ${l.text}`).join('\n') + '\n'
            : '(no errors or warnings)\n',
        )
        files.push(consoleFile)
        if (problems.length) failures++

        const ms = Math.round(performance.now() - tPage)
        const status = problems.length ? 'FAIL' : 'ok  '
        console.log(
          `${status}  ${target.path}  ${ms} ms  (${files.length - 1} images)${problems.length ? '  ' + problems.join('; ') : ''}`,
        )
        index.push(
          `## ${target.path} (${label})${problems.length ? ' — FAIL: ' + problems.join('; ') : ''}`,
          '',
          `- page: ${url}`,
        )
        for (const file of files) index.push(`- \`${path.relative(OUT, file)}\``)
        index.push('')
      }
      // A long run used to hang (Runtime.evaluate got no reply after ~110 pages): Chrome is restarted every
      // `--restart-every` pages, and a page whose protocol call hangs is retried once in a fresh Chrome.
      let sinceRestart = 0
      for (const target of targets) {
        if (++sinceRestart > opts.restartEvery) {
          await restart(v, theme)
          sinceRestart = 1
        }
        try {
          await shoot(target)
        } catch (e) {
          const message = (e as Error).message
          if (!/no reply after|connection closed|Target closed|Session with given id not found/.test(message)) throw e
          console.log(`      ${target.path}: ${message}; restarting Chrome and retrying`)
          await restart(v, theme)
          sinceRestart = 1
          await shoot(target)
        }
      }
    }
  }
  cdp.close()
}

let crashed = false
try {
  await run()
} catch (e) {
  crashed = true
  console.error(`shots: ${(e as Error).message}`)
} finally {
  await teardown()
}
mkdirSync(OUT, { recursive: true })
writeFileSync(
  path.join(OUT, 'index.md'),
  [
    `# Lab screenshots`,
    '',
    `${new Date().toISOString()} · ${shots} images · ${failures} failed page(s)`,
    '',
    ...index,
  ].join('\n'),
)
console.log(
  `\n${shots} images · ${failures} failed page(s) · ${((performance.now() - t0) / 1000).toFixed(1)} s · ${path.relative(REPO, OUT)}/index.md`,
)
process.exit(crashed || failures ? 1 : 0)

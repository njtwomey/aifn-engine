import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import type { Mode } from './palette'
import { ThemeContext, type ThemePreference } from './theme'

const STORAGE_KEY = 'aifn-lab:theme'
const QUERY = '(prefers-color-scheme: dark)'

function readPreference(): ThemePreference {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return v === 'light' || v === 'dark' ? v : 'system'
  } catch {
    return 'system'
  }
}

function systemDark(): boolean {
  try {
    return matchMedia(QUERY).matches
  } catch {
    return false
  }
}

/**
 * The lab's theme: light, dark or the system's scheme, remembered per browser. It toggles the `dark` class on the
 * document root (which switches the CSS tokens) and tells charts which palette to draw with.
 */
export function ThemeProvider({ children, initial }: { children: ReactNode; initial?: ThemePreference }) {
  const [preference, setPreferenceState] = useState<ThemePreference>(() => initial ?? readPreference())
  const [dark, setDark] = useState(systemDark)

  useEffect(() => {
    const mq = matchMedia(QUERY)
    const onChange = () => setDark(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  const resolved: Mode = preference === 'system' ? (dark ? 'dark' : 'light') : preference

  useEffect(() => {
    document.documentElement.classList.toggle('dark', resolved === 'dark')
  }, [resolved])

  const setPreference = useCallback((p: ThemePreference) => {
    setPreferenceState(p)
    try {
      localStorage.setItem(STORAGE_KEY, p)
    } catch {
      // Storage unavailable (private mode): the preference lasts for this session only.
    }
  }, [])

  const value = useMemo(() => ({ preference, resolved, setPreference }), [preference, resolved, setPreference])
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

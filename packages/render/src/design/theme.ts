import { createContext, useContext } from 'react'
import type { Mode } from './palette'

export type ThemePreference = 'light' | 'dark' | 'system'

export type ThemeContextValue = {
  preference: ThemePreference
  /** The theme in effect: the preference, or the system's scheme when the preference is `system`. */
  resolved: Mode
  setPreference: (p: ThemePreference) => void
}

export const ThemeContext = createContext<ThemeContextValue | null>(null)

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext)
  if (!ctx) {
    const isDark =
      typeof document !== 'undefined'
        ? document.documentElement.classList.contains('dark') ||
          (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches)
        : false
    return {
      preference: 'system',
      resolved: isDark ? 'dark' : 'light',
      setPreference: () => {},
    }
  }
  return ctx
}

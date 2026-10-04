import { Moon, Sun } from 'lucide-react'
import { useTheme } from '@render/design/theme'
import { Button } from '@render/ui/button'

/**
 * Switches between light and dark with one click. The icon shows the theme in effect; until the first click the
 * theme follows the system's scheme.
 */
export function ThemeToggle({ className }: { className?: string }) {
  const { resolved, setPreference } = useTheme()
  const next = resolved === 'dark' ? 'light' : 'dark'
  const Icon = resolved === 'dark' ? Moon : Sun
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      className={className}
      onClick={() => setPreference(next)}
    >
      <Icon />
    </Button>
  )
}

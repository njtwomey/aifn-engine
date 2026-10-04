import { Monitor, Moon, Sun } from 'lucide-react'
import { useTheme, type ThemePreference } from '@render/design/theme'
import { Button } from '@render/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from '@render/ui/dropdown-menu'

const CHOICES = [
  { value: 'light', label: 'Light', Icon: Sun },
  { value: 'dark', label: 'Dark', Icon: Moon },
  { value: 'system', label: 'System', Icon: Monitor },
] as const

/** Light, dark or the system's scheme. The icon shows the theme in effect. */
export function ThemeToggle({ className }: { className?: string }) {
  const { preference, resolved, setPreference } = useTheme()
  const Icon = resolved === 'dark' ? Moon : Sun
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={<Button variant="ghost" size="icon-sm" aria-label="Theme" title="Theme" className={className} />}
      >
        <Icon />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-36">
        <DropdownMenuRadioGroup value={preference} onValueChange={(v) => setPreference(v as ThemePreference)}>
          {CHOICES.map(({ value, label, Icon }) => (
            <DropdownMenuRadioItem key={value} value={value}>
              <Icon /> {label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

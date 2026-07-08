import { watch, type Ref } from 'vue'

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'

/**
 * Traps Tab/Shift+Tab focus inside `containerRef` while `active` is true,
 * restores focus to whatever had it beforehand when `active` goes false.
 * Optionally invokes `onEscape` when Escape is pressed inside the trap.
 */
export function useFocusTrap(
  containerRef: Ref<HTMLElement | null>,
  active: Ref<boolean>,
  onEscape?: () => void
) {
  let previouslyFocused: HTMLElement | null = null

  function getFocusable(): HTMLElement[] {
    const el = containerRef.value
    if (!el) return []
    return Array.from(el.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && onEscape) {
      e.preventDefault()
      onEscape()
      return
    }
    if (e.key !== 'Tab') return
    const focusable = getFocusable()
    if (!focusable.length) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault()
      last.focus()
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault()
      first.focus()
    }
  }

  watch(
    active,
    (isActive) => {
      if (isActive) {
        previouslyFocused = document.activeElement as HTMLElement | null
        document.addEventListener('keydown', onKeydown, true)
        requestAnimationFrame(() => getFocusable()[0]?.focus())
      } else {
        document.removeEventListener('keydown', onKeydown, true)
        previouslyFocused?.focus?.()
        previouslyFocused = null
      }
    },
    { immediate: true }
  )
}

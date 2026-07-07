import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useUserSettingsStore } from '@/stores/userSettings'

const STORAGE_KEY = 'lawnberry_user_preferences'

describe('User Settings Store', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
  })

  it('defaults to dark theme with notifications enabled when nothing is stored', () => {
    const store = useUserSettingsStore()

    expect(store.preferences.theme).toBe('dark')
    expect(store.preferences.notifications).toEqual({ mower_status: true, system_alerts: true })
  })

  it('loads and migrates a previously stored preference set', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ theme: 'light' }))

    const store = useUserSettingsStore()

    expect(store.preferences.theme).toBe('light')
    // Missing keys are backfilled from defaults by the migration merge.
    expect(store.preferences.notifications).toEqual({ mower_status: true, system_alerts: true })
    expect(store.preferences.version).toBe(1)
  })

  it('falls back to defaults when stored JSON is corrupt', () => {
    localStorage.setItem(STORAGE_KEY, '{not-json')

    const store = useUserSettingsStore()

    expect(store.preferences.theme).toBe('dark')
  })

  it('updatePreference persists a change to an existing key', () => {
    const store = useUserSettingsStore()

    store.updatePreference('theme', 'light')

    expect(store.preferences.theme).toBe('light')
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).theme).toBe('light')
  })

  it('updatePreference warns and does nothing for an unknown key', () => {
    const store = useUserSettingsStore()
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

    store.updatePreference('does_not_exist', 'value')

    expect(warnSpy).toHaveBeenCalled()
    expect((store.preferences as any).does_not_exist).toBeUndefined()
    warnSpy.mockRestore()
  })

  it('setPreferences merges a partial update and persists it', () => {
    const store = useUserSettingsStore()

    store.setPreferences({ theme: 'light' })

    expect(store.preferences.theme).toBe('light')
    expect(store.preferences.notifications.mower_status).toBe(true)
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).theme).toBe('light')
  })

  it('resetPreferences restores defaults and persists them', () => {
    const store = useUserSettingsStore()
    store.setPreferences({ theme: 'light' })

    store.resetPreferences()

    expect(store.preferences.theme).toBe('dark')
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).theme).toBe('dark')
  })
})

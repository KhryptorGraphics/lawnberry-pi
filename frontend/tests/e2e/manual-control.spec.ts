import { test, expect } from '@playwright/test'
import { MockBackend } from './utils/mock-backend'
import { launchApp, resetAppStores } from './utils/test-setup'

test.describe('Manual control access', () => {
  test.beforeEach(async ({ page }) => {
    await resetAppStores(page)
  })

  test('requires unlock and supports emergency stop workflow', async ({ page }) => {
    const backend = new MockBackend()
    backend.setWebSocketScript([{ message: { event: 'connection.established', client_id: 'control-client' } }])

    await launchApp(page, backend, '/control')

    await expect(page.getByText('Control Access Required')).toBeVisible()
    await page.getByLabel('Confirm Password').fill('admin')
    await page.getByRole('button', { name: 'Unlock Control' }).click()

    const emergencyButton = page.getByRole('button', { name: /EMERGENCY STOP/i })
    await expect(emergencyButton).toBeEnabled()

    await emergencyButton.click()
    expect(backend.emergencyCalls).toHaveLength(1)
  })

  test('raises safety lockout when drive command is blocked', async ({ page }) => {
    const backend = new MockBackend()
    backend.setDriveResult({ result: 'blocked', status_reason: 'SAFETY_LOCKOUT' })
    backend.setWebSocketScript([{ message: { event: 'connection.established', client_id: 'control-client' } }])

    await launchApp(page, backend, '/control')

    await page.getByLabel('Confirm Password').fill('admin')
    await page.getByRole('button', { name: 'Unlock Control' }).click()

    await expect(page.getByRole('heading', { name: 'Movement Controls' })).toBeVisible()

    const joystick = page.getByRole('slider', { name: /joystick/i })
    // VirtualJoystick listens for pointer events; drive a real drag from the
    // centre toward the right edge so it emits a non-zero movement vector.
    const box = await joystick.boundingBox()
    if (!box) throw new Error('joystick not visible')
    const cx = box.x + box.width / 2
    const cy = box.y + box.height / 2
    await page.mouse.move(cx, cy)
    await page.mouse.down()
    await page.mouse.move(box.x + box.width, cy, { steps: 6 })

    await expect.poll(() => backend.driveCommands.length).toBeGreaterThanOrEqual(1)
    const lockoutActive = await page.evaluate(() => {
      return Boolean((window as any).__APP_TEST_HOOKS__?.controlStore?.lockout)
    })
    expect(lockoutActive).toBe(true)
  })
})

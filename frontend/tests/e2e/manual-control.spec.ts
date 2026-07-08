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

  test('stays locked when the unlock endpoint fails (fail-closed)', async ({ page }) => {
    const backend = new MockBackend()
    backend.setManualUnlockFailure(404)
    backend.setWebSocketScript([{ message: { event: 'connection.established', client_id: 'control-client' } }])

    await launchApp(page, backend, '/control')

    await expect(page.getByText('Control Access Required')).toBeVisible()
    await page.getByLabel('Confirm Password').fill('admin')
    await page.getByRole('button', { name: 'Unlock Control' }).click()

    // Fail-closed: an unlock error must never grant control access.
    await expect(page.getByText(/unlock is unavailable/i)).toBeVisible()
    await expect(page.getByText('Control Access Required')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Movement Controls' })).not.toBeVisible()
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
    // VirtualJoystick binds pointerdown on the element and pointermove on window.
    // Dispatch a pointerdown at centre then a pointermove toward the right edge
    // (same pointerId) so it emits a non-zero movement vector and sends a drive.
    const box = await joystick.boundingBox()
    if (!box) throw new Error('joystick not visible')
    const cx = box.x + box.width / 2
    const cy = box.y + box.height / 2
    await joystick.dispatchEvent('pointerdown', { pointerId: 1, clientX: cx, clientY: cy })
    await page.evaluate(
      ({ x, y }) =>
        window.dispatchEvent(
          new PointerEvent('pointermove', { pointerId: 1, clientX: x, clientY: y, bubbles: true })
        ),
      { x: box.x + box.width - 2, y: cy }
    )

    await expect.poll(() => backend.driveCommands.length).toBeGreaterThanOrEqual(1)
    const lockoutActive = await page.evaluate(() => {
      return Boolean((window as any).__APP_TEST_HOOKS__?.controlStore?.lockout)
    })
    expect(lockoutActive).toBe(true)
  })
})

// The manual-control gate used to say "additional authentication based on your security level". It now has to say
// what is true: POST /control/manual-unlock needs a signed-in operator (Bearer) AND a re-entered operator password
// (step-up check), and Cloudflare Access can no longer unlock at all.
import { describe, it, expect } from 'vitest'
import { mount, RouterLinkStub } from '@vue/test-utils'
import SecurityGateCard from '@/components/control/SecurityGateCard.vue'

type AuthLevel = 'password' | 'totp' | 'google' | 'cloudflare'

function mountCard(props: Record<string, unknown> = {}, authLevel: AuthLevel = 'password') {
  return mount(SecurityGateCard, {
    props: {
      securityConfig: {
        auth_level: authLevel,
        session_timeout_minutes: 15,
        require_https: false,
        auto_lock_manual_control: true,
      },
      authenticating: false,
      canAuthenticate: true,
      authError: '',
      formatSecurityLevel: (level: AuthLevel) => level,
      ...props,
    },
    global: { stubs: { RouterLink: RouterLinkStub } },
  })
}

describe('SecurityGateCard sign-in notice', () => {
  it('signed out (signedIn=false): shows a prominent "Sign in first" notice linking to /login', () => {
    const wrapper = mountCard({ signedIn: false })

    const notice = wrapper.find('.sign-in-notice')
    expect(notice.exists()).toBe(true)
    expect(notice.attributes('role')).toBe('alert')
    expect(notice.text()).toContain('Sign in first')
    expect(wrapper.findComponent(RouterLinkStub).props('to')).toBe('/login')
  })

  it('signed in (signedIn=true): no notice', () => {
    const wrapper = mountCard({ signedIn: true })

    expect(wrapper.find('.sign-in-notice').exists()).toBe(false)
    expect(wrapper.findComponent(RouterLinkStub).exists()).toBe(false)
  })

  it('defaults to signed in when the prop is omitted (no notice)', () => {
    const wrapper = mountCard()

    expect(wrapper.find('.sign-in-notice').exists()).toBe(false)
  })

  it('does not disable the unlock button while signed out: the backend decides', () => {
    const wrapper = mountCard({ signedIn: false })

    const unlock = wrapper.findAll('button').find((b) => b.text() === 'Unlock Control')
    expect(unlock?.attributes('disabled')).toBeUndefined()
  })
})

describe('SecurityGateCard password step-up wording', () => {
  it('keeps the "Confirm Password" label bound to #control-auth-password (the Playwright spec finds it by label)', () => {
    const wrapper = mountCard()

    const label = wrapper.find('label[for="control-auth-password"]')
    expect(label.text()).toBe('Confirm Password')
    expect(wrapper.find('input#control-auth-password').exists()).toBe(true)
  })

  it('asks for the operator password to be re-entered, in the placeholder and in a hint the input points at', () => {
    const wrapper = mountCard()

    const input = wrapper.find('input#control-auth-password')
    expect(input.attributes('placeholder')).toBe('Operator password (re-enter to unlock)')
    const hintId = input.attributes('aria-describedby')
    expect(hintId).toBe('control-auth-password-hint')
    expect(wrapper.find(`#${hintId}`).text()).toContain('Re-enter the operator password')
  })

  it('explains the two requirements: signed in AND the re-entered operator credential', () => {
    const text = mountCard().find('.security-info').text()

    expect(text).toContain('signed in')
    expect(text).toContain('re-enter the operator credential')
  })

  it('emits authenticate from the Unlock Control button and from Enter in the password field', async () => {
    const wrapper = mountCard()

    await wrapper.findAll('button').find((b) => b.text() === 'Unlock Control')?.trigger('click')
    await wrapper.find('input#control-auth-password').trigger('keyup.enter')

    expect(wrapper.emitted('authenticate')).toHaveLength(2)
  })
})

describe('SecurityGateCard Cloudflare level', () => {
  it('no longer claims you are already authenticated; says the backend cannot verify Cloudflare Access', () => {
    const text = mountCard({}, 'cloudflare').find('.auth-method').text()

    expect(text).not.toContain('already be authenticated')
    expect(text).toContain('cannot verify Cloudflare Access')
    expect(text).toContain('Choose Password in Settings > Security')
  })

  it('still offers Verify Access (which fails closed) and has no Unlock Control button', async () => {
    const wrapper = mountCard({}, 'cloudflare')

    await wrapper.findAll('button').find((b) => b.text() === 'Verify Access')?.trigger('click')

    expect(wrapper.emitted('verify-cloudflare')).toHaveLength(1)
    expect(wrapper.findAll('button').some((b) => b.text() === 'Unlock Control')).toBe(false)
  })
})

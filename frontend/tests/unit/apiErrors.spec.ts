import { describe, it, expect } from 'vitest'
import { SIGN_IN_REQUIRED_MESSAGE, apiErrorInfo, describeApiError } from '@/utils/apiErrors'

function httpError(status: number, data: unknown = {}, headers: Record<string, unknown> = {}) {
  return { message: `Request failed with status code ${status}`, response: { status, data, headers } }
}

describe('apiErrorInfo', () => {
  it('classifies a 401 as "sign in required" and carries the backend detail', () => {
    const info = apiErrorInfo(httpError(401, { detail: 'Operator authentication required' }))

    expect(info).toMatchObject({
      status: 401,
      detail: 'Operator authentication required',
      authRequired: true,
      rateLimited: false,
      noResponse: false,
    })
  })

  it('reads Retry-After on a 429 whatever the header casing, and ignores junk', () => {
    expect(apiErrorInfo(httpError(429, {}, { 'Retry-After': '30' })).retryAfterSeconds).toBe(30)
    expect(apiErrorInfo(httpError(429, {}, { 'retry-after': '7' })).retryAfterSeconds).toBe(7)
    expect(apiErrorInfo(httpError(429, {}, { 'retry-after': 'soon' })).retryAfterSeconds).toBeNull()
    expect(apiErrorInfo(httpError(429)).retryAfterSeconds).toBeNull()
    // Only a 429 has a meaningful Retry-After.
    expect(apiErrorInfo(httpError(503, {}, { 'retry-after': '5' })).retryAfterSeconds).toBeNull()
  })

  it('flattens a FastAPI 422 validation list into one readable line', () => {
    const info = apiErrorInfo(
      httpError(422, {
        detail: [
          { loc: ['body', 'value'], msg: 'Input should be less than or equal to 1', type: 'less_than_equal' },
          { loc: ['body', 'duration_ms'], msg: 'Input should be a valid integer', type: 'int_parsing' },
        ],
      })
    )

    expect(info.detail).toBe('Input should be less than or equal to 1; Input should be a valid integer')
  })

  it('treats an error without an HTTP response (link down, timeout, abort) as noResponse', () => {
    const info = apiErrorInfo({ message: 'timeout of 500ms exceeded', code: 'ECONNABORTED' })

    expect(info.noResponse).toBe(true)
    expect(info.status).toBeNull()
    expect(info.authRequired).toBe(false)
  })

  it('does not throw on null, undefined or a bare string', () => {
    for (const odd of [null, undefined, 'boom', 42]) {
      expect(apiErrorInfo(odd).noResponse).toBe(true)
    }
  })
})

describe('describeApiError', () => {
  it('turns a 401 into the one recognisable sign-in instruction, whatever the detail says', () => {
    expect(describeApiError(httpError(401, { detail: 'Operator authentication required' }), 'Failed')).toBe(
      SIGN_IN_REQUIRED_MESSAGE
    )
    expect(SIGN_IN_REQUIRED_MESSAGE).toContain('Sign In')
  })

  it('says how long to wait on a 429 when the backend said so, else falls back to its detail', () => {
    expect(describeApiError(httpError(429, {}, { 'retry-after': '12' }), 'Failed')).toBe(
      'Too many attempts: try again in 12 s.'
    )
    expect(describeApiError(httpError(429, { detail: 'Too many requests' }), 'Failed')).toBe('Too many requests')
    expect(describeApiError(httpError(429), 'Failed')).toContain('wait')
  })

  it('names the failed action when nothing answered at all', () => {
    expect(describeApiError({ message: 'Network Error' }, 'Left lever command failed')).toBe(
      'Left lever command failed: no response from the backend.'
    )
  })

  it('prefers the backend detail for any other HTTP error, else the caller fallback', () => {
    expect(describeApiError(httpError(409, { detail: 'Mission already running' }), 'Failed to start')).toBe(
      'Mission already running'
    )
    expect(describeApiError(httpError(500), 'Failed to start')).toBe('Failed to start')
  })
})

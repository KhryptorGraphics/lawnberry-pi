// Turns a failed backend call into something an operator can act on.
//
// The backend now enforces operator authentication on every actuating write
// (OPERATOR_AUTH_REQUIRED=1, see docs/authentication-config.md): an unsigned
// request answers 401 {"detail": "Operator authentication required"}. Before
// this helper every call site rendered that as a bare "Failed to ..." (or, for
// the mission store, as nothing at all), so an operator who simply had not
// signed in could not tell it from a hardware fault.

export interface ApiErrorInfo {
  /** HTTP status, or null when no HTTP response arrived (network error, timeout, abort). */
  status: number | null
  /** The backend's `detail` text, flattened; empty when there is none. */
  detail: string
  /** 401: the backend wants an operator sign-in (a Bearer token from /auth/login). */
  authRequired: boolean
  /** 429: rate limited or locked out. */
  rateLimited: boolean
  /** Seconds from a 429's Retry-After header, when the backend sent one. */
  retryAfterSeconds: number | null
  /** True when no HTTP response arrived at all (link down, timeout, request aborted). */
  noResponse: boolean
}

export const SIGN_IN_REQUIRED_MESSAGE =
  'Sign in required: open "Sign In" (top right), log in with the operator credential, then retry.'

type ErrorLike = {
  response?: {
    status?: number
    data?: { detail?: unknown } | string | null
    headers?: Record<string, unknown> | null
  }
  code?: string
  message?: string
}

function flattenDetail(raw: unknown): string {
  if (typeof raw === 'string') return raw.trim()
  if (Array.isArray(raw)) {
    // FastAPI 422: [{ loc: [...], msg: '...', type: '...' }, ...]
    return raw
      .map((item) => {
        if (typeof item === 'string') return item
        const msg = (item as { msg?: unknown } | null)?.msg
        return typeof msg === 'string' ? msg : ''
      })
      .filter(Boolean)
      .join('; ')
  }
  return ''
}

function readHeader(headers: Record<string, unknown> | null | undefined, name: string): unknown {
  if (!headers) return undefined
  const wanted = name.toLowerCase()
  for (const key of Object.keys(headers)) {
    if (key.toLowerCase() === wanted) return headers[key]
  }
  return undefined
}

export function apiErrorInfo(error: unknown): ApiErrorInfo {
  const err = (error ?? {}) as ErrorLike
  const response = err.response
  const status = typeof response?.status === 'number' ? response.status : null
  const data = response?.data
  const detail = flattenDetail(typeof data === 'string' ? data : data?.detail)

  let retryAfterSeconds: number | null = null
  if (status === 429) {
    const parsed = Number.parseInt(String(readHeader(response?.headers, 'retry-after') ?? ''), 10)
    if (Number.isFinite(parsed) && parsed > 0) retryAfterSeconds = parsed
  }

  return {
    status,
    detail,
    authRequired: status === 401,
    rateLimited: status === 429,
    retryAfterSeconds,
    noResponse: status === null,
  }
}

/**
 * One-line, operator-facing text for a failed call.
 *
 * - 401  -> the sign-in instruction (the same text everywhere, so it is recognisable)
 * - 429  -> "try again in N s" when the backend said how long
 * - no HTTP response -> "no response from the backend" (+ the call's own fallback text)
 * - anything else -> the backend's `detail`, else `fallback`
 */
export function describeApiError(error: unknown, fallback: string): string {
  const info = apiErrorInfo(error)
  if (info.authRequired) return SIGN_IN_REQUIRED_MESSAGE
  if (info.rateLimited) {
    const wait = info.retryAfterSeconds
    return wait ? `Too many attempts: try again in ${wait} s.` : info.detail || 'Too many attempts: wait a moment and retry.'
  }
  if (info.noResponse) return `${fallback}: no response from the backend.`
  return info.detail || fallback
}

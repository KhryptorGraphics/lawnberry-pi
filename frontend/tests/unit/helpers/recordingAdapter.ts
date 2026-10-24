// Test double for axios' network layer.
//
// The REAL ApiService (src/services/api.ts: base headers, request interceptor that adds
// the Bearer from the auth store, response interceptor that refreshes on 401) runs
// unchanged; only the adapter that would touch the network is replaced, so a spec can
// assert exactly what would have gone on the wire and feed back canned answers.
import { AxiosError, AxiosHeaders } from 'axios'
import type { AxiosAdapter, AxiosResponse, InternalAxiosRequestConfig } from 'axios'

export interface RecordedRequest {
  /** Upper-case HTTP method. */
  method: string
  /** The URL exactly as the call site passed it to the ApiService (baseURL is empty). */
  url: string
  /** Header names are lower-cased so assertions do not depend on axios' casing. */
  headers: Record<string, string>
  /** JSON bodies are parsed back into objects; anything else is left as sent. */
  body: unknown
}

export interface CannedResponse {
  status?: number
  data?: unknown
  headers?: Record<string, string>
}

/** Return undefined to answer 200 `{}`. */
export type Responder = (request: RecordedRequest) => CannedResponse | undefined

function lowerCaseHeaders(headers: InternalAxiosRequestConfig['headers']): Record<string, string> {
  const flat = AxiosHeaders.from(headers).toJSON(true)
  return Object.fromEntries(Object.entries(flat).map(([name, value]) => [name.toLowerCase(), String(value)]))
}

function parseBody(data: unknown): unknown {
  if (typeof data !== 'string') return data
  try {
    return JSON.parse(data)
  } catch {
    return data
  }
}

export function createRecordingAdapter(responder: Responder = () => undefined) {
  const requests: RecordedRequest[] = []

  const adapter: AxiosAdapter = async (config: InternalAxiosRequestConfig) => {
    const request: RecordedRequest = {
      method: (config.method ?? 'get').toUpperCase(),
      url: config.url ?? '',
      headers: lowerCaseHeaders(config.headers),
      body: parseBody(config.data),
    }
    requests.push(request)

    const canned = responder(request) ?? {}
    const status = canned.status ?? 200
    const response: AxiosResponse = {
      data: canned.data ?? {},
      status,
      statusText: String(status),
      headers: canned.headers ?? {},
      config,
      request: {},
    }
    // What axios' own settle() does: a status outside validateStatus rejects with an
    // AxiosError that carries the response (this is what the 401 interceptor sees).
    if (!config.validateStatus || config.validateStatus(status)) return response
    throw new AxiosError(
      `Request failed with status code ${status}`,
      status >= 500 ? AxiosError.ERR_BAD_RESPONSE : AxiosError.ERR_BAD_REQUEST,
      config,
      response.request,
      response
    )
  }

  return { adapter, requests }
}

// ApiService keeps its axios instance in a private field; reaching into it is the only
// way to swap the network layer underneath the interceptors.
interface ApiServiceInternals {
  client: { defaults: { adapter: unknown } }
}

/**
 * Swap the network adapter of the ApiService singleton (`useApiService()`).
 * Returns a function that restores the previous adapter.
 */
export function replaceNetworkAdapter(apiService: unknown, adapter: AxiosAdapter): () => void {
  const internals = apiService as ApiServiceInternals
  const previous = internals.client.defaults.adapter
  internals.client.defaults.adapter = adapter
  return () => {
    internals.client.defaults.adapter = previous
  }
}

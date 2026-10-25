# Authentication Configuration Guide

> **Authentication is ENFORCED by the repo unit (`OPERATOR_AUTH_REQUIRED=1`).**
>
> `systemd/lawnberry-backend.service` ships `OPERATOR_AUTH_REQUIRED=1` (2026-10-24 safety
> audit; it was `0`). Every state-changing route needs the operator's bearer token except the
> three E-stop POSTs and the login/refresh/logout routes (below). The default of
> `api/deps.py::require_operator_auth` is enforced when the variable is unset.
>
> **How the flag is read (fail closed).** Only `0`, `false`, `no` or `off` (any case, surrounding
> blanks ignored) disable operator authentication. Unset, `1`, `true`, `yes`, `on` and every
> ambiguous value (an empty `OPERATOR_AUTH_REQUIRED=` line, `2`, `disabled`, a typo) keep it
> enforced; an ambiguous value is logged as a WARNING. `<root>/.env` (the unit's
> `EnvironmentFile`) overrides the unit, so a `0` there silently turns authentication off: the
> backend therefore logs **CRITICAL at start-up whenever it is disabled outside `SIM_MODE`**,
> and `GET /health` (and `/api/v2/health`) report the state under `operator_auth`
> (`operator_auth_required`, `operator_auth_flag_ambiguous`, `operator_login_available`,
> `simulation_open_auth`). Enforcement points: `core/auth_posture.py::operator_auth_required`,
> read by `api/deps.py::require_operator_auth` and `api/routers/auth.py`
> (`_resolve_manual_session`, `manual_unlock`, `manual_unlock_status`).
>
> **The one operator credential.** There is no built-in default. Run
> `systemd/install_services.sh`: it refuses to finish (exit 3, nothing changed) until
> `LAWN_BERRY_OPERATOR_CREDENTIAL` is a credential of at least 12 characters that is not a
> published default (`operator123`, `admin`, `password`, ...), stores it in `<root>/.env`
> (mode 0600) and generates the persistent `LAWN_BERRY_AUTH_SECRET` there. With no usable
> credential the backend still starts but **nobody can log in** (fail closed); the E-stop routes
> stay open. The web UI login form posts `{username, password}`; the password is that credential
> (any username), and there is no `admin/admin` alias. Scripts pass a bearer token
> (`scripts/teleop.py --token`). To run a LAN bench without auth, set `OPERATOR_AUTH_REQUIRED=0`
> (a CRITICAL line is logged at every start).
>
> **What is open, and why** (pinned by `tests/integration/test_operator_auth_coverage.py`, which
> enumerates every POST/PUT/PATCH/DELETE route of the app and fails on an unreviewed one):
> `POST /api/v2/control/emergency`, `/api/v2/control/emergency-stop` and
> `/api/v2/tractor/emergency-stop` (an E-stop must work with no credential; they are also exempt
> from every rate limiter and from request validation, by exact path, see
> `middleware/emergency_paths.py`); the two login routes (they are the login); `/auth/logout` and
> `/auth/refresh` (they verify the bearer themselves). **Everything else needs
> `Authorization: Bearer <token>`**, including missions, `/maintenance/imu/calibrate`, `/ai/*`
> writes, `/nav/*`, the `/debug/*` injectors, `/control/drive`, `/control/blade`, the map, recording
> and `/api/v1` writes, and clearing an E-stop or latch (`/tractor/clear-emergency`,
> `/control/emergency_clear`). A manual-control session id is **not** a credential and
> `X-Session-Id` is no longer read. The `/debug/*` injectors (GPS, geofence, ToF and tilt) answer
> 404 unless `SIM_MODE=1` or `LAWNBERRY_ENABLE_DEBUG_ROUTES=1` (bench use on the mower needs that
> flag in `<root>/.env`; `/health` reports it as `debug_routes_enabled`), and need the bearer
> token on top.
>
> **WebSockets.** A browser cannot set an `Authorization` header on a WebSocket handshake, and the
> login token must not travel in a URL (uvicorn logs the query string; `?access_token=` is no
> longer read). It therefore trades its bearer token for a 30 s, single-use ticket
> (`POST /api/v2/auth/ws-ticket`, bearer required; a ticket dies with the login session it was
> minted for; the route is outside the `/api/*/auth` attempt limiter, see below) and connects with
> `?ticket=...`. Non-browser clients may send `Authorization: Bearer <token>` on the handshake.
> A presented but invalid credential is always refused. **With no credential, no peer is
> admitted, the loopback included** (2026-10-24 review SEC20-02): every ingress the product ships
> (a Cloudflare or ngrok agent pointed at the UI server, nginx) reaches the backend from a
> loopback address, so "loopback" cannot tell the browser on the mower from a remote visitor. The
> read-only streams (`/ws/telemetry`, `/api/v2/ws/telemetry`, `/api/v2/ws/status`) are opened to
> anonymous clients only by an explicit opt-in in `<root>/.env`: `LAWN_BERRY_ANONYMOUS_TELEMETRY=loopback`
> (a kiosk browser on the mower: the RESOLVED client address must be a loopback address, which is
> only as safe as the forwarding setup, see "Forwarders, tunnels and client addresses") or `=any`
> (anyone who can reach the port reads position, battery and state). An unrecognised value (`1`,
> `true`, a typo) stays off and is warned about at start-up, and an anonymous browser must also
> present an allowed `Origin` (CORS allow-list or the page's own host: CORS does not cover
> WebSockets). `/api/v2/ws/control` (the command-capable socket) needs a credential whenever
> operator auth is on, whatever the opt-in. Every socket is bounded: 32 in all and 8 per login
> session (anonymous sockets draw on a separate budget of 4 in all, 2 per peer); over the limit the
> handshake answers 503 + `Retry-After`. A frame over 4 KiB, or faster than 20 per second (burst
> 40), ends the socket; a client holds at most 32 subscriptions with short plain topic names; a
> socket that cannot take a frame within 2 s is dropped; `/ws/status` shares one telemetry sample
> between all its sockets (it used to run a full sensor read per socket and tick: 12 sockets
> pushed the E-stop POST from 4 ms to 560 ms, and one socket sending 330 000 `subscribe` frames
> grew the backend from 241 MiB to 612 MiB, the unit's cap is 512 MiB). No WebSocket handler
> changes mower state today (`/ws/control` only acknowledges).
>
> **Manual-control unlock** (`POST /api/v2/control/manual-unlock`, the Control view's gate) is a
> step-up for a logged-in operator: it needs the bearer token AND the credential again in the
> body (`{"method": "password", "password": ...}`; the shape the UI already sends). The password
> is checked against the bcrypt `password_hash` of the security settings, else
> `MANUAL_CONTROL_PASSWORD`, else the operator credential, and against nothing if none is usable:
> the unlock is refused (401; 400 for a blank password). Wrong passwords are throttled per TCP
> peer like the login (`AUTH_LOCKOUT_FAILURES`, 429 + `Retry-After`). The session id it returns
> is random (it used to be derived from the password) and is only the extra factor
> `/control/drive` and `/control/blade` check in the request body, together with the bearer token.
> **Cloudflare Access is not accepted**: the `CF-Access-Jwt-Assertion` header and the
> `CF_Authorization` cookie were decoded but never verified, so any value unlocked control (and
> the session was accepted as an operator login). `method: cloudflare` is now 501 and
> `GET /control/manual-unlock/status` answers `authorized: false` while auth is enforced. Verifying
> Cloudflare Access needs the Access issuer, audience and signing keys plus RS256 support
> (`cryptography`, not installed here): owner decision, see the audit.
>
> **Limits are per client, not per `X-Client-Id`.** The login lockout, the unlock throttle, the
> `/api/*/auth` limiter and the global limiter never count a header the client writes, and every
> table is bounded. A request that carries a bearer for a LIVE login session is counted in its own
> bucket, `session:<id>` (looked up silently by `AuthService.peek_session`: no log line, no side
> effect); every other request (no bearer, an invalid, expired or revoked one) is counted against
> the client's address (IPv6 per /64). That split is what keeps anonymous junk from a shared
> address (every visitor behind an undeclared same-host forwarder is one address) from draining
> the bucket the operator's lever POSTs use, and the login lockout only ever gates the login
> routes, never a request that carries a valid bearer. The client address is the TCP peer, or,
> only behind a forwarder DECLARED in `LAWN_BERRY_TRUSTED_PROXIES`, the `X-Forwarded-For` hop that
> forwarder wrote (`core/peer_address.py`; the default trusts nobody and the loopback as a whole
> is not trusted). uvicorn's own proxy handling is switched off (`--no-proxy-headers` in the unit)
> because it runs first and takes whatever string it finds. What residual a shared address has:
> with no declared forwarder all visitors share one lockout, so an anonymous visitor can delay a
> LOGIN (a logged-in operator keeps working); declare the forwarder to separate them. Exact
> settings per setup: "Forwarders, tunnels and client addresses" below. `POST /auth/ws-ticket` (the
> UI calls it before every WebSocket (re)connect, with a token that may have expired) is exempt
> from the `/api/*/auth` attempt limiter and lockout
> (`middleware/security.py::UNLIMITED_AUTH_PATHS`): it only redeems a live bearer, and counting it
> would block the login after a few reconnects. Tractor command POSTs and `/control/drive|blade`
> have their own buckets so UI polling cannot starve them. Password checks (bcrypt, ~270 ms) run on
> a small dedicated thread pool, never on the event loop that carries the deadman and the E-stop
> handlers.
>
> **`SIM_MODE` is not a credential.** `SIM_MODE=1` alone no longer makes the login accept any
> credential or admits any WebSocket client as an operator: that needs `LAWN_BERRY_SIM_OPEN_AUTH=1`
> as well (the test suite sets it) and is ignored when `LAWNBERRY_ENV=production`. Start-up logs
> CRITICAL for `SIM_MODE=1` in a production environment or on a non-loopback bind
> (`--host 0.0.0.0`, `UVICORN_HOST`), and for the opt-in on such a process.
>
> The `lawnberry-pi ...` command examples further down describe a CLI that is not shipped (no
> console entry point in `pyproject.toml`); the supported configuration is the environment
> and `.env` described here.

## Forwarders, tunnels and client addresses

The login lockout, every rate limit and the WebSocket budgets need to know who the client is. On
this host that is not obvious: a browser request reaches the backend through up to three hops (a
tunnel agent or nginx, the UI server `frontend/server.mjs`, uvicorn), all of them on
127.0.0.0/8, and any process on the host can connect from any address in that range.

1. **Nobody is trusted by default.** `LAWN_BERRY_TRUSTED_PROXIES` (backend) and
   `UI_TRUSTED_PROXIES` (UI server) default to the empty list: every client behind a forwarder is
   counted as the forwarder's own address (one lockout and one bucket for all of them; logged-in
   operators are still counted per login session). Safe, coarse, and what you get when nothing is
   declared. Trust is declared, never inferred from the loopback.
2. **A declaration is `ADDRESS[/PREFIX][=SHAPE]`, comma separated.** `ADDRESS` is the address the
   forwarder CONNECTS FROM. `SHAPE` is what it does to `X-Forwarded-For`: `single` (the default:
   it replaces the header with exactly one hop; two hops or a non-address means the request did
   not come through it, the header is ignored and a warning is logged) or `last` (it appends the
   address it saw; only the last hop is believed, everything to its left is what the client wrote
   and is never read; trailing hops that are themselves declared forwarders are skipped).
3. **A bad declaration trusts nobody.** A `/0`, a network wider than one LAN (IPv4 shorter than
   /24, IPv6 shorter than /64), host bits set (`192.168.1.64/24`), an unknown shape or something
   that is not an address makes the WHOLE list invalid: ERROR at start-up naming each problem,
   nobody trusted, never a widened list (the backend still starts: it carries the E-stop). The
   old default `127.0.0.0/8,::1` and the old silent acceptance of `0.0.0.0/0` are gone.
4. **Each trusted forwarder gets its own source address.** A forwarder that passes a
   client-written `X-Forwarded-For` through (`ssh -R`, `socat`, `frp`, a custom tunnel command, an
   nginx location without the header) lets its client choose its own identity, bucket and lockout,
   so it must never be listed, and it cannot be told apart from a good one at `127.0.0.1`.

What the repository ships, and what to change for a tunnel (the two settings live in the units:
the backend reads `<root>/.env`, which overrides the unit; the UI server has no `.env`, use a
systemd drop-in, `sudo systemctl edit lawnberry-frontend`, and restart it):

| Setup | Connects from | Declare | Notes |
|---|---|---|---|
| UI server to backend (shipped) | `127.0.0.2` (`BACKEND_SOURCE_ADDRESS`) | backend: `127.0.0.2=single` | the UI server drops every inbound `X-Forwarded-For`/`Forwarded`/`X-Real-IP` and sends exactly one hop; if it cannot bind `127.0.0.2` it logs an ERROR and the backend sees one shared address |
| nginx in front of the UI server (`scripts/setup_https.sh`, `config/nginx.conf`) | `127.0.0.3` (`proxy_bind`) | UI server: `UI_TRUSTED_PROXIES=127.0.0.3=last` (shipped) | every proxying location, the two WebSocket ones included, now sets `X-Forwarded-For $proxy_add_x_forwarded_for` (they sent none: all remote clients were one address) |
| cloudflared or ngrok on this host, pointed at the UI server (port 3000) | `127.0.0.1` | drop-in: `Environment=UI_TRUSTED_PROXIES=127.0.0.3=last,127.0.0.1=last` | only if the tunnel agent is the ONLY process that forwards remote traffic to port 3000 from `127.0.0.1`; the Cloudflare or ngrok edge writes the last hop |
| a tunnel pointed straight at the backend (port 8081) | `127.0.0.1` | `.env`: `LAWN_BERRY_TRUSTED_PROXIES=127.0.0.2=single,127.0.0.3=last,127.0.0.1=last` | same condition; without it every tunnel client is one address (safe, coarse) |
| a tunnel or proxy on another host | its LAN address | list that address (`/32` host entry), shape per its behaviour | a LAN-wide network is refused: list the proxy host |

`CF-Connecting-IP` and `True-Client-IP` are never read. Cloudflare Access is not accepted as a
credential (see above); a tunnel is transport, the operator still logs in.

**Check a setup from outside** (it locks that client out of the login for `AUTH_LOCKOUT_SECONDS`,
30 s by default): send six wrong logins through the tunnel, each with a different
`X-Forwarded-For`. The sixth must be `429` with `Retry-After`: the header was ignored and all six
counted against your real address. Six `401`s mean the forwarder lets the client name its own
address: remove it from the list. The start-up log (`journalctl -u lawnberry-backend | grep -i
"forwarder\|WebSocket"`) states the resolved trusted list, the uvicorn proxy-headers state and the
anonymous-WebSocket mode.

This guide covers setting up and configuring the multi-level authentication system in LawnBerry Pi v2, from basic password authentication to enterprise-grade security.

## Table of Contents

1. [Overview](#overview)
2. [Security Levels](#security-levels)
3. [Password Authentication](#password-authentication)
4. [TOTP Two-Factor Authentication](#totp-two-factor-authentication)
5. [Google OAuth Integration](#google-oauth-integration)
6. [Cloudflare Tunnel Authentication](#cloudflare-tunnel-authentication)
7. [Session Management](#session-management)
8. [Security Best Practices](#security-best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

LawnBerry Pi v2 features a configurable authentication system with four security levels:

1. **Password**: Basic username/password authentication
2. **TOTP**: Password + Time-based One-Time Password (2FA)
3. **Google OAuth**: Google account authentication with domain restrictions
4. **Tunnel Auth**: Cloudflare tunnel authentication (highest security)

Each level provides increasing security and may disable lower-level authentication methods.

### Key Features

- **Progressive Enhancement**: Start simple, upgrade security as needed
- **Session Management**: Configurable timeout and concurrent session limits
- **Audit Logging**: Complete authentication event logging
- **Rate Limiting**: Protection against brute force attacks
- **Backup Recovery**: TOTP backup codes and account recovery options

## Security Levels

### Level 1: Password Authentication

**Use Case**: Basic home networks, initial setup
**Security**: Basic protection against unauthorized access
**Requirements**: Username and password only

### Level 2: TOTP Two-Factor Authentication

**Use Case**: Enhanced home security, remote access scenarios
**Security**: Significantly improved protection with 2FA
**Requirements**: Password + TOTP code from authenticator app

### Level 3: Google OAuth

**Use Case**: Integration with existing Google Workspace/Gmail accounts
**Security**: Enterprise-grade OAuth with optional domain restrictions
**Requirements**: Valid Google account, optional domain membership

### Level 4: Cloudflare Tunnel Authentication

**Use Case**: Maximum security with zero-trust architecture
**Security**: Highest level, relies on Cloudflare Access policies
**Requirements**: Cloudflare tunnel setup with Access policies

## Password Authentication

Basic username/password authentication suitable for initial setup and secure local networks.

### Initial Setup

```bash
# Set initial admin password during first setup
lawnberry-pi config auth --level password --set-password

# Or use interactive setup
lawnberry-pi config auth --setup-wizard
```

### Configuration Options

```bash
# Configure password requirements
lawnberry-pi config auth --password-policy \
    --min-length 12 \
    --require-uppercase \
    --require-lowercase \
    --require-numbers \
    --require-symbols

# Set session timeout
lawnberry-pi config auth --session-timeout 60  # 60 minutes

# Enable password change requirement
lawnberry-pi config auth --require-password-change \
    --change-interval 90  # 90 days
```

### Password Management

```bash
# Change admin password
lawnberry-pi auth change-password

# Reset password (requires physical access)
sudo lawnberry-pi auth reset-password --user admin

# View password policy
lawnberry-pi config auth --show-policy
```

### Example Configuration

```yaml
# /etc/lawnberry/auth-config.yml
security_level: password
password_policy:
  min_length: 12
  require_uppercase: true
  require_lowercase: true
  require_numbers: true
  require_symbols: false
  max_age_days: 90
session_config:
  timeout_minutes: 60
  max_concurrent_sessions: 3
  remember_me_days: 30
```

## TOTP Two-Factor Authentication

Time-based One-Time Password (TOTP) adds significant security by requiring a second factor from your mobile device.

### Setup Process

```bash
# Enable TOTP authentication
lawnberry-pi config auth --level totp --setup-totp

# This will:
# 1. Generate a secret key
# 2. Display QR code for mobile app
# 3. Generate backup codes
# 4. Test TOTP verification
```

### Mobile App Setup

1. **Install Authenticator App**:
   - Google Authenticator (iOS/Android)
   - Authy (iOS/Android/Desktop)
   - Microsoft Authenticator (iOS/Android)
   - 1Password (iOS/Android/Desktop)

2. **Scan QR Code**: Use app to scan the displayed QR code

3. **Verify Setup**: Enter the 6-digit code to confirm

### Backup Codes

TOTP setup generates 10 backup codes for account recovery:

```bash
# Generate new backup codes
lawnberry-pi auth totp --regenerate-backup-codes

# View remaining backup codes
lawnberry-pi auth totp --show-backup-codes

# Use backup code for login (emergency)
# Enter backup code instead of TOTP during login
```

### TOTP Configuration

```bash
# Configure TOTP settings
lawnberry-pi config auth --totp-window 1 \  # Allow 1 step tolerance
    --totp-rate-limit 5 \                   # Max 5 attempts per minute
    --backup-codes-count 10                 # Generate 10 backup codes

# Disable TOTP (downgrade to password)
lawnberry-pi config auth --level password --disable-totp
```

### Recovery Process

If you lose access to your TOTP device:

1. **Use Backup Code**: Use one of your saved backup codes
2. **Physical Reset**: Access Pi directly and reset auth level
3. **Recovery Key**: Use recovery key if configured

```bash
# Physical recovery (requires sudo access on Pi)
sudo lawnberry-pi auth recovery --reset-totp --user admin

# Set up TOTP again after recovery
lawnberry-pi config auth --level totp --setup-totp
```

## Google OAuth Integration

Integrate with Google accounts for seamless authentication using existing corporate or personal Google accounts.

### Prerequisites

1. **Google Cloud Project**: Create project at [console.cloud.google.com](https://console.cloud.google.com)
2. **OAuth 2.0 Client**: Configure OAuth client credentials
3. **Domain Control**: Optional domain restrictions

### Google Cloud Setup

1. **Create Project**:
   - Go to Google Cloud Console
   - Create new project: "LawnBerry Pi Auth"

2. **Enable APIs**:
   ```bash
   # Enable Google+ API for user info
   # Navigate to APIs & Services > Library
   # Search and enable "Google+ API"
   ```

3. **Create OAuth Client**:
   - Go to APIs & Services > Credentials
   - Create OAuth 2.0 Client ID
   - Application type: Web application
   - Authorized redirect URIs:
     - `https://your-domain.com/auth/google/callback`
     - `http://localhost:3000/auth/google/callback` (for local testing)

4. **Download Credentials**: Download JSON credentials file

### LawnBerry Configuration

```bash
# Configure Google OAuth
lawnberry-pi config auth --level google \
    --google-client-id "YOUR_CLIENT_ID.googleusercontent.com" \
    --google-client-secret "YOUR_CLIENT_SECRET" \
    --google-redirect-uri "https://your-domain.com/auth/google/callback"

# Optional: Restrict to specific domains
lawnberry-pi config auth --google-allowed-domains "yourdomain.com,anotherdomain.com"

# Optional: Require specific group membership
lawnberry-pi config auth --google-required-groups "lawnberry-users@yourdomain.com"
```

### Domain Restrictions

```bash
# Allow only specific email domains
lawnberry-pi config auth --google-allowed-domains "company.com"

# Allow specific email addresses
lawnberry-pi config auth --google-allowed-emails "user1@gmail.com,user2@company.com"

# Combine domain and email restrictions
lawnberry-pi config auth --google-allowed-domains "company.com" \
    --google-allowed-emails "contractor@external.com"
```

### Testing Google OAuth

```bash
# Test OAuth configuration
lawnberry-pi auth test-google-oauth --client-id YOUR_CLIENT_ID

# Verify user authentication
lawnberry-pi auth verify-google-user --email user@domain.com
```

### Example Google OAuth Configuration

```yaml
# /etc/lawnberry/auth-config.yml
security_level: google_oauth
google_auth_config:
  client_id: "123456789.googleusercontent.com"
  client_secret: "encrypted_secret"
  allowed_domains:
    - "company.com"
    - "yourdomain.com"
  allowed_emails:
    - "admin@external.com"
  require_verified_email: true
  session_duration_hours: 8
```

## Cloudflare Tunnel Authentication

The highest security level using Cloudflare Access for zero-trust authentication.

### Prerequisites

1. **Cloudflare Account**: With domain management
2. **Cloudflare Tunnel**: Configured and running
3. **Cloudflare Access**: Access subscription (free tier available)

### Cloudflare Access Setup

1. **Enable Cloudflare Access**:
   - Go to Cloudflare Dashboard > Access
   - Set up Access for your domain
   - Configure identity providers (Google, Azure AD, etc.)

2. **Create Access Application**:
   - Name: LawnBerry Pi
   - Domain: `lawnberry.yourdomain.com`
   - Type: Self-hosted

3. **Configure Access Policies**:
   ```yaml
   # Example policy
   name: "LawnBerry Pi Access Policy"
   decision: "allow"
   rules:
     - emails:
         - "admin@yourdomain.com"
         - "user@yourdomain.com"
     - email_domains:
         - "yourdomain.com"
   ```

### LawnBerry Configuration

```bash
# Enable Cloudflare tunnel authentication
lawnberry-pi config auth --level tunnel \
    --tunnel-provider cloudflare \
    --required-headers "CF-Access-Authenticated-User-Email" \
    --verify-jwt-signature

# Configure session settings for tunnel auth
lawnberry-pi config auth --tunnel-session-timeout 120  # 2 hours
```

### Header Validation

Cloudflare provides authentication headers that LawnBerry validates:

```bash
# Configure required headers
lawnberry-pi config auth --tunnel-required-headers \
    "CF-Access-Authenticated-User-Email" \
    "CF-Ray" \
    "CF-Visitor"

# Optional: Configure user mapping
lawnberry-pi config auth --tunnel-user-header "CF-Access-Authenticated-User-Email"
```

### Testing Tunnel Authentication

```bash
# Test tunnel authentication
lawnberry-pi auth test-tunnel --simulate-headers

# Verify Access policy
curl -H "CF-Access-Authenticated-User-Email: user@domain.com" \
     https://lawnberry.yourdomain.com/api/v1/auth/verify
```

## Session Management

Configure session behavior, timeouts, and concurrent access controls.

### Session Configuration

```bash
# Configure session timeouts
lawnberry-pi config auth --session-timeout 60 \      # 60 minutes
    --absolute-timeout 480 \                         # 8 hours absolute max
    --idle-timeout 30                                # 30 minutes idle

# Configure concurrent sessions
lawnberry-pi config auth --max-concurrent-sessions 3 \
    --session-collision-policy "terminate_oldest"

# Configure remember-me functionality
lawnberry-pi config auth --remember-me-duration 30  # 30 days
```

### Session Security

```bash
# Enable secure session features
lawnberry-pi config auth --secure-sessions \
    --rotate-session-ids \
    --bind-to-ip \
    --require-https

# Configure CSRF protection
lawnberry-pi config auth --csrf-protection \
    --csrf-token-lifetime 3600  # 1 hour
```

### Session Monitoring

```bash
# View active sessions
lawnberry-pi auth sessions --list

# Terminate specific session
lawnberry-pi auth sessions --terminate SESSION_ID

# Terminate all sessions for user
lawnberry-pi auth sessions --terminate-user admin

# View session statistics
lawnberry-pi auth sessions --stats
```

## Security Best Practices

### Password Policies

```bash
# Enforce strong password policy
lawnberry-pi config auth --password-policy \
    --min-length 16 \
    --require-uppercase \
    --require-lowercase \
    --require-numbers \
    --require-symbols \
    --no-dictionary-words \
    --no-personal-info

# Set password history
lawnberry-pi config auth --password-history 12  # Remember last 12 passwords
```

### Rate Limiting

```bash
# Configure authentication rate limits
lawnberry-pi config auth --rate-limits \
    --login-attempts 5 \
    --lockout-duration 1800 \  # 30 minutes
    --progressive-delay
```

### Audit Logging

```bash
# Enable comprehensive audit logging
lawnberry-pi config auth --audit-logging \
    --log-successful-logins \
    --log-failed-attempts \
    --log-session-events \
    --log-config-changes

# View audit logs
lawnberry-pi auth audit --tail 100
lawnberry-pi auth audit --filter "failed_login" --since "1 hour ago"
```

### Security Monitoring

```bash
# Enable security monitoring
lawnberry-pi config security --monitoring \
    --alert-failed-logins 10 \
    --alert-concurrent-sessions 5 \
    --alert-suspicious-activity

# Set up email alerts
lawnberry-pi config security --email-alerts admin@yourdomain.com
```

## Troubleshooting

### Common Issues

#### Password Authentication Issues

1. **Forgot Password**:
   ```bash
   # Reset password with physical access
   sudo lawnberry-pi auth reset-password --user admin
   
   # Follow prompts to set new password
   ```

2. **Account Locked**:
   ```bash
   # Check lockout status
   lawnberry-pi auth status --user admin
   
   # Unlock account
   sudo lawnberry-pi auth unlock --user admin
   ```

#### TOTP Issues

1. **TOTP Codes Not Working**:
   ```bash
   # Check time synchronization
   sudo ntpdate -s time.nist.gov
   
   # Verify TOTP configuration
   lawnberry-pi auth totp --verify-config
   
   # Test with backup code
   # Use backup code instead of TOTP code
   ```

2. **Lost TOTP Device**:
   ```bash
   # Use backup code to login
   # Then regenerate TOTP setup
   lawnberry-pi config auth --level totp --setup-totp --force
   ```

#### Google OAuth Issues

1. **OAuth Callback Errors**:
   ```bash
   # Verify redirect URI configuration
   lawnberry-pi config auth --show-google-config
   
   # Check Google Cloud Console settings
   # Ensure redirect URI matches exactly
   ```

2. **Domain Restriction Errors**:
   ```bash
   # Check allowed domains
   lawnberry-pi config auth --show-allowed-domains
   
   # Add domain if needed
   lawnberry-pi config auth --google-allowed-domains "newdomain.com"
   ```

#### Tunnel Authentication Issues

1. **Missing Headers**:
   ```bash
   # Check Cloudflare Access configuration
   # Verify Access policy is applied
   
   # Test headers manually
   curl -v https://lawnberry.yourdomain.com/api/v1/status
   ```

2. **JWT Verification Errors**:
   ```bash
   # Check JWT signature verification
   lawnberry-pi auth test-tunnel --verify-jwt
   
   # Update Cloudflare certificates
   lawnberry-pi auth tunnel --update-certs
   ```

3. **DNS NXDOMAIN for Tunnel Hostname**:

If your browser console shows `net::ERR_NAME_NOT_RESOLVED` for your tunnel domain (e.g., `lawnberry.yourdomain.com`), the public DNS record is missing or misconfigured.

Steps to fix:

1) Verify from the Pi that DNS is missing:

```bash
getent hosts lawnberry.yourdomain.com || host lawnberry.yourdomain.com
```

2) Restore the hostname in Cloudflare Zero Trust:

- Cloudflare Dashboard → Zero Trust → Access → Tunnels → select your tunnel → Public Hostnames → Add application
- Hostname: `lawnberry.yourdomain.com`
- Service: `http://localhost:3000` (frontend server)
- Save and wait ~1–2 minutes for propagation

3) Confirm Cloudflare created the DNS record in the domain’s DNS app:

- DNS → Check for a CNAME `lawnberry` pointing to `<tunnel-uuid>.cfargotunnel.com` with Proxied = ON

4) Restart the connector (optional but helps pick up config):

```bash
sudo systemctl restart cloudflared
sudo systemctl status cloudflared --no-pager
```

5) Re-test resolution and connectivity from the Pi:

```bash
getent hosts lawnberry.yourdomain.com
curl -sS https://lawnberry.yourdomain.com/ | head -c 200
curl -sS -X POST https://lawnberry.yourdomain.com/api/v2/auth/login \
    -H 'Content-Type: application/json' \
    -d '{"username":"admin","password":"<your operator credential>"}'
```

Notes:
- If you prefer direct backend access too, add a second public hostname mapping `lawnberry-backend.yourdomain.com` → `http://localhost:8081`.
- Cloudflare tunnels created via token (no local config.yml) manage hostnames from the dashboard; local `/etc/cloudflared/config.yml` may not exist.

4. **WebSocket 401 Unauthorized (remote access)**:

Browsers cannot set custom headers on WebSocket upgrades, so the UI trades its bearer token for a
single-use ticket and connects with `?ticket=<ticket>`. Ensure:

- You are logged in to the UI (a valid token is stored in localStorage); the UI then calls
  `POST /api/v2/auth/ws-ticket` before every (re)connect. The login token itself is no longer
  accepted in a WebSocket URL.
- A ticket lives 30 s and works once; a retry needs a fresh one (the UI fetches one per attempt).
- A `CF-Access-Jwt-Assertion` header is not a credential on any route (it was never verified), so
  a Cloudflare Access policy does not authorize a WebSocket: log in to the mower UI as well.

If you still see WS failures, live telemetry will fall back to REST polling. Check the browser
console network tab for 401/403 responses on `/api/v2/ws/telemetry` or `/ws/telemetry`, and for
the `POST /api/v2/auth/ws-ticket` request that precedes them.

### Debug Commands

```bash
# Test authentication configuration
lawnberry-pi auth test --all-levels

# Check authentication status
lawnberry-pi auth status --verbose

# View authentication logs
sudo journalctl -u lawnberry-backend | grep auth

# Validate configuration
lawnberry-pi config auth --validate
```

### Recovery Procedures

#### Emergency Access

No emergency password and no `lawnberry-pi auth emergency-access` command exist. With shell
access to the Pi, set a new credential and re-run the installer (it keeps the signing secret;
`--rotate-auth-secret` also ends every existing session):

```bash
f=$(mktemp) && ${EDITOR:-nano} "$f"      # one line: the new credential
sudo bash systemd/install_services.sh --operator-credential-file "$f" --rotate-auth-secret
shred -u "$f"
# restart lawnberry-backend to load it (on the shed Pi only with physical access)
```

#### Configuration Backup/Restore

```bash
# Backup authentication configuration
lawnberry-pi config backup --auth-only --file auth-backup.json

# Restore authentication configuration
lawnberry-pi config restore --auth-only --file auth-backup.json
```

### Getting Help

For authentication issues:

1. **Check logs**: `sudo journalctl -u lawnberry-backend | grep auth`
2. **Verify config**: `lawnberry-pi config auth --validate`
3. **Test connectivity**: `lawnberry-pi auth test --connectivity`
4. **Emergency access**: Use physical recovery if needed

Remember to always test authentication changes in a controlled environment before applying to production systems.
<template>
  <div class="security-gate">
    <div class="card security-card">
      <div class="card-header">
        <h3>🔒 Control Access Required</h3>
      </div>
      <div class="card-body">
        <div class="security-info">
          <p>Manual control access requires additional authentication based on your security level:</p>
          <div class="security-level">
            <strong>Current Security Level:</strong>
            <span class="level-badge" :class="`level-${securityConfig.auth_level}`">
              {{ formatSecurityLevel(securityConfig.auth_level) }}
            </span>
          </div>
        </div>

        <!-- Authentication Methods -->
        <div class="auth-methods">
          <!-- Password Verification -->
          <div v-if="securityConfig.auth_level === 'password'" class="auth-method">
            <label for="control-auth-password">Confirm Password</label>
            <input
              id="control-auth-password"
              v-model="password"
              type="password"
              class="form-control"
              placeholder="Enter your password"
              @keyup.enter="$emit('authenticate')"
            >
          </div>

          <!-- TOTP Verification -->
          <div v-else-if="securityConfig.auth_level === 'totp'" class="auth-method">
            <label for="control-auth-totp">Enter TOTP Code</label>
            <input
              id="control-auth-totp"
              v-model="totpCode"
              type="text"
              class="form-control totp-input"
              placeholder="000000"
              maxlength="6"
              @keyup.enter="$emit('authenticate')"
            >
            <small class="form-text text-muted">
              Use your authenticator app (Google Authenticator, Authy, etc.)
            </small>
          </div>

          <!-- Google Auth -->
          <div v-else-if="securityConfig.auth_level === 'google'" class="auth-method">
            <button class="btn btn-google" @click="$emit('authenticate-google')">
              <span class="google-icon">🔑</span>
              Authenticate with Google
            </button>
          </div>

          <!-- Cloudflare Tunnel Auth -->
          <div v-else-if="securityConfig.auth_level === 'cloudflare'" class="auth-method">
            <div class="info-panel">
              <p>Authentication is handled by Cloudflare Access.</p>
              <p>You should already be authenticated if accessing via Cloudflare tunnel.</p>
            </div>
            <button class="btn btn-primary" @click="$emit('verify-cloudflare')">
              Verify Access
            </button>
          </div>
        </div>

        <div v-if="securityConfig.auth_level !== 'cloudflare'" class="auth-actions">
          <button
            class="btn btn-primary"
            :disabled="authenticating || !canAuthenticate"
            @click="$emit('authenticate')"
          >
            {{ authenticating ? 'Verifying...' : 'Unlock Control' }}
          </button>
        </div>

        <div v-if="authError" class="alert alert-danger">
          {{ authError }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface ManualControlSecurityConfig {
  auth_level: 'password' | 'totp' | 'google' | 'cloudflare'
  session_timeout_minutes: number
  require_https: boolean
  auto_lock_manual_control: boolean
}

defineProps<{
  securityConfig: ManualControlSecurityConfig
  authenticating: boolean
  canAuthenticate: boolean
  authError: string
  formatSecurityLevel: (level: ManualControlSecurityConfig['auth_level']) => string
}>()

defineEmits<{
  authenticate: []
  'authenticate-google': []
  'verify-cloudflare': []
}>()

const password = defineModel<string>('password', { default: '' })
const totpCode = defineModel<string>('totpCode', { default: '' })
</script>

<style scoped>
/*
 * Shared card-shell primitives duplicated from ControlView.vue — Vue scoped
 * CSS only stamps the parent scope-id onto a child's ROOT element (here,
 * .security-gate itself, via that root-leak), not nested elements. The
 * .card/.card-header/.card-body/.btn/.alert* rules below are also kept in
 * ControlView.vue itself since its own remaining template still uses them.
 */
.card {
  background: var(--secondary-dark);
  border: 1px solid var(--primary-light);
  border-radius: 8px;
  margin-bottom: 2rem;
}

.card-header {
  background: var(--primary-dark);
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--primary-light);
  border-radius: 8px 8px 0 0;
}

.card-header h3 {
  margin: 0;
  color: var(--accent-green);
  font-size: 1.25rem;
}

.card-body {
  padding: 1.5rem;
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 4px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-primary {
  background: var(--accent-green);
  color: var(--primary-dark);
}

.btn:hover:not(:disabled) {
  transform: translateY(-2px);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.alert {
  padding: 1rem;
  border-radius: 4px;
  margin-top: 2rem;
}

.alert-danger {
  background: rgba(255, 67, 67, 0.1);
  border: 1px solid #ff4343;
  color: #ff4343;
}

/* Security-gate-specific rules (moved verbatim from ControlView.vue). */
.security-gate {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 60vh;
}

.security-card {
  max-width: 500px;
  width: 100%;
}

.security-info {
  margin-bottom: 2rem;
}

.security-level {
  margin-top: 1rem;
}

.level-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 4px;
  font-weight: 600;
  font-size: 0.875rem;
}

.level-password {
  background: rgba(255, 193, 7, 0.2);
  color: #ffc107;
  border: 1px solid #ffc107;
}

.level-totp {
  background: rgba(0, 123, 255, 0.2);
  color: #007bff;
  border: 1px solid #007bff;
}

.level-google {
  background: rgba(220, 53, 69, 0.2);
  color: #dc3545;
  border: 1px solid #dc3545;
}

.level-cloudflare {
  background: rgba(0, 255, 146, 0.2);
  color: var(--accent-green);
  border: 1px solid var(--accent-green);
}

.auth-methods {
  margin-bottom: 2rem;
}

.auth-method {
  margin-bottom: 1.5rem;
}

.form-control {
  width: 100%;
  padding: 0.75rem;
  background: var(--primary-dark);
  border: 1px solid var(--primary-light);
  border-radius: 4px;
  color: var(--text-color);
  font-size: 1rem;
}

.form-control:focus {
  outline: none;
  border-color: var(--accent-green);
  box-shadow: 0 0 0 2px rgba(0, 255, 146, 0.2);
}

.totp-input {
  text-align: center;
  font-family: monospace;
  font-size: 1.5rem;
  letter-spacing: 0.5rem;
}

.form-text {
  font-size: 0.875rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
}

.btn-google {
  background: #db4437;
  color: white;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.info-panel {
  background: var(--primary-dark);
  padding: 1rem;
  border-radius: 4px;
  margin-bottom: 1rem;
}
</style>

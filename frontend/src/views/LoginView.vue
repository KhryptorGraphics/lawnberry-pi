<template>
  <div class="login-container">
    <div class="login-card">
      <div class="login-header">
        <img :src="iconUrl" alt="LawnBerry Pi" class="login-logo">
        <h1>LawnBerry Pi v2</h1>
        <p>Autonomous Lawn Care System</p>
      </div>
      
      <form class="login-form" @submit.prevent="handleLogin">
        <div class="form-group">
          <label class="form-label" for="username">Username</label>
          <input
            id="username"
            v-model="credentials.username"
            type="text"
            class="form-control"
            :class="{ error: errors.username }"
            required
            :disabled="isLoading"
          >
          <div v-if="errors.username" class="form-error">{{ errors.username }}</div>
        </div>
        
        <div class="form-group">
          <label class="form-label" for="password">Password</label>
          <input
            id="password"
            v-model="credentials.password"
            type="password"
            class="form-control"
            :class="{ error: errors.password }"
            required
            :disabled="isLoading"
          >
          <div v-if="errors.password" class="form-error">{{ errors.password }}</div>
        </div>
        
        <div v-if="authError" class="form-error text-center" role="alert">
          <strong>{{ authError }}</strong>
        </div>
        
        <button
          type="submit"
          class="btn btn-primary w-100"
          :disabled="isLoading"
          :aria-label="isLoading ? 'Signing in, please wait' : 'Sign in to your account'"
        >
          <span v-if="isLoading" class="spinner" aria-hidden="true" />
          {{ isLoading ? 'Signing In...' : 'Sign In' }}
        </button>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import type { LoginCredentials } from '@/types/auth'

const iconUrl = '/branding/LawnBerryPi_icon2.png'

const router = useRouter()
const authStore = useAuthStore()

const isLoading = ref(false)
const authError = ref('')

const credentials = reactive<LoginCredentials>({
  username: '',
  password: ''
})

const errors = reactive({
  username: '',
  password: ''
})

const validateForm = () => {
  errors.username = ''
  errors.password = ''
  
  if (!credentials.username.trim()) {
    errors.username = 'Username is required'
  }
  
  if (!credentials.password) {
    errors.password = 'Password is required'
  }
  
  return !errors.username && !errors.password
}

const handleLogin = async () => {
  authError.value = ''
  
  if (!validateForm()) {
    return
  }
  
  try {
    isLoading.value = true
    
    const success = await authStore.login(credentials)
    
    if (success) {
      router.push('/')
    } else {
      authError.value = authStore.error || 'Login failed'
    }
  } catch (error: any) {
    authError.value = error.message || 'Login failed'
  } finally {
    isLoading.value = false
  }
}
</script>

<style scoped>
.login-container {
  min-height: 60vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
  font-family: 'Courier New', 'Consolas', monospace;
}

.login-card {
  background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 30%, #16213e 70%, #0a0a0a 100%);
  border: 2px solid #00ffff;
  border-radius: 8px;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  box-shadow:
    0 8px 32px rgba(0, 255, 255, 0.3),
    0 0 20px rgba(0, 255, 255, 0.2),
    inset 0 1px 0 rgba(255, 255, 255, 0.1),
    inset 0 0 30px rgba(0, 255, 255, 0.05);
  padding: 2rem;
  width: 100%;
  max-width: 400px;
}

.login-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, #00ffff, transparent);
  animation: borderScan 3s linear infinite;
}

@keyframes borderScan {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

.login-header {
  text-align: center;
  margin-bottom: 2rem;
}

.login-logo {
  height: 64px;
  width: auto;
  margin-bottom: 1rem;
  filter: drop-shadow(0 0 10px #00ffff);
}

.login-header h1 {
  margin-bottom: 0.5rem;
  color: #00ffff;
  font-size: 1.4rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 2px;
  text-shadow: 0 0 10px rgba(0, 255, 255, 0.7);
}

.login-header p {
  color: rgba(0, 255, 255, 0.6);
  margin-bottom: 0;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.login-form {
  margin-bottom: 1.5rem;
}

.login-form .form-label {
  color: #00ffff;
  font-family: inherit;
  font-weight: 700;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 2px;
}

.login-form .form-control {
  background: #0a0a0a;
  border: 1px solid #2d3748;
  border-radius: 4px;
  color: #e6f0ff;
  font-family: inherit;
  min-height: 44px;
}

.login-form .form-control::placeholder {
  color: #9db0c6;
}

.login-form .form-control:focus {
  border-color: #00ff92;
  box-shadow: 0 0 0 2px rgba(0, 255, 146, 0.2);
}

.login-form .form-control.error {
  border-color: #ff0040;
}

.login-form .form-control.error:focus {
  box-shadow: 0 0 0 2px rgba(255, 0, 64, 0.2);
}

.login-form .form-error {
  color: #ff0040;
  font-family: inherit;
}

.login-form .btn-primary {
  background: linear-gradient(135deg, #1a1a2e, #16213e, #0f0f23);
  border: 2px solid #00ffff;
  color: #00ffff;
  font-family: inherit;
  font-weight: 700;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 2px;
  border-radius: 6px;
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 15px rgba(0, 255, 255, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.login-form .btn-primary:hover:not(:disabled) {
  background: linear-gradient(135deg, #00ffff, #0a0a0a);
  color: #000;
  box-shadow: 0 0 20px rgba(0, 255, 255, 0.8);
}

.login-form .btn-primary:focus-visible {
  outline: 2px solid #00ff92;
  outline-offset: 2px;
}

.login-form .spinner {
  border: 2px solid rgba(0, 255, 255, 0.2);
  border-top-color: #00ffff;
}

@media (max-width: 480px) {
  .login-card {
    padding: 1.5rem;
  }

  .login-header h1 {
    font-size: 1.2rem;
  }
}
</style>
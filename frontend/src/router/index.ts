import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'Dashboard',
      component: () => import('@/views/DashboardView.vue')
    },
    {
      path: '/rtk',
      name: 'RtkDiagnostics',
      component: () => import('@/views/RtkDiagnosticsView.vue')
    },
    {
      path: '/control',
      name: 'Control',
      component: () => import('@/views/ControlView.vue')
    },
    {
      path: '/tractor',
      name: 'Tractor',
      component: () => import('@/views/TractorControlView.vue')
    },
    {
      path: '/maps',
      name: 'Maps',
      component: () => import('@/views/MapsView.vue')
    },
    {
      path: '/planning',
      name: 'Planning',
      component: () => import('@/views/PlanningView.vue')
    },
    {
      path: '/settings',
      name: 'Settings',
      component: () => import('@/views/SettingsView.vue')
    },
    {
      path: '/ai',
      name: 'AI',
      component: () => import('@/views/AIView.vue')
    },
    {
      path: '/telemetry',
      name: 'Telemetry',
      component: () => import('@/views/TelemetryView.vue')
    },
    {
      path: '/docs',
      name: 'DocsHub',
      component: () => import('@/views/DocsHubView.vue')
    },
    {
      path: '/mission-planner',
      name: 'MissionPlanner',
      component: () => import('@/views/MissionPlannerView.vue')
    },
    {
      path: '/login',
      name: 'Login',
      component: () => import('@/views/LoginView.vue')
    }
  ]
})

// Auth is not required for this deployment (LAN-only, single operator) --
// no login gate, every route is open.
router.beforeEach((to, from, next) => {
  try { (window as any).__TopProgress?.start() } catch {
    // ignore: TopProgress may not be mounted yet
  }
  next()
})

router.afterEach(() => {
  try { (window as any).__TopProgress?.done() } catch {
    // ignore: TopProgress may not be mounted yet
  }
})

export default router

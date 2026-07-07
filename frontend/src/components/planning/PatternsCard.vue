<template>
  <div class="patterns-card">
    <div class="card">
      <div class="card-header">
        <h3>Mowing Patterns</h3>
      </div>
      <div class="card-body">
        <div class="patterns-grid">
          <div
            v-for="pattern in patterns"
            :key="pattern.id"
            class="pattern-card"
            :class="{ selected: selectedPattern === pattern.id }"
            @click="$emit('update:selectedPattern', pattern.id)"
          >
            <div class="pattern-preview">
              <div class="pattern-visual" :class="`pattern-${pattern.id}`" />
            </div>
            <div class="pattern-info">
              <h4>{{ pattern.name }}</h4>
              <p>{{ pattern.description }}</p>
              <div class="pattern-stats">
                <span>Efficiency: {{ pattern.efficiency }}%</span>
                <span>Coverage: {{ pattern.coverage }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Presentational card for the "Patterns" tab. State/logic lives in
// PlanningView.vue; selection is two-way via v-model:selected-pattern.
interface MowPattern {
  id: string
  name: string
  description: string
  efficiency: number
  coverage: number
}

defineProps<{
  patterns: MowPattern[]
  selectedPattern: string
}>()

defineEmits<{
  'update:selectedPattern': [id: string]
}>()
</script>

<style scoped>
/*
 * Shared card-shell primitives duplicated from PlanningView.vue — see
 * JobsCard.vue for why (Vue scoped CSS only stamps the parent scope-id onto a
 * child's ROOT element). Keep in sync with PlanningView.vue and sibling cards.
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
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h3 {
  margin: 0;
  color: var(--accent-green);
  font-size: 1.25rem;
}

.card-body {
  padding: 1.5rem;
}

/* Patterns-card-specific rules (moved verbatim from PlanningView.vue). */
.patterns-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.pattern-card {
  background: var(--primary-dark);
  border: 1px solid var(--primary-light);
  border-radius: 8px;
  padding: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.pattern-card:hover {
  border-color: var(--accent-green);
}

.pattern-card.selected {
  border-color: var(--accent-green);
  background: rgba(0, 255, 146, 0.1);
}

.pattern-preview {
  height: 100px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1rem;
  background: var(--secondary-dark);
  border-radius: 4px;
}

.pattern-visual {
  width: 60px;
  height: 60px;
  border-radius: 4px;
}

.pattern-parallel {
  background: linear-gradient(to right, var(--accent-green) 0%, transparent 0%, transparent 20%, var(--accent-green) 20%, var(--accent-green) 40%, transparent 40%, transparent 60%, var(--accent-green) 60%, var(--accent-green) 80%, transparent 80%);
}

.pattern-spiral {
  background: radial-gradient(circle, transparent 30%, var(--accent-green) 30%, var(--accent-green) 40%, transparent 40%);
  border: 2px solid var(--accent-green);
  border-radius: 50%;
}

.pattern-random {
  background: var(--accent-green);
  clip-path: polygon(20% 0%, 80% 0%, 100% 20%, 80% 40%, 100% 60%, 75% 100%, 25% 100%, 0% 80%, 20% 60%, 0% 40%);
}

.pattern-edge {
  border: 3px solid var(--accent-green);
  position: relative;
}

.pattern-edge::after {
  content: '';
  position: absolute;
  top: 20%;
  left: 20%;
  right: 20%;
  bottom: 20%;
  background: var(--accent-green);
  opacity: 0.5;
}

.pattern-info h4 {
  margin: 0 0 0.5rem 0;
  color: var(--text-color);
}

.pattern-info p {
  font-size: 0.875rem;
  color: var(--text-muted);
  margin-bottom: 1rem;
}

.pattern-stats {
  display: flex;
  gap: 1rem;
  font-size: 0.875rem;
  color: var(--text-muted);
}

@media (max-width: 768px) {
  .patterns-grid {
    grid-template-columns: 1fr;
  }
}
</style>

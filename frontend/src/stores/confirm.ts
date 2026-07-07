import { defineStore } from 'pinia'

export interface ConfirmOptions {
  message: string
  title?: string
  confirmLabel?: string
  cancelLabel?: string
  danger?: boolean
}

interface PendingConfirm extends ConfirmOptions {
  resolve: (value: boolean) => void
}

export const useConfirmStore = defineStore('confirm', {
  state: () => ({
    pending: null as PendingConfirm | null,
  }),
  actions: {
    ask(options: ConfirmOptions | string): Promise<boolean> {
      const opts = typeof options === 'string' ? { message: options } : options
      return new Promise((resolve) => {
        this.pending = { ...opts, resolve }
      })
    },
    confirm() {
      this.pending?.resolve(true)
      this.pending = null
    },
    cancel() {
      this.pending?.resolve(false)
      this.pending = null
    },
  },
})

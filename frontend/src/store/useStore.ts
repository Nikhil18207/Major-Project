import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ReportHistoryItem, GenerationSettings, ModelStatus, GeneratedReport } from '../types'

interface AppState {
  // Model Status
  modelStatus: ModelStatus | null
  setModelStatus: (status: ModelStatus | null) => void

  // Generation Settings
  settings: GenerationSettings
  updateSettings: (settings: Partial<GenerationSettings>) => void

  // Report History
  history: ReportHistoryItem[]
  addToHistory: (item: ReportHistoryItem) => void
  updateHistoryItem: (id: string, updates: Partial<ReportHistoryItem>) => void
  removeFromHistory: (id: string) => void
  clearHistory: () => void

  // Current Report
  currentReport: GeneratedReport | null
  setCurrentReport: (report: GeneratedReport | null) => void
  currentImage: string | null
  setCurrentImage: (image: string | null) => void

  // UI State
  isGenerating: boolean
  setIsGenerating: (value: boolean) => void
  sidebarOpen: boolean
  setSidebarOpen: (value: boolean) => void
}

const useStore = create<AppState>()(
  persist(
    (set) => ({
      // Model Status
      modelStatus: null,
      setModelStatus: (status) => set({ modelStatus: status }),

      // Generation Settings
      settings: {
        maxLength: 256,
        numBeams: 4,
        temperature: 1.0,
        doSample: false,
      },
      updateSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        })),

      // Report History
      history: [],
      addToHistory: (item) =>
        set((state) => ({
          history: [item, ...state.history].slice(0, 50), // Keep last 50
        })),
      updateHistoryItem: (id, updates) =>
        set((state) => ({
          history: state.history.map((item) =>
            item.id === id ? { ...item, ...updates } : item
          ),
        })),
      removeFromHistory: (id) =>
        set((state) => ({
          history: state.history.filter((item) => item.id !== id),
        })),
      clearHistory: () => set({ history: [] }),

      // Current Report
      currentReport: null,
      setCurrentReport: (report) => set({ currentReport: report }),
      currentImage: null,
      setCurrentImage: (image) => set({ currentImage: image }),

      // UI State
      isGenerating: false,
      setIsGenerating: (value) => set({ isGenerating: value }),
      sidebarOpen: true,
      setSidebarOpen: (value) => set({ sidebarOpen: value }),
    }),
    {
      name: 'xr2text-storage',
      partialize: (state) => ({
        settings: state.settings,
        history: state.history,
      }),
    }
  )
)

export default useStore

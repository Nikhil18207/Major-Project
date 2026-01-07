/**
 * Settings Component with HAQT-ARR Information
 *
 * Features:
 * - Generation parameter configuration
 * - Model information display
 * - HAQT-ARR architecture details (Novel)
 * - GPU memory monitoring
 *
 * Authors: S. Nikhil, Dadhania Omkumar
 */

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import toast from 'react-hot-toast'
import {
  Settings as SettingsIcon,
  Sliders,
  Cpu,
  RefreshCw,
  Check,
  Info,
  Brain,
  Target,
} from 'lucide-react'
import { cn } from '../utils/cn'
import { getModelInfo, getModelStatus } from '../services/api'
import useStore from '../store/useStore'
import type { ModelInfo, ModelStatus } from '../types'
import { ANATOMICAL_REGION_DISPLAY } from '../types'

export default function Settings() {
  const { settings, updateSettings } = useStore()
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    fetchModelData()
  }, [])

  const fetchModelData = async () => {
    setLoading(true)
    try {
      const [info, status] = await Promise.all([
        getModelInfo(),
        getModelStatus(),
      ])
      setModelInfo(info)
      setModelStatus(status)
    } catch (error) {
      toast.error('Failed to fetch model information')
    } finally {
      setLoading(false)
    }
  }

  const handleSave = () => {
    setSaved(true)
    toast.success('Settings saved')
    setTimeout(() => setSaved(false), 2000)
  }

  const isHAQTARR = modelInfo?.projection_type?.includes('HAQT-ARR') ?? false

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
        <p className="mt-1 text-gray-500">
          Configure generation parameters and view model information
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Generation Settings */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-500 to-blue-600 flex items-center justify-center">
              <Sliders className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-xl font-semibold text-gray-900">
              Generation Settings
            </h2>
          </div>

          <div className="space-y-6">
            {/* Max Length */}
            <SettingSlider
              label="Maximum Length"
              description="Maximum number of tokens in generated report"
              value={settings.maxLength}
              min={50}
              max={512}
              step={10}
              onChange={(value) => updateSettings({ maxLength: value })}
            />

            {/* Num Beams */}
            <SettingSlider
              label="Beam Search Width"
              description="Number of beams for beam search (higher = better quality, slower)"
              value={settings.numBeams}
              min={1}
              max={8}
              step={1}
              onChange={(value) => updateSettings({ numBeams: value })}
            />

            {/* Temperature */}
            <SettingSlider
              label="Temperature"
              description="Controls randomness in generation (lower = more focused)"
              value={settings.temperature}
              min={0.1}
              max={2.0}
              step={0.1}
              onChange={(value) => updateSettings({ temperature: value })}
            />

            {/* Do Sample */}
            <SettingToggle
              label="Use Sampling"
              description="Enable sampling instead of greedy/beam search"
              checked={settings.doSample}
              onChange={(checked) => updateSettings({ doSample: checked })}
            />

            {/* Save Button */}
            <button
              onClick={handleSave}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {saved ? (
                <>
                  <Check size={20} />
                  Saved!
                </>
              ) : (
                <>
                  <SettingsIcon size={20} />
                  Save Settings
                </>
              )}
            </button>
          </div>
        </motion.div>

        {/* Model Information */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
                <Cpu className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  Model Information
                </h2>
                {isHAQTARR && (
                  <p className="text-xs text-purple-600">HAQT-ARR Enabled</p>
                )}
              </div>
            </div>

            <button
              onClick={fetchModelData}
              disabled={loading}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              title="Refresh"
            >
              <RefreshCw
                size={20}
                className={cn('text-gray-500', loading && 'animate-spin')}
              />
            </button>
          </div>

          {loading ? (
            <div className="flex items-center justify-center h-64">
              <div className="spinner w-8 h-8" />
            </div>
          ) : modelInfo && modelStatus ? (
            <div className="space-y-4">
              <InfoRow label="Model Name" value={modelInfo.model_name} />
              <InfoRow label="Vision Encoder" value={modelInfo.encoder} />
              <InfoRow label="Text Decoder" value={modelInfo.decoder} />
              <InfoRow
                label="Projection Type"
                value={isHAQTARR ? 'HAQT-ARR (Novel)' : 'Standard'}
                highlight={isHAQTARR}
              />
              <InfoRow
                label="Projection Queries"
                value={modelInfo.projection_queries.toString()}
              />
              <InfoRow
                label="Total Parameters"
                value={`${(modelInfo.total_parameters / 1e6).toFixed(1)}M`}
              />
              <InfoRow
                label="Trainable Parameters"
                value={`${(modelInfo.trainable_parameters / 1e6).toFixed(1)}M`}
              />
              <InfoRow label="Device" value={modelStatus.device} />
              <InfoRow label="Mode" value={modelStatus.mode} />

              {modelStatus.gpu && (
                <>
                  <div className="pt-4 border-t border-gray-100">
                    <h3 className="text-sm font-medium text-gray-900 mb-3">
                      GPU Information
                    </h3>
                  </div>
                  <InfoRow label="GPU Name" value={modelStatus.gpu.name} />
                  <InfoRow
                    label="Memory Allocated"
                    value={`${modelStatus.gpu.memory_allocated_gb.toFixed(2)} GB`}
                  />
                  <InfoRow
                    label="Memory Reserved"
                    value={`${modelStatus.gpu.memory_reserved_gb.toFixed(2)} GB`}
                  />
                  <InfoRow
                    label="Total Memory"
                    value={`${modelStatus.gpu.memory_total_gb.toFixed(2)} GB`}
                  />

                  {/* Memory Usage Bar */}
                  <div className="pt-2">
                    <div className="flex justify-between text-xs text-gray-500 mb-1">
                      <span>GPU Memory Usage</span>
                      <span>
                        {(
                          (modelStatus.gpu.memory_allocated_gb /
                            modelStatus.gpu.memory_total_gb) *
                          100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-primary-500 to-blue-500 transition-all duration-500"
                        style={{
                          width: `${
                            (modelStatus.gpu.memory_allocated_gb /
                              modelStatus.gpu.memory_total_gb) *
                            100
                          }%`,
                        }}
                      />
                    </div>
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              <Info size={48} className="mx-auto mb-4 opacity-50" />
              <p>Unable to load model information</p>
              <p className="text-sm">Make sure the API server is running</p>
            </div>
          )}
        </motion.div>
      </div>

      {/* HAQT-ARR Information Card */}
      {isHAQTARR && modelInfo?.anatomical_regions && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                HAQT-ARR Architecture
              </h2>
              <p className="text-xs text-gray-500">
                Hierarchical Anatomical Query Tokens with Adaptive Region Routing
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Novel Components */}
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Target size={16} className="text-purple-600" />
                Novel Components
              </h3>
              <div className="space-y-3">
                <FeatureItem
                  title="Spatial Prior Generator"
                  description="Learnable 2D Gaussian priors for anatomical locations"
                />
                <FeatureItem
                  title="Anatomical Query Tokens"
                  description="Hierarchical queries: 8 global + 7×4 region-specific"
                />
                <FeatureItem
                  title="Adaptive Region Router"
                  description="Dynamic weighting of anatomical region importance"
                />
                <FeatureItem
                  title="Cross-Region Interaction"
                  description="Transformer layers modeling inter-region dependencies"
                />
              </div>
            </div>

            {/* Anatomical Regions */}
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Target size={16} className="text-purple-600" />
                Anatomical Regions ({modelInfo.anatomical_regions.length})
              </h3>
              <div className="grid grid-cols-2 gap-2">
                {modelInfo.anatomical_regions.map((region) => {
                  const displayInfo = ANATOMICAL_REGION_DISPLAY[region]
                  return (
                    <div
                      key={region}
                      className="flex items-center gap-2 p-2 rounded-lg bg-gray-50"
                    >
                      <div
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: displayInfo?.color || '#6B7280' }}
                      />
                      <span className="text-xs font-medium text-gray-700">
                        {displayInfo?.name || region}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {/* Research Note */}
          <div className="mt-6 p-4 rounded-xl bg-purple-50 border border-purple-100">
            <p className="text-sm text-purple-800">
              <strong>Research Contribution:</strong> HAQT-ARR is a novel projection layer
              that introduces anatomical awareness through learnable spatial priors and
              region-specific query tokens, enabling more clinically-focused report generation.
            </p>
          </div>
        </motion.div>
      )}
    </div>
  )
}

function SettingSlider({
  label,
  description,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  description: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="text-sm font-medium text-gray-900">{label}</label>
        <span className="text-sm text-primary-600 font-medium">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
      />
      <p className="text-xs text-gray-500 mt-1">{description}</p>
    </div>
  )
}

function SettingToggle({
  label,
  description,
  checked,
  onChange,
}: {
  label: string
  description: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <label className="text-sm font-medium text-gray-900">{label}</label>
        <p className="text-xs text-gray-500">{description}</p>
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={cn(
          'w-12 h-6 rounded-full transition-colors relative',
          checked ? 'bg-primary-600' : 'bg-gray-300'
        )}
      >
        <div
          className={cn(
            'absolute top-1 w-4 h-4 bg-white rounded-full transition-transform',
            checked ? 'translate-x-7' : 'translate-x-1'
          )}
        />
      </button>
    </div>
  )
}

function InfoRow({
  label,
  value,
  highlight = false,
}: {
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
      <span className="text-sm text-gray-500">{label}</span>
      <span className={cn(
        'text-sm font-medium',
        highlight ? 'text-purple-600' : 'text-gray-900'
      )}>
        {value}
      </span>
    </div>
  )
}

function FeatureItem({
  title,
  description,
}: {
  title: string
  description: string
}) {
  return (
    <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
      <h4 className="text-sm font-medium text-gray-900">{title}</h4>
      <p className="text-xs text-gray-500 mt-0.5">{description}</p>
    </div>
  )
}

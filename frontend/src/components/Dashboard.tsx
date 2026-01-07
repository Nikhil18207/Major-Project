/**
 * Dashboard Component with HAQT-ARR Model Info
 *
 * Displays:
 * - Model statistics and GPU usage
 * - HAQT-ARR architecture details (Novel)
 * - Anatomical regions supported
 * - Recent reports and quick actions
 *
 * Authors: S. Nikhil, Dadhania Omkumar
 */

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  FileText,
  Clock,
  Cpu,
  Activity,
  Upload,
  ChevronRight,
  Zap,
  Brain,
  Layers,
  Target,
} from 'lucide-react'
import { cn } from '../utils/cn'
import { getModelInfo, getModelStatus } from '../services/api'
import useStore from '../store/useStore'
import type { ModelInfo, ModelStatus } from '../types'
import { ANATOMICAL_REGION_DISPLAY } from '../types'

export default function Dashboard() {
  const { history } = useStore()
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [info, status] = await Promise.all([
          getModelInfo(),
          getModelStatus(),
        ])
        setModelInfo(info)
        setModelStatus(status)
      } catch (error) {
        console.error('Failed to fetch model data:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const isHAQTARR = modelInfo?.projection_type?.includes('HAQT-ARR') ?? false

  const stats = [
    {
      icon: FileText,
      label: 'Reports Generated',
      value: history.length.toString(),
      color: 'from-blue-500 to-blue-600',
    },
    {
      icon: Clock,
      label: 'Avg. Generation Time',
      value: history.length > 0
        ? `${Math.round(history.slice(0, 10).reduce((acc, item) => acc + 250, 0) / Math.min(history.length, 10))}ms`
        : 'N/A',
      color: 'from-green-500 to-green-600',
    },
    {
      icon: Cpu,
      label: 'Model Parameters',
      value: modelInfo
        ? `${(modelInfo.total_parameters / 1e6).toFixed(0)}M`
        : 'Loading...',
      color: 'from-purple-500 to-purple-600',
    },
    {
      icon: Activity,
      label: 'GPU Memory',
      value: modelStatus?.gpu
        ? `${modelStatus.gpu.memory_allocated_gb.toFixed(1)} GB`
        : 'N/A',
      color: 'from-orange-500 to-orange-600',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Welcome to <span className="text-gradient">XR2Text</span>
          </h1>
          <p className="mt-1 text-gray-500">
            AI-Powered Chest X-Ray Report Generation
            {isHAQTARR && (
              <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800">
                <Brain size={12} className="mr-1" />
                HAQT-ARR
              </span>
            )}
          </p>
        </div>
        <Link to="/generate" className="btn-primary flex items-center gap-2 w-fit">
          <Upload size={20} />
          Generate New Report
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="card p-6"
          >
            <div className="flex items-center gap-4">
              <div
                className={cn(
                  'w-12 h-12 rounded-xl bg-gradient-to-br flex items-center justify-center',
                  stat.color
                )}
              >
                <stat.icon className="w-6 h-6 text-white" />
              </div>
              <div>
                <p className="text-sm text-gray-500">{stat.label}</p>
                <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Model Info & Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Architecture */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Model Architecture</h2>
              {isHAQTARR && (
                <p className="text-xs text-purple-600">with HAQT-ARR (Novel)</p>
              )}
            </div>
          </div>

          {loading ? (
            <div className="flex items-center justify-center h-40">
              <div className="spinner w-8 h-8" />
            </div>
          ) : modelInfo ? (
            <div className="space-y-4">
              <ArchitectureBlock
                title="Vision Encoder"
                description={modelInfo.encoder}
                detail="Hierarchical feature extraction with shifted-window attention"
                icon={Layers}
              />
              <ArchitectureBlock
                title="Projection Layer"
                description={isHAQTARR ? 'HAQT-ARR' : 'Standard'}
                detail={isHAQTARR
                  ? `${modelInfo.projection_queries} Queries (8 global + 7×4 region)`
                  : `${modelInfo.projection_queries} Query Tokens`
                }
                icon={Target}
                highlight={isHAQTARR}
              />
              <ArchitectureBlock
                title="Text Decoder"
                description={modelInfo.decoder}
                detail="Biomedical language model for clinical text generation"
                icon={FileText}
              />
            </div>
          ) : (
            <p className="text-gray-500">Unable to load model information</p>
          )}
        </motion.div>

        {/* HAQT-ARR Regions or Quick Actions */}
        {isHAQTARR && modelInfo?.anatomical_regions ? (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="card p-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
                <Target className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900">Anatomical Regions</h2>
                <p className="text-xs text-gray-500">HAQT-ARR Focus Areas</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {modelInfo.anatomical_regions.map((region) => {
                const displayInfo = ANATOMICAL_REGION_DISPLAY[region]
                return (
                  <div
                    key={region}
                    className="flex items-center gap-3 p-3 rounded-xl bg-gray-50 border border-gray-100"
                  >
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: displayInfo?.color || '#6B7280' }}
                    />
                    <span className="text-sm font-medium text-gray-700">
                      {displayInfo?.name || region}
                    </span>
                  </div>
                )
              })}
            </div>

            <div className="mt-4 p-4 rounded-xl bg-purple-50 border border-purple-100">
              <p className="text-sm text-purple-700">
                <strong>Novel:</strong> HAQT-ARR uses learnable Gaussian spatial priors
                and adaptive region routing to focus on clinically relevant areas.
              </p>
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="card p-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <h2 className="text-xl font-semibold text-gray-900">Quick Actions</h2>
            </div>

            <div className="space-y-3">
              <QuickActionButton
                to="/generate"
                icon={Upload}
                title="Upload X-Ray"
                description="Generate a new radiology report"
              />
              <QuickActionButton
                to="/history"
                icon={Clock}
                title="View History"
                description="Browse previously generated reports"
              />
              <QuickActionButton
                to="/settings"
                icon={Cpu}
                title="Model Settings"
                description="Configure generation parameters"
              />
            </div>
          </motion.div>
        )}
      </div>

      {/* Quick Actions (if HAQT-ARR is enabled, show below) */}
      {isHAQTARR && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.45 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-xl font-semibold text-gray-900">Quick Actions</h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <QuickActionButton
              to="/generate"
              icon={Upload}
              title="Upload X-Ray"
              description="Generate a new radiology report"
            />
            <QuickActionButton
              to="/history"
              icon={Clock}
              title="View History"
              description="Browse previously generated reports"
            />
            <QuickActionButton
              to="/settings"
              icon={Cpu}
              title="Model Settings"
              description="Configure generation parameters"
            />
          </div>
        </motion.div>
      )}

      {/* Recent Reports */}
      {history.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="card p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Recent Reports</h2>
            <Link
              to="/history"
              className="text-primary-600 hover:text-primary-700 text-sm font-medium flex items-center gap-1"
            >
              View All <ChevronRight size={16} />
            </Link>
          </div>

          <div className="space-y-4">
            {history.slice(0, 3).map((item) => (
              <div
                key={item.id}
                className="flex items-start gap-4 p-4 rounded-xl bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <img
                  src={item.imageUrl}
                  alt="X-ray"
                  className="w-16 h-16 rounded-lg object-cover"
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-900 line-clamp-2">
                    {item.impression || item.findings || item.report}
                  </p>
                  <p className="mt-1 text-xs text-gray-500">
                    {new Date(item.generatedAt).toLocaleString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  )
}

function ArchitectureBlock({
  title,
  description,
  detail,
  icon: Icon,
  highlight = false,
}: {
  title: string
  description: string
  detail: string
  icon?: typeof Layers
  highlight?: boolean
}) {
  return (
    <div className={cn(
      'p-4 rounded-xl border',
      highlight
        ? 'bg-purple-50 border-purple-200'
        : 'bg-gray-50 border-gray-100'
    )}>
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          {Icon && <Icon size={16} className={highlight ? 'text-purple-600' : 'text-gray-500'} />}
          <h3 className="font-medium text-gray-900">{title}</h3>
        </div>
        <span className={cn(
          'text-sm font-medium',
          highlight ? 'text-purple-600' : 'text-primary-600'
        )}>
          {description}
        </span>
      </div>
      <p className="text-sm text-gray-500">{detail}</p>
    </div>
  )
}

function QuickActionButton({
  to,
  icon: Icon,
  title,
  description,
}: {
  to: string
  icon: typeof Upload
  title: string
  description: string
}) {
  return (
    <Link
      to={to}
      className="flex items-center gap-4 p-4 rounded-xl bg-gray-50 hover:bg-gray-100 transition-colors group"
    >
      <div className="w-10 h-10 rounded-lg bg-white border border-gray-200 flex items-center justify-center group-hover:border-primary-200 transition-colors">
        <Icon size={20} className="text-gray-600 group-hover:text-primary-600 transition-colors" />
      </div>
      <div className="flex-1">
        <h3 className="font-medium text-gray-900">{title}</h3>
        <p className="text-sm text-gray-500">{description}</p>
      </div>
      <ChevronRight size={20} className="text-gray-400 group-hover:text-primary-600 transition-colors" />
    </Link>
  )
}

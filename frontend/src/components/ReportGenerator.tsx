/**
 * Report Generator Component with HAQT-ARR Visualization
 *
 * Features:
 * - X-ray image upload and preview
 * - AI-powered report generation
 * - HAQT-ARR anatomical attention visualization (Novel)
 * - Report editing and feedback submission
 *
 * Authors: S. Nikhil, Dadhania Omkumar
 */

import { useState, useCallback, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import {
  Upload,
  X,
  FileText,
  Clock,
  Edit3,
  Save,
  Send,
  RefreshCw,
  Copy,
  Check,
  Eye,
  Brain,
} from 'lucide-react'
import { cn } from '../utils/cn'
import { generateReport, submitFeedback, getAttentionVisualization, isHAQTARREnabled } from '../services/api'
import useStore from '../store/useStore'
import type { GeneratedReport, AttentionVisualization, AnatomicalRegion } from '../types'
import { ANATOMICAL_REGION_DISPLAY } from '../types'

export default function ReportGenerator() {
  const { settings, addToHistory, setIsGenerating, isGenerating } = useStore()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [report, setReport] = useState<GeneratedReport | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [editedReport, setEditedReport] = useState('')
  const [copied, setCopied] = useState(false)

  // HAQT-ARR State
  const [attentionData, setAttentionData] = useState<AttentionVisualization | null>(null)
  const [isLoadingAttention, setIsLoadingAttention] = useState(false)
  const [showAttention, setShowAttention] = useState(false)
  const [haqtEnabled, setHaqtEnabled] = useState(false)

  // Check if HAQT-ARR is enabled on mount
  useEffect(() => {
    isHAQTARREnabled().then(setHaqtEnabled)
  }, [])

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setReport(null)
      setEditedReport('')
      setIsEditing(false)
      setAttentionData(null)
      setShowAttention(false)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.dicom', '.dcm'],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB
  })

  const handleGenerate = async () => {
    if (!selectedFile) return

    setIsGenerating(true)
    try {
      const result = await generateReport(selectedFile)
      setReport(result)
      setEditedReport(result.report)

      // Fetch HAQT-ARR attention data if enabled
      if (haqtEnabled) {
        fetchAttentionData()
      }

      // Add to history
      addToHistory({
        id: crypto.randomUUID(),
        imageUrl: previewUrl!,
        report: result.report,
        findings: result.findings,
        impression: result.impression,
        generatedAt: new Date(),
        edited: false,
      })

      toast.success(`Report generated in ${result.generation_time_ms.toFixed(0)}ms`)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to generate report')
    } finally {
      setIsGenerating(false)
    }
  }

  const fetchAttentionData = async () => {
    if (!selectedFile) return

    setIsLoadingAttention(true)
    try {
      const data = await getAttentionVisualization(selectedFile)
      setAttentionData(data)
    } catch (error) {
      console.error('Failed to fetch attention data:', error)
      // Don't show error toast - attention is optional
    } finally {
      setIsLoadingAttention(false)
    }
  }

  const handleSaveEdit = async () => {
    if (!report || editedReport === report.report) {
      setIsEditing(false)
      return
    }

    try {
      await submitFeedback({
        original_report: report.report,
        corrected_report: editedReport,
        feedback_notes: 'Radiologist correction',
      })

      toast.success('Feedback submitted successfully')
      setIsEditing(false)
    } catch (error) {
      toast.error('Failed to submit feedback')
    }
  }

  const handleCopy = () => {
    const textToCopy = isEditing ? editedReport : report?.report
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy)
      setCopied(true)
      toast.success('Copied to clipboard')
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleClear = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setReport(null)
    setEditedReport('')
    setIsEditing(false)
    setAttentionData(null)
    setShowAttention(false)
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Generate Report</h1>
        <p className="mt-1 text-gray-500">
          Upload a chest X-ray image to generate an AI-powered radiology report
          {haqtEnabled && (
            <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800">
              <Brain size={12} className="mr-1" />
              HAQT-ARR Enabled
            </span>
          )}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="space-y-6">
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">X-Ray Image</h2>

            {!selectedFile ? (
              <div
                {...getRootProps()}
                className={cn(
                  'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200',
                  isDragActive
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
                )}
              >
                <input {...getInputProps()} />
                <Upload
                  size={48}
                  className={cn(
                    'mx-auto mb-4 transition-colors',
                    isDragActive ? 'text-primary-500' : 'text-gray-400'
                  )}
                />
                <p className="text-gray-600 font-medium">
                  {isDragActive ? 'Drop the image here' : 'Drag & drop an X-ray image'}
                </p>
                <p className="text-sm text-gray-400 mt-2">
                  or click to browse (PNG, JPG, DICOM)
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative rounded-xl overflow-hidden bg-black">
                  <img
                    src={previewUrl!}
                    alt="Uploaded X-ray"
                    className="w-full h-auto max-h-96 object-contain mx-auto"
                  />
                  <button
                    onClick={handleClear}
                    className="absolute top-2 right-2 p-2 bg-black/50 hover:bg-black/70 rounded-full text-white transition-colors"
                  >
                    <X size={20} />
                  </button>
                </div>

                <div className="flex items-center justify-between text-sm text-gray-500">
                  <span className="truncate">{selectedFile.name}</span>
                  <span>{(selectedFile.size / 1024).toFixed(1)} KB</span>
                </div>

                <button
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className="btn-primary w-full flex items-center justify-center gap-2"
                >
                  {isGenerating ? (
                    <>
                      <RefreshCw size={20} className="animate-spin" />
                      Generating Report...
                    </>
                  ) : (
                    <>
                      <Send size={20} />
                      Generate Report
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Generation Settings Quick Info */}
          <div className="card p-4">
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <Clock size={16} />
              <span>
                Settings: {settings.numBeams} beams, max {settings.maxLength} tokens
              </span>
            </div>
          </div>

          {/* HAQT-ARR Attention Visualization */}
          {haqtEnabled && report && (
            <div className="card p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Brain size={20} className="text-purple-600" />
                  <h2 className="text-lg font-semibold text-gray-900">
                    Anatomical Attention
                  </h2>
                </div>
                <button
                  onClick={() => setShowAttention(!showAttention)}
                  className="btn-secondary py-2 px-3 text-sm flex items-center gap-2"
                >
                  <Eye size={16} />
                  {showAttention ? 'Hide' : 'Show'}
                </button>
              </div>

              <AnimatePresence>
                {showAttention && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    {isLoadingAttention ? (
                      <div className="flex items-center justify-center py-8">
                        <RefreshCw size={24} className="animate-spin text-purple-600" />
                        <span className="ml-2 text-gray-500">Loading attention data...</span>
                      </div>
                    ) : attentionData ? (
                      <AttentionVisualizationPanel data={attentionData} />
                    ) : (
                      <div className="text-center py-8 text-gray-400">
                        <Brain size={32} className="mx-auto mb-2 opacity-50" />
                        <p>Attention data not available</p>
                        <button
                          onClick={fetchAttentionData}
                          className="mt-2 text-purple-600 hover:text-purple-700 text-sm"
                        >
                          Retry
                        </button>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>

              {!showAttention && (
                <p className="text-sm text-gray-500">
                  View which anatomical regions the model focused on during report generation.
                </p>
              )}
            </div>
          )}
        </div>

        {/* Report Section */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <FileText size={20} className="text-primary-600" />
              <h2 className="text-lg font-semibold text-gray-900">Generated Report</h2>
            </div>

            {report && (
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                  title="Copy to clipboard"
                >
                  {copied ? (
                    <Check size={18} className="text-green-600" />
                  ) : (
                    <Copy size={18} className="text-gray-500" />
                  )}
                </button>

                {isEditing ? (
                  <button
                    onClick={handleSaveEdit}
                    className="btn-primary py-2 px-4 text-sm flex items-center gap-2"
                  >
                    <Save size={16} />
                    Save
                  </button>
                ) : (
                  <button
                    onClick={() => setIsEditing(true)}
                    className="btn-secondary py-2 px-4 text-sm flex items-center gap-2"
                  >
                    <Edit3 size={16} />
                    Edit
                  </button>
                )}
              </div>
            )}
          </div>

          <AnimatePresence mode="wait">
            {isGenerating ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center h-64"
              >
                <div className="spinner w-12 h-12 mb-4" />
                <p className="text-gray-500">Analyzing X-ray image...</p>
                <p className="text-sm text-gray-400 mt-1">
                  {haqtEnabled
                    ? 'Using HAQT-ARR anatomical attention...'
                    : 'This may take a few seconds'}
                </p>
              </motion.div>
            ) : report ? (
              <motion.div
                key="report"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-4"
              >
                {/* Findings Section */}
                {(report.findings || isEditing) && (
                  <ReportSection
                    title="FINDINGS"
                    content={report.findings || ''}
                    isEditing={isEditing}
                    editedContent={editedReport}
                    onEditChange={setEditedReport}
                  />
                )}

                {/* Impression Section */}
                {(report.impression || isEditing) && (
                  <ReportSection
                    title="IMPRESSION"
                    content={report.impression || ''}
                    isEditing={isEditing}
                    editedContent={editedReport}
                    onEditChange={setEditedReport}
                  />
                )}

                {/* Full Report (if no sections) */}
                {!report.findings && !report.impression && (
                  <ReportSection
                    title="REPORT"
                    content={report.report}
                    isEditing={isEditing}
                    editedContent={editedReport}
                    onEditChange={setEditedReport}
                  />
                )}

                {/* Generation Info */}
                <div className="pt-4 border-t border-gray-100">
                  <div className="flex items-center justify-between text-sm text-gray-500">
                    <span>Generation Time</span>
                    <span className="font-medium">
                      {report.generation_time_ms.toFixed(0)}ms
                    </span>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center h-64 text-gray-400"
              >
                <FileText size={48} className="mb-4 opacity-50" />
                <p>Upload an X-ray image to generate a report</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}

// Report Section Component
function ReportSection({
  title,
  content,
  isEditing,
  editedContent,
  onEditChange,
}: {
  title: string
  content: string
  isEditing: boolean
  editedContent: string
  onEditChange: (value: string) => void
}) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-primary-600 uppercase tracking-wide">
        {title}
      </h3>
      {isEditing ? (
        <textarea
          value={editedContent}
          onChange={(e) => onEditChange(e.target.value)}
          className="textarea min-h-32"
          placeholder={`Enter ${title.toLowerCase()}...`}
        />
      ) : (
        <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
          {content}
        </p>
      )}
    </div>
  )
}

// HAQT-ARR Attention Visualization Panel (Novel)
function AttentionVisualizationPanel({ data }: { data: AttentionVisualization }) {
  // Sort regions by weight (descending)
  const sortedRegions = data.anatomical_regions
    .map((region, index) => ({
      region: region as AnatomicalRegion,
      weight: data.region_weights[index],
    }))
    .sort((a, b) => b.weight - a.weight)

  const maxWeight = Math.max(...data.region_weights)

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-500">
        Region importance weights from HAQT-ARR adaptive routing:
      </p>

      <div className="space-y-3">
        {sortedRegions.map(({ region, weight }) => {
          const displayInfo = ANATOMICAL_REGION_DISPLAY[region]
          const percentage = (weight / maxWeight) * 100

          return (
            <div key={region} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-gray-700">
                  {displayInfo.name}
                </span>
                <span className="text-gray-500">
                  {(weight * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${percentage}%` }}
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                  className="h-full rounded-full"
                  style={{ backgroundColor: displayInfo.color }}
                />
              </div>
            </div>
          )
        })}
      </div>

      <div className="pt-4 border-t border-gray-100">
        <div className="flex items-center justify-between text-xs text-gray-400">
          <span>Visualization Time</span>
          <span>{data.generation_time_ms.toFixed(0)}ms</span>
        </div>
      </div>
    </div>
  )
}

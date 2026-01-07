import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search,
  Trash2,
  FileText,
  Calendar,
  Edit3,
  Eye,
  X,
  AlertCircle,
} from 'lucide-react'
import { cn } from '../utils/cn'
import useStore from '../store/useStore'
import type { ReportHistoryItem } from '../types'

export default function History() {
  const { history, removeFromHistory, clearHistory } = useStore()
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedItem, setSelectedItem] = useState<ReportHistoryItem | null>(null)
  const [showClearConfirm, setShowClearConfirm] = useState(false)

  const filteredHistory = history.filter((item) => {
    const searchLower = searchQuery.toLowerCase()
    return (
      item.report.toLowerCase().includes(searchLower) ||
      item.findings?.toLowerCase().includes(searchLower) ||
      item.impression?.toLowerCase().includes(searchLower)
    )
  })

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Report History</h1>
          <p className="mt-1 text-gray-500">
            {history.length} reports generated
          </p>
        </div>

        {history.length > 0 && (
          <button
            onClick={() => setShowClearConfirm(true)}
            className="btn-secondary flex items-center gap-2 text-red-600 hover:text-red-700 w-fit"
          >
            <Trash2 size={18} />
            Clear All
          </button>
        )}
      </div>

      {/* Search */}
      {history.length > 0 && (
        <div className="relative">
          <Search
            size={20}
            className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400"
          />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search reports..."
            className="input pl-12"
          />
        </div>
      )}

      {/* History Grid */}
      {history.length === 0 ? (
        <EmptyState />
      ) : filteredHistory.length === 0 ? (
        <NoResults searchQuery={searchQuery} />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <AnimatePresence>
            {filteredHistory.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ delay: index * 0.05 }}
              >
                <HistoryCard
                  item={item}
                  onView={() => setSelectedItem(item)}
                  onDelete={() => removeFromHistory(item.id)}
                />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedItem && (
          <DetailModal
            item={selectedItem}
            onClose={() => setSelectedItem(null)}
          />
        )}
      </AnimatePresence>

      {/* Clear Confirmation Modal */}
      <AnimatePresence>
        {showClearConfirm && (
          <ClearConfirmModal
            onConfirm={() => {
              clearHistory()
              setShowClearConfirm(false)
            }}
            onCancel={() => setShowClearConfirm(false)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

function HistoryCard({
  item,
  onView,
  onDelete,
}: {
  item: ReportHistoryItem
  onView: () => void
  onDelete: () => void
}) {
  return (
    <div className="card overflow-hidden group">
      {/* Image */}
      <div className="relative h-48 bg-black">
        <img
          src={item.imageUrl}
          alt="X-ray"
          className="w-full h-full object-contain"
        />
        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3">
          <button
            onClick={onView}
            className="p-3 bg-white rounded-full hover:bg-gray-100 transition-colors"
          >
            <Eye size={20} className="text-gray-700" />
          </button>
          <button
            onClick={onDelete}
            className="p-3 bg-white rounded-full hover:bg-gray-100 transition-colors"
          >
            <Trash2 size={20} className="text-red-600" />
          </button>
        </div>

        {item.edited && (
          <div className="absolute top-2 right-2 px-2 py-1 bg-primary-500 text-white text-xs rounded-full flex items-center gap-1">
            <Edit3 size={12} />
            Edited
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-4">
        <p className="text-sm text-gray-700 line-clamp-3">
          {item.impression || item.findings || item.report}
        </p>

        <div className="flex items-center gap-2 mt-3 text-xs text-gray-500">
          <Calendar size={14} />
          <span>{new Date(item.generatedAt).toLocaleString()}</span>
        </div>
      </div>
    </div>
  )
}

function DetailModal({
  item,
  onClose,
}: {
  item: ReportHistoryItem
  onClose: () => void
}) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <FileText size={24} className="text-primary-600" />
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Report Details</h2>
              <p className="text-sm text-gray-500">
                {new Date(item.generatedAt).toLocaleString()}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X size={24} className="text-gray-500" />
          </button>
        </div>

        {/* Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6 max-h-[calc(90vh-120px)] overflow-y-auto">
          {/* Image */}
          <div className="bg-black rounded-xl overflow-hidden">
            <img
              src={item.imageUrl}
              alt="X-ray"
              className="w-full h-auto max-h-96 object-contain mx-auto"
            />
          </div>

          {/* Report */}
          <div className="space-y-6">
            {item.findings && (
              <div>
                <h3 className="text-sm font-semibold text-primary-600 uppercase tracking-wide mb-2">
                  Findings
                </h3>
                <p className="text-gray-700 whitespace-pre-wrap">{item.findings}</p>
              </div>
            )}

            {item.impression && (
              <div>
                <h3 className="text-sm font-semibold text-primary-600 uppercase tracking-wide mb-2">
                  Impression
                </h3>
                <p className="text-gray-700 whitespace-pre-wrap">{item.impression}</p>
              </div>
            )}

            {!item.findings && !item.impression && (
              <div>
                <h3 className="text-sm font-semibold text-primary-600 uppercase tracking-wide mb-2">
                  Report
                </h3>
                <p className="text-gray-700 whitespace-pre-wrap">{item.report}</p>
              </div>
            )}

            {item.edited && item.editedReport && (
              <div className="pt-4 border-t border-gray-100">
                <h3 className="text-sm font-semibold text-orange-600 uppercase tracking-wide mb-2">
                  Edited Report
                </h3>
                <p className="text-gray-700 whitespace-pre-wrap">{item.editedReport}</p>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}

function ClearConfirmModal({
  onConfirm,
  onCancel,
}: {
  onConfirm: () => void
  onCancel: () => void
}) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50"
      onClick={onCancel}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6"
      >
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center">
            <AlertCircle size={24} className="text-red-600" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Clear History</h2>
            <p className="text-sm text-gray-500">This action cannot be undone</p>
          </div>
        </div>

        <p className="text-gray-600 mb-6">
          Are you sure you want to delete all report history? This will permanently
          remove all generated reports from your local storage.
        </p>

        <div className="flex justify-end gap-3">
          <button onClick={onCancel} className="btn-secondary">
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="btn-primary bg-red-600 hover:bg-red-700"
          >
            Clear All
          </button>
        </div>
      </motion.div>
    </motion.div>
  )
}

function EmptyState() {
  return (
    <div className="card p-12 text-center">
      <FileText size={48} className="mx-auto text-gray-300 mb-4" />
      <h3 className="text-lg font-medium text-gray-900 mb-2">No reports yet</h3>
      <p className="text-gray-500">
        Generated reports will appear here for easy access
      </p>
    </div>
  )
}

function NoResults({ searchQuery }: { searchQuery: string }) {
  return (
    <div className="card p-12 text-center">
      <Search size={48} className="mx-auto text-gray-300 mb-4" />
      <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
      <p className="text-gray-500">
        No reports matching "{searchQuery}" were found
      </p>
    </div>
  )
}

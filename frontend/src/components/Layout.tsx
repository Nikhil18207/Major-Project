import { useState, useEffect } from 'react'
import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Home,
  FileText,
  History,
  Settings,
  Menu,
  X,
  Activity,
  Cpu,
} from 'lucide-react'
import { cn } from '../utils/cn'
import { checkHealth } from '../services/api'
import useStore from '../store/useStore'

const navItems = [
  { path: '/', icon: Home, label: 'Dashboard' },
  { path: '/generate', icon: FileText, label: 'Generate Report' },
  { path: '/history', icon: History, label: 'History' },
  { path: '/settings', icon: Settings, label: 'Settings' },
]

export default function Layout() {
  const location = useLocation()
  const { sidebarOpen, setSidebarOpen, setModelStatus } = useStore()
  const [isOnline, setIsOnline] = useState(false)
  const [gpuName, setGpuName] = useState<string | null>(null)

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const health = await checkHealth()
        setIsOnline(health.status === 'healthy' && health.model_loaded)
        setGpuName(health.gpu_name)
        setModelStatus({
          model_loaded: health.model_loaded,
          device: health.device || 'cpu',
          mode: 'eval',
          gpu: health.gpu_available
            ? {
                name: health.gpu_name || 'Unknown',
                memory_allocated_gb: 0,
                memory_reserved_gb: 0,
                memory_total_gb: 0,
              }
            : undefined,
        })
      } catch {
        setIsOnline(false)
      }
    }

    checkStatus()
    const interval = setInterval(checkStatus, 30000)
    return () => clearInterval(interval)
  }, [setModelStatus])

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Mobile Header */}
      <header className="lg:hidden fixed top-0 left-0 right-0 z-50 glass border-b border-gray-200">
        <div className="flex items-center justify-between px-4 py-3">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
          <h1 className="text-lg font-semibold text-gradient">XR2Text</h1>
          <StatusIndicator isOnline={isOnline} gpuName={gpuName} />
        </div>
      </header>

      {/* Sidebar */}
      <AnimatePresence>
        {(sidebarOpen || window.innerWidth >= 1024) && (
          <motion.aside
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className={cn(
              'fixed top-0 left-0 z-40 h-screen w-72',
              'bg-white border-r border-gray-200 shadow-xl lg:shadow-none',
              'lg:translate-x-0'
            )}
          >
            <div className="flex flex-col h-full">
              {/* Logo */}
              <div className="flex items-center gap-3 px-6 py-6 border-b border-gray-100">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-blue-600 flex items-center justify-center">
                  <Activity className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gradient">XR2Text</h1>
                  <p className="text-xs text-gray-500">Doctor's Dashboard</p>
                </div>
              </div>

              {/* Navigation */}
              <nav className="flex-1 px-4 py-6 space-y-2">
                {navItems.map((item) => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    onClick={() => window.innerWidth < 1024 && setSidebarOpen(false)}
                    className={({ isActive }) =>
                      cn(
                        'flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200',
                        isActive
                          ? 'bg-primary-50 text-primary-700 font-medium'
                          : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                      )
                    }
                  >
                    <item.icon size={20} />
                    <span>{item.label}</span>
                  </NavLink>
                ))}
              </nav>

              {/* Status */}
              <div className="p-4 mx-4 mb-4 rounded-xl bg-gray-50 border border-gray-100">
                <div className="flex items-center gap-2 mb-2">
                  <Cpu size={16} className="text-gray-500" />
                  <span className="text-sm font-medium text-gray-700">System Status</span>
                </div>
                <StatusIndicator isOnline={isOnline} gpuName={gpuName} showDetails />
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main
        className={cn(
          'transition-all duration-300',
          'pt-16 lg:pt-0',
          'lg:ml-72'
        )}
      >
        <div className="p-6 lg:p-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </div>
      </main>

      {/* Backdrop for mobile */}
      {sidebarOpen && window.innerWidth < 1024 && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  )
}

function StatusIndicator({
  isOnline,
  gpuName,
  showDetails = false,
}: {
  isOnline: boolean
  gpuName: string | null
  showDetails?: boolean
}) {
  return (
    <div className={cn('flex items-center gap-2', showDetails && 'flex-col items-start')}>
      <div className="flex items-center gap-2">
        <div
          className={cn(
            'w-2.5 h-2.5 rounded-full',
            isOnline ? 'bg-green-500 animate-pulse' : 'bg-red-500'
          )}
        />
        <span className={cn('text-sm', isOnline ? 'text-green-700' : 'text-red-700')}>
          {isOnline ? 'Online' : 'Offline'}
        </span>
      </div>
      {showDetails && gpuName && (
        <span className="text-xs text-gray-500 truncate max-w-full">{gpuName}</span>
      )}
    </div>
  )
}
